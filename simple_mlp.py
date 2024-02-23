import os
import json
from typing import List, Union, Dict
from functools import partial

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from metrics import AverageDisplacementError, FinalDisplacementError
from scaler import IOScaler


def dump_json_file(data_to_save: dict, save_path: str):
    """save json file"""
    file_o = open(save_path, "w")
    json.dump(data_to_save, file_o)


def get_prediction_input_data(
    train_batch: dict, obs_len: int, inputs: Union[str, List[str]]
) -> Dict:
    """Split trajectory info for trajectory forecaster"""
    inputs = [inputs] if isinstance(inputs, str) else inputs
    observed_tracklet, gt_tracklet = {}, {}
    for input_type, input_data in train_batch.items():
        if input_type == "img":
            observed_tracklet[input_type] = input_data
            gt_tracklet[input_type] = input_data
            continue
        observed_tracklet[input_type] = input_data[:, :obs_len, ...]
        gt_tracklet[input_type] = input_data[:, obs_len:, ...]
    return observed_tracklet, gt_tracklet


def make_mlp(dim_list: List[int], activation, batch_norm=False, dropout=0):
    """
    Generates MLP network:
    Parameters
    ----------
    dim_list : list, list of number for each layer
    activation_list : list, list containing activation function for each layer
    batch_norm : boolean, use batchnorm at each layer, default: False
    dropout : float [0, 1], dropout probability applied on each layer (except last layer)
    Returns
    -------
    nn.Sequential with layers
    """
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        elif activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif activation == "prelu":
            layers.append(nn.PReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class SimpleMLP(nn.Module):
    def __init__(
        self,
        network_cfg: dict,
        input_type: str | List[str],
    ):
        super().__init__()
        hidden_units = network_cfg["hidden_units"]
        self.input_type = input_type if isinstance(input_type, list) else [input_type]
        mlp_dims = [network_cfg["obs_len"] * network_cfg["n_features"]] + hidden_units
        self.model = make_mlp(
            dim_list=mlp_dims,
            activation=network_cfg["activation"],
            batch_norm=network_cfg["batch_norm"],
            dropout=network_cfg["dropout"],
        )
        self.linear_out = nn.Linear(hidden_units[-1], network_cfg["pred_len"] * 2)

    def forward(self, x: dict) -> torch.Tensor:
        inputs_cat = []
        for feature_name, inputs in x.items():
            if feature_name in self.input_type:
                inputs = inputs if inputs.dim() == 3 else inputs.unsqueeze(dim=-1)
                inputs_cat.append(inputs)

        inputs_cat = torch.cat(inputs_cat, dim=-1)
        bs = inputs.size(0)
        model_input = inputs_cat.view(bs, -1)
        extracted_features = self.model(model_input)
        out = self.linear_out(extracted_features)
        return out.view(bs, -1, 2)


class LightPointPredictor(pl.LightningModule):
    def __init__(
        self,
        data_cfg: dict,
        network_cfg: dict,
        hyperparameters_cfg: dict,
    ) -> None:
        super().__init__()
        saved_hyperparams = dict(
            data=data_cfg, network=network_cfg, hyperparameters=hyperparameters_cfg
        )
        self.save_hyperparameters(saved_hyperparams)
        self.model = SimpleMLP(network_cfg, input_type=data_cfg["inputs"])
        self.hyperparameters_cfg = hyperparameters_cfg
        self.scaler = IOScaler(data_cfg["training_data_stats"])
        self.output_type = data_cfg["output"]
        self.get_prediction_data = partial(
            get_prediction_input_data,
            obs_len=data_cfg["obs_len"],
            inputs=data_cfg["inputs"],
        )

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        opt = optim.Adam(
            self.parameters(),
            lr=float(self.hyperparameters_cfg["lr"]),
            weight_decay=1e-4,
        )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=self.hyperparameters_cfg["scheduler_patience"], min_lr=1e-6
        )
        return [opt], [
            dict(scheduler=lr_scheduler, interval="epoch", monitor="train_loss")
        ]

    def training_step(self, train_batch: dict, batch_idx: int) -> torch.Tensor:
        y_gt, y_hat_unscaled = self.common_step(train_batch)
        trajs_2d = y_gt["trajectories"][:, :, :2]
        loss = F.mse_loss(y_hat_unscaled, trajs_2d)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch: dict, batch_idx: int):
        y_gt, y_hat_unscaled = self.common_step(val_batch)
        trajs_2d = y_gt["trajectories"][:, :, :2]
        self.update_metrics(y_hat_unscaled, trajs_2d)
        val_loss = F.mse_loss(y_hat_unscaled, trajs_2d)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

    def test_step(self, test_batch: dict, batch_idx: int) -> torch.Tensor:
        y_gt, y_hat_unscaled = self.common_step(test_batch)
        trajs_2d = y_gt["trajectories"][:, :, :2]
        self.update_metrics(y_hat_unscaled, trajs_2d)

    def on_validation_start(self) -> None:
        self.eval_metrics = dict(
            ade=AverageDisplacementError(), fde=FinalDisplacementError()
        )

    def on_validation_end(self) -> None:
        save_path = os.path.join(self.logger.log_dir, "val_metrics.json")
        val_metrics = self.compute_metrics()
        dump_json_file(val_metrics, save_path)
        self.reset_metrics()

    def on_test_start(self) -> None:
        self.eval_metrics = dict(
            ade=AverageDisplacementError(), fde=FinalDisplacementError()
        )

    def on_test_end(self) -> None:
        save_path = os.path.join(self.logger.log_dir, "test_metrics.json")
        test_metrics = self.compute_metrics()
        dump_json_file(test_metrics, save_path)
        self.reset_metrics()

    def predict_step(self, predict_batch: dict, batch_idx: int) -> torch.Tensor:
        _, y_hat_unscaled = self.common_step(predict_batch)
        return dict(
            gt=predict_batch["trajectories"][:, :, :2].detach(),
            y_hat=[y_hat_unscaled.detach()],
        )

    def common_step(self, batch: dict):
        obs_tracklet_data, y_gt = self.get_prediction_data(batch)
        scaled_train_batch = self.scaler.scale_inputs(obs_tracklet_data)
        y_hat = self(scaled_train_batch).clone()
        if self.output_type == "trajectories":
            y_hat_unscaled = self.scaler.inv_scale_outputs(y_hat, "trajectories")
        elif self.output_type == "speeds":
            y_hat_unscaled = self.scaler.inv_transform_speeds(y_hat, obs_tracklet_data)
        return y_gt, y_hat_unscaled

    def update_metrics(self, y_hat: torch.Tensor, y_gt: torch.Tensor):
        for _, metric in self.eval_metrics.items():
            metric.update(preds=y_hat, target=y_gt)

    def compute_metrics(self) -> dict:
        return {
            met_name: met.compute().item()
            for met_name, met in self.eval_metrics.items()
        }

    def reset_metrics(self) -> None:
        for _, metric in self.eval_metrics.items():
            metric.reset()
