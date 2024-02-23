import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeaturesScaler:
    """Standard scaling of vecotorized positions and velocities"""

    @staticmethod
    def scale_dataset(data: pd.DataFrame):
        """standard scaling features of a trajectories dataset"""
        traj_scaler = StandardScaler()
        vel_scaler = StandardScaler()

        trajectories, velocities = [], []
        for tracklet in data:
            trajectories.append(tracklet[["x", "y", "z"]])
            velocities.append(tracklet[["x_speed", "y_speed", "z_speed"]].values[1:, :])

        trajectories = np.concatenate(trajectories).astype(float)
        velocities = np.concatenate(velocities).astype(float)

        trajectories_scaled = traj_scaler.fit_transform(trajectories)
        velocities_scaled = vel_scaler.fit_transform(velocities)

        statistics = dict(
            trajectories=(traj_scaler.mean_.tolist(), traj_scaler.scale_.tolist()),
            speeds=(vel_scaler.mean_.tolist(), vel_scaler.scale_.tolist()),
        )

        dummy_rows = np.full((1, 3), 0.0)
        traj_len = len(tracklet)
        vel_len = len(tracklet) - 1
        tracklets = []
        for i, _tracklet in enumerate(data):
            tracklet = _tracklet.copy()
            tracklet.loc[:, ["x_scl", "y_scl", "z_scl"]] = trajectories_scaled[
                i * traj_len: (i * traj_len + traj_len)
            ]
            tracklet.loc[:, ["x_speed_scl", "y_speed_scl", "z_speed_scl"]] = np.vstack(
                (
                    dummy_rows,
                    velocities_scaled[i * vel_len: (i * vel_len + vel_len)],
                )
            )
            tracklets.append(tracklet)
        return tracklets, statistics

    def denormalize_feature(self, features_name: str, features: np.array) -> np.array:
        if features_name not in [
            "trajectories",
            "speeds",
        ]:
            raise ValueError(f"{features_name} not available")
        stats = self.statistics[features_name]
        out = features * stats[1] + stats[0]
        return out


class IOScaler:
    """scale inputs/outputs from the network"""

    def __init__(self, statistics: dict):
        self.stats = statistics
        self.available_features = [
            "trajectories",
            "speeds",
        ]

    def _get_stats(self, input_type: str):
        if input_type not in self.available_features:
            raise ValueError(f"{input_type} not in {self.available_features}")
        _mean, _scale = self.stats[input_type]
        _mean = torch.Tensor(_mean).unsqueeze(dim=0).unsqueeze(dim=0)
        _scale = torch.Tensor(_scale).unsqueeze(dim=0).unsqueeze(dim=0)
        return _mean, _scale

    def scale_inputs(self, x: dict) -> dict:
        """scale inputs"""
        out_scaled = {}
        for input_type, input_data in x.items():
            if input_type not in self.stats:
                out_scaled[input_type] = input_data.clone()
                continue
            _mean, _scale = self._get_stats(input_type)
            input_scaled = (input_data.clone() - _mean.to(input_data)) / _scale.to(
                input_data
            )
            out_scaled[input_type] = input_scaled
        return out_scaled

    def inv_scale_outputs(self, x: torch.Tensor, out_type: str) -> torch.Tensor:
        """inverse scaling of outputs"""
        if out_type not in self.stats:
            raise ValueError(f"{out_type} not in statistics")
        _mean, _scale = self._get_stats(out_type)
        out = x.clone() * _scale.to(x)[:, :, :2] + _mean.to(x)[:, :, :2]
        return out

    def inv_transform_speeds(self, speeds: torch.Tensor, observed_tracklet_info: dict):
        """scaled speed -> unscaled speed -> displacements -> locations"""
        trajs_2d = observed_tracklet_info["trajectories"][:, :, :2]
        period, _ = observed_tracklet_info["period"].median(dim=1)
        unscaled_speeds = self.inv_scale_outputs(speeds, out_type="speeds")
        period = period.unsqueeze(dim=1).unsqueeze(dim=2).to(unscaled_speeds)
        displacements = unscaled_speeds * period
        last_observed_points = trajs_2d[:, -1, :].to(displacements)
        displacements[:, 0, :] += last_observed_points
        return torch.cumsum(displacements, dim=1)
