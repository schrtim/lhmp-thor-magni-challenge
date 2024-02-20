import os
import random 
import matplotlib.pyplot as plt

from extract_data import ExtractPredictionDataset
from cvm import ConstantVelocityModel

from typing import List
import pandas as pd
import numpy as np
import torch

from metrics import AverageDisplacementError, FinalDisplacementError

def evaluate_submission(pred_file, annotation_file, obs_len: int):
    """Evaluate dataset"""
    # data: List[pd.DataFrame]
    ade = AverageDisplacementError()
    fde = FinalDisplacementError()
    
    y_hat_list = pred_file
    track_list = np.load(annotation_file, allow_pickle=True)

    metric_list = []

    for c, scenario_prediction in enumerate(y_hat_list):
        # print(f"Scenario {c+1}")
        y_hat = scenario_prediction

        data = track_list[c]
        gt_dataset = list(map(lambda x: x.iloc[obs_len:][["x", "y"]].values, data))

        y_true = np.stack(gt_dataset).astype(float)

        ade.update(preds=torch.from_numpy(y_hat), target=torch.from_numpy(y_true))
        fde.update(preds=torch.from_numpy(y_hat), target=torch.from_numpy(y_true))
        ade_res = ade.compute().item()
        fde_res = fde.compute().item()

        metrics = [ade_res, fde_res]
        metric_list.append(metrics)

        ade.reset()
        fde.reset()

        # print("\t", ade_res, fde_res)

    ades = np.array([m[0] for m in metric_list])
    fdes = np.array([m[1] for m in metric_list])

    for c, scenario_prediction in enumerate(y_hat_list):
        # print(f"Scenario {c+1}")
        y_hat = scenario_prediction

        data = track_list[c]
        gt_dataset = list(map(lambda x: x.iloc[obs_len:][["x", "y"]].values, data))

        y_true = np.stack(gt_dataset).astype(float)

        ade.update(preds=torch.from_numpy(y_hat), target=torch.from_numpy(y_true))
        fde.update(preds=torch.from_numpy(y_hat), target=torch.from_numpy(y_true))
        ade_res = ade.compute().item()
        fde_res = fde.compute().item()

        metrics = [ade_res, fde_res]
        metric_list.append(metrics)

    total_ade = ade_res
    total_fde = fde_res

    output = {}
    output = [
        {
            "ADE_SC1": round(ades[0],4),
            "FDE_SC1": round(fdes[0],4),
            "ADE_SC2": round(ades[1],4),
            "FDE_SC2": round(fdes[1],4),
            "ADE_SC3": round(ades[2],4),
            "FDE_SC3": round(fdes[2],4),
            "ADE_SC4": round(ades[3],4),
            "FDE_SC4": round(fdes[3],4),
            "ADE_SC5": round(ades[4],4),
            "FDE_SC5": round(fdes[4],4),
            "Total_ADE": round(total_ade, 4),
            "Total_FDE": round(total_fde, 4)
        }
    ]

    return output

def evaluate(test_annotation_file, user_submission_file, **kwargs):
    print("Starting Evaluation.....")

    output = {}

    print("Evaluating for Test Phase")
    output = evaluate_submission(user_submission_file, test_annotation_file, obs_len=8)
    print(output)
    print("Completed evaluation for Test Phase")
    return output
