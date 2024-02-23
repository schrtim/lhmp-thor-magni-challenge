import os
from typing import List, Optional, Union
from PIL import Image

import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict


def get_thor_mapping_roles(trajs_concat: pd.DataFrame, collapse_visitors: bool):
    """return mapping between roles and int"""

    roles = trajs_concat.data_label.unique().tolist()
    mapping = defaultdict(lambda: len(roles))  # if unknown -> new class test set
    mapping.update({role: i for i, role in enumerate(roles)})
    if not collapse_visitors:
        return mapping

    # group visitors in the same label
    first_occurence, new_mapping = None, {}
    for role, numerical_label in mapping.items():
        if role.split("_")[0] == "visitors":
            if first_occurence is not None:
                new_mapping[role] = first_occurence
            else:
                first_occurence = numerical_label
                new_mapping[role] = first_occurence

        else:
            new_mapping[role] = (
                max(-1, max(new_mapping.values())) + 1
                if len(new_mapping.values()) > 0
                else 0
            )
    return new_mapping


class PixWorldConverter:
    """Pixel to world converter"""

    def __init__(self, info: dict) -> None:
        self.resolution = info["resolution_pm"]  # 1pix -> m
        self.offset = np.array(info["offset"])

    def convert2pixels(
        self, world_locations: Union[np.array, torch.Tensor]
    ) -> Union[np.array, torch.Tensor]:
        if world_locations.ndim == 2:
            return (world_locations / self.resolution) - self.offset

        new_world_locations = [
            self.convert2pixels(world_location) for world_location in world_locations
        ]
        return (
            torch.stack(new_world_locations)
            if isinstance(world_locations, torch.Tensor)
            else np.stack(new_world_locations)
        )

    def convert2world(
        self, pix_locations: Union[np.array, torch.Tensor]
    ) -> Union[np.array, torch.Tensor]:
        return (pix_locations + self.offset) * self.resolution


class SimpleMagni(Dataset):
    """Default dataset loader object"""

    def __init__(self, trajectories: List[pd.DataFrame]) -> None:
        super().__init__()
        self.input_data = trajectories

    def convert_to_torch(self, arr: np.array) -> torch.Tensor:
        return torch.from_numpy(arr).type(torch.float)

    def get_common_inputs(self, index):
        target_data = self.input_data[index]
        locations = self.convert_to_torch(target_data[["x", "y", "z"]].values)
        speed = self.convert_to_torch(
            target_data[["x_speed", "y_speed", "z_speed"]].values
        )
        period = target_data.index.to_series().diff()
        period = period.fillna(period.median())
        period = self.convert_to_torch(period.values)
        return dict(
            trajectories=locations,
            speeds=speed,
            period=period,
        )

    def __getitem__(self, index):
        return self.get_common_inputs(index)

    def __len__(self):
        return len(self.input_data)


class HardMagni(SimpleMagni):
    """Loads additioanl features such as the osbtacle map or human roles"""

    def __init__(
        self,
        trajectories: List[pd.DataFrame],
        visuals_path: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(trajectories)
        trajs_concat = pd.concat(self.input_data)
        self.imgs, self.window_size, self.pix2word_converters = {}, None, {}
        self.mapping_roles = get_thor_mapping_roles(
            trajs_concat, collapse_visitors=kwargs["collapse_visitors"]
        )
        if visuals_path:
            self.pix2word_converters = dict(May=None, Spetember=None)
            with open(os.path.join(visuals_path, "offsets.json"), "rb") as f:
                offsets = json.load(f)
            for recording_month in ["May", "September"]:
                self.pix2word_converters[recording_month] = PixWorldConverter(
                    dict(resolution_pm=0.01, offset=offsets[recording_month])
                )

            self.obs_len = kwargs["obs_len"]
            self.window_size = int(kwargs["window_size"] / 0.01)
            map_names = trajs_concat["map_name"].unique()
            for map_name in map_names:
                vis_path = os.path.join(visuals_path, map_name)
                img = np.array(Image.open(vis_path))
                self.imgs[map_name] = np.flipud(img[:, :, :3])

    def get_mapping_cat_vars(self, cat_vars: List[str], mapping: dict):
        return self.convert_to_torch(
            np.array([mapping[cat_var] for cat_var in cat_vars])
        )

    def __getitem__(self, index):
        trajectory_data = self.input_data[index]
        new_inputs = self.get_common_inputs(index)
        roles = trajectory_data["data_label"].values
        new_inputs.update(
            {
                "roles": self.get_mapping_cat_vars(roles, self.mapping_roles),
            }
        )
        if len(self.imgs) > 0:
            map_name = trajectory_data["map_name"].iloc[0]
            recording_month = "September" if map_name.startswith("3009") else "May"
            visual = torch.from_numpy(
                self.imgs[map_name].transpose(2, 0, 1).copy()
            ).float()
            visual /= visual.max()
            trajs_pix = self.pix2word_converters[recording_month].convert2pixels(
                new_inputs["trajectories"][:, :2]
            )
            last_pt = trajs_pix[self.obs_len - 1]
            col_min, col_max = (
                max(0, int(last_pt[0]) - self.window_size),
                min(int(last_pt[0]) + self.window_size, visual.shape[1] - 1),
            )
            row_min, row_max = (
                max(0, int(last_pt[1]) - self.window_size),
                min(int(last_pt[1]) + self.window_size, visual.shape[2] - 1),
            )
            target_visual = torch.zeros((3, self.window_size * 2, self.window_size * 2))
            patch = visual[:, row_min:row_max, col_min:col_max]
            target_visual[:, :patch.shape[1], :patch.shape[2]] = patch
            new_inputs.update(dict(img=target_visual, trajectories_pixels=trajs_pix))
        return new_inputs
