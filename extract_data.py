import os
import math
import pandas as pd
import numpy as np


def convert_run_to_mapfile(filename: str):
    experiment_info = filename.split("_")
    date, scenario = experiment_info[1][:4], experiment_info[2]
    month = "May" if date[2:] == "05" else "September"

    scenario_map = "SC3" if scenario in ["SC3A", "SC3B"] else scenario

    map_name = (
        "_".join([date, scenario_map, "map"])
        if month == "May"
        else "_".join([date, "map"])
    )
    return map_name + ".png"


class ExtractPredictionDataset:
    @staticmethod
    def get_groups_continuous_tracking(dynamic_agent_data: pd.DataFrame):
        """get groups of continuous tracking/no-tracking"""
        mask = dynamic_agent_data[["x", "y", "x"]].isna().any(axis=1)
        groups = (mask != mask.shift()).cumsum()
        groups_of_continuous_tracking = dynamic_agent_data.groupby(groups)
        return groups_of_continuous_tracking

    @staticmethod
    def extract_speed(trajectory: pd.DataFrame) -> pd.DataFrame:
        """extract speed feature"""
        speed_df = trajectory.copy()
        delta_df = speed_df[["x", "y", "z"]].diff().add_suffix("_delta")
        delta_df.loc[:, "Time_delta"] = delta_df.index.to_series().diff()
        speed_df = speed_df.join(
            delta_df[["Time_delta", "x_delta", "y_delta", "z_delta"]]
        )
        speed_df.loc[:, ["x_speed", "y_speed", "z_speed"]] = (
            speed_df[["x_delta", "y_delta", "z_delta"]]
            .div(speed_df["Time_delta"].values, axis=0)
            .values
        )
        speed_df["2D_speed"] = (
            np.sqrt(np.square(speed_df[["x_delta", "y_delta"]]).sum(axis=1))
            .div(speed_df["Time_delta"].values, axis=0)
            .values
        )
        speed_df["3D_speed"] = (
            np.sqrt(np.square(speed_df[["x_delta", "y_delta", "z_delta"]]).sum(axis=1))
            .div(speed_df["Time_delta"].values, axis=0)
            .values
        )
        return speed_df

    @staticmethod
    def get_tracklets_scenario(
        path: str, tracklet_len: int, min_speed: float, max_speed: float
    ) -> pd.DataFrame:
        tracklet_id, tracklets = 0, []
        for run in os.listdir(path):
            run_df = pd.read_csv(os.path.join(path, run), index_col="Time")
            helmets_df = run_df.copy()
            helmets_df = helmets_df[helmets_df["ag_id"].str.startswith("Helmet")]

            helmets_df.loc[:, "map_name"] = convert_run_to_mapfile(run)
            helmets_df[["x", "y", "z"]] /= 1000
            agents = helmets_df["ag_id"].unique()
            for agent_id in agents:
                groups_of_continuous_tracking = (
                    ExtractPredictionDataset.get_groups_continuous_tracking(
                        helmets_df[helmets_df["ag_id"] == agent_id]
                    )
                )
                for _, group in groups_of_continuous_tracking:
                    if group[["x", "y", "z"]].isna().any(axis=0).all():
                        continue
                    num_tracklets = int(
                        math.ceil((len(group) - tracklet_len + 1) / tracklet_len)
                    )
                    if num_tracklets == 0:
                        continue

                    for i in range(0, num_tracklets * tracklet_len, tracklet_len):
                        tracklet = group.iloc[i: i + tracklet_len]
                        trajectory = tracklet.copy()
                        trajectory = ExtractPredictionDataset.extract_speed(trajectory)
                        if (trajectory["2D_speed"] < min_speed).any() or (
                            trajectory["2D_speed"] > max_speed
                        ).any():
                            continue
                        trajectory = trajectory.fillna(0.0)
                        trajectory.loc[:, "tracklet_id"] = tracklet_id
                        tracklet_id += 1
                        tracklets.append(trajectory)
        return tracklets
