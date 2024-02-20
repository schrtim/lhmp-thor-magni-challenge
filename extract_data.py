import os
import pandas as pd


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
        return speed_df

    @staticmethod
    def get_tracklets_scenario(path: str, tracklet_len: int) -> pd.DataFrame:
        tracklet_id, tracklets = 0, []
        for run in os.listdir(path):
            run_df = pd.read_csv(os.path.join(path, run), index_col="Time")
            helmets_df = run_df.copy()
            helmets_df = helmets_df[helmets_df["ag_id"].str.startswith("Helmet")]
            helmets_df.loc[:, "filename"] = run
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
                    num_tracklets = len(group) // tracklet_len
                    if num_tracklets == 0:
                        continue
                    for i in range(num_tracklets):
                        tracklet = group.iloc[i * tracklet_len: (i + 1) * tracklet_len]
                        trajectory = tracklet.copy()
                        trajectory = ExtractPredictionDataset.extract_speed(trajectory)
                        trajectory.loc[:, "tracklet_id"] = tracklet_id
                        tracklet_id += 1
                        tracklets.append(trajectory)
        return tracklets
