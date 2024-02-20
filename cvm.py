from tqdm import tqdm
import numpy as np
from scipy.signal.windows import gaussian


class ConstantVelocityModel:
    def __init__(self, cfg) -> None:
        self.v0_mode = cfg["v0_mode"]
        self.v0_sigma = cfg["v0_sigma"]
        self.pred_len = cfg["pred_len"]

    def predict_dataset(self, dataset):
        prediction, position, v_mean = [], [], []
        for n_t, trajectory in tqdm(enumerate(dataset)):
            last_position = trajectory.iloc[-1][["x", "y"]]
            mean_velocity = self.get_weighted_displacement(trajectory)
            position.append(last_position)
            v_mean.append(mean_velocity)
        v_mean = np.squeeze(v_mean)
        last_position = np.array(position)
        for _ in range(self.pred_len):
            new_position = last_position + v_mean
            prediction.append(new_position)
            last_position = new_position
        return np.stack(prediction, axis=1).astype(float)

    def get_weighted_displacement(self, trajectory):
        period = trajectory.index.to_series().diff()[1:].median()
        displacements = trajectory[["x_speed", "y_speed"]].values[1:, :] * period
        weights = np.expand_dims(self.get_w(displacements), axis=0)
        new_displacements = np.dot(weights, displacements)
        return new_displacements

    def get_w(self, velocity):
        velocity_len = len(velocity)
        if self.v0_mode == "linear":
            weights = np.ones(velocity_len) / velocity_len
        elif self.v0_mode == "gaussian":
            window = gaussian(2 * velocity_len, self.v0_sigma)
            w1 = window[0:velocity_len]
            scale = np.sum(w1)
            # scale = np.linalg.norm(w1)
            weights = w1 / scale
            # print(w)
        elif self.v0_mode == "constant":
            weights = np.zeros(velocity_len)
            weights[-1] = 1
        else:
            raise NotImplementedError(self.v0_mode)
        return weights
