import numpy as np
import pandas as pd
import csv
from tabulate import tabulate


class PnCUserDataset(object):
    """
    The class 'PnCUserDataset' handles the creation and retrieval of an empirical dataset for Point-and-Click (PnC) tasks.
    ==> From "Speeding up Inference with User Simulators through Policy Modulation" (CHI 2022) by Moon et al.
    ==> https://github.com/hsmoon121/pnc-dataset
    """

    def __init__(self):
        self.n_user = 20
        self.max_time = 5.0
        self.max_tv = 0.36
        self.max_tr = 0.024
        self.max_dist = 5 * self.max_tr
        self._get_stat_data()
        self._get_traj_data()

    def _get_stat_data(self):
        """
        Read static (fixed-size) behavioral outputs for every trial from CSV file
        It removes the first block and first trial of each block, normalizes the data and identifies outlier trials.
        """
        df_path = "csv/pnc/empirical_data_stat.csv"
        exp_df = pd.read_csv(df_path)
        self.stat_data = list()
        self.outlier_data = list()
        for user in range(self.n_user):
            # remove 1st block & 1st trial of each block
            user_df = exp_df[
                (exp_df["user"] == user) & (exp_df["task"] > 0) & (exp_df["trial"] > 0)
            ]

            max_values = np.array(
                [
                    1,
                    self.max_time,
                    0.52704,
                    0.29646,
                    1,
                    1,
                    0.52704,
                    0.29646,
                    self.max_tv,
                    self.max_tv,
                    self.max_tr,
                ]
            )
            user_data = (
                user_df[
                    [
                        "success",
                        "time",
                        "c_pos_x",
                        "c_pos_y",
                        "c_vel_x",
                        "c_vel_y",
                        "t_pos_x",
                        "t_pos_y",
                        "t_vel_x",
                        "t_vel_y",
                        "radius",
                    ]
                ].to_numpy(copy=True)
                / max_values
            )
            outlier_idx = np.where((user_data[:, 1] > 1.0) | (user_data[:, 1] < 0.05))[
                0
            ]
            mask = np.ones((user_data.shape[0],), bool)
            mask[outlier_idx] = False
            self.outlier_data.append(outlier_idx)
            self.stat_data.append(user_data[mask, :])

    def _get_traj_data(self):
        """
        Read trajectory (variable-size) behavioral outputs for every trial from pickle file
        """
        df_path = "csv/pnc/empirical_data_traj.pkl"
        traj_df = pd.read_pickle(df_path)
        self.traj_data = list()
        for user in range(self.n_user):
            user_traj = []
            len_traj = []
            click_dist = []
            # remove 1st block & 1st trial of each block
            for bl in range(1, 4):
                for tr in range(1, 200):
                    idx = 199 * (bl - 1) + (tr - 1)
                    if idx not in self.outlier_data[user]:
                        traj = traj_df[user][bl][tr]
                        traj[1:, 0] = traj[1:, 0] - traj[:-1, 0]  # relative time
                        user_traj.append(traj)
                        len_traj.append(len(traj))

                        final_d = ((traj[-1, 1:3] - traj[-1, 3:5]) ** 2).sum() ** 0.5
                        click_dist.append(np.clip(final_d, 0, self.max_dist))

            # append click distance to stat_data
            self.stat_data[user] = np.insert(
                self.stat_data[user], 2, np.array(click_dist) / self.max_dist, axis=-1
            )
            self.traj_data.append(user_traj)

    def _sample_user_trial(self, user, n_trial):
        """
        Sample the specified number of trials for a given user.
        If the requested number of trials is larger than the available data,
        it repeats the data until the required number of trials is reached.
        """
        if n_trial > self.stat_data[user].shape[0]:
            stats = self.stat_data[user]

            trajs = list()
            trajs += self.traj_data[user]
            diff = n_trial - self.stat_data[user].shape[0]
            while diff > self.stat_data[user].shape[0]:
                stats = np.concatenate((stats, self.stat_data[user]), axis=0)
                trajs += self.traj_data[user]
                diff -= self.stat_data[user].shape[0]
            stats = np.concatenate((stats, self.stat_data[user][-diff:]), axis=0)
            trajs += self.traj_data[user][-diff:]
        else:
            stats = self.stat_data[user][-n_trial:]
            trajs = self.traj_data[user][-n_trial:]

        return stats, trajs

    def sample(self, n_trial, n_user=20):

        user_data = list()
        for user in range(n_user):
            stats, trajs = self._sample_user_trial(user, n_trial)
            user_data.append([stats, trajs])

        return user_data

    def read_pickle(self):
        df_path = "csv/pnc/empirical_data_traj.pkl"
        traj_df = pd.read_pickle(df_path)
        print(traj_df[0][0][2])


pnc = PnCUserDataset()
sample = pnc.sample(9000, 20)
print(tabulate(sample[0][0]))  # 0번째 샘플의 stat
print(tabulate(sample[0][1][0]))  # 0번째 샘플의 traj
# print(sample[0][1].shape)
