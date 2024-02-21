import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import os
from copy import deepcopy
from tabulate import tabulate
import math
from datetime import datetime
import random
from time import time
from tqdm import tqdm
from joblib import Parallel, delayed
from enum import Enum, auto

# 처음 Userdataset class 선언해서
# 초기화될때 그냥 undersample 다 시켜놓고
# 데이터 sample (불러오기)에선 미리 undersample된것 그냥 바로 가져올 수 있게


class StrEnum(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
        return str(name).lower()

    def __repr__(self):
        return str(self.name).lower()

    def __str__(self):
        return str(self.name).lower()


class DfType(StrEnum):
    STAT = auto()
    TRAJ = auto()


class Stat(StrEnum):
    TRIAL = auto()
    SUCCESS = auto()
    TARGET_P = auto()
    TOTAL_P = auto()
    TARGET_RADIUS = auto()
    ID = auto()
    W = auto()
    D = auto()
    SKIPPED = auto()
    INACCURATE = auto()
    GEN = auto()
    USER = auto()
    A = auto()
    B = auto()
    HEIGHT = auto()
    LEFT = auto()
    MEASURE_DPI = auto()
    POINTER_WEIGHT = auto()
    PPI = auto()
    TIMESTAMP = auto()
    TOP = auto()
    WIDTH = auto()
    TRIAL_COMPLETION_TIME = auto()


class Traj(StrEnum):
    TRIAL = auto()
    TARGET_RADIUS = auto()
    TARGET_X = auto()
    TARGET_Y = auto()
    CURSOR_X = auto()
    CURSOR_Y = auto()
    MOVEMENT_X = auto()
    MOVEMENT_Y = auto()
    SCREEN_X = auto()
    SCREEN_Y = auto()
    BUTTONS = auto()
    DPR = auto()
    FULLSCREEN = auto()
    TARGET_CHANGE = auto()
    ACTUAL_TRIAL = auto()
    NORM_TARGET_X = auto()
    NORM_TARGET_Y = auto()
    NORM_CURSOR_X = auto()
    NORM_CURSOR_Y = auto()
    NORM_TARGET_RADIUS = auto()
    PHYS_TARGET_RADIUS = auto()
    PHYS_TARGET_X = auto()
    PHYS_TARGET_Y = auto()
    PHYS_CURSOR_X = auto()
    PHYS_CURSOR_Y = auto()
    PHYS_SCREEN_X = auto()
    PHYS_SCREEN_Y = auto()
    USER = auto()
    TIME = auto()


class UserDataset:
    def __init__(self, undersample_rate=40, n_jobs=-1):
        ##TQDM써서 progress bar 띄우기
        USER_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        TOTAL_TRIAL = 1

        self.merged_summary_df = pd.DataFrame()
        self.result_df = []
        self.undersample_rate = undersample_rate
        self.n_jobs = n_jobs

        self.load_merged_summary_csv("csv/process/merged_summary.csv")
        trajs_df = self._init_sample(
            user_list=USER_LIST,
            trial_list=list(range(0, TOTAL_TRIAL)),
        )
        self.trajs_df = trajs_df

    def load_merged_summary_csv(self, file_path):
        merged_summary_df = pd.read_csv(file_path)
        self.merged_summary_df = merged_summary_df.fillna(0)

    def merge_trajectory_summary_df(self):
        merged_df = pd.merge(
            self.trajectory_df, self.summary_df, on="trial", how="inner"
        )
        self.merged_df = merged_df

    def search_summary_by_user_and_trial(self, df, user, trial):
        summary_info_df = df[df["user"] == user]
        summary_info_df = summary_info_df[summary_info_df["trial"] == trial]
        return summary_info_df

    def search_trajectory_by_trial(self, df, trial):
        trial_info_df = df[df["trial"] == trial]

        return trial_info_df

    def load_trajectory_by_user(self, user):
        trajectory_file_path = "csv/process/trajectory"
        file_list = os.listdir(trajectory_file_path)
        matched_files = [
            file_name for file_name in file_list if file_name.startswith(str(user))
        ]
        file_path = "csv/process/trajectory/" + matched_files[0]
        df = pd.read_csv(file_path)
        return df

    def _search_trajectory_by_user_and_trial(self, user_list, trial_list):
        trajs_df = pd.DataFrame()
        trajs_df_list = []

        if isinstance(user_list, int):
            user_list = [user_list]

        if isinstance(trial_list, int):
            trial_list = [trial_list]

        for i in tqdm(range(len(user_list))):
            user = user_list[i]
            single_user_df = self.load_trajectory_by_user(user)

            for column in single_user_df.columns:
                if column != "timestamp":
                    single_user_df[column] = single_user_df[column].astype("float32")

            for j in range(len(trial_list)):
                trial = trial_list[j]
                searched_single_user_df = self.search_trajectory_by_trial(
                    single_user_df, trial
                )
                searched_single_user_df = self.undersampling_group_rate(
                    searched_single_user_df, self.undersample_rate
                )

                trajs_df_list.append(searched_single_user_df)

        trajs_df = pd.concat(trajs_df_list).reset_index(drop=True)

        return trajs_df

    def _search_trajectory_by_user_and_trial2(self, user_list, trial_list):
        if isinstance(user_list, int):
            user_list = [user_list]

        if isinstance(trial_list, int):
            trial_list = [trial_list]

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.process_single_user_trial)(user, trial_list)
            for user in tqdm(user_list)
        )
        trajs_df = pd.concat(results).reset_index(drop=True)

        return trajs_df

    def merge_summary_to_single_csv(self):
        merged_summary_df = []
        summary_file_path = "csv/raw/summary"
        trajectory_file_path = "csv/process/trajectory"
        file_list = os.listdir(summary_file_path)
        file_count = len(file_list)
        user_info_df = pd.read_csv("csv/user.csv")

        for i in range(file_count):
            filename = file_list[i]
            index, id = self.get_user_index_and_id_by_filename(filename)
            summary_df = pd.read_csv(summary_file_path + "/" + filename)
            trajectory_df = pd.read_csv(trajectory_file_path + "/" + filename)

            summary_df["user"] = index
            summary_df["prolific_id"] = id
            summary_df = summary_df[
                (summary_df["skipped"] == 0) & (summary_df["inaccurate"] == 0)
            ]
            summary_df = summary_df.drop("reaction_time", axis=1)
            row_count = len(summary_df)

            summary_df = pd.merge(
                summary_df,
                user_info_df,
                left_on="prolific_id",
                right_on="Prolific Id",
                how="left",
            )
            summary_df = self.calculate_trial_completion_time(summary_df, trajectory_df)
            # skipped, inaccurate 삭제하고 row 갯수 900개 확인
            if row_count == 900:
                merged_summary_df.append(summary_df)
                print("succ")
            else:
                print("err.")
                break

        merged_summary_df = pd.concat(merged_summary_df).reset_index(drop=True)
        return merged_summary_df

    def get_user_index_and_id_by_filename(self, filename):
        user_split_arr = filename.split("_")
        index = user_split_arr[0]
        id = user_split_arr[1].split(".")[0]
        return index, id

    def calculate_trial_completion_time(self, summary_df, trajectory_df):
        trajectory_df["timestamp"] = pd.to_datetime(trajectory_df["timestamp"])
        start_times = (
            trajectory_df.groupby("trial")["timestamp"].min().rename("start_time")
        )
        end_times = trajectory_df.groupby("trial")["timestamp"].max().rename("end_time")

        # 시작과 종료 시간을 summary_df에 병합
        summary_df = summary_df.merge(start_times, on="trial", how="left")
        summary_df = summary_df.merge(end_times, on="trial", how="left")

        # trial completion time 계산 (종료 시간 - 시작 시간)
        summary_df["trial_completion_time"] = (
            summary_df["end_time"] - summary_df["start_time"]
        ).dt.total_seconds() * 1000
        return summary_df

    def process_single_user_trial(self, user, trial_list):
        single_user_df = self.load_trajectory_by_user(user)
        for column in single_user_df.columns:
            if column != "timestamp":
                single_user_df[column] = single_user_df[column].astype("float32")

        trajs_df_list = []
        for trial in trial_list:
            searched_single_user_df = self.search_trajectory_by_trial(
                single_user_df, trial
            )
            searched_single_user_df = self.undersampling_group_rate(
                searched_single_user_df, self.undersample_rate
            )
            trajs_df_list.append(searched_single_user_df)

        return pd.concat(trajs_df_list).reset_index(drop=True)

    def undersampling_group_rate(self, group, freq):
        group = group.copy()
        group = group.drop(columns=["time", "index"])
        group["timestamp"] = pd.to_datetime(group["timestamp"])

        freq = round(1000 / freq, 5)
        start_time = group["timestamp"].min()
        end_time = group["timestamp"].max()

        time_range = pd.date_range(
            start=start_time, end=end_time, freq=str(freq) + "ms"
        )
        resampled_group = group.set_index("timestamp").reindex(
            time_range, method="nearest", limit=1
        )

        linear_columns = [
            "target_x",
            "target_y",
            "cursor_x",
            "cursor_y",
            "movement_x",
            "movement_y",
            "norm_cursor_x",
            "norm_cursor_y",
            "norm_target_x",
            "norm_target_y",
            "norm_target_radius",
            "phys_cursor_x",
            "phys_cursor_y",
            "phys_target_x",
            "phys_target_y",
            "phys_target_radius",
            "phys_screen_x",
            "phys_screen_y",
        ]
        for col in linear_columns:
            if col in resampled_group.columns:
                resampled_group[col] = resampled_group[col].interpolate(method="linear")

        nearest_columns = set(resampled_group.columns) - set(linear_columns)
        for col in nearest_columns:
            resampled_group[col] = resampled_group[col].interpolate(method="nearest")
        resampled_group = resampled_group.reset_index().rename(
            columns={"index": "timestamp"}
        )
        resampled_group["time"] = pd.to_datetime(resampled_group["timestamp"]).astype(
            "int64"
        )
        return resampled_group

    def trim_df(self, df):
        df.columns = df.columns.str.replace(
            "(?<=[a-z])(?=[A-Z])", "_", regex=True
        ).str.lower()
        df.columns = df.columns.str.replace("[ ]", "_", regex=True)

        return df

    # def convert(self, df):
    #     transformed_data = {}
    #     for (user, trial), group in df.groupby(["user", "trial"]):
    #         trial_data = group[
    #             [
    #                 "timestamp_x",
    #                 "norm_cursor_x",
    #                 "norm_cursor_y",
    #                 "norm_target_x",
    #                 "norm_target_y",
    #             ]
    #         ].values.tolist()
    #         if user not in transformed_data:
    #             transformed_data[user] = {}
    #         if 0 not in transformed_data[user]:
    #             transformed_data[user][0] = {}
    #         transformed_data[user][0][trial] = trial_data
    #     return transformed_data

    def _init_sample(self, user_list, trial_list):
        trajs_df = self._search_trajectory_by_user_and_trial2(user_list, trial_list)
        return trajs_df

    def sample_user_trial(
        self, stats_df, trajs_df, user_list, trial_list, select_cols, drop_cols
    ):
        if isinstance(user_list, int):
            user_list = [user_list]

        if isinstance(trial_list, int):
            trial_list = [trial_list]

        stat_list = []
        traj_list = []

        for i in tqdm(range(len(user_list))):
            user = user_list[i]
            single_user_df = trajs_df[trajs_df["user"] == user].copy()

            # for column in single_user_df.columns:
            #     if column != "timestamp":
            #         single_user_df[column] = single_user_df[column].astype("float32")

            for j in range(len(trial_list)):
                trial = trial_list[j]
                searched_summary_df = self.search_summary_by_user_and_trial(
                    self.merged_summary_df, user, trial
                )
                searched_single_user_df = self.search_trajectory_by_trial(
                    single_user_df, trial
                )
                searched_single_user_df = searched_single_user_df.drop(
                    "timestamp", axis=1
                )

                searched_summary_df = searched_summary_df.drop(
                    [
                        "prolific_id",
                        "prolific_id.1",
                        "user.1",
                        "date",
                        "start_time",
                        "end_time",
                    ],
                    axis=1,
                )
                print(len(select_cols[DfType.STAT]) > 0)
                print(drop_cols == False)
                if len(select_cols[DfType.STAT]) > 0 and drop_cols == False:
                    print("what")
                    searched_summary_df = searched_summary_df[select_cols[DfType.STAT]]
                elif len(select_cols[DfType.STAT]) > 0 and drop_cols:
                    searched_summary_df = searched_summary_df.drop(
                        select_cols[DfType.STAT], axis=1
                    )

                if len(select_cols[DfType.TRAJ]) > 0 and drop_cols == False:
                    searched_single_user_df = searched_single_user_df[
                        select_cols[DfType.TRAJ]
                    ]
                elif len(select_cols[DfType.TRAJ]) > 0 and drop_cols:
                    searched_single_user_df = searched_single_user_df.drop(
                        select_cols[DfType.TRAJ], axis=1
                    )

                print(tabulate(searched_summary_df, headers="keys"))
                print(tabulate(searched_single_user_df, headers="keys"))

                summary = searched_summary_df.to_numpy()[0]
                traj = searched_single_user_df.to_numpy()
                summary.astype(np.float32)
                traj = traj.astype(np.float32)

                stat_list.append(summary)
                traj_list.append(traj)

        stats = np.array(stat_list)
        trajs = np.array(traj_list, dtype=object)

        return stats, trajs

    def sample(
        self,
        user_list,
        trial_list,
        n_trial=None,
        randomize=False,
        select_cols={DfType.STAT: [], DfType.TRAJ: []},
        drop_cols=False,
    ):

        if len(user_list) * len(trial_list) < n_trial:
            print("sampling trial must be greater than len user * len trial")
            return [], []

        stats_df = self.merged_summary_df
        trajs_df = self.trajs_df
        stats, trajs = self.sample_user_trial(
            stats_df, trajs_df, user_list, trial_list, select_cols, drop_cols
        )

        stats = stats[:n_trial]
        trajs = trajs[:n_trial]

        if randomize:
            seed = datetime.now().timestamp()
            random.Random(seed).shuffle(stats)
            random.Random(seed).shuffle(trajs)

        return stats, trajs


# 32bit float
# jax, torch 다음으로 좋은거임

start_t = time()
user_dataset = UserDataset(undersample_rate=40)
print(time() - start_t)
start_t = time()
print(time() - start_t)
start_t = time()
stats, trajs = user_dataset.sample(
    user_list=[0],
    trial_list=list(range(0, 1)),
    n_trial=1,
    select_cols={DfType.STAT: [Stat.A, Stat.B], DfType.TRAJ: [Traj.DPR]},
    drop_cols=True,
)


"""
#####STATS RETURN#####
trial, success, target_p, total_p, target_radius, id, w, d, skipped, inaccurate,
gen, user, a, b, height, left, measure_dpi, pointer_weight, ppi, timestamp,
top, width, trial_completion_time

#####TRAJS RETURN#####
trial, target_radius, target_x, target_y, cursor_x, cursor_y, movement_x, movement_y, screen_x, screen_y,
buttons, dpr, fullscreen, target_change, actual_trial, norm_target_x, norm_target_y, norm_cursor_x, norm_cursor_y, norm_target_radius,
phys_target_radius, phys_target_x, phys_target_y, phys_cursor_x, phys_cursor_y, phys_screen_x, phys_screen_y, user, time
"""

# print(len(stats))
# print(len(trajs))
