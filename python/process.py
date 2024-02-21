import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


user_data = pd.read_csv("csv/user.csv")
user_data = user_data[["Prolific Id", "Ppi"]]

filenames = [
    ####5차####
    "t63d3fbb8c9da3aa4f9302827_65bcae8f7ea5963401552d24_1706868972021",
    "t6266ac01a2a15cce6ed5fba7_65bcaeb81c3f3a916ad42bc1_1706868839839",
    "t5fbfd480aa43de2e41ef0e41_65bcaef9e7615038bbbf5b21_1706869187489",
    "t654a2cd129403e4f45596b55_65bcade81a54952b3431e2fb_1706868608660",
    "t63d79e8e6d3d2f2d2ffd694c_65bcb3754340997dc7cf1273_1706869631236",
    # ####6차####
    "t657c0568573c24460b180d14_65c07f325247ddc10833da1b_1707119655183",
    "t5c7dd8e0665aaa001230a6c7_65c07d65602dd61ab23410d8_1707119509880",
    "t650764c3a197814ab15f8237_65c07d60a85ed9190a24d39e_1707117453687",
    "t5dd311f27aa0d6327c0a2bdd_65c07cf4defe30b609615037_1707117887047",
    "t63039f41fa5c21d483996be2_65c0bccc77963cfe49335381_1707134685522",
]


def makeFullFilePaths(filenames):
    fullFilePaths = []

    for name in filenames:
        fullFilePaths.append(
            "C:/Users/soomin/AppData/Local/Google/Cloud SDK/point-and-click-20d4c.appspot.com/trajectory/"
            + name
            + ".csv"
        )
    return fullFilePaths


def csvToDataframe(csv):
    return pd.read_csv(csv)


full_file_paths = makeFullFilePaths(filenames)
df_list = [pd.read_csv(f) for f in full_file_paths]


def interpolate_df(df):
    df["time"] = df["timestamp"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    interpolated_dfs = []

    original_row_count = len(df)
    df = df.drop_duplicates("timestamp", keep="last")
    drop_duplicated_row_count = len(df)
    print("drop :", original_row_count - drop_duplicated_row_count)

    for trial, group in df.groupby("trial"):
        start_time = group["timestamp"].min()
        end_time = group["timestamp"].max()
        time_range = pd.date_range(start=start_time, end=end_time, freq="16.666ms")

        resampled_group = group.set_index("timestamp").reindex(
            time_range, method="nearest", limit=1
        )

        # 선형 보간
        resampled_group["cursor_x"] = resampled_group["cursor_x"].interpolate(
            method="linear"
        )
        resampled_group["cursor_y"] = resampled_group["cursor_y"].interpolate(
            method="linear"
        )
        resampled_group["movementX"] = resampled_group["movementX"].interpolate(
            method="linear"
        )
        resampled_group["movementY"] = resampled_group["movementY"].interpolate(
            method="linear"
        )
        ##nearest
        resampled_group["target_x"] = resampled_group["target_x"].interpolate(
            method="nearest"
        )
        resampled_group["target_y"] = resampled_group["target_y"].interpolate(
            method="nearest"
        )
        resampled_group["target_radius"] = resampled_group["target_radius"].interpolate(
            method="nearest"
        )
        resampled_group["screenX"] = resampled_group["screenX"].interpolate(
            method="nearest"
        )
        resampled_group["screenY"] = resampled_group["screenY"].interpolate(
            method="nearest"
        )
        resampled_group["buttons"] = resampled_group["buttons"].interpolate(
            method="nearest"
        )
        resampled_group["dpr"] = resampled_group["dpr"].interpolate(method="nearest")
        resampled_group["fullscreen"] = resampled_group["fullscreen"].interpolate(
            method="nearest"
        )
        resampled_group["trial"] = trial

        interpolated_dfs.append(
            resampled_group.reset_index().rename(columns={"index": "timestamp"})
        )
    # 모든 보간된 DataFrame을 하나로 병합
    interpolated_dfs = pd.concat(interpolated_dfs).reset_index(drop=True)
    interpolated_dfs["buttons"] = interpolated_dfs["buttons"].shift(-1, fill_value=1)
    return interpolated_dfs


def normalize_df(df):
    df["norm_target_x"] = 2 * (df["target_x"] / df["screenX"].max()) - 1
    df["norm_target_y"] = 2 * (df["target_y"] / df["screenY"].max()) - 1
    df["norm_cursor_x"] = 2 * (df["cursor_x"] / df["screenX"].max()) - 1
    df["norm_cursor_y"] = 2 * (df["cursor_y"] / df["screenY"].max()) - 1
    df["norm_target_radius"] = 2 * (df["target_radius"] / 1) - 1
    normalized_df = df
    return normalized_df


def physical_df(df, ppi):
    center_x = df["screenX"].max() / 2
    center_y = df["screenY"].max() / 2

    df["phys_target_radius"] = df["target_radius"] / ppi * 0.0254
    df["phys_target_x"] = (df["target_x"] - center_x) / ppi * 0.0254
    df["phys_target_y"] = (df["target_y"] - center_y) / ppi * 0.0254
    df["phys_cursor_x"] = (df["cursor_x"] - center_x) / ppi * 0.0254
    df["phys_cursor_y"] = (df["cursor_y"] - center_y) / ppi * 0.0254
    df["phys_screenX"] = df["screenX"] / ppi * 0.0254
    df["phys_screenY"] = df["screenY"] / ppi * 0.0254
    return df


def add_user_col_df(df, id):
    df["user"] = id
    return df


def remove_abnormal_trial(df):
    # 타겟 재생성
    df["target_change"] = (df["target_x"] != df["target_x"].shift(1)) | (
        df["target_y"] != df["target_y"].shift(1)
    )
    df["actual_trial"] = df["target_change"].cumsum()
    df["target_change"] = df["target_change"].astype(int)
    # 각 trial의 마지막 블럭 번호 찾기
    last_blocks = df.groupby("trial")["actual_trial"].max().reset_index()

    # 마지막 블럭의 데이터만 필터링
    filtered_trajectory_df = pd.DataFrame()
    for index, row in last_blocks.iterrows():
        trial = row["trial"]
        block = row["actual_trial"]
        last_block_data = df[(df["trial"] == trial) & (df["actual_trial"] == block)]
        filtered_trajectory_df = pd.concat(
            [filtered_trajectory_df, last_block_data], ignore_index=True
        )
    return filtered_trajectory_df


def trim_df(df):
    df.columns = df.columns.str.replace(
        "(?<=[a-z])(?=[A-Z])", "_", regex=True
    ).str.lower()
    df.columns = df.columns.str.replace("[ ]", "_", regex=True)

    return df


def add_time(df):
    dfs = []
    df["time"] = pd.to_datetime(df["timestamp"]).astype("int64") / 1000 / 1000

    for trial, group in df.groupby("trial"):
        start_time = group["time"].min()
        group["time"] = group["time"] - start_time
        dfs.append(group.reset_index())
    # 모든 보간된 DataFrame을 하나로 병합
    dfs = pd.concat(dfs).reset_index(drop=True)
    return dfs


result_df = pd.DataFrame()

for i in range(len(filenames)):
    # for i in range(1):
    df = df_list[i]
    filename = filenames[i].split("_")[0].split("t")[1]
    user = user_data.loc[user_data["Prolific Id"] == filename, ["Ppi"]]
    ppi = user.iloc[0]["Ppi"]

    df = remove_abnormal_trial(df)
    df = interpolate_df(df)
    df = normalize_df(df)
    df = physical_df(df, ppi)
    df = add_user_col_df(df, i)
    df = trim_df(df)
    df = add_time(df)

    print(df)
    result_df = pd.concat([df, result_df])
    df.to_csv("csv/process/trajectory/" + str(i) + "_" + filename + ".csv", index=False)


# df = df[
#     [
#         "timestamp",
#         "trial",
#         "target_radius",
#         "target_x",
#         "target_y",
#         "cursor_x",
#         "cursor_y",
#         "movementX",
#         "movementY",
#         "screenX",
#         "screenY",
#         "buttons",
#         "dpr",
#         "fullscreen",
#     ]
# ]
# df.to_csv("csv/original.csv")
