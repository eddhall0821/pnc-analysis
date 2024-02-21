import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

filenames = [
    # "s60f06a479c4f3ec0a536d40a_65b0c816f1a0063d97444363_1706087803561",
    # "s60fc189df1dcc953f098ab5e_65b0bd847e8310dc03e1da0e_1706087495184",  # 0 거르는애
    # "s610d42ce297f0c4f3542f0c8_65b0be3085ee685874c28b8c_1706086330675", #dpi 오류
    # "s614dc3a3eb1768cb811daeae_65b0bd955fabdb2f869173a4_1706089464234",
    # "s61743b31d0f9de288cc272e6_65b0bf1e5fabdb2f869173f1_1706086255010",
    ####2차####
    # "s5f51587f36e91f38fbf34c23_65b33fb1f2d71e32a9c89fb2_1706249456679",
    # "s58bef6fd31de840001e84ba0_65b3468fda5b722e3b0c057b_1706251355218",
    # "s5daa50967776b10016e1bc9b_65b34c204096e2e85ec37254_1706253560814",
    # "s63fe16762b4c19bea77a0904_65b35e5e8eb5426eaf325fa9_1706263232846",
    # "s65492662e16ad03ea732fd62_65b35f91ce60d65be69f50fd_1706259834523",
    ####3차####
    # "s5fd663588215d7567e027d5b_65ba0c1b87ba96dfb72fd127_1706695411009",
    # "s6473924c2cae5ee4db5082d2_65ba0e32888792af73001c22_1706696353716",
    # "s60a6a76aca6c98b970de90f2_65ba1150fdc9a9c61e3b8692_1706696471489",
    # "s5c0bac19796adb00017dffcf_65ba19d3c7eb2481a9ed51e2_1706700437984",
    # "s656851e04160e3d805b2abb5_65ba350f4bc6b2f1d6338dc9_1706707584794",
    # "s__1706841672031"
    ####4차####
    # "s607c5af095d4f618fc21a1d1_65bb6200e17a7058bbe51054_1706784263123",
    # "s60fd093548ae24bc1a355bf7_65bb61e510406bc3436c2b58_1706784119718",
    # "s5c00043a6d931200019bcb9b_65bb67b44c63047973dc8b39_1706786366634",
    # "s628781515f29a0394e23b15b_65bb793c2b5746e34133d5d2_1706791605712",
    # "s63ee65e3470c23cb401ca89a_65bb82703fe600bff8fe9981_1706791869657",
    ##5,6은 사실상 같은 실험인데 그 전은 아님(타겟 생성 조건이 좀 다름)##
    ####5차####
    "s63d3fbb8c9da3aa4f9302827_65bcae8f7ea5963401552d24_1706868972020",
    "s6266ac01a2a15cce6ed5fba7_65bcaeb81c3f3a916ad42bc1_1706868839838",
    "s5fbfd480aa43de2e41ef0e41_65bcaef9e7615038bbbf5b21_1706869187489",
    "s654a2cd129403e4f45596b55_65bcade81a54952b3431e2fb_1706868608660",
    "s63d79e8e6d3d2f2d2ffd694c_65bcb3754340997dc7cf1273_1706869631235",
    ####6차####
    "s657c0568573c24460b180d14_65c07f325247ddc10833da1b_1707119655182",
    "s5c7dd8e0665aaa001230a6c7_65c07d65602dd61ab23410d8_1707119509880",
    "s650764c3a197814ab15f8237_65c07d60a85ed9190a24d39e_1707117453686",
    "s5dd311f27aa0d6327c0a2bdd_65c07cf4defe30b609615037_1707117887046",
    "s63039f41fa5c21d483996be2_65c0bccc77963cfe49335381_1707134685521",
]


def makeFullFilePaths(filenames):
    fullFilePaths = []

    for name in filenames:
        fullFilePaths.append(
            "C:/Users/soomin/AppData/Local/Google/Cloud SDK/point-and-click-20d4c.appspot.com/summary/"
            + name
            + ".csv"
        )
    print(fullFilePaths)
    return fullFilePaths


full_file_paths = makeFullFilePaths(filenames)
reward_array = [0, 50, 100]

df = pd.concat([pd.read_csv(f) for f in full_file_paths], ignore_index=True)


# graph1 함수
def plot_graph1(df):
    df = df[df["reaction_time"] < 3000]
    # df = df[(df["skipped"] != 1) | (df["inaccurate"] != 1)]
    # df = df[(df["trial"] >= 0) & (df["trial"] <= 299)]

    colors = {0: "blue", 50: "darkorange", 100: "green"}
    for target_value in reward_array:
        subset = df[df["target_p"] == target_value]
        plt.scatter(
            subset["id"],
            subset["reaction_time"],
            color=colors[target_value],
            label=f"Target pence: {target_value / 10}",
            alpha=0.7,
        )
        slope, intercept, _, _, _ = linregress(subset["id"], subset["reaction_time"])
        plt.plot(
            subset["id"], intercept + slope * subset["id"], color=colors[target_value]
        )
        # plt.ylim(0, 1)
    plt.title("Trial Completion Time vs ID with Linear Regression by Target Reward")
    plt.xlabel("ID")
    plt.ylabel("Trial Completion Time")
    plt.legend()
    plt.grid(True)


# graph2 함수
def plot_graph2(df):
    df["target_p_scaled"] = df["target_p"] / 10
    # df = df[(df["skipped"] != 1) | (df["inaccurate"] != 1)]
    # df = df[(df["trial"] >= 0) & (df["trial"] <= 299)]

    error_rates_by_target_p = df.groupby("target_p_scaled")["success"].apply(
        lambda x: 1 - x.mean()
    )
    bin_size = 1
    df["id_bin"] = pd.cut(
        df["id"], bins=np.arange(0, df["id"].max() + bin_size, bin_size), right=False
    )
    # plt.ylim(0, 1)
    plt.xlabel("ID")
    plt.ylabel("Error Rate")
    for target_p_val in reward_array:
        error_rates_by_id_bin = (
            df[df["target_p"] == target_p_val]
            .groupby("id_bin")["success"]
            .apply(lambda x: 1 - x.mean())
        )
        plt.plot(
            error_rates_by_id_bin.index.categories.mid,
            error_rates_by_id_bin.values,
            marker="o",
            label=f"target reward = {target_p_val / 10}",
        )
    plt.legend()
    plt.tight_layout()


# 두 그래프를 나란히 표시
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plot_graph1(df)

plt.subplot(1, 2, 2)
plot_graph2(df)

plt.show()
