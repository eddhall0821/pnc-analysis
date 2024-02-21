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
    ####4차####
    "s607c5af095d4f618fc21a1d1_65bb6200e17a7058bbe51054_1706784263123",
    "s60fd093548ae24bc1a355bf7_65bb61e510406bc3436c2b58_1706784119718",
    "s5c00043a6d931200019bcb9b_65bb67b44c63047973dc8b39_1706786366634",
    "s628781515f29a0394e23b15b_65bb793c2b5746e34133d5d2_1706791605712",
    "s63ee65e3470c23cb401ca89a_65bb82703fe600bff8fe9981_1706791869657",
]


def makeFullFilePaths(filenames):
    fullFilePaths = []

    for name in filenames:
        fullFilePaths.append("C:/Users/soomin/Downloads/ttttaaaaaaddfsfasdffew.csv")
    print(fullFilePaths)
    return fullFilePaths


full_file_paths = makeFullFilePaths(filenames)
reward_array = [0, 50, 100]

df = pd.concat([pd.read_csv(f) for f in full_file_paths], ignore_index=True)

df_0_299 = df[(df["trial"] >= 0) & (df["trial"] <= 299)]
df_300_599 = df[(df["trial"] >= 300) & (df["trial"] <= 599)]
df_600_899 = df[(df["trial"] >= 600) & (df["trial"] <= 899)]

# 각 구간별 target_p 분포 그래프 그리기
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.hist(df_0_299["target_p"], bins=20, alpha=0.7)
plt.title("Target_p Distribution for Trials 0-299")

plt.subplot(3, 1, 2)
plt.hist(df_300_599["target_p"], bins=20, alpha=0.7)
plt.title("Target_p Distribution for Trials 300-599")

plt.subplot(3, 1, 3)
plt.hist(df_600_899["target_p"], bins=20, alpha=0.7)
plt.title("Target_p Distribution for Trials 600-899")

plt.tight_layout()
plt.show()
