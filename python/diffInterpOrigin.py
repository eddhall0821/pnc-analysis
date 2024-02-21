import pandas as pd
import matplotlib.pyplot as plt

interp = pd.read_csv("csv/raw/test/test.csv")
origin = pd.read_csv("csv/raw/test/orgt.csv")

interp["timestamp"] = pd.to_datetime(interp["timestamp_x"])
origin["timestamp"] = pd.to_datetime(origin["timestamp"])

origin["target_x_change"] = origin["target_x"].diff().ne(0)
origin["target_y_change"] = origin["target_y"].diff().ne(0)

# 그래프 그리기
plt.figure(figsize=(12, 6))

plt.plot(
    interp["timestamp"],
    interp["cursor_x"],
    label="removed",
    marker="o",
)
plt.plot(
    origin["timestamp"],
    origin["cursor_x"],
    label="Original",
    alpha=0.7,
    linestyle="--",
    marker="x",
)
plt.title("Comparison of cursor_x over Time")
for idx, row in origin.iterrows():
    if row["target_x_change"] or row["target_y_change"]:
        plt.axvline(x=row["timestamp"], color="r", linestyle="--", alpha=0.5)

plt.legend()

plt.tight_layout()  # 서브플롯 간격 자동 조절
plt.savefig(fname="img.png", pad_inches=0.01)
plt.show()
