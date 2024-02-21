import csv
import os
import re
import pandas as pd

# file_pattern = r"s[a-zA-Z0-9]+_[a-zA-Z0-9]+_\d+\.csv"
file_pattern = r"s([a-zA-Z0-9]+)_[a-zA-Z0-9]+_\d+\.csv"


def extract_last_total_p_no_pandas(folder_path, file_pattern):
    results = []
    for filename in os.listdir(folder_path):
        match = re.match(file_pattern, filename)
        print(match)
        if match:
            user_id = match.group(1)
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, newline="") as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader)
                    total_p_index = header.index("total_p")
                    success_index = header.index("success")
                    success = 0
                    last_row = None
                    for row in reader:
                        success += int(row[success_index])
                        last_row = row
                    if last_row:
                        total_p = last_row[total_p_index]
                        results.append(
                            [f"{user_id}, {int(total_p) / 10000}, {success}"]
                        )
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    print(results)
    return results


path = r"C:\Users\soomin\AppData\Local\Google\Cloud SDK\point-and-click-20d4c.appspot.com\summary"
arr = extract_last_total_p_no_pandas(path, file_pattern)

df = pd.DataFrame(arr)
df.to_csv("sample.csv", index=False)
