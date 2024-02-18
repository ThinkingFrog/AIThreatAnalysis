import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm


def collect_raw_data(n):
    # columns = ["pid", "user", "command", "%cpu", "%mem"]
    columns = ["command", "%cpu", "%mem"]
    data_dir = Path("data/raw")

    for _ in tqdm(range(n)):
        top_output = (
            subprocess.check_output(["top", "-c", "-b", "-n", "1"]).decode().split("\n")
        )

        timestamp = top_output[0].split(" ")[2]

        top_output = top_output[7:-1]

        p_info = []
        for proccess in top_output:
            p_split = proccess.split()
            p_info.append(
                {
                    #     "pid": p_split[0],
                    #     "user": p_split[1],
                    "%cpu": p_split[8],
                    "%mem": p_split[9],
                    "command": p_split[11],
                }
            )

        filename = data_dir / f"top_{timestamp}.csv"

        with Path(filename).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=columns)

            writer.writeheader()
            writer.writerows(p_info)

        time.sleep(1)


# Preprocess data and store it into a file
def preprocess_data():
    processed_tables: Dict[str, pd.DataFrame] = {}
    data_dir = Path("data/processed")
    raw_dir = Path("data/raw")

    for csv_file in raw_dir.rglob("*.csv"):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            command_name = row["command"]
            if command_name not in processed_tables.keys():
                processed_tables[command_name] = pd.DataFrame()

            new_row = pd.DataFrame(
                data={
                    # "timestamp": timestamp,
                    # "pid": row["pid"],
                    # "user": row["user"],
                    "%cpu": row["%cpu"],
                    "%mem": row["%mem"],
                },
                index=[0],
            )

            processed_tables[command_name] = pd.concat(
                [processed_tables[command_name], new_row], ignore_index=True
            )

    timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    filename = f"data_{timestamp}.pkl"
    pd.to_pickle(processed_tables, data_dir / filename)


if __name__ == "__main__":
    n = 100

    collect_raw_data(n)
    preprocess_data()
