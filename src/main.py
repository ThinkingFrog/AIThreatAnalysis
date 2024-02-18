def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

import contextlib
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from tqdm import tqdm

from src.metrics import (
    calc_mean_distance_between_anomaly_values,
    calc_mean_distance_between_clusters,
    calc_mean_distance_between_normal_values,
    calc_standard_deviations,
    tukey,
)


def load_data(path: Path) -> Dict[str, pd.DataFrame]:
    return pd.read_pickle(path)


def main():
    train_data_path = Path("data/processed/data_16-02-2024_12:40:40.pkl")
    data = load_data(train_data_path)
    total_entries = len(data.keys())

    distances_in_normal_clusters = []
    distances_in_anomaly_clusters = []
    distances_between_clusters = []
    mean_anomaly_std = []
    mean_normal_std = []

    processed_entries = 0

    for pr_name, df in tqdm(data.items()):
        with contextlib.suppress(Exception):
            ele = EllipticEnvelope(
                random_state=0, support_fraction=1, assume_centered=True
            )
            ele.fit(df)
            # Calculate distances from the robust mean value
            distances = ele.mahalanobis(df)

            processed_distances = tukey(distances)

            # -1 - upper anomaly, -2 - lower anomaly, +1 - normal
            labels = np.array(
                list(
                    map(
                        lambda x: -1 if np.isposinf(x) else -2 if np.isneginf(x) else 1,
                        processed_distances,
                    )
                )
            )

            processed_entries += 1

            (mns, mas) = calc_standard_deviations(distances, labels)
            mean_normal_std.append(mns)
            mean_anomaly_std.append(mas)

            mdbav = calc_mean_distance_between_anomaly_values(df, labels)
            if mdbav is not None:
                distances_in_anomaly_clusters.append(mdbav)

            mdbc = calc_mean_distance_between_clusters(df, labels)
            if mdbc is not None:
                distances_between_clusters.append(mdbc)

            mdbnv = calc_mean_distance_between_normal_values(df, labels)
            if mdbnv is not None:
                distances_in_normal_clusters.append(mdbnv)

    print(
        f"Processed {processed_entries} entries out of {total_entries}",
    )
    print(
        f"Mean distance in normal clusters calculated on {len(distances_in_normal_clusters)} entries"
    )
    print(
        f"Mean distance in anomaly clusters calculated on {len(distances_in_anomaly_clusters)} entries"
    )
    print(
        f"Mean distance between clusters calculated on {len(distances_between_clusters)} entries"
    )

    timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    plt.figure()
    for name, arr in zip(
        [
            "Distance in normal cluster",
            "Distance in anomaly cluster",
            "Distance between clusters",
        ],
        [
            distances_in_normal_clusters,
            distances_in_anomaly_clusters,
            distances_between_clusters,
        ],
    ):
        x = np.linspace(1, len(arr) + 1, len(arr))
        plt.plot(x, arr, label=name)
    plt.ylabel("Mean distance")
    plt.legend()
    plt.grid()
    plt.savefig(f"results/distances_{timestamp}.png")

    plt.figure()
    for name, arr in zip(
        ["Mean normal std", "Mean anomaly std"], [mean_normal_std, mean_anomaly_std]
    ):
        x = np.linspace(1, len(arr) + 1, len(arr))
        plt.plot(x, arr, label=name)
    plt.ylabel("Mean std")
    plt.legend()
    plt.grid()
    plt.savefig(f"results/std_{timestamp}.png")

    diff_std = [abs(x - y) for x in mean_anomaly_std for y in mean_normal_std]
    plt.figure()
    x = np.linspace(1, len(diff_std) + 1, len(diff_std))
    plt.plot(x, diff_std)
    plt.ylabel("Mean std diff")
    plt.grid()
    plt.savefig(f"results/std_diff_{timestamp}.png")

    plt.show()


if __name__ == "__main__":
    main()
