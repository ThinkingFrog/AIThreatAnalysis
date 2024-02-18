from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

distance_metric = "mahalanobis"


def tukey(arr: np.ndarray, k=1.5) -> np.ndarray:
    x = np.array(arr).copy().astype(float)
    first_quartile = np.quantile(x, 0.25)
    third_quartile = np.quantile(x, 0.75)

    # Define IQR
    iqr = third_quartile - first_quartile

    ### Define the allowed limits for 'Normal Data'
    lower_allowed_limit = first_quartile - (k * iqr)
    upper_allowed_limit = third_quartile + (k * iqr)

    # set values below the lower limit/above the upper limit as +-inf
    x[x < lower_allowed_limit] = -np.inf
    x[x > upper_allowed_limit] = np.inf
    return x


def calc_mean_distance_between_normal_values(
    df: pd.DataFrame, labels: np.ndarray
) -> float:
    normal_df = df[labels == 1]

    cov = np.cov(normal_df.values, rowvar=False, ddof=0)
    vi = np.linalg.inv(cov)
    dist_matrix = cdist(normal_df, normal_df, metric=distance_metric, VI=vi)
    return np.mean(dist_matrix)


def calc_mean_distance_between_anomaly_values(
    df: pd.DataFrame, labels: np.ndarray
) -> Optional[float]:
    lower_anomaly_df = df[labels == -2]
    upper_anomaly_df = df[labels == -1]

    lower_anomaly_exist = len(lower_anomaly_df.index) != 0
    upper_anomaly_exist = len(upper_anomaly_df.index) != 0

    if lower_anomaly_exist:
        lower_cov = np.cov(lower_anomaly_df.values, rowvar=False, ddof=0)
        lower_vi = np.linalg.inv(lower_cov)
        lower_dist_matrix = cdist(
            lower_anomaly_df,
            lower_anomaly_df,
            metric=distance_metric,
            VI=lower_vi,
        )

    if upper_anomaly_exist:
        upper_cov = np.cov(upper_anomaly_df.values, rowvar=False, ddof=0)
        upper_vi = np.linalg.inv(upper_cov)
        upper_dist_matrix = cdist(
            upper_anomaly_df,
            upper_anomaly_df,
            metric=distance_metric,
            VI=upper_vi,
        )

    match (lower_anomaly_exist, upper_anomaly_exist):
        case (True, True):
            return (np.mean(lower_dist_matrix) + np.mean(upper_dist_matrix)) / 2
        case (False, False):
            return None
        case (True, False):
            return np.mean(lower_dist_matrix)
        case (False, True):
            return np.mean(upper_dist_matrix)


def calc_mean_distance_between_clusters(
    df: pd.DataFrame, labels: np.ndarray
) -> Optional[float]:
    normal_df = df[labels == 1]
    normal_mean = normal_df.mean()

    lower_anomaly_df = df[labels == -2]
    upper_anomaly_df = df[labels == -1]

    lower_anomaly_exist = len(lower_anomaly_df.index) != 0
    upper_anomaly_exist = len(upper_anomaly_df.index) != 0

    if lower_anomaly_exist:
        lower_mean = lower_anomaly_df.mean()
        normal_lower_cov = pd.concat(
            [normal_df, lower_anomaly_df], ignore_index=True
        ).cov()
        normal_lower_vi = np.linalg.inv(normal_lower_cov)
        normal_lower_dist = cdist(
            [normal_mean],
            [lower_mean],
            metric=distance_metric,
            VI=normal_lower_vi,
        )

    if upper_anomaly_exist:
        upper_mean = upper_anomaly_df.mean()
        normal_upper_cov = pd.concat(
            [normal_df, upper_anomaly_df], ignore_index=True
        ).cov()
        normal_upper_vi = np.linalg.inv(normal_upper_cov)
        normal_upper_dist = cdist(
            [normal_mean],
            [upper_mean],
            metric=distance_metric,
            VI=normal_upper_vi,
        )

    match (lower_anomaly_exist, upper_anomaly_exist):
        case (True, True):
            return (np.mean(normal_lower_dist) + np.mean(normal_upper_dist)) / 2
        case (False, False):
            return None
        case (True, False):
            return np.mean(normal_lower_dist)
        case (False, True):
            return np.mean(normal_upper_dist)


def calc_standard_deviations(
    distances: np.ndarray, labels: np.ndarray
) -> Tuple[float, float]:
    mean = distances.mean()

    variances = [(x - mean) ** 2 for x in distances]

    anomaly_data = [
        x for idx, x in enumerate(variances) if labels[idx] == -1 or labels[idx] == -2
    ]
    if len(anomaly_data) != 0:
        mean_anomaly_var = np.mean(anomaly_data)
    else:
        mean_anomaly_var = 0
    mean_normal_var = np.mean(
        [x for idx, x in enumerate(variances) if labels[idx] == 1]
    )

    return mean_normal_var, mean_anomaly_var
