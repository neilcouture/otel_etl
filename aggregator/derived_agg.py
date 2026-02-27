"""Derived metrics - computed metrics like error rate, success rate."""

from typing import TypedDict
import pandas as pd
import numpy as np


class DerivedMetricsResult(TypedDict):
    """Result of derived metrics computation."""
    error_rate: float
    success_rate: float
    total_count: float
    error_count: float
    success_count: float


def compute_derived_metrics(
    df: pd.DataFrame,
    status_col: str = "status_bucket",
    count_col: str = "count",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute derived metrics like error rate from aggregated data.

    Args:
        df: DataFrame with status bucket and count columns
        status_col: Name of status bucket column
        count_col: Name of count column
        group_cols: Columns to group by (excluding status)

    Returns:
        DataFrame with derived metrics
    """
    if df.empty:
        return pd.DataFrame()

    if group_cols is None:
        return _compute_single_group(df, status_col, count_col)

    results = []

    for group_key, group_df in df.groupby(group_cols, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        derived = _compute_single_group(group_df, status_col, count_col)

        row = dict(zip(group_cols, group_key))
        if not derived.empty:
            row.update(derived.iloc[0].to_dict())
        results.append(row)

    return pd.DataFrame(results)


def _compute_single_group(
    df: pd.DataFrame,
    status_col: str,
    count_col: str,
) -> pd.DataFrame:
    """Compute derived metrics for a single group.

    Args:
        df: DataFrame for single group
        status_col: Status bucket column name
        count_col: Count column name

    Returns:
        Single-row DataFrame with derived metrics
    """
    if df.empty:
        return pd.DataFrame([{
            "error_rate": float("nan"),
            "success_rate": float("nan"),
            "total_count": 0.0,
            "error_count": 0.0,
            "success_count": 0.0,
        }])

    total_count = df[count_col].sum()

    error_statuses = ["client_error", "server_error", "error"]
    success_statuses = ["success"]

    error_mask = df[status_col].isin(error_statuses)
    success_mask = df[status_col].isin(success_statuses)

    error_count = df.loc[error_mask, count_col].sum()
    success_count = df.loc[success_mask, count_col].sum()

    error_rate = error_count / total_count if total_count > 0 else 0.0
    success_rate = success_count / total_count if total_count > 0 else 0.0

    return pd.DataFrame([{
        "error_rate": float(error_rate),
        "success_rate": float(success_rate),
        "total_count": float(total_count),
        "error_count": float(error_count),
        "success_count": float(success_count),
    }])


def compute_error_rate(
    error_count: float,
    total_count: float,
) -> float:
    """Compute error rate from counts.

    Args:
        error_count: Number of errors
        total_count: Total number of requests

    Returns:
        Error rate (0-1)
    """
    if total_count <= 0:
        return 0.0
    return error_count / total_count


def compute_success_rate(
    success_count: float,
    total_count: float,
) -> float:
    """Compute success rate from counts.

    Args:
        success_count: Number of successes
        total_count: Total number of requests

    Returns:
        Success rate (0-1)
    """
    if total_count <= 0:
        return 0.0
    return success_count / total_count


def compute_availability(
    success_count: float,
    total_count: float,
    include_partial: bool = False,
) -> float:
    """Compute availability metric.

    Args:
        success_count: Number of successful requests
        total_count: Total number of requests
        include_partial: Whether to include partial successes

    Returns:
        Availability (0-1)
    """
    return compute_success_rate(success_count, total_count)


def compute_throughput(
    count: float,
    window_seconds: float,
) -> float:
    """Compute throughput (requests per second).

    Args:
        count: Number of requests in window
        window_seconds: Window duration in seconds

    Returns:
        Throughput (requests per second)
    """
    if window_seconds <= 0:
        return 0.0
    return count / window_seconds


def compute_error_budget_consumption(
    current_error_rate: float,
    slo_target: float = 0.999,
) -> float:
    """Compute error budget consumption.

    Args:
        current_error_rate: Current error rate (0-1)
        slo_target: SLO target (e.g., 0.999 for 99.9%)

    Returns:
        Error budget consumption (0-1, >1 means budget exceeded)
    """
    error_budget = 1.0 - slo_target
    if error_budget <= 0:
        return float("inf") if current_error_rate > 0 else 0.0
    return current_error_rate / error_budget


def add_derived_features(
    df: pd.DataFrame,
    rate_col: str = "rate_per_sec",
    count_col: str = "count",
    status_col: str = "status_bucket",
) -> pd.DataFrame:
    """Add derived feature columns to a DataFrame.

    Args:
        df: Input DataFrame
        rate_col: Rate column name
        count_col: Count column name
        status_col: Status bucket column name

    Returns:
        DataFrame with additional derived columns
    """
    result = df.copy()

    if status_col in result.columns and count_col in result.columns:
        error_mask = result[status_col].isin(["client_error", "server_error", "error"])
        result["is_error"] = error_mask.astype(int)

    return result
