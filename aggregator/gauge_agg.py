"""Gauge aggregation - summary statistics for gauge metrics."""

from typing import TypedDict
import pandas as pd
import numpy as np


class GaugeAggResult(TypedDict):
    """Result of gauge aggregation."""
    last: float
    mean: float
    min: float
    max: float
    stddev: float


def aggregate_gauge(
    values: pd.Series,
    timestamps: pd.Series | None = None,
) -> GaugeAggResult:
    """Aggregate gauge values over a time window.

    Args:
        values: Series of gauge values
        timestamps: Optional series of timestamps (used to determine 'last')

    Returns:
        GaugeAggResult with summary statistics
    """
    if len(values) == 0:
        return GaugeAggResult(
            last=float("nan"),
            mean=float("nan"),
            min=float("nan"),
            max=float("nan"),
            stddev=float("nan"),
        )

    clean_values = values.dropna()

    if len(clean_values) == 0:
        return GaugeAggResult(
            last=float("nan"),
            mean=float("nan"),
            min=float("nan"),
            max=float("nan"),
            stddev=float("nan"),
        )

    if timestamps is not None and len(timestamps) == len(values):
        sorted_idx = timestamps.argsort()
        last_val = values.iloc[sorted_idx.iloc[-1]]
    else:
        last_val = values.iloc[-1]

    return GaugeAggResult(
        last=float(last_val),
        mean=float(clean_values.mean()),
        min=float(clean_values.min()),
        max=float(clean_values.max()),
        stddev=float(clean_values.std()) if len(clean_values) > 1 else 0.0,
    )


def compute_gauge_stats(
    df: pd.DataFrame,
    value_col: str = "value",
    timestamp_col: str = "timestamp",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute gauge statistics for a DataFrame.

    Args:
        df: DataFrame with gauge data
        value_col: Name of value column
        timestamp_col: Name of timestamp column
        group_cols: Columns to group by

    Returns:
        DataFrame with gauge statistics
    """
    if group_cols is None:
        result = aggregate_gauge(df[value_col], df.get(timestamp_col))
        return pd.DataFrame([result])

    results = []

    for group_key, group_df in df.groupby(group_cols, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        agg_result = aggregate_gauge(
            group_df[value_col],
            group_df.get(timestamp_col),
        )

        row = dict(zip(group_cols, group_key))
        row.update(agg_result)
        results.append(row)

    return pd.DataFrame(results)


def compute_gauge_change(
    old_result: GaugeAggResult,
    new_result: GaugeAggResult,
) -> dict[str, float]:
    """Compute change between two gauge aggregations.

    Args:
        old_result: Previous gauge result
        new_result: Current gauge result

    Returns:
        Dictionary with change values
    """
    def safe_pct_change(old: float, new: float) -> float:
        if np.isnan(old) or np.isnan(new):
            return float("nan")
        if old == 0:
            return float("nan") if new == 0 else float("inf")
        return (new - old) / abs(old)

    return {
        "last_delta": new_result["last"] - old_result["last"],
        "mean_delta": new_result["mean"] - old_result["mean"],
        "last_pct_change": safe_pct_change(old_result["last"], new_result["last"]),
        "mean_pct_change": safe_pct_change(old_result["mean"], new_result["mean"]),
    }


def detect_gauge_anomaly(
    values: pd.Series,
    threshold_stddev: float = 3.0,
) -> pd.Series:
    """Detect anomalous values using standard deviation.

    Args:
        values: Series of gauge values
        threshold_stddev: Number of standard deviations for anomaly threshold

    Returns:
        Boolean series indicating anomalies
    """
    clean_values = values.dropna()

    if len(clean_values) < 2:
        return pd.Series(False, index=values.index)

    mean = clean_values.mean()
    std = clean_values.std()

    if std == 0:
        return pd.Series(False, index=values.index)

    z_scores = (values - mean).abs() / std
    return z_scores > threshold_stddev
