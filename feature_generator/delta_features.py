"""Delta features - time-window comparison features."""

from typing import Any
import pandas as pd
import numpy as np


def compute_delta_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    entity_col: str | None = "entity_key",
    feature_cols: list[str] | None = None,
    delta_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute delta features comparing to previous time windows.

    Args:
        df: Wide-format DataFrame with timestamp and entity columns
        timestamp_col: Name of timestamp column
        entity_col: Name of entity key column (None if no entity grouping)
        feature_cols: Feature columns to compute deltas for
        delta_windows: Window sizes in minutes (default: [5, 60])

    Returns:
        DataFrame with additional delta columns
    """
    if df.empty:
        return df.copy()

    delta_windows = delta_windows or [5, 60]

    exclude_cols = [timestamp_col]
    if entity_col:
        exclude_cols.append(entity_col)

    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols
        ]

    result = df.copy()
    if entity_col:
        result = result.sort_values([entity_col, timestamp_col])
    else:
        result = result.sort_values([timestamp_col])

    # Collect all new columns first to avoid fragmentation
    new_cols = {}
    for window_minutes in delta_windows:
        window_suffix = f"delta_{window_minutes}m"

        for col in feature_cols:
            delta_col_name = f"{col}__{window_suffix}"
            if entity_col:
                shifted = result.groupby(entity_col)[col].shift(window_minutes)
            else:
                shifted = result[col].shift(window_minutes)
            new_cols[delta_col_name] = result[col] - shifted

    # Add all columns at once
    if new_cols:
        result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)

    return result


def compute_pct_change_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    entity_col: str | None = "entity_key",
    feature_cols: list[str] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute percentage change features.

    Args:
        df: Wide-format DataFrame
        timestamp_col: Name of timestamp column
        entity_col: Name of entity key column (None if no entity grouping)
        feature_cols: Feature columns to compute changes for
        windows: Window sizes in minutes

    Returns:
        DataFrame with percentage change columns
    """
    if df.empty:
        return df.copy()

    windows = windows or [60]

    exclude_cols = [timestamp_col]
    if entity_col:
        exclude_cols.append(entity_col)

    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols
        ]

    result = df.copy()
    if entity_col:
        result = result.sort_values([entity_col, timestamp_col])
    else:
        result = result.sort_values([timestamp_col])

    # Collect all new columns first to avoid fragmentation
    new_cols = {}
    for window_minutes in windows:
        window_suffix = f"pct_change_{window_minutes}m"

        for col in feature_cols:
            pct_col_name = f"{col}__{window_suffix}"
            if entity_col:
                shifted = result.groupby(entity_col)[col].shift(window_minutes)
            else:
                shifted = result[col].shift(window_minutes)

            with np.errstate(divide="ignore", invalid="ignore"):
                pct_change = (result[col] - shifted) / shifted.abs()
                pct_change = pct_change.replace([np.inf, -np.inf], np.nan)

            new_cols[pct_col_name] = pct_change

    # Add all columns at once
    if new_cols:
        result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)

    return result


def compute_rolling_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    entity_col: str | None = "entity_key",
    feature_cols: list[str] | None = None,
    window_size: int = 5,
) -> pd.DataFrame:
    """Compute rolling statistics features.

    Args:
        df: Wide-format DataFrame
        timestamp_col: Name of timestamp column
        entity_col: Name of entity key column (None if no entity grouping)
        feature_cols: Feature columns to compute rolling stats for
        window_size: Rolling window size (number of rows)

    Returns:
        DataFrame with rolling statistic columns
    """
    if df.empty:
        return df.copy()

    exclude_cols = [timestamp_col]
    if entity_col:
        exclude_cols.append(entity_col)

    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols
        ]

    result = df.copy()
    if entity_col:
        result = result.sort_values([entity_col, timestamp_col])
    else:
        result = result.sort_values([timestamp_col])

    # Collect all new columns first to avoid fragmentation
    new_cols = {}
    for col in feature_cols:
        if entity_col:
            grouped = result.groupby(entity_col)[col]
            new_cols[f"{col}__rolling_mean_{window_size}"] = grouped.transform(
                lambda x: x.rolling(window_size, min_periods=1).mean()
            )
            new_cols[f"{col}__rolling_std_{window_size}"] = grouped.transform(
                lambda x: x.rolling(window_size, min_periods=2).std()
            )
        else:
            new_cols[f"{col}__rolling_mean_{window_size}"] = result[col].rolling(
                window_size, min_periods=1
            ).mean()
            new_cols[f"{col}__rolling_std_{window_size}"] = result[col].rolling(
                window_size, min_periods=2
            ).std()

    # Add all columns at once
    if new_cols:
        result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)

    return result


def compute_lag_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    entity_col: str | None = "entity_key",
    feature_cols: list[str] | None = None,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Compute lagged feature values.

    Args:
        df: Wide-format DataFrame
        timestamp_col: Name of timestamp column
        entity_col: Name of entity key column (None if no entity grouping)
        feature_cols: Feature columns to create lags for
        lags: Lag amounts (default: [1, 5])

    Returns:
        DataFrame with lagged feature columns
    """
    if df.empty:
        return df.copy()

    lags = lags or [1, 5]

    exclude_cols = [timestamp_col]
    if entity_col:
        exclude_cols.append(entity_col)

    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols
        ]

    result = df.copy()
    if entity_col:
        result = result.sort_values([entity_col, timestamp_col])
    else:
        result = result.sort_values([timestamp_col])

    # Collect all new columns first to avoid fragmentation
    new_cols = {}
    for col in feature_cols:
        for lag in lags:
            lag_col_name = f"{col}__lag_{lag}"
            if entity_col:
                new_cols[lag_col_name] = result.groupby(entity_col)[col].shift(lag)
            else:
                new_cols[lag_col_name] = result[col].shift(lag)

    # Add all columns at once
    if new_cols:
        result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)

    return result


class DeltaFeatureGenerator:
    """Generates delta and comparison features."""

    def __init__(
        self,
        timestamp_col: str = "timestamp",
        entity_col: str | None = "entity_key",
        delta_windows: list[int] | None = None,
        pct_change_windows: list[int] | None = None,
        include_rolling: bool = False,
        rolling_window: int = 5,
    ):
        """Initialize delta feature generator.

        Args:
            timestamp_col: Timestamp column name
            entity_col: Entity column name (None for no entity grouping)
            delta_windows: Windows for delta features (minutes)
            pct_change_windows: Windows for pct change features (minutes)
            include_rolling: Whether to include rolling statistics
            rolling_window: Rolling window size
        """
        self.timestamp_col = timestamp_col
        self.entity_col = entity_col
        self.delta_windows = delta_windows or [5, 60]
        self.pct_change_windows = pct_change_windows or [60]
        self.include_rolling = include_rolling
        self.rolling_window = rolling_window

    def generate(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Generate all delta features.

        Args:
            df: Input wide-format DataFrame
            feature_cols: Columns to generate features for

        Returns:
            DataFrame with delta features added
        """
        result = compute_delta_features(
            df,
            self.timestamp_col,
            self.entity_col,
            feature_cols,
            self.delta_windows,
        )

        result = compute_pct_change_features(
            result,
            self.timestamp_col,
            self.entity_col,
            feature_cols,
            self.pct_change_windows,
        )

        if self.include_rolling:
            result = compute_rolling_features(
                result,
                self.timestamp_col,
                self.entity_col,
                feature_cols,
                self.rolling_window,
            )

        return result
