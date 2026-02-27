"""Wide formatter - pivots aggregated data to wide format."""

from typing import Any
import pandas as pd
import numpy as np


def pivot_to_wide(
    df: pd.DataFrame,
    index_cols: list[str],
    feature_col: str,
    value_col: str,
    fill_value: Any = np.nan,
) -> pd.DataFrame:
    """Pivot DataFrame from long to wide format.

    Args:
        df: Long-format DataFrame
        index_cols: Columns to keep as index (e.g., ['timestamp', 'entity_key'])
        feature_col: Column containing feature names
        value_col: Column containing feature values
        fill_value: Value to use for missing features

    Returns:
        Wide-format DataFrame with features as columns
    """
    if df.empty:
        return pd.DataFrame()

    pivoted = df.pivot_table(
        index=index_cols,
        columns=feature_col,
        values=value_col,
        aggfunc="first",
        fill_value=fill_value,
    )

    pivoted = pivoted.reset_index()

    pivoted.columns.name = None

    return pivoted


def melt_from_wide(
    df: pd.DataFrame,
    id_cols: list[str],
    var_name: str = "feature",
    value_name: str = "value",
) -> pd.DataFrame:
    """Melt DataFrame from wide to long format.

    Args:
        df: Wide-format DataFrame
        id_cols: Columns to keep as identifiers
        var_name: Name for the variable column
        value_name: Name for the value column

    Returns:
        Long-format DataFrame
    """
    return pd.melt(
        df,
        id_vars=id_cols,
        var_name=var_name,
        value_name=value_name,
    )


def align_columns(
    df: pd.DataFrame,
    expected_columns: list[str],
    fill_value: Any = np.nan,
) -> pd.DataFrame:
    """Align DataFrame columns to expected schema.

    Adds missing columns (filled with fill_value) and reorders
    to match expected order.

    Args:
        df: Input DataFrame
        expected_columns: List of expected column names in order
        fill_value: Value for missing columns

    Returns:
        DataFrame with aligned columns
    """
    result = df.copy()

    for col in expected_columns:
        if col not in result.columns:
            result[col] = fill_value

    available_expected = [c for c in expected_columns if c in result.columns]
    extra_cols = [c for c in result.columns if c not in expected_columns]

    return result[available_expected + extra_cols]


def create_wide_dataframe(
    aggregated_data: list[dict],
    index_cols: list[str],
    feature_values: dict[str, Any],
) -> pd.DataFrame:
    """Create wide-format DataFrame from aggregated data.

    Args:
        aggregated_data: List of dicts with index values
        index_cols: Names of index columns
        feature_values: Mapping of feature name to value dict keyed by index

    Returns:
        Wide-format DataFrame
    """
    rows = []

    for item in aggregated_data:
        row = {col: item[col] for col in index_cols if col in item}

        index_key = tuple(item.get(col) for col in index_cols)

        for feature_name, values_by_key in feature_values.items():
            if index_key in values_by_key:
                row[feature_name] = values_by_key[index_key]
            else:
                row[feature_name] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


class WideFormatter:
    """Transforms aggregated metrics into wide-format DataFrames."""

    def __init__(
        self,
        index_cols: list[str] | None = None,
        column_order: list[str] | None = None,
    ):
        """Initialize wide formatter.

        Args:
            index_cols: Columns to use as index
            column_order: Expected column order for output
        """
        self.index_cols = index_cols or ["timestamp", "entity_key"]
        self.column_order = column_order or []

    def format(
        self,
        long_df: pd.DataFrame,
        feature_col: str = "feature",
        value_col: str = "value",
    ) -> pd.DataFrame:
        """Format long DataFrame to wide.

        Args:
            long_df: Long-format DataFrame
            feature_col: Column with feature names
            value_col: Column with values

        Returns:
            Wide-format DataFrame
        """
        wide_df = pivot_to_wide(
            long_df,
            self.index_cols,
            feature_col,
            value_col,
        )

        if self.column_order:
            wide_df = align_columns(wide_df, self.index_cols + self.column_order)

        return wide_df

    def update_column_order(self, df: pd.DataFrame) -> None:
        """Update expected column order from DataFrame.

        Args:
            df: DataFrame with current columns
        """
        feature_cols = [c for c in df.columns if c not in self.index_cols]
        self.column_order = sorted(set(self.column_order + feature_cols))


def compute_row_completeness(df: pd.DataFrame, index_cols: list[str]) -> pd.Series:
    """Compute completeness ratio for each row.

    Args:
        df: Wide-format DataFrame
        index_cols: Index columns to exclude from calculation

    Returns:
        Series with completeness ratio per row
    """
    feature_cols = [c for c in df.columns if c not in index_cols]

    if not feature_cols:
        return pd.Series(1.0, index=df.index)

    non_null_count = df[feature_cols].notna().sum(axis=1)
    return non_null_count / len(feature_cols)


def compute_column_completeness(df: pd.DataFrame, index_cols: list[str]) -> pd.Series:
    """Compute completeness ratio for each feature column.

    Args:
        df: Wide-format DataFrame
        index_cols: Index columns to exclude

    Returns:
        Series with completeness ratio per column
    """
    feature_cols = [c for c in df.columns if c not in index_cols]

    if not feature_cols:
        return pd.Series(dtype=float)

    non_null_count = df[feature_cols].notna().sum()
    return non_null_count / len(df)
