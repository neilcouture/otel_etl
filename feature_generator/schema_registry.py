"""Schema registry - tracks and stabilizes column schema."""

from typing import Any
from datetime import datetime, timezone
import yaml
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SchemaRegistry:
    """Maintains a stable column schema across pipeline runs."""

    def __init__(
        self,
        index_cols: list[str] | None = None,
        columns: list[str] | None = None,
    ):
        """Initialize schema registry.

        Args:
            index_cols: Index columns (always present)
            columns: Initial feature column list
        """
        self.index_cols = index_cols or ["timestamp", "entity_key"]
        self._columns: list[str] = list(columns) if columns else []
        self._column_set: set[str] = set(self._columns)
        self._created_at: str | None = None
        self._updated_at: str | None = None

    @property
    def columns(self) -> list[str]:
        """Get registered columns in order."""
        return list(self._columns)

    @property
    def all_columns(self) -> list[str]:
        """Get all columns including index."""
        return self.index_cols + self._columns

    def register(self, column: str) -> int:
        """Register a new column.

        Args:
            column: Column name to register

        Returns:
            Index of the column
        """
        if column in self._column_set:
            return self._columns.index(column)

        self._columns.append(column)
        self._column_set.add(column)
        self._updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        logger.debug(f"Registered new column: {column}")
        return len(self._columns) - 1

    def register_many(self, columns: list[str]) -> list[int]:
        """Register multiple columns.

        Args:
            columns: Column names to register

        Returns:
            List of column indices
        """
        return [self.register(col) for col in columns]

    def contains(self, column: str) -> bool:
        """Check if column is registered.

        Args:
            column: Column name

        Returns:
            True if column is registered
        """
        return column in self._column_set

    def align_dataframe(
        self,
        df: pd.DataFrame,
        fill_value: Any = np.nan,
        register_new: bool = True,
    ) -> pd.DataFrame:
        """Align DataFrame to registered schema.

        Args:
            df: Input DataFrame
            fill_value: Value for missing columns
            register_new: Whether to register new columns found in df

        Returns:
            Aligned DataFrame with stable column order
        """
        result = df.copy()

        if register_new:
            new_cols = [
                c for c in df.columns
                if c not in self._column_set and c not in self.index_cols
            ]
            if new_cols:
                self.register_many(sorted(new_cols))
                logger.info(f"Registered {len(new_cols)} new columns")

        for col in self._columns:
            if col not in result.columns:
                result[col] = fill_value

        ordered_cols = []
        for col in self.index_cols:
            if col in result.columns:
                ordered_cols.append(col)

        ordered_cols.extend(self._columns)

        return result[ordered_cols]

    def save(self, path: str) -> None:
        """Save registry to YAML file.

        Args:
            path: Output file path
        """
        data = {
            "created_at": self._created_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "updated_at": self._updated_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "index_cols": self.index_cols,
            "columns": self._columns,
            "column_count": len(self._columns),
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved schema registry to {path} ({len(self._columns)} columns)")

    @classmethod
    def load(cls, path: str) -> "SchemaRegistry":
        """Load registry from YAML file.

        Args:
            path: Input file path

        Returns:
            SchemaRegistry instance
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        registry = cls(
            index_cols=data.get("index_cols", ["timestamp", "entity_key"]),
            columns=data.get("columns", []),
        )
        registry._created_at = data.get("created_at")
        registry._updated_at = data.get("updated_at")

        logger.info(f"Loaded schema registry from {path} ({len(registry._columns)} columns)")
        return registry

    def get_new_columns(self, df: pd.DataFrame) -> list[str]:
        """Get columns in DataFrame that are not in registry.

        Args:
            df: DataFrame to check

        Returns:
            List of new column names
        """
        return [
            c for c in df.columns
            if c not in self._column_set and c not in self.index_cols
        ]

    def get_missing_columns(self, df: pd.DataFrame) -> list[str]:
        """Get registered columns missing from DataFrame.

        Args:
            df: DataFrame to check

        Returns:
            List of missing column names
        """
        return [c for c in self._columns if c not in df.columns]

    def diff(self, other: "SchemaRegistry") -> dict[str, Any]:
        """Compare with another registry.

        Args:
            other: Other SchemaRegistry

        Returns:
            Dictionary with differences
        """
        self_set = self._column_set
        other_set = other._column_set

        return {
            "added": sorted(other_set - self_set),
            "removed": sorted(self_set - other_set),
            "common": sorted(self_set & other_set),
            "self_count": len(self._columns),
            "other_count": len(other._columns),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "column_count": len(self._columns),
            "index_cols": self.index_cols,
            "created_at": self._created_at,
            "updated_at": self._updated_at,
        }
