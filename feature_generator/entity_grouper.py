"""Entity grouper - computes entity keys for grouping metrics."""

from typing import Any
import pandas as pd

from otel_etl.profiler.semantic_classifier import is_entity_label


def compute_entity_key(
    labels: dict[str, str],
    entity_labels: list[str] | None = None,
) -> str:
    """Compute entity key from labels.

    Entity key is a stable identifier for grouping metrics by resource.
    Format: label1=value1::label2=value2::...

    Args:
        labels: Dictionary of label name -> value
        entity_labels: Optional explicit list of labels to use

    Returns:
        Entity key string
    """
    if entity_labels is None:
        entity_labels = [k for k in labels.keys() if is_entity_label(k)]

    if not entity_labels:
        default_labels = ["service_name", "service", "job", "app", "instance"]
        entity_labels = [l for l in default_labels if l in labels]

    parts = []
    for label in sorted(entity_labels):
        if label in labels:
            parts.append(f"{label}={labels[label]}")

    return "::".join(parts) if parts else "default"


def add_entity_key_column(
    df: pd.DataFrame,
    labels_col: str = "labels",
    entity_labels: list[str] | None = None,
    output_col: str = "entity_key",
) -> pd.DataFrame:
    """Add entity key column to DataFrame.

    Args:
        df: DataFrame with labels column
        labels_col: Name of column containing label dicts
        entity_labels: Optional explicit list of entity labels
        output_col: Name of output column

    Returns:
        DataFrame with entity_key column added
    """
    result = df.copy()

    result[output_col] = result[labels_col].apply(
        lambda labels: compute_entity_key(labels, entity_labels)
    )

    return result


def get_entity_label_combinations(
    df: pd.DataFrame,
    labels_col: str = "labels",
    entity_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Get all unique combinations of entity labels.

    Args:
        df: DataFrame with labels column
        labels_col: Name of column containing label dicts
        entity_labels: Optional explicit list of entity labels

    Returns:
        DataFrame with unique entity combinations
    """
    if entity_labels is None:
        all_labels = set()
        for labels in df[labels_col]:
            all_labels.update(k for k in labels.keys() if is_entity_label(k))
        entity_labels = sorted(all_labels)

    rows = []
    for labels in df[labels_col]:
        row = {l: labels.get(l) for l in entity_labels}
        rows.append(row)

    entities_df = pd.DataFrame(rows)
    return entities_df.drop_duplicates().reset_index(drop=True)


def infer_entity_labels(
    df: pd.DataFrame,
    labels_col: str = "labels",
    cardinality_threshold: int = 10,
) -> list[str]:
    """Infer which labels should be used as entity identifiers.

    Uses heuristics based on semantic classification and cardinality.

    Args:
        df: DataFrame with labels column
        labels_col: Name of column containing label dicts
        cardinality_threshold: Max cardinality for entity labels

    Returns:
        List of label names to use as entity identifiers
    """
    label_values: dict[str, set] = {}

    for labels in df[labels_col]:
        for k, v in labels.items():
            if k not in label_values:
                label_values[k] = set()
            label_values[k].add(v)

    entity_candidates = []

    for label, values in label_values.items():
        if not is_entity_label(label):
            continue

        if len(values) <= cardinality_threshold:
            entity_candidates.append((label, len(values)))

    entity_candidates.sort(key=lambda x: x[1])

    return [label for label, _ in entity_candidates]


class EntityGrouper:
    """Groups metrics by entity key."""

    def __init__(
        self,
        entity_labels: list[str] | None = None,
        separator: str = "::",
    ):
        """Initialize entity grouper.

        Args:
            entity_labels: Labels to use for entity key
            separator: Separator between label=value pairs
        """
        self.entity_labels = entity_labels
        self.separator = separator

    def compute_key(self, labels: dict[str, str]) -> str:
        """Compute entity key from labels.

        Args:
            labels: Label dictionary

        Returns:
            Entity key string
        """
        return compute_entity_key(labels, self.entity_labels)

    def group_dataframe(
        self,
        df: pd.DataFrame,
        labels_col: str = "labels",
    ) -> dict[str, pd.DataFrame]:
        """Group DataFrame by entity key.

        Args:
            df: DataFrame with labels column
            labels_col: Name of labels column

        Returns:
            Dictionary mapping entity key to DataFrame
        """
        df_with_key = add_entity_key_column(df, labels_col, self.entity_labels)

        groups = {}
        for key, group_df in df_with_key.groupby("entity_key", sort=False):
            groups[key] = group_df

        return groups

    def get_entity_count(self, df: pd.DataFrame, labels_col: str = "labels") -> int:
        """Count unique entities in DataFrame.

        Args:
            df: DataFrame with labels column
            labels_col: Name of labels column

        Returns:
            Number of unique entities
        """
        df_with_key = add_entity_key_column(df, labels_col, self.entity_labels)
        return df_with_key["entity_key"].nunique()
