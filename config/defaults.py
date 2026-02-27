"""Default configuration values for the OTel ETL pipeline."""

from typing import TypedDict


class CardinalityThresholds(TypedDict):
    """Thresholds for cardinality tier classification."""
    tier1_max: int  # Max cardinality for tier 1 (always keep, pivot freely)
    tier2_max: int  # Max cardinality for tier 2 (keep but bucket values)
    tier3_max: int  # Max cardinality for tier 3 (top-N only or drop)
    # tier4: anything above tier3_max (drop or hash-bucket)


# Default cardinality thresholds (configurable per deployment)
DEFAULT_CARDINALITY_THRESHOLDS: CardinalityThresholds = {
    "tier1_max": 10,   # 1-10: always keep, pivot freely
    "tier2_max": 50,   # 11-50: keep but bucket values
    "tier3_max": 200,  # 51-200: top-N only or drop
    # 200+: drop or hash-bucket
}

# Default actions per tier
DEFAULT_TIER_ACTIONS = {
    1: "keep",      # Always keep, pivot freely
    2: "bucket",    # Keep but bucket values
    3: "top_n",     # Top-N only or drop
    4: "drop",      # Drop or hash-bucket
}

# Default N for top-N filtering
DEFAULT_TOP_N = 20

# Default profiling window in hours
DEFAULT_PROFILING_WINDOW_HOURS = 1.0

# Default aggregation window in seconds
DEFAULT_AGGREGATION_WINDOW_SECONDS = 60

# Histogram percentiles to compute
DEFAULT_PERCENTILES = [0.5, 0.75, 0.9, 0.95, 0.99]

# Feature layer definitions
FEATURE_LAYERS = {
    1: "Entity (tier-1 labels) x metric x agg x status_bucket",
    2: "+ method_bucket, operation_bucket",
    3: "+ top-N routes/endpoints",
}


def get_tier(cardinality: int, thresholds: CardinalityThresholds) -> int:
    """Determine the tier for a given cardinality.

    Args:
        cardinality: Number of distinct values
        thresholds: Cardinality threshold configuration

    Returns:
        Tier number (1-4)
    """
    if cardinality <= thresholds["tier1_max"]:
        return 1
    if cardinality <= thresholds["tier2_max"]:
        return 2
    if cardinality <= thresholds["tier3_max"]:
        return 3
    return 4


def get_action(tier: int, tier_actions: dict[int, str] | None = None) -> str:
    """Get the default action for a tier.

    Args:
        tier: Tier number (1-4)
        tier_actions: Optional custom tier actions mapping

    Returns:
        Action string (keep, bucket, top_n, drop)
    """
    actions = tier_actions or DEFAULT_TIER_ACTIONS
    return actions.get(tier, "drop")
