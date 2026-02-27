"""Streaming utilities for continuous metric processing."""

from typing import Any, Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import time

import pandas as pd

from otel_etl.main import denormalize_metrics
from otel_etl.utils.prometheus_client import PrometheusClient
from otel_etl.profiler.schema_generator import load_schema, SchemaConfig
from otel_etl.feature_generator.schema_registry import SchemaRegistry

logger = logging.getLogger(__name__)


class StreamingETL:
    """Streaming ETL processor for continuous metric transformation.

    Maintains state across invocations for efficient streaming processing.
    """

    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        schema_config_path: Optional[str] = None,
        column_registry_path: Optional[str] = None,
        layers: list[int] = None,
        window_seconds: float = 60,
        include_deltas: bool = False,
        lookback_periods: int = 5,
    ):
        """Initialize streaming ETL processor.

        Args:
            prometheus_url: Prometheus server URL
            schema_config_path: Path to schema config (generated once, reused)
            column_registry_path: Path to column registry (persisted between calls)
            layers: Feature layers to include
            window_seconds: Aggregation window in seconds
            include_deltas: Whether to compute delta features
            lookback_periods: Number of previous periods to keep for delta computation
        """
        self.prometheus_url = prometheus_url
        self.client = PrometheusClient(prometheus_url)

        self.schema_config_path = schema_config_path
        self.column_registry_path = column_registry_path
        self.layers = layers or [1, 2, 3]
        self.window_seconds = window_seconds
        self.include_deltas = include_deltas
        self.lookback_periods = lookback_periods

        # State management
        self.schema: Optional[SchemaConfig] = None
        self.registry: Optional[SchemaRegistry] = None
        self.last_fetch_time: Optional[datetime] = None
        self.historical_data: list[pd.DataFrame] = []

        # Load schema and registry if paths provided
        self._load_state()

    def _load_state(self):
        """Load schema config and column registry from disk."""
        if self.schema_config_path and Path(self.schema_config_path).exists():
            self.schema = load_schema(self.schema_config_path)
            logger.info(f"Loaded schema config from {self.schema_config_path}")

        if self.column_registry_path:
            if Path(self.column_registry_path).exists():
                self.registry = SchemaRegistry.load(self.column_registry_path)
                logger.info(f"Loaded column registry from {self.column_registry_path}")
            else:
                self.registry = SchemaRegistry()
                logger.info("Created new column registry")

    def _save_state(self):
        """Save column registry to disk."""
        if self.registry and self.column_registry_path:
            self.registry.save(self.column_registry_path)
            logger.debug(f"Saved column registry to {self.column_registry_path}")

    def fetch_window(
        self,
        metric_names: list[str],
        end_time: Optional[datetime] = None,
        step: str = "60s",
    ) -> pd.DataFrame:
        """Fetch metrics for a single time window.

        Args:
            metric_names: List of metrics to fetch
            end_time: End time (default: now)
            step: Query resolution

        Returns:
            Raw metrics DataFrame
        """
        if end_time is None:
            end_time = datetime.now(timezone.utc)

        # Fetch one window worth of data
        start_time = end_time - timedelta(seconds=self.window_seconds)

        logger.info(f"Fetching metrics from {start_time} to {end_time}")

        raw_df = self.client.fetch_metrics_range(
            metric_names,
            start_time,
            end_time,
            step,
        )

        self.last_fetch_time = end_time

        return raw_df

    def process_window(
        self,
        raw_df: pd.DataFrame,
        save_state: bool = True,
    ) -> pd.DataFrame:
        """Process a single window of metrics.

        Args:
            raw_df: Raw metrics DataFrame
            save_state: Whether to save state after processing

        Returns:
            Transformed features DataFrame
        """
        if raw_df.empty:
            logger.warning("Empty input DataFrame")
            return pd.DataFrame()

        # Transform metrics
        features_df = denormalize_metrics(
            raw_df,
            schema_config=self.schema,
            column_registry=self.registry,
            layers=self.layers,
            window_seconds=self.window_seconds,
            include_deltas=False,  # We'll compute deltas ourselves
        )

        # Store for delta computation
        if self.include_deltas:
            self.historical_data.append(features_df.copy())
            # Keep only recent history
            if len(self.historical_data) > self.lookback_periods:
                self.historical_data = self.historical_data[-self.lookback_periods:]

            # Compute deltas if we have history
            if len(self.historical_data) > 1:
                features_df = self._compute_streaming_deltas(features_df)

        # Save state
        if save_state:
            self._save_state()

        return features_df

    def _compute_streaming_deltas(self, current_df: pd.DataFrame) -> pd.DataFrame:
        """Compute delta features using historical data.

        Args:
            current_df: Current window's features

        Returns:
            DataFrame with delta features added
        """
        if len(self.historical_data) < 2:
            return current_df

        result = current_df.copy()

        # Get previous window
        prev_df = self.historical_data[-2]

        feature_cols = [
            c for c in current_df.columns
            if c not in ['timestamp', 'entity_key']
        ]

        # Merge with previous window on entity_key
        merged = result.merge(
            prev_df[['entity_key'] + feature_cols],
            on='entity_key',
            how='left',
            suffixes=('', '_prev'),
        )

        # Compute deltas
        for col in feature_cols:
            prev_col = f"{col}_prev"
            if prev_col in merged.columns:
                result[f"{col}__delta_1w"] = merged[col] - merged[prev_col]

        return result

    def run_once(
        self,
        metric_names: list[str],
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch and process metrics for one window.

        Args:
            metric_names: List of metrics to fetch
            end_time: End time (default: now)

        Returns:
            Transformed features DataFrame
        """
        raw_df = self.fetch_window(metric_names, end_time)
        return self.process_window(raw_df)

    def run_continuous(
        self,
        metric_names: list[str],
        interval_seconds: float = 60,
        callback: Optional[Any] = None,
        max_iterations: Optional[int] = None,
    ):
        """Run continuous streaming processing.

        Args:
            metric_names: List of metrics to fetch
            interval_seconds: Time between fetches
            callback: Optional callback function(features_df) to call with each result
            max_iterations: Maximum iterations (None = infinite)

        Yields:
            Transformed features DataFrames
        """
        iteration = 0

        logger.info(f"Starting continuous streaming (interval={interval_seconds}s)")

        while max_iterations is None or iteration < max_iterations:
            try:
                start_time = time.time()

                # Process window
                features_df = self.run_once(metric_names)

                # Call callback if provided
                if callback:
                    callback(features_df)

                yield features_df

                # Sleep until next interval
                elapsed = time.time() - start_time
                sleep_time = max(0, interval_seconds - elapsed)

                if sleep_time > 0:
                    logger.debug(f"Sleeping for {sleep_time:.1f}s")
                    time.sleep(sleep_time)

                iteration += 1

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}", exc_info=True)
                time.sleep(interval_seconds)  # Back off on error

        logger.info(f"Streaming stopped after {iteration} iterations")

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the streaming processor.

        Returns:
            Dictionary with stats
        """
        return {
            "last_fetch_time": self.last_fetch_time.isoformat() if self.last_fetch_time else None,
            "historical_windows": len(self.historical_data),
            "schema_loaded": self.schema is not None,
            "registry_columns": len(self.registry.columns) if self.registry else 0,
        }


def create_streaming_processor(
    prometheus_url: str = "http://localhost:9090",
    config_dir: str = "./streaming_state",
) -> StreamingETL:
    """Create a streaming processor with default configuration.

    Args:
        prometheus_url: Prometheus server URL
        config_dir: Directory to store state files

    Returns:
        StreamingETL instance
    """
    config_path = Path(config_dir)
    config_path.mkdir(parents=True, exist_ok=True)

    return StreamingETL(
        prometheus_url=prometheus_url,
        schema_config_path=str(config_path / "schema_config.yaml"),
        column_registry_path=str(config_path / "column_registry.yaml"),
        include_deltas=True,
        lookback_periods=5,
    )
