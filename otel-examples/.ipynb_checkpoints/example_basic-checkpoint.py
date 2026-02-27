#!/usr/bin/env python3
"""
Basic example of using the OTel ETL pipeline.

This script:
1. Runs the profiler to generate schema config
2. Fetches metrics from Prometheus
3. Transforms them into ML-ready features
4. Saves the output
"""

import sys
sys.path.insert(0, '..')

from datetime import datetime, timedelta
import pandas as pd

from otel_etl import run_profiler, fetch_and_denormalize
from otel_etl.feature_generator.schema_registry import SchemaRegistry

# Configuration
PROMETHEUS_URL = "http://localhost:9090"
OUTPUT_DIR = "output"

def main():
    print("=" * 80)
    print("OTel Metrics ETL - Basic Example")
    print("=" * 80)

    # Step 1: Run profiler to generate schema config
    print("\n[1/3] Running profiler...")
    try:
        schema = run_profiler(
            prometheus_url=PROMETHEUS_URL,
            output_path=f"{OUTPUT_DIR}/schema_config.yaml",
            profiling_window_hours=1,
            cardinality_thresholds={
                "tier1_max": 10,
                "tier2_max": 50,
                "tier3_max": 200,
            },
        )
        print(f"✓ Schema generated with {len(schema['metrics'])} metric families")
    except Exception as e:
        print(f"✗ Profiler failed: {e}")
        print("\nMake sure Prometheus is running at", PROMETHEUS_URL)
        return 1

    # Step 2: Fetch and transform metrics
    print("\n[2/3] Fetching and transforming metrics...")
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)

        features_df = fetch_and_denormalize(
            prometheus_url=PROMETHEUS_URL,
            start=start_time,
            end=end_time,
            step="60s",
            schema_config=f"{OUTPUT_DIR}/schema_config.yaml",
            column_registry=f"{OUTPUT_DIR}/column_registry.yaml",
            layers=[1, 2, 3],
            window_seconds=60,
            include_deltas=True,
        )

        print(f"✓ Generated {len(features_df)} rows × {len(features_df.columns)} columns")
        print(f"  Entities: {features_df['entity_key'].nunique()}")
        print(f"  Time range: {features_df['timestamp'].min()} to {features_df['timestamp'].max()}")

    except Exception as e:
        print(f"✗ Transformation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 3: Save output
    print("\n[3/3] Saving output...")
    try:
        features_df.to_parquet(f"{OUTPUT_DIR}/features.parquet")
        features_df.to_csv(f"{OUTPUT_DIR}/features.csv", index=False)
        print(f"✓ Saved features to {OUTPUT_DIR}/features.parquet and .csv")

        # Save column list for reference
        with open(f"{OUTPUT_DIR}/feature_columns.txt", "w") as f:
            for col in features_df.columns:
                f.write(f"{col}\n")
        print(f"✓ Saved feature list to {OUTPUT_DIR}/feature_columns.txt")

    except Exception as e:
        print(f"✗ Save failed: {e}")
        return 1

    # Print sample data
    print("\n" + "=" * 80)
    print("Sample Output (first 3 rows, first 5 feature columns):")
    print("=" * 80)

    feature_cols = [c for c in features_df.columns if c not in ['timestamp', 'entity_key']]
    sample_cols = ['timestamp', 'entity_key'] + feature_cols[:5]
    print(features_df[sample_cols].head(3))

    print("\n" + "=" * 80)
    print("Summary Statistics:")
    print("=" * 80)

    completeness = features_df[feature_cols].notna().mean(axis=1).mean()
    print(f"Overall completeness: {completeness:.1%}")
    print(f"Sparse columns (>50% NaN): {(features_df[feature_cols].isna().mean() > 0.5).sum()}")

    print("\n✓ Example complete!")
    print(f"\nNext steps:")
    print(f"  - View schema: cat {OUTPUT_DIR}/schema_config.yaml")
    print(f"  - View features: cat {OUTPUT_DIR}/feature_columns.txt")
    print(f"  - Load data: pd.read_parquet('{OUTPUT_DIR}/features.parquet')")

    return 0


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sys.exit(main())
