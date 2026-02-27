# OTel Metrics ETL

A self-configuring ETL pipeline that transforms high-cardinality OpenTelemetry metrics from Prometheus into stable, ML-ready pandas DataFrames.

## Features

- **Automatic Schema Discovery**: Profiles Prometheus metrics to learn structure, cardinality, and semantics
- **Intelligent Cardinality Reduction**: Buckets status codes, HTTP methods, operations; parameterizes routes; applies top-N filtering
- **Stable Feature Schema**: Column registry ensures consistent output across runs
- **ML-Ready Output**: Wide-format DataFrames with timestamp and entity_key indices
- **Time-Window Features**: Optional delta and percentage change features
- **Type-Aware Aggregation**: Handles counters, histograms, and gauges appropriately

## Installation

```bash
# Requires Python 3.10+
pip install pandas numpy pyyaml
```

## Quick Start

### 1. Profile Your Metrics

Run the profiler to discover metrics and generate a schema configuration:

```python
from otel_etl import run_profiler

# Generate schema config from Prometheus
schema = run_profiler(
    prometheus_url="http://localhost:9090",
    output_path="schema_config.yaml",
    profiling_window_hours=1,
    cardinality_thresholds={
        "tier1_max": 10,   # Low cardinality: always keep
        "tier2_max": 50,   # Medium: bucket values
        "tier3_max": 200,  # High: top-N only
    }
)
```

This creates `schema_config.yaml` with discovered metrics, labels, cardinality tiers, and transformation rules.

### 2. Transform Metrics

Transform raw Prometheus data into ML-ready features:

```python
from otel_etl import denormalize_metrics, fetch_and_denormalize
from datetime import datetime, timedelta

# Option A: Fetch and transform in one call
features_df = fetch_and_denormalize(
    prometheus_url="http://localhost:9090",
    start=datetime.utcnow() - timedelta(hours=1),
    end=datetime.utcnow(),
    step="60s",
    schema_config="schema_config.yaml",
    column_registry="column_registry.yaml",
    layers=[1, 2, 3],
    include_deltas=True,
)

# Option B: Transform pre-fetched data
# raw_df = fetch_from_prometheus(...)  # Your data fetching logic
features_df = denormalize_metrics(
    raw_df,
    schema_config="schema_config.yaml",
    column_registry="column_registry.yaml",
)
```

### 3. Use the Features

```python
# Features are in wide format: timestamp x entity_key x features
print(features_df.shape)  # (rows, columns)
print(features_df.columns)  # ['timestamp', 'entity_key', 'http_server_duration__p99__service_name_frontend__status_bucket_success', ...]

# Features are ML-ready
X = features_df.drop(columns=['timestamp', 'entity_key'])
y = features_df['http_server_duration__p99__service_name_frontend__status_bucket_error']
```

## Architecture

### Pipeline Flow

```
Prometheus → Profiler → Schema Config
                            ↓
Raw Metrics → Transformers → Aggregators → Feature Generator → Wide DataFrame
                                                ↓
                                        Column Registry (stability)
```

### Module Structure

- **profiler/**: Schema discovery and cardinality analysis
- **transformer/**: Label value bucketing and parameterization
- **aggregator/**: Type-aware metric aggregation (counter, histogram, gauge)
- **feature_generator/**: Entity grouping, feature naming, wide formatting
- **config/**: Default thresholds and user overrides

## Key Concepts

### Cardinality Tiers

Labels are classified into tiers based on distinct value counts:

| Tier | Cardinality | Action | Example |
|------|-------------|--------|---------|
| 1 | 1-10 | Always keep | service_name, http_method |
| 2 | 11-50 | Bucket values | http_status_code → success/error |
| 3 | 51-200 | Top-N only | http_route (keep top 20 routes) |
| 4 | 200+ | Drop | trace_id, request_id |

### Semantic Classification

Labels are automatically classified by name patterns:

- **Resource**: service_name, instance, pod → Entity identifiers
- **Signal**: status_code, error_type → Bucketed to categories
- **Dimension**: http_method, operation → Bucketed or kept
- **Correlation**: trace_id, request_id → Dropped for aggregation

### Transformations

**Status Code Bucketing**: `200` → `success`, `500` → `server_error`

**HTTP Method Bucketing**: `GET` → `read`, `POST` → `write`

**Operation Bucketing**: `SELECT` → `read`, `INSERT` → `write`

**Route Parameterization**: `/api/users/12345` → `/api/users/{id}`

**Top-N Filtering**: Keep top 20 routes, bucket rest as `__other__`

### Entity Keys

Metrics are grouped by entity_key, computed from tier-1 labels:

```
service_name=frontend → entity_key: "service_name=frontend"
service_name=checkout::instance=10.0.1.5 → entity_key: "instance=10.0.1.5::service_name=checkout"
```

### Feature Names

Features follow a stable naming convention:

```
{metric_family}__{aggregation}__{label1}_{value1}__{label2}_{value2}
```

Examples:
- `http_server_duration__p99__service_name_frontend__status_bucket_success`
- `http_server_requests__rate__http_method_read__service_name_checkout`
- `db_client_duration__mean__operation_bucket_read__service_name_cart`

## Configuration

### Schema Config (Generated)

`schema_config.yaml` is generated by `run_profiler()`:

```yaml
profiled_at: "2024-01-15T10:00:00Z"
profiling_window_hours: 1
cardinality_thresholds:
  tier1_max: 10
  tier2_max: 50
  tier3_max: 200

metrics:
  http_server_request_duration:
    type: histogram
    labels:
      service_name:
        tier: 1
        cardinality: 5
        action: keep
      http_status_code:
        tier: 1
        cardinality: 8
        action: bucket
        bucket_type: status_code
      http_route:
        tier: 2
        cardinality: 45
        action: top_n
        top_values: ["/api/products", "/api/cart", ...]
```

### User Overrides (Optional)

`config/overrides.yaml` allows customization:

```yaml
force_drop_labels:
  - internal_request_id
  - debug_flag

vip_values:
  http_route:
    - /health
    - /ready
    - /metrics

tier_overrides:
  customer_tier:
    tier: 1
    action: keep
```

### Column Registry (Stability)

`column_registry.yaml` tracks feature columns across runs:

```yaml
created_at: "2024-01-15T10:00:00Z"
updated_at: "2024-01-15T11:00:00Z"
index_cols: [timestamp, entity_key]
columns:
  - http_server_duration__p99__service_name_frontend
  - http_server_requests__rate__service_name_checkout
  - ...
```

New features are appended; old features remain (may become NaN if no longer observed).

## Advanced Usage

### Filtering Metrics

```python
# Only profile specific metrics
schema = run_profiler(
    prometheus_url="http://localhost:9090",
    include_patterns=[r"^http_.*", r"^grpc_.*"],
    exclude_patterns=[r".*test.*"],
)
```

### Custom Entity Labels

```python
# Use specific labels for entity grouping
features_df = denormalize_metrics(
    raw_df,
    entity_labels=["service_name", "environment"],
)
```

### Layer Control

Feature layers control granularity:

- **Layer 1**: Entity × metric × aggregation × status_bucket (50-150 features)
- **Layer 2**: + method_bucket, operation_bucket (150-400 features)
- **Layer 3**: + top-N routes/endpoints (400-800 features)

```python
# Use only coarse-grained features
features_df = denormalize_metrics(raw_df, layers=[1, 2])
```

### Delta Features

```python
# Add time-window comparison features
features_df = denormalize_metrics(
    raw_df,
    include_deltas=True,  # Adds __delta_5m, __delta_1h, __pct_change_1h
)
```

### Schema Evolution

```python
from otel_etl.profiler.schema_generator import load_schema, diff_schemas

old_schema = load_schema("schema_config_v1.yaml")
new_schema = run_profiler(prometheus_url="http://localhost:9090")

# Check what changed
diff = diff_schemas(old_schema, new_schema)
print(f"Added metrics: {diff['added_metrics']}")
print(f"Tier changes: {diff['tier_changes']}")
```

## Examples

See `examples/` directory for complete examples:

- `example_basic.py` - Basic profiling and transformation
- `example_streaming.py` - Streaming metric processing
- `example_ml_pipeline.py` - Full ML pipeline with anomaly detection

## Design Principles

1. **Discover, don't hardcode** — Profile the data to learn its shape
2. **Classify by cardinality** — Automatic tiering based on distinct value counts
3. **Classify by semantics** — Label name patterns determine handling
4. **Bucket values, not labels** — Keep the dimension, compress its values
5. **Parameterize paths** — `/users/123` → `/users/{id}`
6. **Top-N with escape hatch** — Bounded cardinality, `__other__` for the tail
7. **Stable schema** — Column registry ensures consistent output shape
8. **Layered features** — Coarse (stable) to fine (more signal, more risk)

## References

- Design document: `../coding-plans/002-claude-opus.md`
- OpenTelemetry Demo: https://github.com/open-telemetry/opentelemetry-demo
# otel_etl
