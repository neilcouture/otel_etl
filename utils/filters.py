"""Filtering utilities for metrics DataFrames."""

import re
import pandas as pd
from typing import Callable

from otel_etl.utils.name_sanitizer import extract_metric_family


def convert_wide_to_otel_format(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    metric_col: str = 'metric',
    value_col: str = 'value',
) -> pd.DataFrame:
    """Convert wide-format DataFrame to otel_etl format.

    Use this if your DataFrame has separate columns for each label
    (e.g., from fetch_metrics_range_filtered) and you need to convert
    it to the format expected by denormalize_metrics().

    Args:
        df: DataFrame with separate label columns
        timestamp_col: Name of timestamp column
        metric_col: Name of metric column
        value_col: Name of value column

    Returns:
        DataFrame with 'labels' dict column

    Example:
        >>> # Your CSV has: timestamp, metric, value, service_name, status_code, http_method
        >>> df = pd.read_csv("metrics.csv")
        >>> otel_df = convert_wide_to_otel_format(df)
        >>> # Now otel_df has: timestamp, metric, labels (dict), value
        >>> features = denormalize_metrics(otel_df)
    """
    # Identify label columns (everything except timestamp, metric, value)
    reserved_cols = {timestamp_col, metric_col, value_col}
    label_cols = [col for col in df.columns if col not in reserved_cols]

    # Create labels dict for each row
    def make_labels_dict(row):
        return {col: str(row[col]) for col in label_cols if pd.notna(row[col])}

    result = pd.DataFrame({
        'timestamp': df[timestamp_col],
        'metric': df[metric_col],
        'labels': df.apply(make_labels_dict, axis=1),
        'value': df[value_col],
    })

    # Convert timestamp to datetime if needed
    if result['timestamp'].dtype != 'datetime64[ns]':
        result['timestamp'] = pd.to_datetime(result['timestamp'])

    return result


def filter_by_labels(
    df: pd.DataFrame,
    **label_filters: str | list[str],
) -> pd.DataFrame:
    """Filter DataFrame by label values.

    Args:
        df: Raw metrics DataFrame with 'labels' column
        **label_filters: Keyword arguments where key is label name and value is the value(s) to keep

    Returns:
        Filtered DataFrame

    Example:
        >>> filtered = filter_by_labels(df, service_name='frontend')
        >>> filtered = filter_by_labels(df, service_name=['frontend', 'checkout'])
        >>> filtered = filter_by_labels(df, service_name='frontend', status_code='200')
    """
    result = df.copy()

    for label_name, values in label_filters.items():
        if isinstance(values, str):
            values = [values]

        mask = result['labels'].apply(
            lambda x: x.get(label_name) in values if isinstance(x, dict) else False
        )
        result = result[mask]

    return result


def exclude_by_labels(
    df: pd.DataFrame,
    **label_filters: str | list[str],
) -> pd.DataFrame:
    """Exclude rows matching label values.

    Args:
        df: Raw metrics DataFrame with 'labels' column
        **label_filters: Keyword arguments where key is label name and value is the value(s) to exclude

    Returns:
        Filtered DataFrame

    Example:
        >>> filtered = exclude_by_labels(df, service_name='test')
        >>> filtered = exclude_by_labels(df, service_name=['test', 'debug'])
    """
    result = df.copy()

    for label_name, values in label_filters.items():
        if isinstance(values, str):
            values = [values]

        mask = result['labels'].apply(
            lambda x: x.get(label_name) not in values if isinstance(x, dict) else True
        )
        result = result[mask]

    return result


def filter_by_service(
    df: pd.DataFrame,
    services: str | list[str],
    service_label: str = 'service_name',
) -> pd.DataFrame:
    """Filter DataFrame by service name(s).

    Args:
        df: Raw metrics DataFrame
        services: Service name(s) to keep
        service_label: Label name for service (default: 'service_name', could also be 'job', 'app', etc.)

    Returns:
        Filtered DataFrame

    Example:
        >>> filtered = filter_by_service(df, 'frontend')
        >>> filtered = filter_by_service(df, ['frontend', 'checkout'])
        >>> filtered = filter_by_service(df, 'my-app', service_label='job')
    """
    return filter_by_labels(df, **{service_label: services})


def filter_by_metrics(
    df: pd.DataFrame,
    patterns: str | list[str],
    exclude: bool = False,
) -> pd.DataFrame:
    """Filter DataFrame by metric name patterns.

    Args:
        df: Raw metrics DataFrame
        patterns: Regex pattern(s) to match metric names
        exclude: If True, exclude matching metrics instead of including

    Returns:
        Filtered DataFrame

    Example:
        >>> # Only HTTP metrics
        >>> filtered = filter_by_metrics(df, r'^http_')
        >>> # Only HTTP and gRPC metrics
        >>> filtered = filter_by_metrics(df, [r'^http_', r'^grpc_'])
        >>> # Exclude test metrics
        >>> filtered = filter_by_metrics(df, r'_test_', exclude=True)
    """
    if isinstance(patterns, str):
        patterns = [patterns]

    def matches_any_pattern(metric_name):
        return any(re.search(pattern, str(metric_name)) for pattern in patterns)

    mask = df['metric'].apply(matches_any_pattern)

    if exclude:
        mask = ~mask

    return df[mask]


def filter_by_custom(
    df: pd.DataFrame,
    filter_func: Callable[[pd.Series], bool],
) -> pd.DataFrame:
    """Filter DataFrame by custom function.

    Args:
        df: Raw metrics DataFrame
        filter_func: Function that takes a row (pd.Series) and returns True to keep it

    Returns:
        Filtered DataFrame

    Example:
        >>> # Keep only metrics with high values
        >>> filtered = filter_by_custom(df, lambda row: row['value'] > 100)
        >>> # Complex label filtering
        >>> filtered = filter_by_custom(
        ...     df,
        ...     lambda row: row['labels'].get('service') == 'frontend' and
        ...                 row['labels'].get('status_code') == '200'
        ... )
    """
    return df[df.apply(filter_func, axis=1)]


def sample_by_time(
    df: pd.DataFrame,
    sample_rate: str = '5min',
    timestamp_col: str = 'timestamp',
) -> pd.DataFrame:
    """Downsample DataFrame by time intervals.

    Args:
        df: Raw metrics DataFrame
        sample_rate: Sampling interval (e.g., '1min', '5min', '1h')
        timestamp_col: Name of timestamp column

    Returns:
        Downsampled DataFrame

    Example:
        >>> # Sample every 5 minutes
        >>> sampled = sample_by_time(df, '5min')
        >>> # Sample every hour
        >>> sampled = sample_by_time(df, '1h')
    """
    df_copy = df.copy()

    if df_copy[timestamp_col].dtype != 'datetime64[ns]':
        df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])

    df_copy['time_bucket'] = df_copy[timestamp_col].dt.floor(sample_rate)

    # Take first value in each time bucket for each metric/label combination
    def make_key(row):
        labels_str = str(sorted(row['labels'].items())) if isinstance(row['labels'], dict) else str(row['labels'])
        return f"{row['metric']}::{labels_str}"

    df_copy['group_key'] = df_copy.apply(make_key, axis=1)

    sampled = df_copy.groupby(['time_bucket', 'group_key']).first().reset_index(drop=True)
    sampled = sampled.drop(columns=['time_bucket', 'group_key'])

    return sampled


def get_available_services(
    df: pd.DataFrame,
    service_label: str = 'service_name',
) -> list[str]:
    """Get list of available services in the DataFrame.

    Args:
        df: Raw metrics DataFrame
        service_label: Label name for service

    Returns:
        Sorted list of unique service names

    Example:
        >>> services = get_available_services(df)
        >>> print(f"Available services: {services}")
    """
    services = set()

    for labels in df['labels']:
        if isinstance(labels, dict) and service_label in labels:
            services.add(labels[service_label])

    return sorted(services)


def get_label_values(
    df: pd.DataFrame,
    label_name: str,
) -> list[str]:
    """Get all distinct values for a label.

    Args:
        df: Raw metrics DataFrame
        label_name: Name of the label

    Returns:
        Sorted list of unique values

    Example:
        >>> status_codes = get_label_values(df, 'status_code')
        >>> print(f"Status codes: {status_codes}")
    """
    values = set()

    for labels in df['labels']:
        if isinstance(labels, dict) and label_name in labels:
            values.add(labels[label_name])

    return sorted(values)


def filter_salient_metrics(
    metric_names: list[str],
    prefer_patterns: list[str] | None = None,
    drop_patterns: list[str] | None = None,
    keep_one_per_group: bool = True,
    keep_latency_and_throughput: bool = True,
) -> list[str]:
    """Filter metrics to keep only the most salient ones per category.

    Groups metrics by broad prefix and keeps the most important one per group,
    preferring duration/latency metrics over size/count metrics.

    Args:
        metric_names: List of metric names
        prefer_patterns: Regex patterns for preferred metrics (default: duration, latency)
        drop_patterns: Regex patterns for metrics to always drop
        keep_one_per_group: If True, keep only ONE metric family per prefix group
        keep_latency_and_throughput: If True, keep BOTH latency and throughput metrics per group

    Returns:
        Filtered list of metric names

    Example:
        >>> metrics = ['rpc_client_duration_ms_bucket', 'rpc_client_request_size_bytes_bucket', ...]
        >>> salient = filter_salient_metrics(metrics)
        >>> # Returns only rpc_client_duration_ms_* metrics
    """
    # Latency patterns (timing)
    latency_patterns = [r'duration', r'latency', r'time_seconds']
    # Throughput patterns (counts/rates)
    throughput_patterns = [r'calls_total$', r'requests_total$', r'_total$']

    if prefer_patterns is None:
        prefer_patterns = [
            r'duration', r'latency',           # timing metrics - highest priority
            r'calls_total$', r'requests_total', # throughput
            r'exceptions', r'errors',           # error signals
            r'memory_usage', r'memory_used',    # memory
            r'gc_collections',                  # GC activity
        ]

    if drop_patterns is None:
        drop_patterns = [
            # Metadata
            r'^target_info$',

            # Size metrics (prefer duration/latency instead)
            r'_size_bytes',
            r'_per_rpc',

            # Go runtime (keep only critical ones)
            r'^go_memory_(?!used)',  # keep only memory_used
            r'^go_config',
            r'^go_processor',

            # .NET runtime (keep only exceptions, gc_collections, memory_working_set)
            r'^aspnetcore_',
            r'^dotnet_assembly',
            r'^dotnet_jit',
            r'^dotnet_monitor',
            r'^dotnet_thread_pool',
            r'^dotnet_timer',
            r'^dotnet_process_cpu',

            # JVM runtime (keep only gc_duration, memory_used, exceptions)
            r'^jvm_class',
            r'^jvm_cpu_(?!recent_utilization)',  # keep only utilization
            r'^jvm_thread_count$',

            # Node.js/V8 runtime (keep only eventloop delay percentiles, gc_duration, heap_used)
            r'^nodejs_eventloop_(?!delay_(p90|p99))',  # keep only p90/p99
            r'^nodejs_eventloop_time',
            r'^v8js_memory_(?!heap_used)',  # keep only heap_used

            # Python runtime (keep only gc_collections)
            r'^cpython_gc_(?!collections_total)',

            # Process metrics (keep only memory_usage, cpu_time)
            r'^process_cpu_count',
            r'^process_thread_count',
            r'^process_open_file',
            r'^process_context_switches',
            r'^process_runtime_',
            r'^process_disk',

            # System metrics (keep only cpu/memory utilization, network errors)
            r'^system_cpu_(?!(utilization|load_average))',
            r'^system_disk_',
            r'^system_filesystem_',
            r'^system_paging_',
            r'^system_processes_',
            r'^system_swap_',
            r'^system_thread_count',
            r'^system_uptime',
            r'^system_memory_(?!usage)',  # keep only usage
            r'^system_network_(?!(errors|dropped))',  # keep errors/dropped

            # Kestrel (keep only duration)
            r'^kestrel_queued',
            r'^kestrel_active',

            # Feature flags
            r'^feature_flag_',

            # Traces (redundant with direct instrumentation)
            r'^traces_span_metrics_',

            # OTel Collector internals
            r'^otelcol_',
            r'^otlp_exporter',
            r'^processedLogs',
            r'^processedSpans',
            r'^queueSize',

            # Container metrics (infrastructure)
            r'^container_',

            # HTTP check (infrastructure)
            r'^httpcheck_',

            # Nginx (infrastructure, unless it's your app server)
            r'^nginx_',

            # Jaeger internals
            r'^jaeger_storage_',

            # Kafka - drop most internal metrics, keep lag/latency/throughput
            r'^kafka_consumer_(?!(records_lag|fetch_latency|records_consumed))',
            r'^kafka_controller',
            r'^kafka_isr',
            r'^kafka_leaderElection',
            r'^kafka_logs',
            r'^kafka_message_count',
            r'^kafka_network_io',
            r'^kafka_partition_(?!offline)',
            r'^kafka_purgatory',
            r'^kafka_request_(?!(time_99p|failed))',

            # PostgreSQL - keep only critical metrics
            r'^postgresql_bgwriter',
            r'^postgresql_blks',
            r'^postgresql_blocks',
            r'^postgresql_database_count',
            r'^postgresql_index_(?!scans)',
            r'^postgresql_table_(?!size)',
            r'^postgresql_tup_',

            # Redis - keep only critical metrics
            r'^redis_clients_(?!connected)',
            r'^redis_cpu',
            r'^redis_db_(?!keys)',
            r'^redis_keys_(?!evicted|expired)',
            r'^redis_keyspace_(?!hits|misses)',
            r'^redis_latest_fork',
            r'^redis_memory_(?!used)',
            r'^redis_net_',
            r'^redis_rdb_',
            r'^redis_replication_',
            r'^redis_slaves',
            r'^redis_uptime',
        ]

    # Get unique metric families
    families = {}
    for metric in metric_names:
        family = extract_metric_family(metric)
        if family not in families:
            families[family] = []
        families[family].append(metric)

    # Group families by BROAD prefix
    prefix_groups = {}
    for family in families:
        prefix = _get_broad_prefix(family)
        if prefix not in prefix_groups:
            prefix_groups[prefix] = []
        prefix_groups[prefix].append(family)

    # For each prefix group, select the most salient family
    selected_families = set()

    for prefix, group_families in prefix_groups.items():
        # Filter out families matching drop patterns
        kept_families = []
        for family in group_families:
            should_drop = False
            for pattern in drop_patterns:
                if re.search(pattern, family):
                    should_drop = True
                    break
            if not should_drop:
                kept_families.append(family)

        if not kept_families:
            continue

        # Score each family by preference (lower = better)
        def score_family(family):
            for i, pattern in enumerate(prefer_patterns):
                if re.search(pattern, family):
                    return i
            return len(prefer_patterns) + 1  # no match = lowest priority

        kept_families.sort(key=score_family)

        if keep_one_per_group:
            # Keep the best one
            selected_families.add(kept_families[0])

            # Also keep throughput if we kept latency (and vice versa)
            if keep_latency_and_throughput:
                best = kept_families[0]
                is_latency = any(re.search(p, best) for p in latency_patterns)
                is_throughput = any(re.search(p, best) for p in throughput_patterns)

                for family in kept_families[1:]:
                    if is_latency and any(re.search(p, family) for p in throughput_patterns):
                        selected_families.add(family)
                        break
                    if is_throughput and any(re.search(p, family) for p in latency_patterns):
                        selected_families.add(family)
                        break
        else:
            # Keep all that match any prefer pattern
            for family in kept_families:
                if score_family(family) < len(prefer_patterns):
                    selected_families.add(family)
            # If none matched, keep the first one
            if not any(f in selected_families for f in kept_families):
                selected_families.add(kept_families[0])

    # Return all metrics belonging to selected families
    result = []
    for metric in metric_names:
        family = extract_metric_family(metric)
        if family in selected_families:
            result.append(metric)

    return result


def _get_broad_prefix(family: str) -> str:
    """Get broad prefix for grouping metric families.

    Examples:
        app_cart_add_item_latency_seconds -> app_cart
        rpc_client_duration_milliseconds -> rpc_client
        dotnet_gc_collections -> dotnet
        go_memory_used_bytes -> go
        http_server_request_duration -> http_server
        kafka_consumer_records_lag -> kafka_consumer
        postgresql_db_size_bytes -> postgresql
        redis_memory_used_bytes -> redis
    """
    parts = family.split('_')

    if len(parts) == 1:
        return family

    first = parts[0]

    # Single-word prefixes for runtime/system/infrastructure metrics
    runtime_prefixes = [
        'go', 'dotnet', 'jvm', 'nodejs', 'v8js', 'cpython',  # runtimes
        'process', 'system',  # OS metrics
        'kestrel', 'aspnetcore',  # .NET server
        'traces', 'feature',  # OTel/features
        'postgresql', 'redis', 'nginx',  # infrastructure
        'otelcol', 'otlp', 'jaeger',  # observability backend
        'container', 'httpcheck',  # infrastructure
        'gen',  # gen_ai -> gen
    ]

    if first in runtime_prefixes:
        return first

    # Two-word prefixes for protocol/client-server metrics
    if first in ['rpc', 'http', 'grpc', 'db', 'dns']:
        if len(parts) >= 2:
            return '_'.join(parts[:2])  # rpc_client, http_server, db_client, etc.

    # Kafka - distinguish consumer vs other
    if first == 'kafka':
        if len(parts) >= 2:
            return '_'.join(parts[:2])  # kafka_consumer, kafka_controller, etc.

    # App-specific: use first two parts to distinguish services
    if first == 'app' and len(parts) >= 2:
        return '_'.join(parts[:2])  # app_cart, app_checkout, app_payment, etc.

    # Default: first two parts
    if len(parts) >= 2:
        return '_'.join(parts[:2])

    return family


def explain_salient_filtering(metric_names: list[str]) -> None:
    """Print explanation of how metrics would be filtered.

    Shows prefix groups, families in each group, and what gets selected.
    """
    families = {}
    for metric in metric_names:
        family = extract_metric_family(metric)
        if family not in families:
            families[family] = []
        families[family].append(metric)

    prefix_groups = {}
    for family in families:
        prefix = _get_broad_prefix(family)
        if prefix not in prefix_groups:
            prefix_groups[prefix] = []
        prefix_groups[prefix].append(family)

    print(f"Found {len(families)} metric families in {len(prefix_groups)} prefix groups:\n")

    for prefix in sorted(prefix_groups.keys()):
        group_families = prefix_groups[prefix]
        print(f"[{prefix}] ({len(group_families)} families)")
        for family in sorted(group_families):
            print(f"    - {family}")
        print()


def filter_salient_metrics_verbose(
    metric_names: list[str],
    **kwargs,
) -> tuple[list[str], dict]:
    """Same as filter_salient_metrics but returns debug info.

    Returns:
        Tuple of (filtered_metrics, debug_info)
        debug_info contains: prefix_groups, selected_families, dropped_families
    """
    # Get families and groups
    families = {}
    for metric in metric_names:
        family = extract_metric_family(metric)
        if family not in families:
            families[family] = []
        families[family].append(metric)

    prefix_groups = {}
    for family in families:
        prefix = _get_broad_prefix(family)
        if prefix not in prefix_groups:
            prefix_groups[prefix] = []
        prefix_groups[prefix].append(family)

    # Run the filter
    result = filter_salient_metrics(metric_names, **kwargs)
    selected_families = set(extract_metric_family(m) for m in result)

    dropped_families = set(families.keys()) - selected_families

    debug_info = {
        'prefix_groups': prefix_groups,
        'selected_families': sorted(selected_families),
        'dropped_families': sorted(dropped_families),
        'original_count': len(metric_names),
        'filtered_count': len(result),
        'family_count_before': len(families),
        'family_count_after': len(selected_families),
    }

    return result, debug_info


def get_metric_families(metric_names: list[str]) -> dict[str, list[str]]:
    """Group metric names by their family.

    Args:
        metric_names: List of metric names

    Returns:
        Dict mapping family name to list of metrics

    Example:
        >>> get_metric_families(['http_request_duration_bucket', 'http_request_duration_sum'])
        {'http_request_duration': ['http_request_duration_bucket', 'http_request_duration_sum']}
    """
    families = {}
    for metric in metric_names:
        family = extract_metric_family(metric)
        if family not in families:
            families[family] = []
        families[family].append(metric)
    return families

def filter_salient_metrics_llm_openai(
        metric_names: list[str],
        model_id: str = "gpt-4o",
        max_families: int = 10) -> tuple[list[str], str]:
    """Filter metrics using OpenAI LLM to select the most salient ones.

    Args:
        metric_names: List of metric names to filter
        model_id: OpenAI model ID to use
        max_families: Maximum number of metric families to keep

    Returns:
        Tuple of (filtered_metrics, explanation) where explanation describes the LLM's reasoning

    Example:
        >>> metrics, explanation = filter_salient_metrics_llm_openai(metric_names)
        >>> print(f"Kept {len(metrics)} metrics")
        >>> print(explanation)
    """
    return filter_salient_metrics_llm(metric_names, _openai_llm_call, model_id, max_families)


def filter_salient_metrics_llm_claude(
        metric_names: list[str],
        model_id: str = "claude-sonnet-4-5-20250929",
        max_families: int = 10) -> tuple[list[str], str]:
    """Filter metrics using Claude LLM to select the most salient ones.

    Args:
        metric_names: List of metric names to filter
        model_id: Claude model ID to use
        max_families: Maximum number of metric families to keep

    Returns:
        Tuple of (filtered_metrics, explanation) where explanation describes the LLM's reasoning

    Example:
        >>> metrics, explanation = filter_salient_metrics_llm_claude(metric_names)
        >>> print(f"Kept {len(metrics)} metrics")
        >>> print(explanation)
    """
    return filter_salient_metrics_llm(metric_names, model_id=model_id, max_families=max_families)

def filter_salient_metrics_llm(
    metric_names: list[str],
    llm_func=None,
    model_id = None,
    max_families: int = 10,
) -> tuple[list[str], str]:
    """Use LLM to semantically filter metrics to most salient ones.

    Args:
        metric_names: List of metric names
        llm_func: Callable that takes a prompt string and returns LLM response text.
                  If None, uses Anthropic Claude API (requires ANTHROPIC_API_KEY env var)
        model_id: Model ID to use (passed to llm_func if provided)
        max_families: Maximum number of metric families to keep

    Returns:
        Tuple of (filtered_metrics, explanation) where explanation describes the LLM's reasoning

    Example:
        >>> # Using custom LLM
        >>> def my_llm(prompt):
        ...     return openai.chat.completions.create(...)
        >>> salient, explanation = filter_salient_metrics_llm(metrics, llm_func=my_llm)

        >>> # Using default (Claude API)
        >>> salient, explanation = filter_salient_metrics_llm(metrics)
    """
    # Get unique families
    families_dict = get_metric_families(metric_names)
    family_names = sorted(families_dict.keys())

    if len(family_names) <= max_families:
        return metric_names, "All metric families kept (count <= max_families threshold)."

    # Build LLM prompt
    prompt = f"""Analyze these {len(family_names)} Prometheus/OpenTelemetry metric families and select the {max_families} MOST IMPORTANT ones for monitoring service health.

Metric families (without _bucket/_count/_sum/_total suffixes):
{chr(10).join(f'- {f}' for f in family_names)}

Selection criteria (in priority order):
1. Latency/duration metrics (request/response times)
2. Error/exception counters
3. Throughput metrics (requests/calls per second)
4. Resource usage (memory, connections)
5. Drop: internal runtime details, redundant counters, size metrics

Respond with a JSON object containing:
1. "selected": a JSON array of the metric family names to keep
2. "explanation": a brief explanation of your selection reasoning

Example response:
{{
  "selected": ["http_server_request_duration_seconds", "rpc_client_duration_milliseconds"],
  "explanation": "Selected latency metrics (http_server_request_duration, rpc_client_duration) as they directly measure user-facing performance. Included error counters for reliability monitoring. Dropped internal runtime metrics as they provide less actionable insights."
}}"""

    # Call LLM
    if llm_func is None:
        llm_func = _default_llm_call

    if model_id is not None:
        response_text = llm_func(prompt, model_id)
    else:
        response_text = llm_func(prompt)

    # Parse JSON response
    try:
        import json

        # Extract JSON object from response (handles markdown code blocks)
        text_to_parse = response_text
        if '```' in response_text:
            # Extract from markdown code block
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                text_to_parse = response_text[start:end]

        parsed = json.loads(text_to_parse)

        # Handle both old format (array) and new format (object with selected/explanation)
        if isinstance(parsed, list):
            selected_families = parsed
            explanation = "No explanation provided by LLM."
        elif isinstance(parsed, dict):
            selected_families = parsed.get("selected", [])
            explanation = parsed.get("explanation", "No explanation provided by LLM.")
            if not isinstance(selected_families, list):
                raise ValueError("LLM response 'selected' field is not a JSON array")
        else:
            raise ValueError("LLM did not return a JSON array or object")

    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {response_text}")

    # Filter metrics to only include selected families
    selected_families_set = set(selected_families)
    result = []
    for metric in metric_names:
        family = extract_metric_family(metric)
        if family in selected_families_set:
            result.append(metric)

    return result, explanation


def _default_llm_call(prompt: str, model_id:str="claude-sonnet-4-5-20250929") -> str:
    """Default LLM call using Anthropic Claude API."""
    try:
        import anthropic
        import os
    except ImportError:
        raise ImportError(
            "anthropic package required for default LLM. "
            "Install with: pip install anthropic\n"
            "Or provide your own llm_func parameter."
        )

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set it or provide your own llm_func parameter."
        )

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model_id,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


def _openai_llm_call(prompt: str, model_id:str="gpt-4o") -> str:
    """LLM call using OpenAI via langchain API.

    Args:
        prompt: The prompt to send to the LLM

    Returns:
        The LLM's text response

    Raises:
        ImportError: If langchain_openai is not installed
        ValueError: If OPENAI_API_KEY environment variable is not set
    """
    try:
        from langchain_openai import ChatOpenAI
        import os
    except ImportError:
        raise ImportError(
            "langchain_openai package required. "
            "Install with: pip install langchain-openai"
        )

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it or provide your own llm_func parameter."
        )

    # Initialize ChatOpenAI model
    llm = ChatOpenAI(
        model=model_id,
        temperature=0,
        max_tokens=1024,
        api_key=api_key
    )

    # Invoke the model with the prompt
    response = llm.invoke(prompt)

    return response.content
