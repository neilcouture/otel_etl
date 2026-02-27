"""Prometheus API client wrapper."""

from typing import Any, Optional
from datetime import datetime, timedelta, timezone
import urllib.request
import urllib.parse
import json
import logging
import time

import pandas as pd
import requests
import statistics
import yaml

logger = logging.getLogger(__name__)


def _detect_metric_type(metric_name, labels=None):
    """
    Detect metric type based on naming conventions and labels.

    Returns: (type, subtype)
    - type: 'counter', 'gauge', 'histogram', 'summary'
    - subtype: 'bucket', 'sum', 'count', or None
    """
    # Check for histogram
    if metric_name.endswith('_bucket'):
        return 'histogram', 'bucket'
    if metric_name.endswith('_count'):
        # Could be histogram or summary count
        base_name = metric_name[:-6]
        return 'histogram', 'count'  # Assume histogram
    if metric_name.endswith('_sum'):
        base_name = metric_name[:-4]
        return 'histogram', 'sum'  # Assume histogram

    # Check labels for histogram/summary
    if labels:
        if 'le' in labels:
            return 'histogram', 'bucket'
        if 'quantile' in labels:
            return 'summary', 'quantile'

    # Check for counter
    if metric_name.endswith('_total'):
        return 'counter', None
    if 'total' in metric_name.lower():
        return 'counter', None

    # Default to gauge
    return 'gauge', None


def _format_timestamp(dt: datetime) -> str:
    """Format datetime for Prometheus API.

    Handles both naive and timezone-aware datetimes.
    """
    if dt.tzinfo is not None:
        # Convert to UTC and format without the +00:00 suffix
        utc_dt = dt.astimezone(timezone.utc)
        return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        # Naive datetime - assume UTC
        return dt.isoformat() + "Z"


def get_prometheus_scrape_interval(prometheus_url="http://10.0.0.5:9090"):
    """Get scrape_interval from Prometheus config."""
    response = requests.get(f"{prometheus_url}/api/v1/status/config")
    config = response.json()['data']['yaml']

    # Parse YAML to find scrape_interval
    config_dict = yaml.safe_load(config)
    scrape_interval = config_dict['global']['scrape_interval']

    # Convert to seconds (e.g., "60s" -> 60)
    if scrape_interval.endswith('s'):
        return int(scrape_interval[:-1])
    elif scrape_interval.endswith('m'):
        return int(scrape_interval[:-1]) * 60

    return 60


def detect_scrape_interval(prometheus_url, metric_name, samples=5):
    """
    Detect scrape interval by querying recent data points.

    Fetches last N samples and calculates time delta between them.
    """
    query = f'{metric_name}[{samples}m]'  # Last N minutes
    response = requests.get(
        f"{prometheus_url}/api/v1/query",
        params={'query': query}
    )

    result = response.json()['data']['result'][0]
    values = result['values']  # [[timestamp, value], ...]

    # Calculate deltas
    deltas = []
    for i in range(1, len(values)):
        delta = values[i][0] - values[i-1][0]
        deltas.append(delta)

    # Most common delta = scrape interval
    return int(statistics.mode(deltas))


class PrometheusClient:
    """Wrapper for Prometheus HTTP API."""

    def __init__(self, base_url: str = "http://localhost:9090"):
        """Initialize client with Prometheus server URL.

        Args:
            base_url: Prometheus server URL (e.g., http://localhost:9090)
        """
        self.base_url = base_url.rstrip("/")

    def _request(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make HTTP request to Prometheus API.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            Parsed JSON response

        Raises:
            RuntimeError: If the request fails or returns an error status
        """
        url = f"{self.base_url}{endpoint}"
        if params:
            query_string = urllib.parse.urlencode(params)
            url = f"{url}?{query_string}"

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Prometheus API error: {e.code} {e.reason}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to Prometheus: {e.reason}")

        if data.get("status") != "success":
            error_msg = data.get("error", "Unknown error")
            raise RuntimeError(f"Prometheus query failed: {error_msg}")

        return data

    def get_metric_names(self) -> list[str]:
        """Get all metric names from Prometheus.

        Returns:
            List of metric names
        """
        data = self._request("/api/v1/label/__name__/values")
        return data.get("data", [])

    def get_labels_for_metric(self, metric_name: str) -> list[str]:
        """Get all label names for a specific metric.

        Args:
            metric_name: The metric name to query

        Returns:
            List of label names (excluding __name__)
        """
        params = {"match[]": metric_name}
        data = self._request("/api/v1/labels", params)
        labels = data.get("data", [])
        return [l for l in labels if l != "__name__"]

    def get_label_values(
        self, label_name: str, metric_name: Optional[str] = None
    ) -> list[str]:
        """Get distinct values for a label.

        Args:
            label_name: The label to query
            metric_name: Optional metric name to filter by

        Returns:
            List of distinct label values
        """
        params = {}
        if metric_name:
            params["match[]"] = metric_name

        data = self._request(f"/api/v1/label/{label_name}/values", params)
        return data.get("data", [])

    def query(self, promql: str) -> list[dict]:
        """Execute an instant query.

        Args:
            promql: PromQL query string

        Returns:
            List of result vectors
        """
        params = {"query": promql}
        data = self._request("/api/v1/query", params)
        return data.get("data", {}).get("result", [])

    def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step: str = "60s",
    ) -> list[dict]:
        """Execute a range query.

        Args:
            promql: PromQL query string
            start: Start time
            end: End time
            step: Query resolution (e.g., '60s', '1m', '5m')

        Returns:
            List of result matrices
        """
        params = {
            "query": promql,
            "start": _format_timestamp(start),
            "end": _format_timestamp(end),
            "step": step,
        }
        data = self._request("/api/v1/query_range", params)
        return data.get("data", {}).get("result", [])

    def get_series(
        self,
        match: str | list[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> list[dict]:
        """Get time series matching selectors.

        Args:
            match: Metric selector(s)
            start: Optional start time
            end: Optional end time

        Returns:
            List of series label sets
        """
        if isinstance(match, str):
            match = [match]

        params = {f"match[]": m for m in match}
        if start:
            params["start"] = _format_timestamp(start)
        if end:
            params["end"] = _format_timestamp(end)

        data = self._request("/api/v1/series", params)
        return data.get("data", [])

    def count_label_cardinality(
        self,
        metric_name: str,
        label_name: str,
        window_hours: float = 1.0,
    ) -> int:
        """Count distinct values for a label within a time window.

        Args:
            metric_name: Metric to query
            label_name: Label to count values for
            window_hours: Time window in hours

        Returns:
            Count of distinct label values
        """
        query = f'count(count by ({label_name}) ({metric_name}))'
        result = self.query(query)
        if result and len(result) > 0:
            return int(float(result[0].get("value", [0, 0])[1]))
        return 0

    def get_top_n_values(
        self,
        metric_name: str,
        label_name: str,
        n: int = 20,
        window_hours: float = 1.0,
    ) -> list[str]:
        """Get top N label values by volume.

        Args:
            metric_name: Metric to query
            label_name: Label to rank
            n: Number of top values to return
            window_hours: Time window in hours

        Returns:
            List of top N label values ordered by volume
        """
        query = f'topk({n}, sum by ({label_name}) (rate({metric_name}[{int(window_hours)}h])))'
        result = self.query(query)

        values = []
        for series in result:
            metric = series.get("metric", {})
            if label_name in metric:
                values.append(metric[label_name])
        return values

    def fetch_metrics_range(
        self,
        metric_names: list[str],
        start: datetime,
        end: datetime,
        step: str = "60s",
    ) -> pd.DataFrame:
        """Fetch multiple metrics over a time range.

        Args:
            metric_names: List of metrics to fetch
            start: Start time
            end: End time
            step: Query resolution

        Returns:
            DataFrame with columns: timestamp, metric, labels (dict), value
        """
        rows = []

        for metric_name in metric_names:
            try:
                result = self.query_range(metric_name, start, end, step)
                for series in result:
                    metric = series.get("metric", {})
                    labels = {k: v for k, v in metric.items() if k != "__name__"}
                    actual_name = metric.get("__name__", metric_name)

                    for ts, value in series.get("values", []):
                        rows.append({
                            "timestamp": datetime.fromtimestamp(ts),
                            "metric": actual_name,
                            "labels": labels,
                            "value": float(value) if value != "NaN" else float("nan"),
                        })
            except RuntimeError as e:
                logger.warning(f"Failed to fetch {metric_name}: {e}")
                continue

        if not rows:
            return pd.DataFrame(columns=["timestamp", "metric", "labels", "value"])

        return pd.DataFrame(rows)

    def fetch_metrics_filtered(
        self,
        services: list[str],
        metric_names: list[str],
    ) -> pd.DataFrame:
        """Fetch latest values for specific services and metrics.

        Args:
            services: List of service names to filter by
            metric_names: List of metrics to fetch

        Returns:
            DataFrame with columns: timestamp, metric, labels (dict), value
        """
        rows = []
        services_set = set(services)

        for metric_name in metric_names:
            try:
                # Query the metric (instant query for latest values)
                result = self.query(metric_name)

                for series in result:
                    metric = series.get("metric", {})

                    # Extract service name from labels (check common service label names)
                    service = (
                        metric.get("service_name") or
                        metric.get("service") or
                        metric.get("job") or
                        metric.get("container") or
                        "unknown"
                    )

                    # Only include if service matches our filter
                    if service not in services_set:
                        continue

                    labels = {k: v for k, v in metric.items() if k != "__name__"}
                    actual_name = metric.get("__name__", metric_name)

                    # Get timestamp and value from instant query result
                    value_data = series.get("value", [])
                    if len(value_data) >= 2:
                        ts, value = value_data[0], value_data[1]
                        rows.append({
                            "timestamp": datetime.fromtimestamp(ts),
                            "metric": actual_name,
                            "labels": labels,
                            "value": float(value) if value != "NaN" else float("nan"),
                        })
            except RuntimeError as e:
                logger.warning(f"Failed to fetch {metric_name}: {e}")
                continue

        if not rows:
            return pd.DataFrame(columns=["timestamp", "metric", "labels", "value"])

        return pd.DataFrame(rows)


def get_metrics_dataframe2(prometheus_url="http://10.0.0.5:9090"):
    """
    Get all metrics and their services from Prometheus - FAST version.

    Uses /api/v1/series endpoint to fetch ALL metrics in ONE request
    instead of one request per metric.

    Returns DataFrame with columns: ['service', 'metric', 'type', 'subtype']
    """
    import requests

    api_url = f"{prometheus_url.rstrip('/')}/api/v1"

    # ONE request to get ALL series with all labels (last 5 min only)
    now = time.time()
    response = requests.get(f"{api_url}/series", params={
        'match[]': '{__name__=~".+"}',
        'start': now - 300,
        'end': now
    })
    series = response.json()['data']

    data = []
    seen = set()

    for labels in series:
        metric = labels.get('__name__', '')
        service = (labels.get('service_name') or
                   labels.get('service') or
                   labels.get('job') or
                   labels.get('container') or
                   'unknown')

        key = (service, metric)
        if key in seen:
            continue
        seen.add(key)

        metric_type, metric_subtype = _detect_metric_type(metric, labels)
        data.append({
            'service': service,
            'metric': metric,
            'type': metric_type,
            'subtype': metric_subtype
        })

    return pd.DataFrame(data).sort_values(['service', 'metric']).reset_index(drop=True)
