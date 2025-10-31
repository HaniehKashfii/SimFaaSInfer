"""Metrics collection and aggregation."""

import numpy as np
from typing import Dict, List
from collections import defaultdict

from ..utils.logger import setup_logger


class MetricsCollector:
    """Collect and aggregate simulation metrics.

    Tracks both request-level metrics (latency, TTFT, TBT) and
    cluster-level metrics (utilization, throughput).
    """

    def __init__(self, config: Dict):
        """Initialize metrics collector.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)

        # Request-level metrics
        self.latencies = []
        self.ttfts = []  # Time to first token
        self.tbts = []  # Time between tokens
        self.scheduling_delays = []
        self.prefill_times = []
        self.decode_times = []
        self.normalized_latencies = []

        # Cluster-level metrics (time-series)
        self.cluster_metrics = defaultdict(list)
        self.timestamps = []

        # Percentiles to compute
        self.percentiles = config['metrics'].get('percentiles', [50, 90, 95, 99])

    def record_request(self, request) -> None:
        """Record metrics for a completed request.

        Args:
            request: Completed request object
        """
        if not request.is_complete():
            return

        # End-to-end latency
        latency = request.completion_time - request.arrival_time
        self.latencies.append(latency * 1000)  # Convert to ms

        # Normalized latency (latency per output token)
        if request.decode_tokens > 0:
            normalized = (latency * 1000) / request.decode_tokens
            self.normalized_latencies.append(normalized)

        # Time to first token
        if hasattr(request, 'first_token_time') and request.first_token_time:
            ttft = request.first_token_time - request.arrival_time
            self.ttfts.append(ttft * 1000)

        # Time between tokens
        if hasattr(request, 'decode_start_time') and request.decode_start_time:
            if request.decode_tokens > 1:
                decode_duration = request.completion_time - request.decode_start_time
                tbt = (decode_duration * 1000) / (request.decode_tokens - 1)
                self.tbts.append(tbt)

        # Scheduling delay
        if hasattr(request, 'scheduling_delay'):
            self.scheduling_delays.append(request.scheduling_delay * 1000)

        # Phase times
        if hasattr(request, 'prefill_time'):
            self.prefill_times.append(request.prefill_time * 1000)
        if hasattr(request, 'decode_time'):
            self.decode_times.append(request.decode_time * 1000)

    def record_cluster_metrics(self, timestamp: float, metrics: Dict) -> None:
        """Record cluster-level metrics at a point in time.

        Args:
            timestamp: Current simulation time
            metrics: Dictionary of cluster metrics
        """
        self.timestamps.append(timestamp)

        for key, value in metrics.items():
            self.cluster_metrics[key].append(value)

    def compute_metrics(self) -> Dict:
        """Compute aggregate metrics from collected data.

        Returns:
            Dictionary of computed metrics
        """
        results = {}

        # Request-level metrics
        if self.latencies:
            results.update(self._compute_distribution_metrics(
                'latency', self.latencies
            ))

        if self.ttfts:
            results.update(self._compute_distribution_metrics(
                'ttft', self.ttfts
            ))

        if self.tbts:
            results.update(self._compute_distribution_metrics(
                'tbt', self.tbts
            ))

        if self.normalized_latencies:
            results.update(self._compute_distribution_metrics(
                'normalized_latency', self.normalized_latencies
            ))

        if self.scheduling_delays:
            results.update(self._compute_distribution_metrics(
                'scheduling_delay', self.scheduling_delays
            ))

        # Throughput
        if self.latencies:
            total_time = self.timestamps[-1] - self.timestamps[0] if self.timestamps else 1.0
            results['throughput'] = len(self.latencies) / total_time  # QPS

        # Cluster-level metrics
        for metric_name, values in self.cluster_metrics.items():
            if values:
                results[f'mean_{metric_name}'] = np.mean(values)
                results[f'max_{metric_name}'] = np.max(values)
                results[f'min_{metric_name}'] = np.min(values)

        return results

    def _compute_distribution_metrics(self, name: str, values: List[float]) -> Dict:
        """Compute distribution statistics for a metric.

        Args:
            name: Metric name
            values: List of values

        Returns:
            Dictionary with mean, median, and percentiles
        """
        if not values:
            return {}

        results = {
            f'mean_{name}': np.mean(values),
            f'median_{name}': np.median(values),
            f'std_{name}': np.std(values),
        }

        for p in self.percentiles:
            results[f'p{p}_{name}'] = np.percentile(values, p)

        return results

    def get_summary(self) -> str:
        """Get human-readable summary of metrics.

        Returns:
            Formatted string with key metrics
        """
        if not self.latencies:
            return "No metrics collected"

        summary = [
            "=== Metrics Summary ===",
            f"Requests: {len(self.latencies)}",
            f"Median Latency: {np.median(self.latencies):.2f} ms",
            f"P95 Latency: {np.percentile(self.latencies, 95):.2f} ms",
        ]

        if self.ttfts:
            summary.append(f"Median TTFT: {np.median(self.ttfts):.2f} ms")
        if self.tbts:
            summary.append(f"Median TBT: {np.median(self.tbts):.2f} ms")
        if self.timestamps:
            duration = self.timestamps[-1] - self.timestamps[0]
            throughput = len(self.latencies) / duration
            summary.append(f"Throughput: {throughput:.2f} QPS")

        return "\n".join(summary)