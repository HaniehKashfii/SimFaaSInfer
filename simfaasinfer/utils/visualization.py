"""Visualization utilities for simulation results."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_results(results: Dict, output_dir: Path) -> None:
    """Generate all visualization plots.

    Args:
        results: Results dictionary from simulation
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot latency distribution
    plot_latency_distribution(results, output_dir / "latency_distribution.png")

    # Plot metrics over time (if available)
    if 'timestamps' in results:
        plot_metrics_timeline(results, output_dir / "metrics_timeline.png")

    # Plot cost breakdown (if available)
    if 'total_cost' in results:
        plot_cost_breakdown(results, output_dir / "cost_breakdown.png")


def plot_latency_distribution(results: Dict, output_path: Path) -> None:
    """Plot latency distribution histogram.

    Args:
        results: Results dictionary
        output_path: Output file path
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Latency
    if 'p50_latency' in results:
        ax = axes[0, 0]
        metrics = ['p50_latency', 'p90_latency', 'p95_latency', 'p99_latency']
        values = [results.get(m, 0) for m in metrics]
        labels = ['P50', 'P90', 'P95', 'P99']

        ax.bar(labels, values, color='steelblue')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('End-to-End Latency Distribution')
        ax.grid(axis='y', alpha=0.3)

    # TTFT
    if 'p50_ttft' in results:
        ax = axes[0, 1]
        metrics = ['p50_ttft', 'p90_ttft', 'p95_ttft', 'p99_ttft']
        values = [results.get(m, 0) for m in metrics]

        ax.bar(labels, values, color='coral')
        ax.set_ylabel('TTFT (ms)')
        ax.set_title('Time to First Token')
        ax.grid(axis='y', alpha=0.3)

    # TBT
    if 'p50_tbt' in results:
        ax = axes[1, 0]
        metrics = ['p50_tbt', 'p90_tbt', 'p95_tbt', 'p99_tbt']
        values = [results.get(m, 0) for m in metrics]

        ax.bar(labels, values, color='lightgreen')
        ax.set_ylabel('TBT (ms)')
        ax.set_title('Time Between Tokens')
        ax.grid(axis='y', alpha=0.3)

    # Summary metrics
    ax = axes[1, 1]
    metrics_text = [
        f"Throughput: {results.get('throughput', 0):.2f} QPS",
        f"Mean MFU: {results.get('mean_mfu', 0):.2%}",
        f"Mean Memory Util: {results.get('mean_memory_util', 0):.2%}",
        f"Completed: {results.get('completed_requests', 0)}/{results.get('total_requests', 0)}",
    ]

    if 'qps_per_dollar' in results:
        metrics_text.append(f"QPS/$: {results.get('qps_per_dollar', 0):.4f}")

    ax.text(0.1, 0.5, '\n'.join(metrics_text), fontsize=12,
            verticalalignment='center', family='monospace')
    ax.axis('off')
    ax.set_title('Summary Metrics')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_timeline(results: Dict, output_path: Path) -> None:
    """Plot metrics over time.

    Args:
        results: Results dictionary with time-series data
        output_path: Output file path
    """
    if 'timestamps' not in results:
        return

    timestamps = results['timestamps']

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Utilization over time
    if 'mfu_timeline' in results:
        ax = axes[0]
        ax.plot(timestamps, results['mfu_timeline'], label='MFU', linewidth=2)
        if 'memory_util_timeline' in results:
            ax.plot(timestamps, results['memory_util_timeline'],
                    label='Memory Util', linewidth=2)
        ax.set_ylabel('Utilization')
        ax.set_title('Resource Utilization Over Time')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])

    # Request rate over time
    if 'request_rate_timeline' in results:
        ax = axes[1]
        ax.plot(timestamps, results['request_rate_timeline'],
                linewidth=2, color='coral')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Request Rate (QPS)')
        ax.set_title('Request Arrival Rate Over Time')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_cost_breakdown(results: Dict, output_path: Path) -> None:
    """Plot cost breakdown.

    Args:
        results: Results dictionary
        output_path: Output file path
    """
    if 'total_cost' not in results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Cost breakdown pie chart
    costs = []
    labels = []

    if 'compute_cost' in results:
        costs.append(results['compute_cost'])
        labels.append('Compute')
    if 'idle_cost' in results:
        costs.append(results['idle_cost'])
        labels.append('Idle')

    if costs:
        ax1.pie(costs, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Cost Breakdown')

    # Cost metrics
    cost_text = [
        f"Total Cost: ${results.get('total_cost', 0):.2f}",
        f"GPU Hours: {results.get('gpu_hours', 0):.2f}",
        f"Cost/GPU/Hour: ${results.get('cost_per_gpu_hour', 0):.2f}",
        f"QPS per Dollar: {results.get('qps_per_dollar', 0):.4f}",
        f"Duration: {results.get('simulation_duration', 0):.0f}s",
    ]

    ax2.text(0.1, 0.5, '\n'.join(cost_text), fontsize=12,
             verticalalignment='center', family='monospace')
    ax2.axis('off')
    ax2.set_title('Cost Metrics')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()