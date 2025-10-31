"""Capacity planning example - find optimal configuration."""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simfaasinfer.core.simulator import Simulator
from simfaasinfer.utils.logger import setup_logger
from configs import load_config, merge_configs


def run_configuration(base_config: Dict, model_config: Dict,
                      modifications: Dict) -> Dict:
    """Run simulation with specific configuration.

    Args:
        base_config: Base configuration
        model_config: Model configuration
        modifications: Configuration modifications

    Returns:
        Results dictionary
    """
    # Merge configurations
    config = merge_configs(base_config, model_config)

    # Apply modifications
    for key, value in modifications.items():
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            current = current[k]
        current[keys[-1]] = value

    # Run simulation
    simulator = Simulator(config)
    results = simulator.run()

    return results


def search_optimal_batch_size(base_config: Dict, model_config: Dict,
                              batch_sizes: List[int]) -> Tuple[int, Dict]:
    """Search for optimal batch size.

    Args:
        base_config: Base configuration
        model_config: Model configuration
        batch_sizes: List of batch sizes to try

    Returns:
        Tuple of (best_batch_size, best_results)
    """
    logger = setup_logger("CapacityPlanning")
    logger.info("=== Searching for Optimal Batch Size ===")

    best_batch_size = None
    best_qps_per_dollar = 0
    best_results = None

    for batch_size in batch_sizes:
        logger.info(f"\nTrying batch size: {batch_size}")

        modifications = {
            'scheduler.max_batch_size': batch_size,
        }

        results = run_configuration(base_config, model_config, modifications)

        # Check if SLOs are met
        ttft_p90 = results.get('p90_ttft', 0)
        tbt_p99 = results.get('p99_tbt', 0)

        slo_met = ttft_p90 < 2000 and tbt_p99 < 200  # 2s TTFT, 200ms TBT

        if not slo_met:
            logger.info(f"  SLOs not met: TTFT P90={ttft_p90:.0f}ms, TBT P99={tbt_p99:.0f}ms")
            continue

        qps_per_dollar = results.get('qps_per_dollar', 0)
        logger.info(f"  QPS per Dollar: {qps_per_dollar:.4f}")
        logger.info(f"  Throughput: {results['throughput']:.2f} QPS")
        logger.info(f"  P95 Latency: {results['p95_latency']:.2f} ms")

        if qps_per_dollar > best_qps_per_dollar:
            best_batch_size = batch_size
            best_qps_per_dollar = qps_per_dollar
            best_results = results

    return best_batch_size, best_results


def compare_schedulers(base_config: Dict, model_config: Dict,
                       schedulers: List[str]) -> Dict[str, Dict]:
    """Compare different scheduling strategies.

    Args:
        base_config: Base configuration
        model_config: Model configuration
        schedulers: List of scheduler types

    Returns:
        Dictionary mapping scheduler to results
    """
    logger = setup_logger("CapacityPlanning")
    logger.info("\n=== Comparing Schedulers ===")

    results_by_scheduler = {}

    for scheduler in schedulers:
        logger.info(f"\nTesting {scheduler} scheduler...")

        modifications = {
            'scheduler.type': scheduler,
        }

        results = run_configuration(base_config, model_config, modifications)
        results_by_scheduler[scheduler] = results

        logger.info(f"  Throughput: {results['throughput']:.2f} QPS")
        logger.info(f"  P95 Latency: {results['p95_latency']:.2f} ms")
        logger.info(f"  QPS per Dollar: {results.get('qps_per_dollar', 0):.4f}")

    return results_by_scheduler


def main():
    """Run capacity planning analysis."""
    logger = setup_logger("CapacityPlanning")

    logger.info("=== LLM Inference Capacity Planning ===")

    # Load base configuration
    base_config = load_config("configs/default.yaml")
    model_config = load_config("configs/models/llama2_7b.yaml")

    # Customize for capacity planning
    base_config['simulation']['duration'] = 600  # 10 minutes
    base_config['simulation']['warmup_duration'] = 60
    base_config['workload']['arrival_rate'] = 10

    # 1. Search for optimal batch size
    batch_sizes = [32, 64, 128, 256, 512]
    best_batch_size, best_results = search_optimal_batch_size(
        base_config, model_config, batch_sizes
    )

    if best_batch_size:
        logger.info(f"\n=== Best Batch Size: {best_batch_size} ===")
        logger.info(f"QPS per Dollar: {best_results['qps_per_dollar']:.4f}")
        logger.info(f"Throughput: {best_results['throughput']:.2f} QPS")
        logger.info(f"P95 Latency: {best_results['p95_latency']:.2f} ms")
    else:
        logger.warning("No configuration met SLOs!")

    # 2. Compare schedulers
    schedulers = ['vllm', 'orca', 'sarathi']
    scheduler_results = compare_schedulers(base_config, model_config, schedulers)

    # Find best scheduler
    best_scheduler = max(
        scheduler_results.items(),
        key=lambda x: x[1].get('qps_per_dollar', 0)
    )

    logger.info(f"\n=== Best Scheduler: {best_scheduler[0]} ===")
    logger.info(f"QPS per Dollar: {best_scheduler[1]['qps_per_dollar']:.4f}")

    # 3. Print summary
    logger.info("\n=== Capacity Planning Summary ===")
    logger.info(f"Recommended Configuration:")
    logger.info(f"  Model: {model_config['model']['name']}")
    logger.info(f"  Batch Size: {best_batch_size}")
    logger.info(f"  Scheduler: {best_scheduler[0]}")
    logger.info(f"  Expected QPS: {best_scheduler[1]['throughput']:.2f}")
    logger.info(f"  Expected P95 Latency: {best_scheduler[1]['p95_latency']:.2f} ms")
    logger.info(f"  Cost Efficiency: {best_scheduler[1]['qps_per_dollar']:.4f} QPS/$")


if __name__ == "__main__":
    main()