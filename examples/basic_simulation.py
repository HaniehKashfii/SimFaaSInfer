"""Basic simulation example."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simfaasinfer.core.simulator import Simulator
from simfaasinfer.utils.logger import setup_logger
from configs import load_config, merge_configs


def main():
    """Run a basic simulation."""
    logger = setup_logger("BasicSimulation")

    logger.info("=== Basic LLM Inference Simulation ===")

    # Load configuration
    base_config = load_config("configs/default.yaml")
    model_config = load_config("configs/models/llama2_7b.yaml")

    # Merge configurations
    config = merge_configs(base_config, model_config)

    # Customize for this example
    config['simulation']['duration'] = 300  # 5 minutes
    config['workload']['arrival_rate'] = 5  # 5 QPS
    config['cold_start']['enabled'] = True
    config['scaling']['enabled'] = False

    logger.info(f"Running simulation for {config['simulation']['duration']}s")
    logger.info(f"Arrival rate: {config['workload']['arrival_rate']} QPS")
    logger.info(f"Model: {config['model']['name']}")

    # Create and run simulator
    simulator = Simulator(config)
    results = simulator.run()

    # Print results
    logger.info("\n=== Results ===")
    logger.info(f"Total requests: {results['total_requests']}")
    logger.info(f"Completed requests: {results['completed_requests']}")
    logger.info(f"Completion rate: {results['completion_rate']:.1%}")
    logger.info(f"\nThroughput: {results['throughput']:.2f} QPS")
    logger.info(f"\nLatency:")
    logger.info(f"  Median: {results['median_latency']:.2f} ms")
    logger.info(f"  P95: {results['p95_latency']:.2f} ms")
    logger.info(f"  P99: {results['p99_latency']:.2f} ms")

    if 'median_ttft' in results:
        logger.info(f"\nTime to First Token:")
        logger.info(f"  Median: {results['median_ttft']:.2f} ms")
        logger.info(f"  P95: {results['p95_ttft']:.2f} ms")

    if 'median_tbt' in results:
        logger.info(f"\nTime Between Tokens:")
        logger.info(f"  Median: {results['median_tbt']:.2f} ms")
        logger.info(f"  P95: {results['p95_tbt']:.2f} ms")

    logger.info(f"\nResource Utilization:")
    logger.info(f"  Mean MFU: {results['mean_mfu']:.2%}")
    logger.info(f"  Mean Memory Util: {results['mean_memory_util']:.2%}")
    logger.info(f"  Mean KV-Cache Util: {results['mean_kv_cache_util']:.2%}")

    if 'total_cost' in results:
        logger.info(f"\nCost:")
        logger.info(f"  Total: ${results['total_cost']:.2f}")
        logger.info(f"  QPS per Dollar: {results['qps_per_dollar']:.4f}")

    logger.info("\nSimulation complete!")


if __name__ == "__main__":
    main()