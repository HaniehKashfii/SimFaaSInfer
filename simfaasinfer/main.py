"""Main entry point for SimFaaSInfer simulator."""

import argparse
import sys
from pathlib import Path
import yaml

from simfaasinfer.core.simulator import Simulator
from simfaasinfer.utils.logger import setup_logger
from simfaasinfer.utils.visualization import plot_results
from configs import load_config, merge_configs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SimFaaSInfer: Serverless LLM Inference Simulator"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/models/llama2_7b.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger("SimFaaSInfer", level=log_level)

    logger.info("=== SimFaaSInfer: Serverless LLM Inference Simulator ===")
    logger.info(f"Loading configuration from {args.config}")

    try:
        # Load configurations
        base_config = load_config(args.config)
        model_config = load_config(args.model_config)

        # Merge configurations
        config = merge_configs(base_config, model_config)

        logger.info(f"Model: {config['model']['name']}")
        logger.info(f"Workload: {config['workload']['type']}")
        logger.info(f"Scheduler: {config['scheduler']['type']}")

        # Create simulator
        simulator = Simulator(config)

        # Run simulation
        logger.info("Starting simulation...")
        results = simulator.run()

        # Print results
        logger.info("\n=== Simulation Results ===")
        logger.info(f"Total Requests: {results['total_requests']}")
        logger.info(f"Completed Requests: {results['completed_requests']}")
        logger.info(f"Throughput: {results['throughput']:.2f} QPS")
        logger.info(f"Median Latency: {results['median_latency']:.2f} ms")
        logger.info(f"P95 Latency: {results['p95_latency']:.2f} ms")
        logger.info(f"P99 Latency: {results['p99_latency']:.2f} ms")
        logger.info(f"Median TTFT: {results['median_ttft']:.2f} ms")
        logger.info(f"P95 TTFT: {results['p95_ttft']:.2f} ms")
        logger.info(f"Median TBT: {results['median_tbt']:.2f} ms")
        logger.info(f"P95 TBT: {results['p95_tbt']:.2f} ms")
        logger.info(f"Mean MFU: {results['mean_mfu']:.2%}")
        logger.info(f"Mean Memory Util: {results['mean_memory_util']:.2%}")
        logger.info(f"Mean KV-Cache Util: {results['mean_kv_cache_util']:.2%}")

        if config['cost']['compute_cost']:
            logger.info(f"Total Cost: ${results['total_cost']:.2f}")
            logger.info(f"QPS per Dollar: {results['qps_per_dollar']:.4f}")

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / "results.yaml"
        with open(results_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        logger.info(f"Results saved to {results_file}")

        # Generate visualizations
        if args.visualize:
            logger.info("Generating visualization plots...")
            plot_results(results, output_dir)
            logger.info(f"Plots saved to {output_dir}")

        logger.info("Simulation completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())