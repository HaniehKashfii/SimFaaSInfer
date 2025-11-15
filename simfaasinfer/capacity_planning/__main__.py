"""
CLI interface for capacity planning module.

Usage:
    python -m simfaasinfer.capacity_planning --profile <path> --output <path>
"""

import sys
import argparse
from pathlib import Path

from .planner import CapacityPlanner, run_capacity_planning
from .report_writer import ReportWriter
from ..utils.logger import setup_logger


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SimFaaSInfer Capacity Planning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic capacity planning
  python -m simfaasinfer.capacity_planning \\
    --profile data/profiling/compute/a100/llama-7b/fitted_profile.json \\
    --output results/capacity_plan.json

  # With custom workload and constraints
  python -m simfaasinfer.capacity_planning \\
    --profile data/profiling/compute/a100/llama-7b/fitted_profile.json \\
    --workload configs/workload/production.yaml \\
    --constraints configs/constraints/strict_slos.yaml \\
    --output results/capacity_plan.json

  # With production telemetry calibration
  python -m simfaasinfer.capacity_planning \\
    --profile data/profiling/compute/a100/llama-7b/fitted_profile.json \\
    --telemetry data/telemetry/prod_samples.json \\
    --output results/capacity_plan.json
        """
    )

    parser.add_argument(
        '--profile',
        type=str,
        required=True,
        help='Path to FaaSInfer fitted_profile.json'
    )

    parser.add_argument(
        '--workload',
        type=str,
        default=None,
        help='Path to workload config YAML (optional)'
    )

    parser.add_argument(
        '--constraints',
        type=str,
        default=None,
        help='Path to constraints config YAML (optional)'
    )

    parser.add_argument(
        '--telemetry',
        type=str,
        default=None,
        help='Path to production telemetry for calibration (optional)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save capacity planning report (JSON)'
    )

    parser.add_argument(
        '--gpu-types',
        nargs='+',
        default=['A100'],
        help='GPU types to evaluate (default: A100)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    logger = setup_logger("CapacityPlanning", verbose=args.verbose)

    # Check profile exists
    if not Path(args.profile).exists():
        logger.error(f"Profile not found: {args.profile}")
        return 1

    # Use simplified interface if workload config provided
    if args.workload:
        if not Path(args.workload).exists():
            logger.error(f"Workload config not found: {args.workload}")
            return 1

        logger.info("Running capacity planning with config files...")

        try:
            results = run_capacity_planning(
                profile_path=args.profile,
                workload_config=args.workload,
                output_path=args.output,
                constraints_config=args.constraints,
                telemetry_path=args.telemetry,
                config_space={'gpu_types': args.gpu_types}
            )

            logger.info("✓ Capacity planning complete!")
            return 0

        except Exception as e:
            logger.error(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Otherwise use programmatic interface with defaults
    else:
        logger.info("Running capacity planning with default parameters...")

        try:
            # Initialize planner
            planner = CapacityPlanner(args.profile)

            # Optional calibration
            if args.telemetry:
                logger.info(f"Calibrating with telemetry: {args.telemetry}")
                planner.calibrate(args.telemetry)

            # Default workload
            workload_desc = {
                'arrival_rate_range': [10, 100],
                'max_qps': 100,
                'prompt_length': {
                    'distribution': 'normal',
                    'mean': 512,
                    'std': 128
                },
                'output_length': {
                    'distribution': 'normal',
                    'mean': 256,
                    'std': 64
                }
            }

            # Default constraints
            constraints = {
                'max_cost_per_hour': 100.0,
                'min_qps': 10.0,
                'max_ttft_p95_ms': 2000,
                'max_tbt_p99_ms': 200,
                'slo_threshold_ms': 1000
            }

            # Default config space
            config_space = {
                'gpu_types': args.gpu_types,
                'num_gpus': [1, 2, 4, 8],
                'tp_sizes': [1, 2, 4],
                'pp_sizes': [1],
                'num_replicas': [1, 2, 4]
            }

            # Run search
            results = planner.search(
                config_space=config_space,
                constraints=constraints,
                workload_desc=workload_desc
            )

            # Save outputs
            planner.save_report(results, args.output)

            summary_path = Path(args.output).with_suffix('.txt')
            planner.save_summary(results, str(summary_path))

            # Print summary
            writer = ReportWriter()
            summary = writer.generate_summary(results)
            print("\n" + summary)

            logger.info("✓ Capacity planning complete!")
            return 0

        except Exception as e:
            logger.error(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    sys.exit(main())
