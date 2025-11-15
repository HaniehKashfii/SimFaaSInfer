#!/usr/bin/env python3
"""
Complete capacity planning workflow.

This script demonstrates the full capacity planning workflow:
1. Load FaaSInfer profiling outputs
2. Configure workload and constraints
3. Search for optimal configurations
4. Generate capacity planning reports

Usage:
    # Basic usage with defaults
    python examples/run_capacity_planning.py

    # With custom workload and constraints
    python examples/run_capacity_planning.py \
        --profile data/profiling/compute/a100/llama-7b/fitted_profile.json \
        --workload configs/workload/production.yaml \
        --constraints configs/constraints/strict_slos.yaml \
        --output results/capacity_plan.json

    # With production telemetry calibration
    python examples/run_capacity_planning.py \
        --profile data/profiling/compute/a100/llama-7b/fitted_profile.json \
        --telemetry data/telemetry/prod_samples.json \
        --output results/capacity_plan_calibrated.json
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simfaasinfer.capacity_planning import CapacityPlanner, ReportWriter
from simfaasinfer.utils.logger import setup_logger


def main():
    """Run capacity planning workflow."""
    parser = argparse.ArgumentParser(
        description="Run capacity planning for serverless LLM inference"
    )

    parser.add_argument(
        '--profile',
        type=str,
        default='data/profiling/compute/a100/llama-7b/fitted_profile.json',
        help='Path to FaaSInfer fitted_profile.json'
    )

    parser.add_argument(
        '--telemetry',
        type=str,
        default=None,
        help='Optional path to production telemetry for calibration'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results/capacity_plan.json',
        help='Path to save capacity planning report'
    )

    parser.add_argument(
        '--gpu-types',
        nargs='+',
        default=['A100'],
        help='GPU types to evaluate (e.g., A100 H100 A10G)'
    )

    parser.add_argument(
        '--max-cost',
        type=float,
        default=100.0,
        help='Maximum cost per hour ($)'
    )

    parser.add_argument(
        '--min-qps',
        type=float,
        default=10.0,
        help='Minimum required QPS'
    )

    parser.add_argument(
        '--ttft-slo',
        type=int,
        default=2000,
        help='Time to First Token P95 SLO (ms)'
    )

    parser.add_argument(
        '--tbt-slo',
        type=int,
        default=200,
        help='Time Between Tokens P99 SLO (ms)'
    )

    parser.add_argument(
        '--prompt-len',
        type=int,
        default=512,
        help='Average prompt length (tokens)'
    )

    parser.add_argument(
        '--output-len',
        type=int,
        default=256,
        help='Average output length (tokens)'
    )

    parser.add_argument(
        '--arrival-rate',
        type=float,
        default=50.0,
        help='Target arrival rate (QPS) to search up to'
    )

    args = parser.parse_args()

    logger = setup_logger("CapacityPlanning")

    logger.info("=" * 70)
    logger.info("   SIMFAASINFER CAPACITY PLANNING")
    logger.info("=" * 70)
    logger.info("")

    # Check if profile exists
    profile_path = Path(args.profile)
    if not profile_path.exists():
        logger.error(f"Profile not found: {profile_path}")
        logger.info("\nAvailable profiles:")
        profile_dir = Path("data/profiling/compute")
        if profile_dir.exists():
            for p in profile_dir.rglob("fitted_profile.json"):
                logger.info(f"  - {p}")
        sys.exit(1)

    logger.info(f"Loading profile: {args.profile}")

    # Initialize planner
    planner = CapacityPlanner(args.profile)

    # Optional calibration
    if args.telemetry:
        logger.info(f"Calibrating with telemetry: {args.telemetry}")
        planner.calibrate(args.telemetry)
        logger.info("✓ Calibration complete")
    else:
        logger.info("⚠ Running without calibration (use --telemetry for better accuracy)")

    logger.info("")

    # Configure workload
    workload_desc = {
        'arrival_rate_range': [args.min_qps, args.arrival_rate],
        'max_qps': args.arrival_rate,
        'prompt_length': {
            'distribution': 'normal',
            'mean': args.prompt_len,
            'std': args.prompt_len // 4
        },
        'output_length': {
            'distribution': 'normal',
            'mean': args.output_len,
            'std': args.output_len // 4
        }
    }

    logger.info("Workload Configuration:")
    logger.info(f"  Target QPS:        {args.arrival_rate}")
    logger.info(f"  Prompt Length:     {args.prompt_len} tokens (avg)")
    logger.info(f"  Output Length:     {args.output_len} tokens (avg)")
    logger.info("")

    # Configure constraints
    constraints = {
        'max_cost_per_hour': args.max_cost,
        'min_qps': args.min_qps,
        'max_ttft_p95_ms': args.ttft_slo,
        'max_tbt_p99_ms': args.tbt_slo,
        'slo_threshold_ms': 1000  # For Vidur search scheduling delay
    }

    logger.info("Constraints:")
    logger.info(f"  Max Cost:          ${args.max_cost}/hour")
    logger.info(f"  Min QPS:           {args.min_qps}")
    logger.info(f"  TTFT P95 SLO:      <{args.ttft_slo}ms")
    logger.info(f"  TBT P99 SLO:       <{args.tbt_slo}ms")
    logger.info("")

    # Configure search space
    config_space = {
        'gpu_types': args.gpu_types,
        'num_gpus': [1, 2, 4, 8],
        'tp_sizes': [1, 2, 4],
        'pp_sizes': [1],
        'num_replicas': [1, 2, 4]
    }

    logger.info("Configuration Search Space:")
    logger.info(f"  GPU Types:         {', '.join(args.gpu_types)}")
    logger.info(f"  GPU Counts:        {config_space['num_gpus']}")
    logger.info(f"  Tensor Parallel:   {config_space['tp_sizes']}")
    logger.info(f"  Replicas:          {config_space['num_replicas']}")
    logger.info("")

    # Run search
    logger.info("━" * 70)
    logger.info("Starting capacity planning search...")
    logger.info("━" * 70)
    logger.info("")

    try:
        results = planner.search(
            config_space=config_space,
            constraints=constraints,
            workload_desc=workload_desc
        )

        # Save reports
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("")
        logger.info("━" * 70)
        logger.info("Search Complete!")
        logger.info("━" * 70)
        logger.info("")

        # Generate and display summary
        writer = ReportWriter()
        summary = writer.generate_summary(results)

        print(summary)

        # Save JSON report
        planner.save_report(results, str(output_path))
        logger.info(f"\n✓ JSON report saved: {output_path}")

        # Save text summary
        summary_path = output_path.with_suffix('.txt')
        planner.save_summary(results, str(summary_path))
        logger.info(f"✓ Text summary saved: {summary_path}")

        # Show Pareto frontier if multiple configs
        pareto_points = results.get('pareto', {}).get('points', [])
        if len(pareto_points) > 1:
            pareto_table = writer.generate_pareto_table(pareto_points)
            print(pareto_table)

        logger.info("")
        logger.info("=" * 70)
        logger.info("Capacity planning complete!")
        logger.info("=" * 70)

        # Return success
        return 0

    except Exception as e:
        logger.error(f"\n✗ Capacity planning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
