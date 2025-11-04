"""CLI helper to run SimFaaSInfer with calibrated estimators and trace workloads."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import yaml

from configs import load_config as load_yaml_config, merge_configs
from simfaasinfer import Simulator
from simfaasinfer.utils.io import load_json

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def ensure_full_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure simulation + workload settings exist by merging defaults if needed."""
    if config.get("simulation") and config.get("workload"):
        return config

    if not DEFAULT_CONFIG_PATH.exists():
        raise ValueError(
            "Simulation/workload settings missing and default config not found at "
            f"{DEFAULT_CONFIG_PATH}"
        )

    base_config = load_yaml_config(str(DEFAULT_CONFIG_PATH))
    return merge_configs(base_config, config)


def attach_trace_workload(config: Dict[str, Any], trace_path: str) -> None:
    """Configure workload section to replay a trace file."""
    workload_cfg = config.setdefault("workload", {})
    workload_cfg["type"] = "trace"
    workload_cfg["trace_path"] = trace_path

    try:
        trace_data = load_json(trace_path)
    except FileNotFoundError:
        trace_data = []

    if isinstance(trace_data, list):
        num_entries = len(trace_data)
    elif isinstance(trace_data, dict):
        num_entries = len(
            trace_data.get("requests")
            or trace_data.get("entries")
            or trace_data.get("trace", [])
        )
    else:
        num_entries = 0

    if num_entries and not workload_cfg.get("num_requests"):
        workload_cfg["num_requests"] = num_entries


def compute_startup_stats(results: Dict[str, Any]) -> Dict[str, float]:
    """Extract startup latency metrics from simulator results."""
    samples: List[float] = results.get("startup_latency_samples") or []
    if samples:
        ordered = sorted(samples)

        def percentile(data: List[float], pct: float) -> float:
            if not data:
                return float("nan")
            k = (len(data) - 1) * (pct / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return data[int(k)]
            return data[f] * (c - k) + data[c] * (k - f)

        return {
            "mean_startup_latency": sum(ordered) / len(ordered),
            "p95_startup_latency": percentile(ordered, 95),
            "p99_startup_latency": percentile(ordered, 99),
        }

    stats = {
        key: results.get(key)
        for key in ("mean_startup_latency", "p95_startup_latency", "p99_startup_latency")
        if results.get(key) is not None
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SimFaaSInfer simulation with calibrated estimator.")
    parser.add_argument(
        "--config",
        required=True,
        help="Simulation config or profiled model YAML path.",
    )
    parser.add_argument(
        "--workload",
        required=True,
        help="Trace JSON file to replay.",
    )
    parser.add_argument(
        "--estimator_dir",
        default="/tmp/calibrated_estimator",
        help="Path to calibrated estimator directory.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = ensure_full_config(cfg or {})

    estimator_cfg = cfg.setdefault("estimator", {})
    estimator_cfg["path"] = args.estimator_dir

    attach_trace_workload(cfg, args.workload)

    simulator = Simulator(cfg)
    results = simulator.run()

    startup_stats = compute_startup_stats(results)
    if startup_stats:
        print("Startup latency stats:", startup_stats)

    print("Simulation results:", json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
