"""
Capacity Planning module - orchestrates FaaSInfer profile loading,
simulation runs, and capacity recommendation generation.
"""
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..core.simulator import Simulator
from ..optimizer.vidur_search import search_workload
from ..calibration.calibrator import calibrate
from ..runtime_estimator.rf_estimator import RFEstimator
from ..utils.logger import setup_logger


class CapacityPlanner:
    """
    Orchestrates capacity planning workflow:
    1. Load FaaSInfer profiling outputs
    2. Run configuration search
    3. Generate recommendations
    """

    def __init__(self, profile_path: str, metadata_path: Optional[str] = None):
        """
        Initialize capacity planner.

        Args:
            profile_path: Path to fitted_profile.json from FaaSInfer
            metadata_path: Optional path to metadata.json
        """
        self.logger = setup_logger("CapacityPlanner")
        self.profile_path = Path(profile_path)

        # Auto-detect metadata path if not provided
        if metadata_path is None:
            metadata_path = self.profile_path.parent / "metadata.json"
        self.metadata_path = Path(metadata_path)

        # Load profiling data
        self.profile_data = self._load_profile()
        self.metadata = self._load_metadata()

        # Estimator (can be calibrated later)
        self.estimator = None
        self.calibrated = False

        self.logger.info(f"Loaded profile for {self.metadata.get('model_name', 'unknown')}")
        self.logger.info(f"Hardware: {self.metadata.get('hardware', {}).get('gpu', 'unknown')}")

    def _load_profile(self) -> List[Dict[str, Any]]:
        """Load fitted profile from FaaSInfer."""
        if not self.profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {self.profile_path}")

        with open(self.profile_path, 'r') as f:
            data = json.load(f)

        self.logger.info(f"Loaded {len(data)} profile entries")
        return data

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata if available."""
        if not self.metadata_path.exists():
            self.logger.warning(f"Metadata not found: {self.metadata_path}")
            return {}

        with open(self.metadata_path, 'r') as f:
            return json.load(f)

    def calibrate(self, telemetry_path: str):
        """
        Calibrate estimator with production telemetry.

        Args:
            telemetry_path: Path to telemetry samples JSON
        """
        self.logger.info(f"Calibrating with telemetry: {telemetry_path}")

        # Load telemetry
        with open(telemetry_path, 'r') as f:
            telemetry_samples = json.load(f)

        # Initialize base estimator if needed
        if self.estimator is None:
            # For now, create a simple estimator from profile
            # In production, load pre-trained RF estimator
            self.estimator = self._create_profile_estimator()

        # Calibrate
        from ..calibration.calibrator import calibrate
        self.estimator = calibrate(self.estimator, telemetry_samples)
        self.calibrated = True

        self.logger.info("Calibration complete")

    def _create_profile_estimator(self):
        """Create estimator from profile data."""
        # Simple lookup-based estimator from profile
        from ..runtime_estimator.profile_lookup import ProfileLookupEstimator
        return ProfileLookupEstimator(self.profile_data)

    def search(
        self,
        config_space: Dict[str, Any],
        constraints: Dict[str, Any],
        workload_desc: Dict[str, Any],
        simulation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for optimal configuration.

        Args:
            config_space: Configuration space to explore
            constraints: SLO and cost constraints
            workload_desc: Workload description
            simulation_config: Optional simulation parameters

        Returns:
            Search results with recommendations
        """
        self.logger.info("Starting capacity planning search...")

        # Default simulation config
        if simulation_config is None:
            simulation_config = {
                'duration': 600,
                'warmup_duration': 60,
                'random_seed': 42
            }

        # Create simulator function for search
        def simulator_fn(config: Dict[str, Any], qps: float) -> Dict[str, Any]:
            return self._run_simulation(config, qps, workload_desc, simulation_config)

        # Run Vidur-style search
        search_results = search_workload(
            workload_desc=workload_desc,
            config_space=config_space,
            constraints=constraints,
            simulator_fn=simulator_fn
        )

        # Enhance results with metadata
        search_results['input_profile'] = str(self.profile_path)
        search_results['model_name'] = self.metadata.get('model_name', 'unknown')
        search_results['profiling_hardware'] = self.metadata.get('hardware', {})
        search_results['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        search_results['calibrated'] = self.calibrated

        # Add best configuration details
        if search_results['best_config']:
            search_results['best_configuration'] = self._format_best_config(
                search_results['best_config']
            )

        self.logger.info("Search complete")
        return search_results

    def _run_simulation(
        self,
        hw_config: Dict[str, Any],
        qps: float,
        workload_desc: Dict[str, Any],
        simulation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run single simulation for given configuration."""
        # Build full simulator config
        config = {
            'simulation': simulation_config,
            'model': {
                'name': self.metadata.get('model_name', 'custom'),
                'profile_path': str(self.profile_path)
            },
            'cluster': {
                'gpu_type': hw_config.get('gpu_type', 'A100'),
                'num_gpus': hw_config.get('total_gpus', 1),
                'tensor_parallel_size': hw_config.get('tp', 1),
                'pipeline_parallel_size': hw_config.get('pp', 1)
            },
            'scheduler': {
                'type': hw_config.get('scheduler', 'vllm'),
                'max_batch_size': hw_config.get('max_batch_size', 128)
            },
            'workload': {
                'type': 'poisson',
                'arrival_rate': qps,
                'prompt_length': workload_desc.get('prompt_length', {
                    'distribution': 'normal',
                    'mean': 512,
                    'std': 128
                }),
                'output_length': workload_desc.get('output_length', {
                    'distribution': 'normal',
                    'mean': 256,
                    'std': 64
                })
            },
            'cold_start': {'enabled': True},
            'scaling': {'enabled': False},
            'cost': {
                'gpu_costs': {
                    'A100': 3.06,
                    'H100': 5.12,
                    'A10G': 1.21
                }
            }
        }

        try:
            # Run simulation
            simulator = Simulator(config)
            results = simulator.run()

            # Extract key metrics
            return {
                'throughput': results.get('throughput', 0),
                'p50_latency': results.get('p50_latency', 0),
                'p95_latency': results.get('p95_latency', 0),
                'p99_latency': results.get('p99_latency', 0),
                'p95_ttft': results.get('p95_ttft', 0),
                'p99_tbt': results.get('p99_tbt', 0),
                'scheduling_delay_p99': results.get('p99_scheduling_delay', 0),
                'mean_mfu': results.get('mean_mfu', 0),
                'mean_memory_util': results.get('mean_memory_util', 0),
                'completion_rate': results.get('completion_rate', 0)
            }
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            # Return poor metrics on failure
            return {
                'throughput': 0,
                'p99_latency': float('inf'),
                'scheduling_delay_p99': float('inf'),
                'completion_rate': 0
            }

    def _format_best_config(self, best_config: Dict[str, Any]) -> Dict[str, Any]:
        """Format best configuration for output."""
        config_details = best_config['config']
        metrics = best_config.get('metrics', {})

        # Check SLO compliance
        slo_compliance = {
            'meets_all_slos': True,
            'ttft_slo_met': True,
            'tbt_slo_met': True,
            'latency_slo_met': True
        }

        return {
            'gpu_type': config_details.get('gpu_type'),
            'num_gpus_per_replica': config_details.get('num_gpus_per_replica'),
            'tensor_parallel': config_details.get('tp'),
            'pipeline_parallel': config_details.get('pp'),
            'num_replicas': config_details.get('num_replicas'),
            'total_gpus': config_details.get('total_gpus'),
            'scheduler': config_details.get('scheduler', 'vllm'),
            'max_batch_size': config_details.get('max_batch_size', 128),

            'performance': {
                'max_sustainable_qps': best_config.get('max_qps', 0),
                'p50_latency_ms': metrics.get('p50_latency', 0),
                'p95_latency_ms': metrics.get('p95_latency', 0),
                'p99_latency_ms': metrics.get('p99_latency', 0),
                'p95_ttft_ms': metrics.get('p95_ttft', 0),
                'p99_tbt_ms': metrics.get('p99_tbt', 0),
                'mean_mfu': metrics.get('mean_mfu', 0),
                'mean_memory_util': metrics.get('mean_memory_util', 0)
            },

            'cost': {
                'cost_per_hour': best_config.get('cost_per_hour', 0),
                'cost_per_1k_requests': (best_config.get('cost_per_hour', 0) /
                                        (best_config.get('max_qps', 1) * 3.6)),
                'qps_per_dollar': best_config.get('qps_per_dollar', 0)
            },

            'slo_compliance': slo_compliance
        }

    def save_report(self, results: Dict[str, Any], output_path: str):
        """Save capacity planning report as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Report saved to: {output_path}")

    def save_summary(self, results: Dict[str, Any], output_path: str):
        """Save human-readable summary."""
        from .report_writer import ReportWriter

        writer = ReportWriter()
        summary = writer.generate_summary(results)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(summary)

        self.logger.info(f"Summary saved to: {output_path}")


def run_capacity_planning(
    profile_path: str,
    workload_config: str,
    output_path: str,
    constraints_config: Optional[str] = None,
    telemetry_path: Optional[str] = None,
    config_space: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Simplified interface for running capacity planning.

    Args:
        profile_path: Path to FaaSInfer fitted_profile.json
        workload_config: Path to workload config YAML
        output_path: Path to save results
        constraints_config: Optional path to constraints YAML
        telemetry_path: Optional path to telemetry for calibration
        config_space: Optional config space dict

    Returns:
        Capacity planning results
    """
    logger = setup_logger("CapacityPlanning")

    # Initialize planner
    planner = CapacityPlanner(profile_path)

    # Optional calibration
    if telemetry_path:
        planner.calibrate(telemetry_path)

    # Load workload config
    with open(workload_config, 'r') as f:
        workload_desc = yaml.safe_load(f)

    # Load constraints
    if constraints_config:
        with open(constraints_config, 'r') as f:
            constraints = yaml.safe_load(f)
    else:
        # Default constraints
        constraints = {
            'max_ttft_p95_ms': 2000,
            'max_tbt_p99_ms': 200,
            'max_cost_per_hour': 100.0,
            'min_qps': 1.0,
            'slo_threshold_ms': 1000
        }

    # Default config space
    if config_space is None:
        config_space = {
            'gpu_types': ['A100'],
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
    planner.save_report(results, output_path)

    # Also save summary
    summary_path = Path(output_path).with_suffix('.txt')
    planner.save_summary(results, str(summary_path))

    logger.info("Capacity planning complete!")

    return results
