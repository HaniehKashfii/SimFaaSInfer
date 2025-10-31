"""Cost calculation for serverless LLM inference."""

from typing import Dict

from ..utils.logger import setup_logger


class CostCalculator:
    """Calculate costs for serverless LLM inference.

    Includes:
    - GPU compute costs
    - Idle time costs
    - Data transfer costs (optional)
    """

    def __init__(self, config: Dict):
        """Initialize cost calculator.

        Args:
            config: Cost configuration
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)

        # Pricing (per GPU per hour)
        self.gpu_costs = config.get('gpu_cost_per_hour', {
            'A100': 3.06,
            'H100': 5.12,
        })

        self.compute_cost_enabled = config.get('compute_cost', True)
        self.idle_cost_enabled = config.get('idle_cost', True)
        self.idle_cost_factor = config.get('idle_cost_factor', 0.5)  # 50% of compute cost when idle

    def calculate_total_cost(self, duration: float, num_instances: int,
                             gpu_type: str, num_gpus_per_instance: int,
                             utilization: float = 1.0) -> Dict:
        """Calculate total cost for a simulation run.

        Args:
            duration: Duration in seconds
            num_instances: Number of instances
            gpu_type: GPU type (A100, H100)
            num_gpus_per_instance: GPUs per instance
            utilization: Average utilization (0-1)

        Returns:
            Dictionary with cost breakdown
        """
        duration_hours = duration / 3600.0

        # Base GPU cost
        gpu_cost_per_hour = self.gpu_costs.get(gpu_type, 3.06)
        total_gpu_hours = duration_hours * num_instances * num_gpus_per_instance

        # Compute cost (utilization-weighted)
        if self.compute_cost_enabled:
            compute_cost = total_gpu_hours * gpu_cost_per_hour * utilization
        else:
            compute_cost = 0.0

        # Idle cost
        if self.idle_cost_enabled:
            idle_time_hours = total_gpu_hours * (1 - utilization)
            idle_cost = idle_time_hours * gpu_cost_per_hour * self.idle_cost_factor
        else:
            idle_cost = 0.0

        total_cost = compute_cost + idle_cost

        return {
            'total_cost': total_cost,
            'compute_cost': compute_cost,
            'idle_cost': idle_cost,
            'gpu_hours': total_gpu_hours,
            'cost_per_gpu_hour': gpu_cost_per_hour,
            'duration_hours': duration_hours,
        }

    def calculate_qps_per_dollar(self, throughput: float, total_cost: float) -> float:
        """Calculate queries per second per dollar.

        Args:
            throughput: Throughput in QPS
            total_cost: Total cost in dollars

        Returns:
            QPS per dollar
        """
        if total_cost <= 0:
            return 0.0

        return throughput / total_cost

    def calculate_cost_per_request(self, total_cost: float, num_requests: int) -> float:
        """Calculate cost per request.

        Args:
            total_cost: Total cost in dollars
            num_requests: Number of requests

        Returns:
            Cost per request
        """
        if num_requests <= 0:
            return 0.0

        return total_cost / num_requests