"""Cold start simulation for serverless functions."""

import numpy as np
from typing import Dict

from ..models.model_config import ModelConfig
from ..utils.logger import setup_logger


class ColdStartSimulator:
    """Simulates cold start delays for serverless LLM inference.

    Cold start includes:
    - Container initialization
    - Model loading
    - GPU memory allocation
    - KV-cache setup
    """

    def __init__(self, config: Dict, model_config: ModelConfig):
        """Initialize cold start simulator.

        Args:
            config: Cold start configuration
            model_config: Model configuration
        """
        self.config = config
        self.model_config = model_config
        self.logger = setup_logger(self.__class__.__name__)

        self.enabled = config.get('enabled', True)
        self.base_cold_start_time = config.get('model_load_time', 30.0)

        # Additional delays
        self.container_init_time = config.get('container_init_time', 5.0)
        self.memory_alloc_time = config.get('memory_alloc_time', 2.0)

    def simulate_cold_start(self) -> float:
        """Simulate a cold start and return the delay.

        Returns:
            Cold start time in seconds
        """
        if not self.enabled:
            return 0.0

        # Base time for model loading (scales with model size)
        model_load_time = self.base_cold_start_time * (
                self.model_config.model_memory_gb / 14.0  # Normalized to 7B model
        )

        # Container initialization
        container_time = self.container_init_time

        # Memory allocation time
        memory_time = self.memory_alloc_time

        # Total cold start time
        total_time = model_load_time + container_time + memory_time

        # Add some variance (±10%)
        noise = np.random.uniform(0.9, 1.1)
        total_time *= noise

        self.logger.debug(f"Cold start simulated: {total_time:.2f}s")

        return total_time

    def get_warm_start_time(self) -> float:
        """Get warm start time (request processing on warm instance).

        Returns:
            Warm start time in seconds (typically very small)
        """
        return 0.001  # 1ms overhead