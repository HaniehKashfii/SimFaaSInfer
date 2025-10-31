"""Request arrival process modeling."""

import numpy as np
from typing import List

from ..utils.logger import setup_logger


class ArrivalProcess:
    """Models request arrival patterns.

    Supports various arrival processes:
    - Poisson (memoryless arrivals)
    - Uniform (evenly spaced)
    - Gamma (bursty)
    """

    def __init__(self, process_type: str = 'poisson', rate: float = 10.0):
        """Initialize arrival process.

        Args:
            process_type: Type of arrival process
            rate: Arrival rate (requests per second)
        """
        self.process_type = process_type
        self.rate = rate
        self.logger = setup_logger(self.__class__.__name__)

    def generate_arrivals(self, start_time: float, end_time: float) -> List[float]:
        """Generate arrival times.

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            List of arrival times
        """
        if self.process_type == 'poisson':
            return self._poisson_arrivals(start_time, end_time)
        elif self.process_type == 'uniform':
            return self._uniform_arrivals(start_time, end_time)
        elif self.process_type == 'gamma':
            return self._gamma_arrivals(start_time, end_time)
        else:
            raise ValueError(f"Unknown arrival process: {self.process_type}")

    def _poisson_arrivals(self, start_time: float, end_time: float) -> List[float]:
        """Generate Poisson arrivals (exponential inter-arrival times).

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            List of arrival times
        """
        arrivals = []
        current_time = start_time

        while current_time < end_time:
            # Exponential inter-arrival time
            inter_arrival = np.random.exponential(1.0 / self.rate)
            current_time += inter_arrival

            if current_time < end_time:
                arrivals.append(current_time)

        return arrivals

    def _uniform_arrivals(self, start_time: float, end_time: float) -> List[float]:
        """Generate uniformly spaced arrivals.

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            List of arrival times
        """
        duration = end_time - start_time
        num_arrivals = int(duration * self.rate)
        inter_arrival = 1.0 / self.rate

        arrivals = [start_time + i * inter_arrival for i in range(num_arrivals)]
        return arrivals

    def _gamma_arrivals(self, start_time: float, end_time: float) -> List[float]:
        """Generate arrivals with gamma-distributed inter-arrival times (bursty).

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            List of arrival times
        """
        arrivals = []
        current_time = start_time

        # Gamma distribution parameters for burstiness
        mean_inter_arrival = 1.0 / self.rate
        shape = 0.5  # Lower shape = more bursty
        scale = mean_inter_arrival / shape

        while current_time < end_time:
            inter_arrival = np.random.gamma(shape, scale)
            current_time += inter_arrival

            if current_time < end_time:
                arrivals.append(current_time)

        return arrivals