"""Request generation for simulation workloads."""

import numpy as np
from typing import List, Dict

from ..serverless.function_instance import Request
from .arrival_process import ArrivalProcess
from ..utils.logger import setup_logger


class RequestGenerator:
    """Generate inference requests for simulation.

    Supports multiple generation strategies:
    - Poisson arrivals with configurable distributions
    - Trace-based replay
    - Synthetic workloads
    """

    def __init__(self, config: Dict):
        """Initialize request generator.

        Args:
            config: Workload configuration
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)

        self.workload_type = config.get('type', 'poisson')
        self.arrival_rate = config.get('arrival_rate', 10)
        self.num_requests = config.get('num_requests', 1000)

        # Request length distributions
        self.prompt_config = config.get('prompt_length', {})
        self.decode_config = config.get('decode_length', {})

        # Arrival process
        self.arrival_process = ArrivalProcess(self.workload_type, self.arrival_rate)

        self.request_counter = 0

    def generate(self, start_time: float = 0.0, end_time: float = 3600.0) -> List[Request]:
        """Generate requests for the simulation period.

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            List of generated requests
        """
        self.logger.info(f"Generating {self.workload_type} workload...")

        if self.workload_type == 'poisson':
            return self._generate_poisson(start_time, end_time)
        elif self.workload_type == 'trace':
            return self._generate_from_trace(start_time, end_time)
        elif self.workload_type == 'synthetic':
            return self._generate_synthetic(start_time, end_time)
        else:
            raise ValueError(f"Unknown workload type: {self.workload_type}")

    def _generate_poisson(self, start_time: float, end_time: float) -> List[Request]:
        """Generate requests with Poisson arrivals.

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            List of requests
        """
        # Generate arrival times
        arrival_times = self.arrival_process.generate_arrivals(
            start_time, end_time
        )

        requests = []
        for arrival_time in arrival_times:
            request = self._create_request(arrival_time)
            requests.append(request)

        self.logger.info(f"Generated {len(requests)} Poisson requests")
        return requests

    def _generate_from_trace(self, start_time: float, end_time: float) -> List[Request]:
        """Generate requests from trace file.

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            List of requests
        """
        from .trace_loader import TraceLoader

        trace_path = self.config.get('trace_path')
        if not trace_path:
            raise ValueError("trace_path required for trace workload")

        loader = TraceLoader(trace_path)
        trace_data = loader.load()

        # Scale to simulation time
        duration = end_time - start_time
        requests = []

        for i, entry in enumerate(trace_data[:self.num_requests]):
            # Scale arrival time to simulation duration
            arrival_time = start_time + (entry['timestamp'] % duration)

            request = Request(
                request_id=i,
                arrival_time=arrival_time,
                prompt_tokens=entry['prompt_tokens'],
                decode_tokens=entry['decode_tokens']
            )
            requests.append(request)

        # Sort by arrival time
        requests.sort(key=lambda r: r.arrival_time)

        self.logger.info(f"Generated {len(requests)} requests from trace")
        return requests

    def _generate_synthetic(self, start_time: float, end_time: float) -> List[Request]:
        """Generate synthetic workload with specific patterns.

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            List of requests
        """
        # Generate burst patterns or specific distributions
        pattern = self.config.get('pattern', 'uniform')

        if pattern == 'burst':
            return self._generate_burst_pattern(start_time, end_time)
        elif pattern == 'uniform':
            return self._generate_poisson(start_time, end_time)
        else:
            raise ValueError(f"Unknown synthetic pattern: {pattern}")

    def _generate_burst_pattern(self, start_time: float, end_time: float) -> List[Request]:
        """Generate bursty traffic pattern.

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            List of requests
        """
        burst_duration = self.config.get('burst_duration', 60)  # seconds
        idle_duration = self.config.get('idle_duration', 120)
        burst_rate = self.arrival_rate * 3  # 3x normal rate during bursts

        requests = []
        current_time = start_time
        in_burst = True

        while current_time < end_time:
            if in_burst:
                # Generate burst
                burst_end = min(current_time + burst_duration, end_time)
                arrival_process = ArrivalProcess('poisson', burst_rate)
                arrivals = arrival_process.generate_arrivals(current_time, burst_end)

                for arrival_time in arrivals:
                    request = self._create_request(arrival_time)
                    requests.append(request)

                current_time = burst_end
                in_burst = False
            else:
                # Idle period
                current_time = min(current_time + idle_duration, end_time)
                in_burst = True

        self.logger.info(f"Generated {len(requests)} requests in burst pattern")
        return requests

    def _create_request(self, arrival_time: float) -> Request:
        """Create a single request with sampled parameters.

        Args:
            arrival_time: Request arrival time

        Returns:
            Created request
        """
        prompt_tokens = self._sample_length(self.prompt_config)
        decode_tokens = self._sample_length(self.decode_config)

        request = Request(
            request_id=self.request_counter,
            arrival_time=arrival_time,
            prompt_tokens=prompt_tokens,
            decode_tokens=decode_tokens
        )

        self.request_counter += 1
        return request

    def _sample_length(self, config: Dict) -> int:
        """Sample token length from configured distribution.

        Args:
            config: Length configuration

        Returns:
            Sampled length
        """
        distribution = config.get('distribution', 'gamma')
        mean = config.get('mean', 512)
        std = config.get('std', 256)
        min_val = config.get('min', 1)
        max_val = config.get('max', 4096)

        if distribution == 'gamma':
            # Use gamma distribution (common for text lengths)
            shape = (mean / std) ** 2
            scale = std ** 2 / mean
            sample = np.random.gamma(shape, scale)
        elif distribution == 'normal':
            sample = np.random.normal(mean, std)
        elif distribution == 'uniform':
            sample = np.random.uniform(min_val, max_val)
        elif distribution == 'constant':
            return mean
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Clip to valid range
        sample = np.clip(sample, min_val, max_val)
        return int(sample)