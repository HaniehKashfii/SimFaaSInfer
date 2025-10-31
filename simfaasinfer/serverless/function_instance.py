"""Function instance management for serverless LLM inference."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

from ..models.model_config import ModelConfig
from ..models.execution_time_predictor import ExecutionTimePredictor
from ..utils.logger import setup_logger


class InstanceState(Enum):
    """States of a function instance."""
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    IDLE = "idle"
    SHUTDOWN = "shutdown"


@dataclass
class Request:
    """Represents an inference request."""
    request_id: int
    arrival_time: float
    prompt_tokens: int
    decode_tokens: int

    # State tracking
    enqueue_time: Optional[float] = None
    schedule_time: Optional[float] = None
    prefill_start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    decode_start_time: Optional[float] = None
    completion_time: Optional[float] = None

    # Metrics
    scheduling_delay: float = 0.0
    prefill_time: float = 0.0
    decode_time: float = 0.0

    # Processing state
    tokens_generated: int = 0
    kv_cache_tokens: int = 0
    assigned_instance: Optional[int] = None

    def is_complete(self) -> bool:
        """Check if request is complete."""
        return self.tokens_generated >= self.decode_tokens

    def __repr__(self) -> str:
        return f"Request(id={self.request_id}, prompt={self.prompt_tokens}, decode={self.decode_tokens})"


class FunctionInstance:
    """Represents a serverless function instance running LLM inference.

    Each instance can process batches of requests using the configured
    scheduling policy and parallelization strategy.
    """

    def __init__(self, instance_id: int, config: Dict, model_config: ModelConfig):
        """Initialize function instance.

        Args:
            instance_id: Unique instance identifier
            config: Configuration dictionary
            model_config: Model configuration
        """
        self.instance_id = instance_id
        self.config = config
        self.model_config = model_config
        self.logger = setup_logger(f"Instance-{instance_id}")

        # State
        self.state = InstanceState.STARTING
        self.start_time = None
        self.ready_time = None

        # Resources
        self.gpu_memory_gb = config['cluster']['gpu_memory_gb']
        self.num_gpus = config['cluster']['num_gpus_per_replica']

        # Memory allocation
        reserved_memory = config['memory'].get('reserved_memory_gb', 2.0)
        self.total_memory_gb = self.gpu_memory_gb * self.num_gpus
        self.available_memory_gb = self.total_memory_gb - model_config.model_memory_gb - reserved_memory
        self.kv_cache_capacity_tokens = int(
            (self.available_memory_gb * 1024) / model_config.get_kv_cache_size_per_token()
        )

        # Current state
        self.current_batch: List[Request] = []
        self.kv_cache_used_tokens = 0
        self.is_processing = False

        # Execution time predictor
        self.predictor = ExecutionTimePredictor(
            model_config,
            config['parallelization']
        )

        # Statistics
        self.total_requests_processed = 0
        self.total_compute_time = 0.0
        self.total_idle_time = 0.0
        self.last_activity_time = 0.0

    def mark_ready(self, current_time: float) -> None:
        """Mark instance as ready to serve requests.

        Args:
            current_time: Current simulation time
        """
        self.state = InstanceState.READY
        self.ready_time = current_time
        self.last_activity_time = current_time
        self.logger.info(f"Instance {self.instance_id} ready at {current_time:.2f}s")

    def can_fit_request(self, request: Request) -> bool:
        """Check if request can fit in available KV-cache.

        Args:
            request: Request to check

        Returns:
            True if request fits
        """
        required_tokens = request.prompt_tokens + request.decode_tokens
        return (self.kv_cache_used_tokens + required_tokens) <= self.kv_cache_capacity_tokens

    def add_request(self, request: Request, current_time: float) -> bool:
        """Add request to instance.

        Args:
            request: Request to add
            current_time: Current simulation time

        Returns:
            True if request was added
        """
        if not self.can_fit_request(request):
            return False

        self.current_batch.append(request)
        request.assigned_instance = self.instance_id
        request.schedule_time = current_time

        # Allocate KV-cache
        required_tokens = request.prompt_tokens + request.decode_tokens
        self.kv_cache_used_tokens += required_tokens

        return True

    def execute_batch(self, batch: List[Request], current_time: float) -> float:
        """Execute a batch of requests.

        Args:
            batch: List of requests to execute
            current_time: Current simulation time

        Returns:
            Predicted execution time in seconds
        """
        if not batch:
            return 0.0

        self.is_processing = True
        self.state = InstanceState.BUSY

        # Separate prefill and decode
        prefill_requests = [r for r in batch if r.tokens_generated == 0]
        decode_requests = [r for r in batch if r.tokens_generated > 0]

        # Calculate tokens
        prefill_tokens = sum(r.prompt_tokens for r in prefill_requests)
        decode_tokens = len(decode_requests)  # One token per request
        kv_cache_tokens = sum(r.kv_cache_tokens + r.tokens_generated for r in decode_requests)

        # Predict execution time
        execution_time_ms = self.predictor.predict_batch_time(
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            kv_cache_tokens=kv_cache_tokens,
            batch_size=len(batch)
        )

        # Update request states
        for request in prefill_requests:
            if not request.prefill_start_time:
                request.prefill_start_time = current_time
                request.kv_cache_tokens = request.prompt_tokens

        for request in decode_requests:
            if not request.first_token_time and request.tokens_generated == 0:
                request.first_token_time = current_time
            if not request.decode_start_time:
                request.decode_start_time = current_time

            request.tokens_generated += 1
            request.kv_cache_tokens += 1

        # Update statistics
        execution_time_sec = execution_time_ms / 1000.0
        self.total_compute_time += execution_time_sec
        self.last_activity_time = current_time + execution_time_sec

        return execution_time_sec

    def remove_completed_requests(self) -> List[Request]:
        """Remove and return completed requests.

        Returns:
            List of completed requests
        """
        completed = [r for r in self.current_batch if r.is_complete()]
        self.current_batch = [r for r in self.current_batch if not r.is_complete()]

        # Free KV-cache
        for request in completed:
            freed_tokens = request.prompt_tokens + request.decode_tokens
            self.kv_cache_used_tokens -= freed_tokens
            self.total_requests_processed += 1

        if not self.current_batch:
            self.state = InstanceState.IDLE
            self.is_processing = False

        return completed

    def get_utilization(self) -> float:
        """Get compute utilization.

        Returns:
            Utilization as fraction (0-1)
        """
        if self.ready_time is None:
            return 0.0

        total_time = self.last_activity_time - self.ready_time
        if total_time <= 0:
            return 0.0

        return min(1.0, self.total_compute_time / total_time)

    def get_memory_utilization(self) -> float:
        """Get memory utilization.

        Returns:
            Memory utilization as fraction (0-1)
        """
        return self.kv_cache_used_tokens / self.kv_cache_capacity_tokens

    def get_metrics(self) -> Dict:
        """Get instance metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            'instance_id': self.instance_id,
            'state': self.state.value,
            'utilization': self.get_utilization(),
            'memory_utilization': self.get_memory_utilization(),
            'requests_in_progress': len(self.current_batch),
            'total_requests_processed': self.total_requests_processed,
            'kv_cache_used': self.kv_cache_used_tokens,
            'kv_cache_capacity': self.kv_cache_capacity_tokens,
        }


class FunctionInstanceManager:
    """Manages a pool of function instances."""

    def __init__(self, config: Dict, model_config: ModelConfig, cold_start_sim):
        """Initialize instance manager.

        Args:
            config: Configuration dictionary
            model_config: Model configuration
            cold_start_sim: Cold start simulator
        """
        self.config = config
        self.model_config = model_config
        self.cold_start_sim = cold_start_sim
        self.logger = setup_logger(self.__class__.__name__)

        self.instances: Dict[int, FunctionInstance] = {}
        self.next_instance_id = 0

    def create_instance(self, instance_id: Optional[int] = None,
                        start_time: float = 0.0) -> FunctionInstance:
        """Create a new function instance.

        Args:
            instance_id: Optional instance ID (auto-generated if None)
            start_time: Start time

        Returns:
            Created instance
        """
        if instance_id is None:
            instance_id = self.next_instance_id
            self.next_instance_id += 1

        instance = FunctionInstance(instance_id, self.config, self.model_config)
        instance.start_time = start_time
        self.instances[instance_id] = instance

        self.logger.debug(f"Created instance {instance_id}")
        return instance

    def get_instance(self, instance_id: int) -> Optional[FunctionInstance]:
        """Get instance by ID.

        Args:
            instance_id: Instance ID

        Returns:
            Instance or None
        """
        return self.instances.get(instance_id)

    def remove_instance(self, instance_id: int) -> None:
        """Remove an instance.

        Args:
            instance_id: Instance ID
        """
        if instance_id in self.instances:
            del self.instances[instance_id]
            self.logger.debug(f"Removed instance {instance_id}")

    def get_ready_instances(self) -> List[FunctionInstance]:
        """Get all ready instances.

        Returns:
            List of ready instances
        """
        return [
            inst for inst in self.instances.values()
            if inst.state in [InstanceState.READY, InstanceState.IDLE]
        ]

    def get_least_utilized_instance(self) -> Optional[int]:
        """Get ID of least utilized instance.

        Returns:
            Instance ID or None
        """
        ready_instances = self.get_ready_instances()
        if not ready_instances:
            return None

        return min(ready_instances, key=lambda inst: inst.get_utilization()).instance_id

    def num_instances(self) -> int:
        """Get number of instances.

        Returns:
            Instance count
        """
        return len(self.instances)

    def get_cluster_metrics(self) -> Dict:
        """Get cluster-level metrics.

        Returns:
            Dictionary of cluster metrics
        """
        if not self.instances:
            return {
                'num_instances': 0,
                'mfu': 0.0,
                'memory_util': 0.0,
                'kv_cache_util': 0.0,
            }

        utilizations = [inst.get_utilization() for inst in self.instances.values()]
        memory_utils = [inst.get_memory_utilization() for inst in self.instances.values()]

        return {
            'num_instances': len(self.instances),
            'mfu': sum(utilizations) / len(utilizations) if utilizations else 0.0,
            'memory_util': sum(memory_utils) / len(memory_utils) if memory_utils else 0.0,
            'kv_cache_util': sum(memory_utils) / len(memory_utils) if memory_utils else 0.0,
        }