"""Request scheduling and routing."""

from typing import List, Dict, Optional
from collections import deque

from ..serverless.function_instance import Request, FunctionInstance
from ..models.model_config import ModelConfig
from .batching_strategy import BatchingStrategy, VLLMBatcher, OrcaBatcher, SarathiBatcher, FasterTransformerBatcher
from ..utils.logger import setup_logger
from ..core.event_queue import Event, EventType


class GlobalScheduler:
    """Global scheduler for routing requests to instances.

    Implements load balancing and request routing strategies.
    """

    def __init__(self, config: Dict, instance_manager, model_config: ModelConfig):
        """Initialize global scheduler.

        Args:
            config: Scheduler configuration
            instance_manager: Function instance manager
            model_config: Model configuration
        """
        self.config = config
        self.instance_manager = instance_manager
        self.model_config = model_config
        self.logger = setup_logger(self.__class__.__name__)

        self.routing_policy = config.get('routing_policy', 'round_robin')

        # Request queues
        self.pending_requests = deque()

        # Replica schedulers (one per instance)
        self.replica_schedulers: Dict[int, ReplicaScheduler] = {}

        # Round-robin state
        self.next_instance_id = 0

    def schedule_request(self, request: Request, current_time: float) -> bool:
        """Schedule a request to an instance.

        Args:
            request: Request to schedule
            current_time: Current simulation time

        Returns:
            True if request was scheduled
        """
        # Get available instances
        ready_instances = self.instance_manager.get_ready_instances()

        if not ready_instances:
            self.pending_requests.append(request)
            return False

        # Select instance based on routing policy
        if self.routing_policy == 'round_robin':
            instance = self._round_robin_select(ready_instances)
        elif self.routing_policy == 'least_loaded':
            instance = self._least_loaded_select(ready_instances)
        elif self.routing_policy == 'random':
            import random
            instance = random.choice(ready_instances)
        else:
            instance = ready_instances[0]

        # Get or create replica scheduler for this instance
        if instance.instance_id not in self.replica_schedulers:
            self.replica_schedulers[instance.instance_id] = ReplicaScheduler(
                instance, self.config, self.model_config
            )

        replica_scheduler = self.replica_schedulers[instance.instance_id]

        # Try to schedule to replica
        success = replica_scheduler.add_request(request, current_time)

        if not success:
            self.pending_requests.append(request)

        return success

    def _round_robin_select(self, instances: List[FunctionInstance]) -> FunctionInstance:
        """Select instance using round-robin.

        Args:
            instances: Available instances

        Returns:
            Selected instance
        """
        if not instances:
            return None

        instance = instances[self.next_instance_id % len(instances)]
        self.next_instance_id += 1
        return instance

    def _least_loaded_select(self, instances: List[FunctionInstance]) -> FunctionInstance:
        """Select least loaded instance.

        Args:
            instances: Available instances

        Returns:
            Selected instance
        """
        return min(instances, key=lambda inst: inst.get_utilization())

    def trigger_batch_execution(self, current_time: float, event_queue) -> None:
        """Trigger batch execution on all instances.

        Args:
            current_time: Current simulation time
            event_queue: Event queue for scheduling batch events
        """
        for instance_id, replica_scheduler in self.replica_schedulers.items():
            batch = replica_scheduler.get_next_batch(current_time)

            if batch:
                event_queue.push(Event(
                    time=current_time,
                    event_type=EventType.BATCH_START,
                    data={
                        'instance_id': instance_id,
                        'batch': batch
                    },
                    priority=50
                ))


class ReplicaScheduler:
    """Scheduler for a single replica instance.

    Handles batching and memory management for one instance.
    """

    def __init__(self, instance: FunctionInstance, config: Dict,
                 model_config: ModelConfig):
        """Initialize replica scheduler.

        Args:
            instance: Function instance
            config: Scheduler configuration
            model_config: Model configuration
        """
        self.instance = instance
        self.config = config
        self.model_config = model_config
        self.logger = setup_logger(f"ReplicaScheduler-{instance.instance_id}")

        # Create batching strategy
        scheduler_type = config.get('type', 'vllm')
        self.batcher = self._create_batcher(scheduler_type)

        # Request queues
        self.waiting_requests = deque()
        self.running_requests = []

    def _create_batcher(self, scheduler_type: str) -> BatchingStrategy:
        """Create batching strategy.

        Args:
            scheduler_type: Type of scheduler

        Returns:
            Batching strategy instance
        """
        if scheduler_type == 'vllm':
            return VLLMBatcher(self.config, self.model_config, self.instance)
        elif scheduler_type == 'orca':
            return OrcaBatcher(self.config, self.model_config, self.instance)
        elif scheduler_type == 'sarathi':
            return SarathiBatcher(self.config, self.model_config, self.instance)
        elif scheduler_type == 'fastertransformer':
            return FasterTransformerBatcher(self.config, self.model_config, self.instance)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def add_request(self, request: Request, current_time: float) -> bool:
        """Add request to scheduler.

        Args:
            request: Request to add
            current_time: Current simulation time

        Returns:
            True if request was added
        """
        # Check if request fits in memory
        if not self.instance.can_fit_request(request):
            return False

        # Add to instance and waiting queue
        success = self.instance.add_request(request, current_time)
        if success:
            self.waiting_requests.append(request)

        return success

    def get_next_batch(self, current_time: float) -> Optional[List[Request]]:
        """Get next batch to execute.

        Args:
            current_time: Current simulation time

        Returns:
            Batch of requests or None
        """
        # Use batching strategy to form batch
        batch = self.batcher.form_batch(
            waiting_requests=list(self.waiting_requests),
            running_requests=self.running_requests,
            current_time=current_time
        )

        if batch:
            # Remove scheduled requests from waiting queue
            for req in batch:
                if req in self.waiting_requests:
                    self.waiting_requests.remove(req)
                if req not in self.running_requests:
                    self.running_requests.append(req)

        return batch