"""Main simulator class orchestrating the discrete event simulation."""

import time
from typing import Dict, List, Optional
import numpy as np

from .event_queue import Event, EventType, EventQueue
from .metrics_collector import MetricsCollector
from ..models.model_config import ModelConfig
from ..serverless.function_instance import FunctionInstanceManager
from ..serverless.cold_start_simulator import ColdStartSimulator
from ..serverless.scaling_policy import ScalingPolicy
from ..serverless.cost_calculator import CostCalculator
from ..workload.request_generator import RequestGenerator
from ..scheduling.scheduler import GlobalScheduler
from ..utils.logger import setup_logger


class Simulator:
    """Main discrete event simulator for serverless LLM inference.

    This class orchestrates the entire simulation, managing:
    - Event processing
    - Request lifecycle
    - Instance management
    - Metrics collection
    - Cost tracking
    """

    def __init__(self, config: Dict):
        """Initialize simulator.

        Args:
            config: Simulation configuration dictionary
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)

        # Simulation state
        self.current_time = 0.0
        self.event_queue = EventQueue()
        self.simulation_duration = config['simulation']['duration']
        self.warmup_duration = config['simulation'].get('warmup_duration', 0)

        # Random seed for reproducibility
        np.random.seed(config['simulation'].get('random_seed', 42))

        # Initialize components
        self.model_config = ModelConfig(config['model'])
        self.metrics_collector = MetricsCollector(config)
        self.request_generator = RequestGenerator(config['workload'])

        # Serverless components
        self.cold_start_sim = ColdStartSimulator(config['cold_start'], self.model_config)
        self.instance_manager = FunctionInstanceManager(
            config, self.model_config, self.cold_start_sim
        )
        self.scaling_policy = ScalingPolicy(config['scaling'])
        self.cost_calculator = CostCalculator(config['cost'])

        # Scheduler
        self.scheduler = GlobalScheduler(
            config['scheduler'],
            self.instance_manager,
            self.model_config
        )

        # Statistics
        self.completed_requests = 0
        self.total_requests = 0

        self.logger.info("Simulator initialized")
        self.logger.info(f"Model: {self.model_config.name}")
        self.logger.info(f"Simulation duration: {self.simulation_duration}s")

    def run(self) -> Dict:
        """Run the simulation.

        Returns:
            Dictionary containing simulation results and metrics
        """
        start_time = time.time()
        self.logger.info("Starting simulation...")

        # Initialize simulation
        self._initialize()

        # Main simulation loop
        while not self.event_queue.is_empty():
            event = self.event_queue.pop()
            self.current_time = event.time

            # Stop if we've reached simulation end
            if event.event_type == EventType.SIMULATION_END:
                break

            # Process event
            self._process_event(event)

        # Collect final metrics
        results = self._finalize()

        elapsed_time = time.time() - start_time
        self.logger.info(f"Simulation completed in {elapsed_time:.2f}s")

        return results

    def _initialize(self) -> None:
        """Initialize the simulation."""
        self.logger.info("Initializing simulation...")

        # Schedule simulation end event
        self.event_queue.push(Event(
            time=self.simulation_duration,
            event_type=EventType.SIMULATION_END,
            priority=999
        ))

        # Schedule metrics collection
        metrics_interval = self.config['metrics'].get('collect_interval', 1.0)
        for t in np.arange(0, self.simulation_duration, metrics_interval):
            self.event_queue.push(Event(
                time=t,
                event_type=EventType.METRICS_COLLECTION,
                priority=100
            ))

        # Schedule scaling decisions if enabled
        if self.config['scaling']['enabled']:
            cooldown = self.config['scaling']['cooldown_period']
            for t in np.arange(0, self.simulation_duration, cooldown):
                self.event_queue.push(Event(
                    time=t,
                    event_type=EventType.SCALING_DECISION,
                    priority=90
                ))

        # Generate request arrival events
        self._generate_requests()

        # Start initial instances
        initial_instances = self.config['cold_start'].get('initial_instances', 1)
        for i in range(initial_instances):
            self.event_queue.push(Event(
                time=0.0,
                event_type=EventType.INSTANCE_START,
                data={'instance_id': i},
                priority=0
            ))

    def _generate_requests(self) -> None:
        """Generate request arrival events."""
        self.logger.info("Generating request arrivals...")

        requests = self.request_generator.generate(
            start_time=0.0,
            end_time=self.simulation_duration
        )

        self.total_requests = len(requests)
        self.logger.info(f"Generated {self.total_requests} requests")

        for request in requests:
            self.event_queue.push(Event(
                time=request.arrival_time,
                event_type=EventType.REQUEST_ARRIVAL,
                data={'request': request},
                priority=50
            ))

    def _process_event(self, event: Event) -> None:
        """Process a single event.

        Args:
            event: Event to process
        """
        handler = {
            EventType.REQUEST_ARRIVAL: self._handle_request_arrival,
            EventType.REQUEST_SCHEDULED: self._handle_request_scheduled,
            EventType.BATCH_START: self._handle_batch_start,
            EventType.BATCH_COMPLETE: self._handle_batch_complete,
            EventType.REQUEST_COMPLETE: self._handle_request_complete,
            EventType.INSTANCE_START: self._handle_instance_start,
            EventType.INSTANCE_READY: self._handle_instance_ready,
            EventType.INSTANCE_SHUTDOWN: self._handle_instance_shutdown,
            EventType.SCALING_DECISION: self._handle_scaling_decision,
            EventType.METRICS_COLLECTION: self._handle_metrics_collection,
        }.get(event.event_type)

        if handler:
            handler(event)

    def _handle_request_arrival(self, event: Event) -> None:
        """Handle request arrival."""
        request = event.data['request']
        request.enqueue_time = self.current_time

        # Try to schedule immediately
        scheduled = self.scheduler.schedule_request(request, self.current_time)

        if scheduled:
            # Calculate scheduling delay
            scheduling_delay = self.current_time - request.arrival_time
            request.scheduling_delay = scheduling_delay

            # Schedule batch execution if needed
            self.scheduler.trigger_batch_execution(self.current_time, self.event_queue)

    def _handle_request_scheduled(self, event: Event) -> None:
        """Handle request being scheduled to an instance."""
        pass  # Handled in scheduler

    def _handle_batch_start(self, event: Event) -> None:
        """Handle batch execution start."""
        instance_id = event.data['instance_id']
        batch = event.data['batch']

        instance = self.instance_manager.get_instance(instance_id)
        if instance:
            # Calculate batch execution time
            execution_time = instance.execute_batch(batch, self.current_time)

            # Schedule batch completion
            self.event_queue.push(Event(
                time=self.current_time + execution_time,
                event_type=EventType.BATCH_COMPLETE,
                data={
                    'instance_id': instance_id,
                    'batch': batch,
                    'execution_time': execution_time
                },
                priority=50
            ))

    def _handle_batch_complete(self, event: Event) -> None:
        """Handle batch execution completion."""
        instance_id = event.data['instance_id']
        batch = event.data['batch']

        # Process completed requests
        for request in batch:
            if request.is_complete():
                self.event_queue.push(Event(
                    time=self.current_time,
                    event_type=EventType.REQUEST_COMPLETE,
                    data={'request': request},
                    priority=40
                ))

        # Schedule next batch
        self.scheduler.trigger_batch_execution(self.current_time, self.event_queue)

    def _handle_request_complete(self, event: Event) -> None:
        """Handle request completion."""
        request = event.data['request']
        request.completion_time = self.current_time

        # Record metrics (only after warmup)
        if self.current_time >= self.warmup_duration:
            self.metrics_collector.record_request(request)
            self.completed_requests += 1

    def _handle_instance_start(self, event: Event) -> None:
        """Handle instance startup (cold start)."""
        instance_id = event.data['instance_id']

        # Simulate cold start delay
        cold_start_time = self.cold_start_sim.simulate_cold_start()

        instance = self.instance_manager.create_instance(
            instance_id, self.current_time
        )

        # Schedule instance ready event
        self.event_queue.push(Event(
            time=self.current_time + cold_start_time,
            event_type=EventType.INSTANCE_READY,
            data={'instance_id': instance_id, 'cold_start_time': cold_start_time},
            priority=10
        ))

    def _handle_instance_ready(self, event: Event) -> None:
        """Handle instance becoming ready."""
        instance_id = event.data['instance_id']
        cold_start_time = event.data['cold_start_time']

        instance = self.instance_manager.get_instance(instance_id)
        if instance:
            instance.mark_ready(self.current_time)
            self.logger.debug(
                f"Instance {instance_id} ready after {cold_start_time:.2f}s cold start"
            )

    def _handle_instance_shutdown(self, event: Event) -> None:
        """Handle instance shutdown."""
        instance_id = event.data['instance_id']
        self.instance_manager.remove_instance(instance_id)
        self.logger.debug(f"Instance {instance_id} shutdown")

    def _handle_scaling_decision(self, event: Event) -> None:
        """Handle scaling decision point."""
        if not self.config['scaling']['enabled']:
            return

        # Get current metrics
        metrics = self.instance_manager.get_cluster_metrics()

        # Make scaling decision
        decision = self.scaling_policy.decide(
            current_instances=self.instance_manager.num_instances(),
            metrics=metrics,
            current_time=self.current_time
        )

        if decision == 'scale_up':
            # Start new instance
            new_id = self.instance_manager.num_instances()
            self.event_queue.push(Event(
                time=self.current_time,
                event_type=EventType.INSTANCE_START,
                data={'instance_id': new_id},
                priority=0
            ))
            self.logger.info(f"Scaling up: starting instance {new_id}")

        elif decision == 'scale_down':
            # Remove least utilized instance
            instance_id = self.instance_manager.get_least_utilized_instance()
            if instance_id is not None:
                self.event_queue.push(Event(
                    time=self.current_time,
                    event_type=EventType.INSTANCE_SHUTDOWN,
                    data={'instance_id': instance_id},
                    priority=0
                ))
                self.logger.info(f"Scaling down: stopping instance {instance_id}")

    def _handle_metrics_collection(self, event: Event) -> None:
        """Handle periodic metrics collection."""
        # Collect cluster-level metrics
        cluster_metrics = self.instance_manager.get_cluster_metrics()

        if self.current_time >= self.warmup_duration:
            self.metrics_collector.record_cluster_metrics(
                self.current_time, cluster_metrics
            )

    def _finalize(self) -> Dict:
        """Finalize simulation and compute results.

        Returns:
            Dictionary containing all results and metrics
        """
        self.logger.info("Finalizing simulation...")

        # Compute metrics
        metrics = self.metrics_collector.compute_metrics()

        # Calculate costs
        cost_data = self.cost_calculator.calculate_total_cost(
            duration=self.simulation_duration - self.warmup_duration,
            num_instances=self.instance_manager.num_instances(),
            gpu_type=self.config['cluster']['gpu_type'],
            num_gpus_per_instance=self.config['cluster']['num_gpus_per_replica'],
            utilization=metrics.get('mean_mfu', 0.0)
        )

        # Combine all results
        results = {
            'total_requests': self.total_requests,
            'completed_requests': self.completed_requests,
            'completion_rate': self.completed_requests / self.total_requests if self.total_requests > 0 else 0,
            **metrics,
            **cost_data,
            'simulation_duration': self.simulation_duration - self.warmup_duration,
            'warmup_duration': self.warmup_duration,
        }

        return results