"""Tests for the main simulator."""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import yaml

from simfaasinfer.core.simulator import Simulator
from simfaasinfer.core.event_queue import Event, EventType, EventQueue
from configs import load_config


class TestSimulator(unittest.TestCase):
    """Test cases for Simulator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'simulation': {
                'duration': 100,
                'warmup_duration': 10,
                'random_seed': 42,
            },
            'workload': {
                'type': 'poisson',
                'arrival_rate': 5,
                'num_requests': 50,
                'prompt_length': {
                    'distribution': 'constant',
                    'mean': 512,
                },
                'decode_length': {
                    'distribution': 'constant',
                    'mean': 128,
                },
            },
            'model': {
                'name': 'test-model',
                'num_layers': 32,
                'hidden_size': 4096,
                'intermediate_size': 11008,
                'num_attention_heads': 32,
                'num_kv_heads': 32,
                'vocab_size': 32000,
                'max_position_embeddings': 2048,
                'attention_type': 'mha',
                'architecture': 'llama',
                'head_dim': 128,
                'kv_channels': 4096,
                'profiling': {
                    'attention_prefill_time_per_token_sq': 0.000012,
                    'attention_decode_time_per_token': 0.00015,
                    'mlp_time_per_token': 0.00025,
                    'layernorm_time_per_token': 0.00002,
                    'embedding_time_per_token': 0.00003,
                    'allreduce_time_per_gb': 5.0,
                    'allgather_time_per_gb': 4.5,
                    'send_recv_time_per_gb': 3.0,
                    'model_memory_gb': 14.0,
                    'kv_cache_memory_per_token_mb': 0.0005,
                },
            },
            'cluster': {
                'num_replicas': 1,
                'gpu_type': 'A100',
                'gpu_memory_gb': 80,
                'num_gpus_per_replica': 1,
            },
            'parallelization': {
                'tensor_parallel_size': 1,
                'pipeline_parallel_size': 1,
            },
            'scheduler': {
                'type': 'vllm',
                'max_batch_size': 128,
                'max_tokens_per_batch': 2048,
                'scheduling_policy': 'fcfs',
            },
            'memory': {
                'kv_cache_dtype': 'float16',
                'reserved_memory_gb': 2.0,
            },
            'cold_start': {
                'enabled': True,
                'model_load_time': 5.0,
                'initial_instances': 1,
            },
            'scaling': {
                'enabled': False,
                'min_instances': 1,
                'max_instances': 4,
            },
            'cost': {
                'compute_cost': True,
                'idle_cost': True,
                'gpu_cost_per_hour': {
                    'A100': 3.06,
                },
            },
            'metrics': {
                'collect_interval': 10.0,
                'percentiles': [50, 90, 95, 99],
            },
        }

    def test_simulator_initialization(self):
        """Test simulator initialization."""
        simulator = Simulator(self.config)

        self.assertIsNotNone(simulator)
        self.assertEqual(simulator.simulation_duration, 100)
        self.assertEqual(simulator.warmup_duration, 10)
        self.assertEqual(simulator.total_requests, 0)

    def test_simulator_run(self):
        """Test running a complete simulation."""
        simulator = Simulator(self.config)
        results = simulator.run()

        # Check that results are returned
        self.assertIsInstance(results, dict)
        self.assertIn('total_requests', results)
        self.assertIn('completed_requests', results)
        self.assertIn('throughput', results)

        # Check that some requests were processed
        self.assertGreater(results['total_requests'], 0)
        self.assertGreater(results['completed_requests'], 0)

    def test_event_queue_ordering(self):
        """Test event queue maintains correct order."""
        queue = EventQueue()

        # Add events in random order
        queue.push(Event(time=3.0, event_type=EventType.REQUEST_ARRIVAL))
        queue.push(Event(time=1.0, event_type=EventType.REQUEST_ARRIVAL))
        queue.push(Event(time=2.0, event_type=EventType.REQUEST_ARRIVAL))

        # Pop events and verify order
        event1 = queue.pop()
        event2 = queue.pop()
        event3 = queue.pop()

        self.assertEqual(event1.time, 1.0)
        self.assertEqual(event2.time, 2.0)
        self.assertEqual(event3.time, 3.0)

    def test_metrics_collection(self):
        """Test metrics are collected properly."""
        simulator = Simulator(self.config)
        results = simulator.run()

        # Check key metrics exist
        self.assertIn('median_latency', results)
        self.assertIn('p95_latency', results)
        self.assertIn('mean_mfu', results)

        # Check metrics are reasonable
        if results['completed_requests'] > 0:
            self.assertGreater(results['median_latency'], 0)
            self.assertGreaterEqual(results['mean_mfu'], 0)
            self.assertLessEqual(results['mean_mfu'], 1.0)


class TestEventQueue(unittest.TestCase):
    """Test cases for EventQueue."""

    def test_empty_queue(self):
        """Test empty queue behavior."""
        queue = EventQueue()

        self.assertTrue(queue.is_empty())
        self.assertEqual(queue.size(), 0)
        self.assertIsNone(queue.peek())

    def test_push_pop(self):
        """Test push and pop operations."""
        queue = EventQueue()

        event = Event(time=1.0, event_type=EventType.REQUEST_ARRIVAL)
        queue.push(event)

        self.assertFalse(queue.is_empty())
        self.assertEqual(queue.size(), 1)

        popped = queue.pop()
        self.assertEqual(popped.time, 1.0)
        self.assertTrue(queue.is_empty())

    def test_priority_ordering(self):
        """Test events with same time are ordered by priority."""
        queue = EventQueue()

        queue.push(Event(time=1.0, priority=2, event_type=EventType.REQUEST_ARRIVAL))
        queue.push(Event(time=1.0, priority=1, event_type=EventType.BATCH_START))
        queue.push(Event(time=1.0, priority=3, event_type=EventType.REQUEST_COMPLETE))

        event1 = queue.pop()
        event2 = queue.pop()
        event3 = queue.pop()

        self.assertEqual(event1.priority, 1)
        self.assertEqual(event2.priority, 2)
        self.assertEqual(event3.priority, 3)


if __name__ == '__main__':
    unittest.main()