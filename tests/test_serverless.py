"""Tests for serverless components."""

import unittest
import numpy as np

from simfaasinfer.serverless.function_instance import (
    FunctionInstance, FunctionInstanceManager, Request, InstanceState
)
from simfaasinfer.serverless.cold_start_simulator import ColdStartSimulator
from simfaasinfer.serverless.scaling_policy import ScalingPolicy
from simfaasinfer.serverless.cost_calculator import CostCalculator
from simfaasinfer.models.model_config import ModelConfig


class TestFunctionInstance(unittest.TestCase):
    """Test cases for FunctionInstance."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'cluster': {
                'gpu_memory_gb': 80,
                'num_gpus_per_replica': 1,
            },
            'memory': {
                'reserved_memory_gb': 2.0,
            },
            'parallelization': {
                'tensor_parallel_size': 1,
                'pipeline_parallel_size': 1,
            },
            'scheduler': {
                'max_batch_size': 64,
                'max_tokens_per_batch': 2048,
            },
        }

        self.model_config = ModelConfig({
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
                'model_memory_gb': 14.0,
                'kv_cache_memory_per_token_mb': 0.0005,
            },
        })

    def test_instance_initialization(self):
        """Test instance initialization."""
        instance = FunctionInstance(0, self.config, self.model_config)

        self.assertEqual(instance.instance_id, 0)
        self.assertEqual(instance.state, InstanceState.STARTING)
        self.assertGreater(instance.kv_cache_capacity_tokens, 0)

    def test_instance_ready(self):
        """Test marking instance as ready."""
        instance = FunctionInstance(0, self.config, self.model_config)
        instance.mark_ready(10.0)

        self.assertEqual(instance.state, InstanceState.READY)
        self.assertEqual(instance.ready_time, 10.0)

    def test_can_fit_request(self):
        """Test request fitting logic."""
        instance = FunctionInstance(0, self.config, self.model_config)
        instance.mark_ready(0.0)

        # Small request should fit
        small_request = Request(
            request_id=0,
            arrival_time=0.0,
            prompt_tokens=512,
            decode_tokens=128
        )
        self.assertTrue(instance.can_fit_request(small_request))

        # Very large request should not fit
        large_request = Request(
            request_id=1,
            arrival_time=0.0,
            prompt_tokens=100000,
            decode_tokens=100000
        )
        self.assertFalse(instance.can_fit_request(large_request))

    def test_add_request(self):
        """Test adding request to instance."""
        instance = FunctionInstance(0, self.config, self.model_config)
        instance.mark_ready(0.0)

        request = Request(
            request_id=0,
            arrival_time=0.0,
            prompt_tokens=512,
            decode_tokens=128
        )

        success = instance.add_request(request, 0.0)

        self.assertTrue(success)
        self.assertEqual(len(instance.current_batch), 1)
        self.assertEqual(request.assigned_instance, 0)
        self.assertGreater(instance.kv_cache_used_tokens, 0)

    def test_execute_batch(self):
        """Test batch execution."""
        instance = FunctionInstance(0, self.config, self.model_config)
        instance.mark_ready(0.0)

        request = Request(
            request_id=0,
            arrival_time=0.0,
            prompt_tokens=512,
            decode_tokens=128
        )
        instance.add_request(request, 0.0)

        batch = [request]
        execution_time = instance.execute_batch(batch, 0.0)

        self.assertGreater(execution_time, 0)
        self.assertTrue(instance.is_processing)


class TestColdStartSimulator(unittest.TestCase):
    """Test cases for ColdStartSimulator."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'enabled': True,
            'model_load_time': 30.0,
            'container_init_time': 5.0,
            'memory_alloc_time': 2.0,
        }

        self.model_config = ModelConfig({
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
                'model_memory_gb': 14.0,
            },
        })

    def test_cold_start_enabled(self):
        """Test cold start when enabled."""
        sim = ColdStartSimulator(self.config, self.model_config)
        cold_start_time = sim.simulate_cold_start()

        self.assertGreater(cold_start_time, 0)
        self.assertGreater(cold_start_time, 30)  # At least base time

    def test_cold_start_disabled(self):
        """Test cold start when disabled."""
        config = self.config.copy()
        config['enabled'] = False

        sim = ColdStartSimulator(config, self.model_config)
        cold_start_time = sim.simulate_cold_start()

        self.assertEqual(cold_start_time, 0.0)


class TestScalingPolicy(unittest.TestCase):
    """Test cases for ScalingPolicy."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'enabled': True,
            'min_instances': 1,
            'max_instances': 10,
            'scale_up_threshold': 0.8,
            'scale_down_threshold': 0.2,
            'cooldown_period': 60,
        }

    def test_scale_up_decision(self):
        """Test scale up decision."""
        policy = ScalingPolicy(self.config)

        metrics = {'mfu': 0.85}
        decision = policy.decide(
            current_instances=2,
            metrics=metrics,
            current_time=0.0
        )

        # First call won't scale due to consecutive threshold
        self.assertIsNone(decision)

        # Second call should scale
        policy.decide(2, metrics, 1.0)
        decision = policy.decide(2, metrics, 2.0)
        self.assertIsNone(decision)  # Still in cooldown

        # After cooldown
        decision = policy.decide(2, metrics, 100.0)
        self.assertEqual(decision, 'scale_up')

    def test_scale_down_decision(self):
        """Test scale down decision."""
        policy = ScalingPolicy(self.config)

        metrics = {'mfu': 0.15}

        # Need 3 consecutive low utilization readings
        policy.decide(5, metrics, 0.0)
        policy.decide(5, metrics, 1.0)
        policy.decide(5, metrics, 2.0)
        decision = policy.decide(5, metrics, 100.0)

        self.assertEqual(decision, 'scale_down')

    def test_min_max_instances(self):
        """Test min/max instance constraints."""
        policy = ScalingPolicy(self.config)

        # Can't scale below min
        metrics = {'mfu': 0.1}
        for _ in range(5):
            policy.decide(1, metrics, 0.0)

        decision = policy.decide(1, metrics, 200.0)
        self.assertIsNone(decision)  # At minimum

        # Can't scale above max
        metrics = {'mfu': 0.9}
        for _ in range(5):
            policy.decide(10, metrics, 300.0)

        decision = policy.decide(10, metrics, 500.0)
        self.assertIsNone(decision)  # At maximum


class TestCostCalculator(unittest.TestCase):
    """Test cases for CostCalculator."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'compute_cost': True,
            'idle_cost': True,
            'idle_cost_factor': 0.5,
            'gpu_cost_per_hour': {
                'A100': 3.06,
                'H100': 5.12,
            },
        }

    def test_calculate_total_cost(self):
        """Test total cost calculation."""
        calculator = CostCalculator(self.config)

        cost_data = calculator.calculate_total_cost(
            duration=3600,  # 1 hour
            num_instances=2,
            gpu_type='A100',
            num_gpus_per_instance=4,
            utilization=0.7
        )

        self.assertIn('total_cost', cost_data)
        self.assertIn('compute_cost', cost_data)
        self.assertIn('idle_cost', cost_data)
        self.assertGreater(cost_data['total_cost'], 0)

        # 2 instances * 4 GPUs * 1 hour * $3.06/GPU/hour
        expected_gpu_hours = 8
        self.assertEqual(cost_data['gpu_hours'], expected_gpu_hours)

    def test_qps_per_dollar(self):
        """Test QPS per dollar calculation."""
        calculator = CostCalculator(self.config)

        qps_per_dollar = calculator.calculate_qps_per_dollar(
            throughput=10.0,
            total_cost=50.0
        )

        self.assertEqual(qps_per_dollar, 0.2)

    def test_cost_per_request(self):
        """Test cost per request calculation."""
        calculator = CostCalculator(self.config)

        cost_per_request = calculator.calculate_cost_per_request(
            total_cost=100.0,
            num_requests=1000
        )

        self.assertEqual(cost_per_request, 0.1)


if __name__ == '__main__':
    unittest.main()