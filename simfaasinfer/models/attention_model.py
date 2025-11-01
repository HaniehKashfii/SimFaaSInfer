# simfaasinfer/models/attention_model.py
"""
Attention cost model with separate prefill and decode behavior.

Public:
    class AttentionModel:
        prefill_cost(...)
        decode_cost(...)
"""
from typing import List, Dict, Any
import numpy as np


class AttentionModel:
    def __init__(self):
        # params or learned coefficients can be stored here
        self.params = {
            'prefill_coeff': 0.000015,  # ms per token^2
            'decode_coeff': 0.0002,  # ms per token
            'memory_latency_ms': 0.01
        }

    def prefill_cost(self, seq_lens: List[int], model_config: Dict[str, Any], hw_params: Dict[str, Any]) -> float:
        """
        Predict runtime (ms) for a batch of prefills using Vidur heuristic:
        compose cost based on Σ p_i^2 and model params.
        """
        # Vidur transform: sum of squares
        sum_sq = sum([l * l for l in seq_lens])
        
        # Get model parameters
        num_layers = model_config.get('num_layers', 32)
        hidden_size = model_config.get('hidden_size', 4096)
        num_heads = model_config.get('num_attention_heads', 32)
        
        # Get hardware parameters
        tflops = hw_params.get('tflops', 312)
        mem_bw_gb_s = hw_params.get('mem_bw_gb_s', 2000)
        
        # Attention computation: O(sum_sq * d_model)
        # QK^T: sum_sq * hidden_size FLOPs
        # Softmax: sum_sq FLOPs
        # Output: sum_sq * hidden_size FLOPs
        total_flops = 2 * sum_sq * hidden_size * num_layers
        
        # Compute time
        compute_time_ms = (total_flops / (tflops * 1e12)) * 1000
        
        # Memory time (for loading K, V)
        total_tokens = sum(seq_lens)
        kv_memory_gb = (total_tokens * hidden_size * 2 * 4 * num_layers) / 1e9  # K and V, float32
        memory_time_ms = (kv_memory_gb / mem_bw_gb_s) * 1000
        
        # Total is max of compute and memory bound
        total_time = max(compute_time_ms, memory_time_ms) + self.params['memory_latency_ms']
        
        return total_time

    def decode_cost(self, kv_fetch_bytes: int, model_config: Dict[str, Any], hw_params: Dict[str, Any]) -> float:
        """
        Predict decode runtime (ms) based on total KV fetch bytes and memory bandwidth.
        Decode is typically memory-bound.
        """
        # Get hardware parameters
        mem_bw_gb_s = hw_params.get('mem_bw_gb_s', 2000)
        
        # Memory transfer time
        kv_fetch_gb = kv_fetch_bytes / 1e9
        memory_time_ms = (kv_fetch_gb / mem_bw_gb_s) * 1000
        
        # Add latency term
        total_time = memory_time_ms + self.params['memory_latency_ms']
        
        return total_time
    
    def update_params(self, new_params: Dict[str, float]):
        """Update model parameters based on calibration."""
        self.params.update(new_params)