# simfaasinfer/runtime_estimator/estimator_utils.py
"""
Feature engineering utilities for the RF estimator.
"""
from typing import Dict, Any, List
import numpy as np


def features_for_token_op(model_spec: Dict[str, Any], input_desc: Dict[str, Any], hw: Dict[str, Any]) -> Dict[str, Any]:
    """Return feature dict for token-level operator."""
    features = {}
    
    # Input features
    features['batch_tokens'] = input_desc.get('batch_tokens', 128)
    features['hidden_size'] = input_desc.get('hidden_size', model_spec.get('hidden_size', 4096))
    
    # Model features
    features['d_model'] = model_spec.get('hidden_size', 4096)
    features['intermediate_size'] = model_spec.get('intermediate_size', 11008)
    
    # Hardware features
    features['tflops'] = hw.get('tflops', 312)
    features['mem_bw_gb_s'] = hw.get('mem_bw_gb_s', 2000)
    
    # Derived features
    features['total_flops'] = features['batch_tokens'] * features['d_model'] * features['intermediate_size']
    features['memory_traffic_gb'] = (features['batch_tokens'] * features['d_model'] * 4) / 1e9
    
    return features


def features_for_sequence_op(model_spec: Dict[str, Any], input_desc: Dict[str, Any], 
                             hw: Dict[str, Any], phase: str = 'prefill') -> Dict[str, Any]:
    """Return feature dict for sequence-level operator; include sum_sq_tokens for prefill."""
    features = {}
    
    features['hidden_size'] = input_desc.get('hidden_size', model_spec.get('hidden_size', 4096))
    features['num_heads'] = model_spec.get('num_attention_heads', 32)
    features['head_dim'] = features['hidden_size'] // features['num_heads']
    
    # Hardware
    features['tflops'] = hw.get('tflops', 312)
    features['mem_bw_gb_s'] = hw.get('mem_bw_gb_s', 2000)
    
    if phase == 'prefill':
        # Use sum of squares for prefill
        if 'sum_sq_tokens' in input_desc:
            features['sum_sq_tokens'] = input_desc['sum_sq_tokens']
        elif 'seq_lens' in input_desc:
            seq_lens = input_desc['seq_lens']
            if isinstance(seq_lens, list):
                features['sum_sq_tokens'] = sum([s**2 for s in seq_lens])
            else:
                features['sum_sq_tokens'] = seq_lens ** 2
        else:
            seq_len = input_desc.get('seq_len', 512)
            features['sum_sq_tokens'] = seq_len ** 2
        
        # Batch size
        if 'seq_lens' in input_desc and isinstance(input_desc['seq_lens'], list):
            features['batch_size'] = len(input_desc['seq_lens'])
        else:
            features['batch_size'] = input_desc.get('batch_size', 1)
        
        # Compute features
        features['attention_flops'] = features['sum_sq_tokens'] * features['hidden_size']
        
    else:  # decode
        # Memory-bound for decode
        features['total_kv_bytes'] = input_desc.get('total_kv_bytes', 1024 * 1024)
        features['batch_size'] = input_desc.get('batch_size', 16)
        
        # Effective bandwidth
        features['kv_fetch_gb'] = features['total_kv_bytes'] / 1e9
        features['decode_tokens'] = features['batch_size']
    
    return features


def features_for_comm_op(input_desc: Dict[str, Any], hw: Dict[str, Any]) -> Dict[str, Any]:
    """Return feature dict for communication operators."""
    features = {}
    
    # Message size
    features['msg_bytes'] = input_desc.get('msg_bytes', 1024 * 1024)
    features['msg_mb'] = features['msg_bytes'] / (1024 * 1024)
    features['msg_gb'] = features['msg_bytes'] / (1024 * 1024 * 1024)
    
    # TP size
    features['tp_size'] = input_desc.get('tp_size', 2)
    
    # Hardware
    features['nvlink_bw_gb_s'] = hw.get('nvlink_bw_gb_s', 600)
    features['pcie_bw_gb_s'] = hw.get('pcie_bw_gb_s', 32)
    
    # Effective bandwidth (depends on collective type)
    # AllReduce has 2(n-1)/n factor
    if features['tp_size'] > 1:
        features['allreduce_factor'] = 2 * (features['tp_size'] - 1) / features['tp_size']
    else:
        features['allreduce_factor'] = 0
    
    return features


def normalize_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize features to reasonable ranges."""
    normalized = features.copy()
    
    # Log-scale for large values
    for key in ['batch_tokens', 'sum_sq_tokens', 'total_kv_bytes', 'msg_bytes', 'total_flops']:
        if key in normalized and normalized[key] > 0:
            normalized[f'{key}_log'] = np.log10(normalized[key] + 1)
    
    # Normalize bandwidth features to [0, 1] range
    if 'mem_bw_gb_s' in normalized:
        normalized['mem_bw_normalized'] = normalized['mem_bw_gb_s'] / 3000.0  # Max ~3TB/s
    
    if 'nvlink_bw_gb_s' in normalized:
        normalized['nvlink_bw_normalized'] = normalized['nvlink_bw_gb_s'] / 900.0
    
    # Normalize TFLOPS
    if 'tflops' in normalized:
        normalized['tflops_normalized'] = normalized['tflops'] / 500.0
    
    return normalized