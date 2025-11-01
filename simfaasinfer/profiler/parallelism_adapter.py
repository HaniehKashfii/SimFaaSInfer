# simfaasinfer/profiler/parallelism_adapter.py
"""
Map model operators to per-device operator shapes for TP/PP configurations.

Public:
    device_operator_mapping(model_spec: dict, tp:int, pp:int) -> Dict[str,Any]
"""
from typing import Dict, Any, List


def device_operator_mapping(model_spec: Dict[str, Any], tp: int, pp: int) -> Dict[str, Any]:
    """
    Given model_spec and TP/PP dims, return a mapping describing what each device computes.

    Returns a mapping with keys like:
      {
        "device_0": [{"operator":"attention","shard_shape": {...}}, ...],
        ...
      }
    """
    num_layers = model_spec.get('num_layers', 32)
    hidden_size = model_spec.get('hidden_size', 4096)
    intermediate_size = model_spec.get('intermediate_size', 11008)
    
    # Calculate layers per pipeline stage
    layers_per_stage = num_layers // pp
    
    # Calculate tensor shard sizes
    hidden_shard = hidden_size // tp
    intermediate_shard = intermediate_size // tp
    
    mapping = {}
    
    # Generate mapping for each device
    total_devices = tp * pp
    for device_id in range(total_devices):
        # Determine PP stage and TP rank
        pp_stage = device_id // tp
        tp_rank = device_id % tp
        
        # Layer range for this PP stage
        start_layer = pp_stage * layers_per_stage
        end_layer = start_layer + layers_per_stage
        
        device_ops = []
        
        # Attention operator
        device_ops.append({
            'operator': 'attention',
            'layers': list(range(start_layer, end_layer)),
            'shard_shape': {
                'hidden_size': hidden_shard,
                'full_hidden': hidden_size,
                'tp_rank': tp_rank,
                'tp_size': tp
            }
        })
        
        # MLP operators
        device_ops.append({
            'operator': 'mlp_up',
            'layers': list(range(start_layer, end_layer)),
            'shard_shape': {
                'input_size': hidden_shard if tp > 1 else hidden_size,
                'output_size': intermediate_shard,
                'tp_rank': tp_rank
            }
        })
        
        device_ops.append({
            'operator': 'mlp_down',
            'layers': list(range(start_layer, end_layer)),
            'shard_shape': {
                'input_size': intermediate_shard,
                'output_size': hidden_shard if tp > 1 else hidden_size,
                'tp_rank': tp_rank
            }
        })
        
        # Communication operators for TP
        if tp > 1:
            device_ops.append({
                'operator': 'allreduce',
                'count_per_layer': 2,  # One after attention, one after MLP
                'message_size': hidden_size * 4  # float32
            })
        
        # Communication operators for PP
        if pp > 1 and pp_stage < pp - 1:
            device_ops.append({
                'operator': 'send',
                'target_stage': pp_stage + 1,
                'message_size': hidden_size * 4
            })
        
        if pp > 1 and pp_stage > 0:
            device_ops.append({
                'operator': 'recv',
                'source_stage': pp_stage - 1,
                'message_size': hidden_size * 4
            })
        
        mapping[f'device_{device_id}'] = device_ops
    
    return mapping