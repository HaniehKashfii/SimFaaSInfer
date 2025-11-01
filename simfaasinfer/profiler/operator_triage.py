# simfaasinfer/profiler/operator_triage.py
"""
Operator triage and profiling plan generator.

Public:
    generate_profiling_plan(model_spec: dict, parallel_config: dict) -> ProfilingPlan
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import numpy as np


@dataclass
class OperatorInput:
    operator: str
    operator_class: str  # 'token'|'sequence'|'comm'
    phase: str  # 'prefill'|'decode'|'both'
    inputs: Dict[str, Any]  # e.g. {"batch_tokens":64, "seq_len":512}
    tp: int = 1
    pp: int = 1


@dataclass
class ProfilingPlan:
    operators: List[OperatorInput]
    parallelism_scenarios: List[Dict[str, int]]

    def to_dict(self):
        return {
            "operators": [asdict(o) for o in self.operators],
            "parallelism_scenarios": self.parallelism_scenarios
        }


def generate_profiling_plan(model_spec: Dict[str, Any], parallel_config: Dict[str, Any]) -> ProfilingPlan:
    """
    Construct a compact profiling plan from a model spec and requested parallelism scenarios.

    Args:
        model_spec: dictionary describing model architecture (num_layers, d_model, etc).
        parallel_config: dict or list describing TP/PP scenarios to consider.

    Returns:
        ProfilingPlan with operator entries and parallelism scenarios.
    """
    operators = []
    
    # Extract model parameters
    num_layers = model_spec.get('num_layers', 32)
    hidden_size = model_spec.get('hidden_size', 4096)
    intermediate_size = model_spec.get('intermediate_size', 11008)
    
    # Get profiling config
    profiling_cfg = model_spec.get('profiling', {})
    token_ops = profiling_cfg.get('token_ops', ['mlp_up', 'mlp_down', 'layernorm'])
    seq_ops = profiling_cfg.get('seq_ops', ['attention'])
    
    # Token sample points
    token_samples = profiling_cfg.get('token_sample_tokens', [32, 64, 128, 256, 512, 1024])
    prefill_lengths = profiling_cfg.get('prefill_lengths', [128, 256, 512, 1024, 2048])
    decode_kv_sizes = profiling_cfg.get('decode_kv_sizes', [512, 1024, 2048, 4096])
    
    # Parallelism scenarios
    if isinstance(parallel_config, list):
        parallelism_scenarios = parallel_config
    else:
        tp_sizes = parallel_config.get('tp_sizes', [1, 2, 4])
        pp_sizes = parallel_config.get('pp_sizes', [1])
        parallelism_scenarios = [
            {'tp': tp, 'pp': pp} 
            for tp in tp_sizes 
            for pp in pp_sizes
        ]
    
    # Generate token-level operator inputs
    for op_name in token_ops:
        for batch_tokens in token_samples:
            for scenario in parallelism_scenarios:
                operators.append(OperatorInput(
                    operator=op_name,
                    operator_class='token',
                    phase='both',
                    inputs={'batch_tokens': batch_tokens, 'hidden_size': hidden_size},
                    tp=scenario['tp'],
                    pp=scenario['pp']
                ))
    
    # Generate sequence-level operator inputs (attention prefill)
    for seq_len in prefill_lengths:
        for scenario in parallelism_scenarios:
            # Use sum of squares for varied batch compositions
            batch_size = np.random.randint(1, 8)
            seq_lens = np.random.randint(seq_len // 2, seq_len, size=batch_size)
            sum_sq = int(np.sum(seq_lens ** 2))
            
            operators.append(OperatorInput(
                operator='attention_prefill',
                operator_class='sequence',
                phase='prefill',
                inputs={
                    'seq_lens': seq_lens.tolist(),
                    'sum_sq_tokens': sum_sq,
                    'hidden_size': hidden_size
                },
                tp=scenario['tp'],
                pp=scenario['pp']
            ))
    
    # Generate decode operator inputs
    for kv_bytes in decode_kv_sizes:
        for scenario in parallelism_scenarios:
            batch_size = np.random.randint(1, 32)
            operators.append(OperatorInput(
                operator='attention_decode',
                operator_class='sequence',
                phase='decode',
                inputs={
                    'batch_size': batch_size,
                    'total_kv_bytes': kv_bytes * 1024,  # KB to bytes
                    'hidden_size': hidden_size
                },
                tp=scenario['tp'],
                pp=scenario['pp']
            ))
    
    # Generate communication operator inputs if TP > 1
    comm_sizes_mb = [1, 4, 16, 64, 256]
    for scenario in parallelism_scenarios:
        if scenario['tp'] > 1:
            for size_mb in comm_sizes_mb:
                # AllReduce
                operators.append(OperatorInput(
                    operator='allreduce',
                    operator_class='comm',
                    phase='both',
                    inputs={'msg_bytes': size_mb * 1024 * 1024, 'tp_size': scenario['tp']},
                    tp=scenario['tp'],
                    pp=scenario['pp']
                ))
                
                # AllGather
                operators.append(OperatorInput(
                    operator='allgather',
                    operator_class='comm',
                    phase='both',
                    inputs={'msg_bytes': size_mb * 1024 * 1024, 'tp_size': scenario['tp']},
                    tp=scenario['tp'],
                    pp=scenario['pp']
                ))
    
    plan = ProfilingPlan(operators=operators, parallelism_scenarios=parallelism_scenarios)
    return plan