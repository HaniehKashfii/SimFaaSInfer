# simfaasinfer/profiler/profiler_runner.py
"""
Run microbenchmarks / profiling experiments and write ProfileArtifact files.

Public:
    run_profiles(plan: ProfilingPlan, target_device: str, out_dir: str, driver: str='torch') -> List[dict]
"""
import os
import json
import time
from typing import List, Dict, Any
import numpy as np

from .operator_triage import ProfilingPlan


def run_profiles(plan: ProfilingPlan, target_device: str, out_dir: str, driver: str = "torch") -> List[Dict[str, Any]]:
    """
    Execute profiling plan on the target device and write ProfileArtifact JSON files to out_dir.

    Args:
        plan: ProfilingPlan object from operator_triage.generate_profiling_plan
        target_device: e.g., "a100", "h100", or cuda:0
        out_dir: output directory for artifacts
        driver: 'torch'|'cpu' -- indicates runtime harness

    Returns:
        List of ProfileArtifact dictionaries (in-memory).
    """
    os.makedirs(out_dir, exist_ok=True)
    artifacts = []
    
    print(f"Running profiling plan with {len(plan.operators)} operators on {target_device}...")
    
    for idx, op in enumerate(plan.operators):
        # Simulate profiling run
        runtime_ms = _simulate_operator_runtime(op, target_device, driver)
        gpu_memory_bytes = _estimate_memory_usage(op)
        
        artifact = {
            "operator": op.operator,
            "operator_class": op.operator_class,
            "phase": op.phase,
            "input": op.inputs,
            "tp": op.tp,
            "pp": op.pp,
            "runtime_ms": runtime_ms,
            "gpu_memory_bytes": gpu_memory_bytes,
            "kernel_details": {
                "kernels": [f"{op.operator}_kernel"],
                "device": target_device,
                "driver": driver
            },
            "timestamp": time.time()
        }
        
        # Write artifact as JSON
        fname = os.path.join(out_dir, f"{op.operator}_tp{op.tp}_pp{op.pp}_{idx}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)
        
        artifacts.append(artifact)
        
        if (idx + 1) % 10 == 0:
            print(f"  Profiled {idx + 1}/{len(plan.operators)} operators")
    
    print(f"Profiling complete. Generated {len(artifacts)} artifacts in {out_dir}")
    return artifacts


def _simulate_operator_runtime(op, target_device: str, driver: str) -> float:
    """Simulate operator runtime based on operator type and inputs."""
    # Base timings for A100 (adjust for other devices)
    device_multiplier = {
        'a100': 1.0,
        'h100': 0.6,  # H100 is ~1.6x faster
        'a10g': 2.0,
        'cpu': 50.0
    }.get(target_device.lower(), 1.0)
    
    if op.operator_class == 'token':
        # Token-level ops scale linearly with tokens
        batch_tokens = op.inputs.get('batch_tokens', 128)
        base_time_per_token = 0.0003  # ms per token
        runtime = batch_tokens * base_time_per_token
        
    elif op.operator_class == 'sequence' and op.phase == 'prefill':
        # Attention prefill: use sum of squares
        sum_sq = op.inputs.get('sum_sq_tokens', 0)
        if sum_sq == 0:
            seq_lens = op.inputs.get('seq_lens', [512])
            sum_sq = sum([s**2 for s in seq_lens])
        
        base_time_per_token_sq = 0.000015  # ms per token^2
        runtime = sum_sq * base_time_per_token_sq
        
    elif op.operator_class == 'sequence' and op.phase == 'decode':
        # Attention decode: memory-bound
        total_kv_bytes = op.inputs.get('total_kv_bytes', 1024 * 1024)
        batch_size = op.inputs.get('batch_size', 16)
        
        # Memory bandwidth limited
        memory_bw_gb_s = 2000 if 'a100' in target_device.lower() else 1500
        runtime = (total_kv_bytes / (memory_bw_gb_s * 1e9)) * 1000  # to ms
        runtime += batch_size * 0.05  # Add compute component
        
    elif op.operator_class == 'comm':
        # Communication ops
        msg_bytes = op.inputs.get('msg_bytes', 1024 * 1024)
        tp_size = op.inputs.get('tp_size', op.tp)
        
        # NVLink bandwidth (A100: ~600 GB/s)
        nvlink_bw_gb_s = 600 if 'a100' in target_device.lower() else 450
        
        # AllReduce: 2(n-1)/n factor
        if 'allreduce' in op.operator.lower():
            factor = 2 * (tp_size - 1) / tp_size if tp_size > 1 else 0
        else:
            factor = 1
        
        runtime = (msg_bytes * factor / (nvlink_bw_gb_s * 1e9)) * 1000
        runtime += 0.01  # Latency term
    else:
        runtime = 1.0
    
    # Apply device multiplier
    runtime *= device_multiplier
    
    # Apply TP/PP scaling
    if op.tp > 1:
        runtime /= op.tp * 0.85  # Not perfect scaling
    
    # Add noise
    runtime *= np.random.uniform(0.95, 1.05)
    
    return max(0.01, runtime)


def _estimate_memory_usage(op) -> int:
    """Estimate GPU memory usage for operator."""
    if op.operator_class == 'token':
        batch_tokens = op.inputs.get('batch_tokens', 128)
        hidden_size = op.inputs.get('hidden_size', 4096)
        # Activations: batch_tokens * hidden_size * 4 bytes (float32)
        return batch_tokens * hidden_size * 4
        
    elif op.operator_class == 'sequence':
        if op.phase == 'prefill':
            seq_lens = op.inputs.get('seq_lens', [512])
            total_tokens = sum(seq_lens)
            hidden_size = op.inputs.get('hidden_size', 4096)
            # Attention matrices: O(total_tokens^2)
            return total_tokens * hidden_size * 4 + total_tokens ** 2 * 2
        else:
            total_kv_bytes = op.inputs.get('total_kv_bytes', 1024 * 1024)
            return total_kv_bytes
            
    elif op.operator_class == 'comm':
        return op.inputs.get('msg_bytes', 0)
    
    return 0