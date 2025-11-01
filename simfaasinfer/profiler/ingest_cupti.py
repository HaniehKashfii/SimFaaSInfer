# simfaasinfer/profiler/ingest_cupti.py
"""
Parsers to convert raw CUPTI or PyTorch profiler traces into ProfileArtifact format.
"""
import json
from typing import List, Dict, Any


def parse_cupti_trace(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a raw CUPTI trace file and produce operator-level ProfileArtifact dicts.

    Returns:
        List of ProfileArtifact dictionaries compatible with schema.
    """
    artifacts = []
    
    try:
        with open(file_path, 'r') as f:
            trace_data = json.load(f)
        
        # Parse CUPTI trace format (simplified)
        if 'traceEvents' in trace_data:
            events = trace_data['traceEvents']
            
            # Group by operator name
            operator_groups = {}
            for event in events:
                if event.get('cat') == 'kernel' or event.get('ph') == 'X':
                    name = event.get('name', 'unknown')
                    dur = event.get('dur', 0) / 1000.0  # microseconds to ms
                    
                    if name not in operator_groups:
                        operator_groups[name] = []
                    operator_groups[name].append(dur)
            
            # Create artifacts
            for op_name, durations in operator_groups.items():
                artifact = {
                    'operator': op_name,
                    'operator_class': _infer_operator_class(op_name),
                    'phase': 'both',
                    'input': {},
                    'tp': 1,
                    'pp': 1,
                    'runtime_ms': sum(durations) / len(durations),
                    'gpu_memory_bytes': None,
                    'kernel_details': {
                        'kernel_count': len(durations),
                        'total_time_ms': sum(durations)
                    },
                    'timestamp': 0
                }
                artifacts.append(artifact)
    
    except Exception as e:
        print(f"Error parsing CUPTI trace: {e}")
    
    return artifacts


def _infer_operator_class(op_name: str) -> str:
    """Infer operator class from name."""
    name_lower = op_name.lower()
    if 'attention' in name_lower or 'attn' in name_lower:
        return 'sequence'
    elif 'mlp' in name_lower or 'linear' in name_lower or 'matmul' in name_lower:
        return 'token'
    elif 'reduce' in name_lower or 'gather' in name_lower or 'comm' in name_lower:
        return 'comm'
    else:
        return 'token'