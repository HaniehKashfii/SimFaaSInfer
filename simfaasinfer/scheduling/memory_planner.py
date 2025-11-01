# simfaasinfer/scheduling/memory_planner.py
"""
Compute KV cache budgets per replica and advise microbatch sizes.
"""
from typing import Dict, Any


def compute_kv_budget(model_config: Dict[str, Any], instance_spec: Dict[str, Any], tp: int, pp: int) -> int:
    """
    Return KV budget in bytes for a replica given model and instance memory.
    """
    # Get model memory footprint
    model_memory_gb = model_config.get('model_memory_gb', 14.0)
    
    # Get instance memory
    gpu_memory_gb = instance_spec.get('gpu_memory_gb', 80)
    num_gpus = instance_spec.get('num_gpus_per_replica', 1)
    
    # Total memory available
    total_memory_gb = gpu_memory_gb * num_gpus
    
    # Reserve memory for model, activations, and overhead
    reserved_gb = model_memory_gb / tp  # Model sharded by TP
    activation_memory_gb = 2.0  # Reserve for activations
    system_reserved_gb = 2.0
    
    # KV cache budget
    kv_budget_gb = total_memory_gb - reserved_gb - activation_memory_gb - system_reserved_gb
    kv_budget_gb = max(0, kv_budget_gb)
    
    # Convert to bytes
    kv_budget_bytes = int(kv_budget_gb * 1024 * 1024 * 1024)
    
    return kv_budget_bytes


def suggest_microbatch_sizes(kv_budget_bytes: int, avg_request_kv_bytes: int, max_batch_size: int = 256) -> Dict[str, Any]:
    """
    Suggest microbatch sizes given KV budget and average request size.
    """
    if avg_request_kv_bytes <= 0:
        return {'microbatch_size': max_batch_size}
    
    # Calculate how many requests fit in budget
    max_requests = kv_budget_bytes // avg_request_kv_bytes
    max_requests = min(max_requests, max_batch_size)
    
    # Suggest microbatch size
    microbatch_size = max(1, max_requests // 4)  # Use 1/4 of capacity for microbatch
    
    return {
        'microbatch_size': microbatch_size,
        'max_requests_in_memory': max_requests,
        'kv_budget_gb': kv_budget_bytes / (1024**3)
    }


class MemoryPlanner:
    """Memory planning helper class."""
    
    def __init__(self, kv_budget_bytes: int):
        self.kv_budget_bytes = kv_budget_bytes
        self.current_usage_bytes = 0
    
    def can_fit(self, request_kv_bytes: int) -> bool:
        """Check if request fits in current budget."""
        return (self.current_usage_bytes + request_kv_bytes) <= self.kv_budget_bytes
    
    def allocate(self, request_kv_bytes: int) -> bool:
        """Allocate memory for request."""
        if self.can_fit(request_kv_bytes):
            self.current_usage_bytes += request_kv_bytes
            return True
        return False
    
    def free(self, request_kv_bytes: int):
        """Free memory from completed request."""
        self.current_usage_bytes = max(0, self.current_usage_bytes - request_kv_bytes)
    
    def get_utilization(self) -> float:
        """Get memory utilization."""
        if self.kv_budget_bytes == 0:
            return 0.0
        return self.current_usage_bytes / self.kv_budget_bytes