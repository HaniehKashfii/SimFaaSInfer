# simfaasinfer/scheduling/hierarchical_scheduler.py
"""
Three-tier hierarchical scheduler skeleton.

Classes:
    GlobalScheduler
    ReplicaScheduler
    ReplicaStageScheduler
"""
from typing import Any, Dict, List, Optional


class GlobalScheduler:
    def __init__(self, replicas: List[Any], policy: str = 'round_robin'):
        self.replicas = replicas
        self.policy = policy
        self.next_idx = 0

    def route(self, request: Dict[str, Any]) -> Any:
        """
        Decide target replica for a request. Could be round-robin, least-loaded, or stateful.
        """
        if not self.replicas:
            return None
        
        if self.policy == 'round_robin':
            replica = self.replicas[self.next_idx % len(self.replicas)]
            self.next_idx += 1
            return replica
        elif self.policy == 'least_loaded':
            return min(self.replicas, key=lambda r: r.get_load())
        else:
            return self.replicas[0]


class ReplicaScheduler:
    def __init__(self, replica_id: str, memory_planner=None):
        self.replica_id = replica_id
        self.memory_planner = memory_planner
        self.queue = []
        self.running_requests = []

    def form_batch(self, queue: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Given queued requests, form a batch respecting KV memory and batching policy.
        Returns Batch metadata.
        """
        batch = []
        total_kv = 0
        
        # Simple greedy batching
        for req in queue:
            req_kv = req.get('kv_bytes', 0)
            if self.memory_planner and not self.memory_planner.can_fit(req_kv):
                continue
            
            batch.append(req)
            total_kv += req_kv
            
            if len(batch) >= 32:  # Max batch size
                break
        
        return {
            'requests': batch,
            'total_kv_bytes': total_kv,
            'batch_size': len(batch)
        }

    def execute_batch(self, batch: Dict[str, Any], runtime_estimator=None):
        """
        Simulate batch execution: query runtime_estimator for operator times and emit metrics.
        """
        if not batch or not batch.get('requests'):
            return 0.0
        
        # If we have runtime estimator, use it
        if runtime_estimator:
            # Construct operator queries for this batch
            total_time = 0.0
            
            # Prefill phase
            prefill_reqs = [r for r in batch['requests'] if r.get('phase') == 'prefill']
            if prefill_reqs:
                query = {
                    'operator_class': 'sequence',
                    'phase': 'prefill',
                    'inputs': {
                        'seq_lens': [r.get('prefill_len', 512) for r in prefill_reqs]
                    },
                    'tp': 1,
                    'pp': 1,
                    'model_spec': {},
                    'hardware': {}
                }
                total_time += runtime_estimator.predict(query)
            
            # Decode phase
            decode_reqs = [r for r in batch['requests'] if r.get('phase') == 'decode']
            if decode_reqs:
                query = {
                    'operator_class': 'sequence',
                    'phase': 'decode',
                    'inputs': {
                        'batch_size': len(decode_reqs),
                        'total_kv_bytes': sum([r.get('kv_bytes', 0) for r in decode_reqs])
                    },
                    'tp': 1,
                    'pp': 1,
                    'model_spec': {},
                    'hardware': {}
                }
                total_time += runtime_estimator.predict(query)
            
            return total_time
        
        # Fallback
        return 10.0  # ms


class ReplicaStageScheduler:
    def __init__(self, stage_id: int):
        self.stage_id = stage_id

    def schedule_microbatch(self, microbatch: Dict[str, Any]):
        """
        Schedule microbatch (for pipeline parallelism).
        """
        # For pipeline parallelism, schedule microbatches across stages
        pass