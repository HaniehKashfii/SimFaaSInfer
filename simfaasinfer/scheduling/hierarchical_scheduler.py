# simfaasinfer/scheduling/hierarchical_scheduler.py
"""
Three-tier hierarchical scheduler skeleton.

Classes:
    GlobalScheduler
    ReplicaScheduler
    ReplicaStageScheduler
"""
from typing import Any, Dict, List, Optional

from simfaasinfer.loader.loading_estimator import estimate_loading_time
from simfaasinfer.loader.migration_estimator import estimate_resume_time


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
        elif self.policy == 'least_loaded':
            replica = min(self.replicas, key=lambda r: r.get_load())
        elif self.policy == 'startup_time':
            replica = self._select_min_startup_time(request)
        else:
            replica = self.replicas[0]

        if replica and self.policy == 'startup_time':
            # Update queue delay to reflect newly assigned load.
            load_time = self._estimate_startup_components(replica, request)[0]
            self._bump_loading_queue(replica, load_time)

        return replica

    def _select_min_startup_time(self, request: Dict[str, Any]) -> Any:
        """Pick replica with the lowest startup time estimate."""
        best_replica = None
        best_startup = float('inf')
        for replica in self.replicas:
            _, _, startup_time = self._estimate_startup_components(replica, request)
            if startup_time < best_startup:
                best_startup = startup_time
                best_replica = replica
        return best_replica

    def _estimate_startup_components(self, replica: Any, request: Optional[Dict[str, Any]]):
        """Return (load_time, resume_path_time, min_startup_time)."""
        model_spec = self._extract_model_spec(replica, request)
        server_state = self._extract_server_state(replica)

        load_time = estimate_loading_time(model_spec, server_state)

        tin, tout = self._extract_migration_times(server_state)
        resume_time = estimate_resume_time(model_spec, tin, tout)
        network_transfer = float(server_state.get('migration_network_transfer_s', 0.0))
        pause_latency = float(server_state.get('pause_latency_s', 0.0))
        resume_path = resume_time + network_transfer + pause_latency

        startup_time = min(load_time, resume_path)
        return load_time, resume_path, startup_time

    def _extract_model_spec(self, replica: Any, request: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Favor request-level spec, otherwise fall back to replica metadata."""
        if request and request.get('model_spec'):
            return request['model_spec']

        if isinstance(replica, dict):
            return replica.get('model_spec', {})

        return getattr(replica, 'model_spec', {})

    def _extract_server_state(self, replica: Any) -> Dict[str, Any]:
        """Extract server state with queue delay fallback."""
        state = {}
        if isinstance(replica, dict):
            state = replica.get('server_state', replica.get('state', {}))
        else:
            state = getattr(replica, 'server_state', {})

        state = dict(state or {})
        if 'loading_queue_estimated_delay' not in state:
            pending = state.get('pending_load_time_s', 0.0)
            state['loading_queue_estimated_delay'] = float(pending)
        return state

    def _extract_migration_times(self, server_state: Dict[str, Any]):
        """Pull migration tin/tout from server state."""
        stats = server_state.get('migration_stats', {})
        tin = stats.get('tin', stats.get('tin_s', server_state.get('last_migration_tin_s', 0.0)))
        tout = stats.get('tout', stats.get('tout_s', server_state.get('last_migration_tout_s', 0.0)))
        return float(tin or 0.0), float(tout or 0.0)

    def _bump_loading_queue(self, replica: Any, additional_time: float) -> None:
        """Accumulate queue delay on the replica's server_state."""
        if additional_time <= 0:
            return

        if isinstance(replica, dict):
            state = replica.setdefault('server_state', {})
        else:
            state = getattr(replica, 'server_state', None)
            if state is None:
                state = {}
                setattr(replica, 'server_state', state)

        state['loading_queue_estimated_delay'] = state.get('loading_queue_estimated_delay', 0.0) + additional_time


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
