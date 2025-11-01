# simfaasinfer/optimizer/vidur_search.py
"""
Vidur-Search style optimizer: enumerate configs, binary-search capacity per config,
compute QPS per dollar, return Pareto frontier.
"""
from typing import Dict, Any, List, Callable
import multiprocessing as mp
import math
import itertools


def _capacity_binary_search(simulator_fn: Callable, config: Dict[str, Any], 
                            qps_lo: float, qps_hi: float, threshold: float, 
                            max_iterations: int = 20) -> Dict[str, Any]:
    """
    Internal helper: binary search for maximum sustainable QPS that keeps scheduling delay below threshold.
    simulator_fn(config, qps) -> metrics dict with 'scheduling_delay_p99' etc.
    """
    best_qps = qps_lo
    best_metrics = None
    
    for iteration in range(max_iterations):
        mid_qps = (qps_lo + qps_hi) / 2.0
        
        # Run simulation at this QPS
        metrics = simulator_fn(config, mid_qps)
        
        # Check if meets SLO
        scheduling_delay = metrics.get('scheduling_delay_p99', math.inf)
        p99_latency = metrics.get('p99_latency', math.inf)
        
        # Combined threshold check
        meets_slo = (scheduling_delay <= threshold)
        
        if meets_slo:
            # Can handle this load, try higher
            best_qps = mid_qps
            best_metrics = metrics
            qps_lo = mid_qps
        else:
            # Overloaded, try lower
            qps_hi = mid_qps
        
        # Convergence check
        if (qps_hi - qps_lo) < 0.1:
            break
    
    return {
        'max_qps': best_qps,
        'metrics': best_metrics or {},
        'iterations': iteration + 1
    }


def search_workload(workload_desc: Dict[str, Any], config_space: Dict[str, Any], 
                   constraints: Dict[str, Any], simulator_fn: Callable = None) -> Dict[str, Any]:
    """
    Orchestrate search over config_space. Returns SearchResult:
      {
        "candidates": [...],
        "pareto": {...},
        "raw": {...}
      }
    
    Args:
        workload_desc: Workload description (arrival rate, request sizes, etc)
        config_space: Configuration space to search
        constraints: Constraints (max cost, min QPS, SLO threshold, etc)
        simulator_fn: Function that runs simulation and returns metrics
    """
    print("Starting Vidur-Search capacity planning...")
    
    # Extract constraints
    slo_threshold_ms = constraints.get('slo_threshold_ms', 1000)
    max_cost_per_hour = constraints.get('max_cost_per_hour', float('inf'))
    min_qps = constraints.get('min_qps', 1.0)
    
    # Generate candidate configurations
    candidates = _generate_candidates(config_space)
    print(f"Generated {len(candidates)} candidate configurations")
    
    # If no simulator provided, use mock
    if simulator_fn is None:
        simulator_fn = _mock_simulator
    
    # Evaluate each candidate
    results = []
    for idx, config in enumerate(candidates):
        print(f"Evaluating config {idx + 1}/{len(candidates)}: {config}")
        
        # Calculate cost
        cost_per_hour = _calculate_cost(config)
        
        if cost_per_hour > max_cost_per_hour:
            print(f"  Skipping: cost ${cost_per_hour:.2f}/hr exceeds max ${max_cost_per_hour:.2f}/hr")
            continue
        
        # Binary search for max sustainable QPS
        qps_lo = min_qps
        qps_hi = workload_desc.get('max_qps', 1000)
        
        capacity_result = _capacity_binary_search(
            simulator_fn, config, qps_lo, qps_hi, slo_threshold_ms
        )
        
        max_qps = capacity_result['max_qps']
        metrics = capacity_result['metrics']
        
        # Calculate QPS per dollar
        qps_per_dollar = max_qps / cost_per_hour if cost_per_hour > 0 else 0
        
        result = {
            'config': config,
            'max_qps': max_qps,
            'cost_per_hour': cost_per_hour,
            'qps_per_dollar': qps_per_dollar,
            'metrics': metrics,
            'search_iterations': capacity_result['iterations']
        }
        
        results.append(result)
        
        print(f"  Max QPS: {max_qps:.2f}, Cost: ${cost_per_hour:.2f}/hr, QPS/$: {qps_per_dollar:.4f}")
    
    # Sort by QPS per dollar
    results.sort(key=lambda x: x['qps_per_dollar'], reverse=True)
    
    # Compute Pareto frontier
    pareto = _compute_pareto_frontier(results)
    
    search_result = {
        'candidates': results,
        'pareto': pareto,
        'raw': {
            'num_configs_evaluated': len(results),
            'config_space_size': len(candidates)
        },
        'best_config': results[0] if results else None
    }
    
    print(f"\nSearch complete. Best config: QPS/$ = {results[0]['qps_per_dollar']:.4f}")
    
    return search_result


def _generate_candidates(config_space: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate candidate configurations from config space."""
    candidates = []
    
    # Extract dimensions
    gpu_types = config_space.get('gpu_types', ['A100'])
    num_gpus_options = config_space.get('num_gpus', [1, 2, 4, 8])
    tp_sizes = config_space.get('tp_sizes', [1, 2, 4])
    pp_sizes = config_space.get('pp_sizes', [1])
    num_replicas = config_space.get('num_replicas', [1, 2, 4])
    
    # Generate all combinations
    for gpu_type in gpu_types:
        for num_gpus in num_gpus_options:
            for tp in tp_sizes:
                for pp in pp_sizes:
                    for replicas in num_replicas:
                        # Validate configuration
                        if tp * pp > num_gpus:
                            continue
                        
                        config = {
                            'gpu_type': gpu_type,
                            'num_gpus_per_replica': num_gpus,
                            'tp': tp,
                            'pp': pp,
                            'num_replicas': replicas,
                            'total_gpus': num_gpus * replicas
                        }
                        candidates.append(config)
    
    return candidates


def _calculate_cost(config: Dict[str, Any]) -> float:
    """Calculate hourly cost for configuration."""
    gpu_costs = {
        'A100': 3.06,
        'H100': 5.12,
        'A10G': 1.21,
    }
    
    gpu_type = config['gpu_type']
    total_gpus = config['total_gpus']
    
    cost_per_gpu = gpu_costs.get(gpu_type, 3.0)
    total_cost = cost_per_gpu * total_gpus
    
    return total_cost


def _compute_pareto_frontier(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Pareto frontier (QPS vs Cost)."""
    if not results:
        return {'points': []}
    
    # Sort by cost
    sorted_results = sorted(results, key=lambda x: x['cost_per_hour'])
    
    pareto_points = []
    max_qps_seen = 0
    
    for result in sorted_results:
        if result['max_qps'] > max_qps_seen:
            pareto_points.append({
                'config': result['config'],
                'max_qps': result['max_qps'],
                'cost_per_hour': result['cost_per_hour'],
                'qps_per_dollar': result['qps_per_dollar']
            })
            max_qps_seen = result['max_qps']
    
    return {
        'points': pareto_points,
        'count': len(pareto_points)
    }


def _mock_simulator(config: Dict[str, Any], qps: float) -> Dict[str, Any]:
    """Mock simulator for testing."""
    import random
    
    # Simulate decreasing performance as QPS increases
    max_capacity = config['total_gpus'] * 100  # ~100 QPS per GPU
    utilization = qps / max_capacity
    
    # Scheduling delay increases with utilization
    base_delay = 10  # ms
    scheduling_delay = base_delay * (1 + utilization ** 3)
    
    # Add noise
    scheduling_delay *= random.uniform(0.9, 1.1)
    
    return {
        'scheduling_delay_p99': scheduling_delay,
        'p99_latency': scheduling_delay + 50,
        'throughput': qps,
        'mfu': min(utilization, 0.95)
    }