# Capacity Planning with SimFaaSInfer

## Overview

SimFaaSInfer provides comprehensive capacity planning capabilities for serverless LLM inference. This guide explains how to use FaaSInfer profiling outputs to generate capacity planning recommendations.

## Workflow

The capacity planning workflow consists of three stages:

```
FaaSInfer Profiling → SimFaaSInfer Capacity Planning → Deployment Recommendations
```

### Stage 1: FaaSInfer Profiling

FaaSInfer profiles your LLM workload and generates:
- **Fitted Profiles**: Runtime statistics for prefill/decode operations
- **Model Memory**: Memory footprint and KV-cache requirements
- **Metadata**: Hardware info, profiling date, trace locations

### Stage 2: SimFaaSInfer Capacity Planning

SimFaaSInfer reads the profiling outputs and performs:
- **Configuration Search**: Explores hardware configs (GPU types, parallelism)
- **Load Testing**: Binary search for maximum sustainable QPS
- **SLO Validation**: Ensures latency targets are met (TTFT, TBT)
- **Cost Optimization**: Finds Pareto-optimal QPS-per-dollar configurations

### Stage 3: Deployment Recommendations

SimFaaSInfer generates structured outputs:
- **Best Configuration**: Recommended hardware/software setup
- **Capacity Estimates**: Max QPS, expected latencies
- **Cost Projections**: Hourly costs, QPS per dollar
- **Pareto Frontier**: Trade-off curves for different configs

## Input Formats

### FaaSInfer Profiling Outputs

#### 1. Fitted Profile (`fitted_profile.json`)

```json
[
  {
    "type": "prefill",
    "seq_len": 512,
    "batch_size": 8,
    "mean_ms": 69.75,
    "median_ms": 69.76,
    "std_ms": 0.03
  },
  {
    "type": "decode",
    "kv_cache": 512,
    "new_tokens": 1,
    "batch_size": 8,
    "mean_ms": 19.34,
    "median_ms": 19.32,
    "std_ms": 0.04
  }
]
```

**Fields:**
- `type`: Operation type (`prefill` or `decode`)
- `seq_len`: Sequence length for prefill
- `kv_cache`: KV-cache tokens for decode
- `new_tokens`: New tokens to generate
- `batch_size`: Batch size
- `mean_ms`, `median_ms`, `std_ms`: Runtime statistics

#### 2. Metadata (`metadata.json`)

```json
{
  "model_name": "llama-7b",
  "fitted_profile": "fitted_profile.json",
  "hardware": {
    "gpu": "NVIDIA A100"
  },
  "profiling_date": "2025-11-04T21:06:41Z",
  "raw_trace_location": "/data/profiling/faasinfer/...",
  "notes": "..."
}
```

#### 3. Model Memory (`model_memory.json`) [Optional]

```json
{
  "model_params": 7000000000,
  "activation_memory_mb": 2048,
  "kv_cache_per_token_bytes": 256,
  "total_model_memory_gb": 14.5
}
```

### Capacity Planning Request

```yaml
# capacity_planning_request.yaml

workload:
  # Workload characteristics
  arrival_rate_range: [1, 100]  # QPS range to explore
  prompt_length:
    distribution: "normal"
    mean: 512
    std: 128
  output_length:
    distribution: "normal"
    mean: 256
    std: 64

constraints:
  # Service Level Objectives (SLOs)
  max_ttft_p95_ms: 2000  # Time to First Token P95 < 2s
  max_tbt_p99_ms: 200    # Time Between Tokens P99 < 200ms
  max_latency_p99_ms: 10000  # Total latency P99 < 10s

  # Cost constraints
  max_cost_per_hour: 100.0  # Maximum hourly cost
  min_qps: 10.0             # Minimum required throughput

config_space:
  # Hardware options to explore
  gpu_types: ["A100", "H100", "A10G"]
  num_gpus: [1, 2, 4, 8]

  # Parallelism strategies
  tensor_parallel: [1, 2, 4, 8]
  pipeline_parallel: [1, 2]
  num_replicas: [1, 2, 4, 8]

  # Scheduler options
  schedulers: ["vllm", "orca", "sarathi"]
  max_batch_sizes: [32, 64, 128, 256]

simulation:
  # Simulation parameters
  duration: 600  # seconds
  warmup_duration: 60
  random_seed: 42
```

## Output Formats

### Capacity Planning Report

```json
{
  "timestamp": "2025-11-15T10:30:00Z",
  "input_profile": "data/profiling/compute/a100/llama-7b/fitted_profile.json",

  "best_configuration": {
    "gpu_type": "A100",
    "num_gpus_per_replica": 4,
    "tensor_parallel": 4,
    "pipeline_parallel": 1,
    "num_replicas": 2,
    "total_gpus": 8,
    "scheduler": "vllm",
    "max_batch_size": 128,

    "performance": {
      "max_sustainable_qps": 87.5,
      "p50_latency_ms": 245.3,
      "p95_latency_ms": 512.7,
      "p99_latency_ms": 789.2,
      "p95_ttft_ms": 1456.8,
      "p99_tbt_ms": 145.3,
      "mean_mfu": 0.67,
      "mean_memory_util": 0.82
    },

    "cost": {
      "cost_per_hour": 24.48,
      "cost_per_1k_requests": 0.28,
      "qps_per_dollar": 3.58
    },

    "slo_compliance": {
      "meets_all_slos": true,
      "ttft_slo_met": true,
      "tbt_slo_met": true,
      "latency_slo_met": true
    }
  },

  "pareto_frontier": [
    {
      "config_id": 1,
      "max_qps": 45.2,
      "cost_per_hour": 6.12,
      "qps_per_dollar": 7.39,
      "gpu_type": "A10G",
      "total_gpus": 2
    },
    {
      "config_id": 2,
      "max_qps": 87.5,
      "cost_per_hour": 24.48,
      "qps_per_dollar": 3.58,
      "gpu_type": "A100",
      "total_gpus": 8
    }
  ],

  "all_candidates": [
    {
      "config": {...},
      "max_qps": 87.5,
      "cost_per_hour": 24.48,
      "qps_per_dollar": 3.58,
      "metrics": {...},
      "slo_compliance": {...}
    }
  ],

  "summary": {
    "num_configs_evaluated": 45,
    "num_configs_meeting_slos": 12,
    "total_simulation_time_s": 342.5,
    "recommendation": "Use A100 (x8) with vLLM scheduler for best cost-efficiency while meeting SLOs"
  }
}
```

### Human-Readable Summary

```
===============================================
   CAPACITY PLANNING REPORT
   Model: llama-7b on NVIDIA A100
   Date: 2025-11-15 10:30:00
===============================================

RECOMMENDED CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hardware:
  GPU Type:         NVIDIA A100 (80GB)
  Total GPUs:       8 (2 replicas × 4 GPUs each)
  Parallelism:      TP=4, PP=1

Software:
  Scheduler:        vLLM (continuous batching)
  Max Batch Size:   128

PERFORMANCE ESTIMATES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Throughput:         87.5 QPS (requests/second)
Latency (P95):      512.7 ms
TTFT (P95):         1456.8 ms ✓ (target: <2000ms)
TBT (P99):          145.3 ms ✓ (target: <200ms)
Resource Util:      67% MFU, 82% Memory

COST ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hourly Cost:        $24.48/hour
Cost per 1K Req:    $0.28
QPS per Dollar:     3.58

SLO COMPLIANCE:     ✓ ALL SLOS MET

ALTERNATIVE CONFIGURATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Budget Option:
  A10G (x2): 45 QPS @ $6.12/hr (7.39 QPS/$)

Premium Option:
  H100 (x8): 142 QPS @ $40.96/hr (3.47 QPS/$)

RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Deploy with A100 (x8) configuration using vLLM
scheduler. This provides optimal cost-efficiency
(3.58 QPS/$) while meeting all SLO requirements.

For lower traffic (<50 QPS), consider A10G (x2)
for better cost efficiency (7.39 QPS/$).
```

## Usage Examples

### Basic Capacity Planning

```python
from simfaasinfer.capacity_planning import run_capacity_planning

# Run capacity planning
results = run_capacity_planning(
    profile_path="data/profiling/compute/a100/llama-7b/fitted_profile.json",
    workload_config="configs/workload/production.yaml",
    output_path="results/capacity_plan.json"
)

# Print recommendation
print(f"Recommended: {results['best_configuration']['gpu_type']}")
print(f"Max QPS: {results['best_configuration']['performance']['max_sustainable_qps']}")
print(f"Cost: ${results['best_configuration']['cost']['cost_per_hour']}/hr")
```

### Advanced Workflow with Calibration

```python
from simfaasinfer.capacity_planning import CapacityPlanner

# Initialize planner
planner = CapacityPlanner(
    profile_path="data/profiling/compute/a100/llama-7b/fitted_profile.json"
)

# Optional: Calibrate with production telemetry
planner.calibrate(telemetry_path="data/telemetry/production_samples.json")

# Define search space
config_space = {
    'gpu_types': ['A100', 'H100'],
    'num_gpus': [2, 4, 8],
    'schedulers': ['vllm', 'sarathi']
}

# Define constraints
constraints = {
    'max_cost_per_hour': 50.0,
    'max_ttft_p95_ms': 2000,
    'max_tbt_p99_ms': 200
}

# Run search
results = planner.search(
    config_space=config_space,
    constraints=constraints,
    workload_desc={'arrival_rate_range': [10, 100]}
)

# Save outputs
planner.save_report(results, "results/capacity_plan.json")
planner.save_summary(results, "results/capacity_plan.txt")
```

### Command-Line Interface

```bash
# Run capacity planning from CLI
python -m simfaasinfer.capacity_planning \
  --profile data/profiling/compute/a100/llama-7b/fitted_profile.json \
  --workload configs/workload/production.yaml \
  --constraints configs/constraints/strict_slos.yaml \
  --output results/capacity_plan.json

# With calibration
python -m simfaasinfer.capacity_planning \
  --profile data/profiling/compute/a100/llama-7b/fitted_profile.json \
  --telemetry data/telemetry/prod_samples.json \
  --workload configs/workload/production.yaml \
  --output results/capacity_plan.json
```

## Best Practices

### 1. Profile Representative Workloads

Ensure FaaSInfer profiling captures:
- Realistic prompt/output length distributions
- Peak load scenarios
- Various batch sizes (1, 2, 4, 8, 16, 32)
- Both prefill and decode phases

### 2. Set Realistic SLOs

Common SLO targets:
- **Interactive Chat**: TTFT P95 < 1s, TBT P99 < 100ms
- **Document Generation**: TTFT P95 < 2s, TBT P99 < 200ms
- **Batch Processing**: TTFT P95 < 5s, TBT P99 < 500ms

### 3. Consider Cost-Performance Trade-offs

- **High Efficiency**: Fewer powerful GPUs (e.g., 2×A100)
- **High Throughput**: More GPUs with replication (e.g., 8×A100)
- **Budget**: Smaller GPUs (e.g., A10G)

### 4. Validate with Production Telemetry

If available, calibrate the simulator with real production data:
```python
planner.calibrate(telemetry_path="prod_telemetry.json")
```

This improves prediction accuracy by 15-30%.

### 5. Plan for Growth

Search over a range of QPS values:
```yaml
workload:
  arrival_rate_range: [current_qps, current_qps * 2]
```

## Interpreting Results

### QPS per Dollar

Higher is better. Typical values:
- **Budget GPUs (A10G)**: 5-10 QPS/$
- **Mid-tier (A100)**: 3-5 QPS/$
- **Premium (H100)**: 2-4 QPS/$

### Model FLOPs Utilization (MFU)

Indicates how efficiently the GPU is used:
- **<50%**: Consider larger batches or better scheduling
- **50-70%**: Good for variable workloads
- **>70%**: Excellent, but watch for SLO violations

### Memory Utilization

- **<60%**: Over-provisioned, consider smaller GPUs
- **60-85%**: Healthy range
- **>85%**: Risk of OOM, reduce batch size

## Troubleshooting

### No Configurations Meet SLOs

1. Relax SLO constraints
2. Add more powerful GPUs to search space
3. Increase number of replicas
4. Reduce target QPS

### High Costs

1. Try smaller GPU types (A10G instead of A100)
2. Reduce parallelism (lower TP/PP)
3. Optimize batch sizes
4. Consider on-demand vs. spot pricing

### Low QPS Estimates

1. Check if batch sizes are too conservative
2. Verify profiling data is from correct hardware
3. Consider more aggressive schedulers (vLLM, Sarathi)
4. Increase parallelism or replicas

## Related Documentation

- [Profiling Guide](profiling.md): How to generate FaaSInfer profiles
- [Metrics Reference](metrics.md): Understanding simulation metrics
- [Scheduler Comparison](schedulers.md): Choosing the right scheduler
- [Cost Optimization](cost_optimization.md): Advanced cost-saving strategies
