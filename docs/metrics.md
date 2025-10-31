# Metrics Guide

## Request-Level Metrics

### Latency
- **Definition**: End-to-end time from arrival to completion
- **Formula**: `completion_time - arrival_time`
- **Unit**: Milliseconds
- **Importance**: Primary user-facing metric

### Time to First Token (TTFT)
- **Definition**: Time from arrival to first output token
- **Formula**: `first_token_time - arrival_time`
- **Unit**: Milliseconds
- **Importance**: User-perceived responsiveness

### Time Between Tokens (TBT)
- **Definition**: Average time between consecutive output tokens
- **Formula**: `(completion_time - first_token_time) / (decode_tokens - 1)`
- **Unit**: Milliseconds
- **Importance**: Streaming quality

### Normalized Latency
- **Definition**: Latency per output token
- **Formula**: `latency / decode_tokens`
- **Unit**: Milliseconds per token
- **Importance**: Fair comparison across request sizes

### Scheduling Delay
- **Definition**: Time waiting in queue
- **Formula**: `schedule_time - arrival_time`
- **Unit**: Milliseconds
- **Importance**: System load indicator

## Cluster-Level Metrics

### Model FLOPs Utilization (MFU)
- **Definition**: Fraction of peak FLOPs achieved
- **Formula**: `actual_compute / peak_compute`
- **Range**: [0, 1]
- **Importance**: Hardware efficiency

### Memory Utilization
- **Definition**: Fraction of GPU memory used
- **Formula**: `used_memory / total_memory`
- **Range**: [0, 1]
- **Importance**: Resource efficiency

### KV-Cache Utilization
- **Definition**: Fraction of KV-cache capacity used
- **Formula**: `kv_cache_used / kv_cache_capacity`
- **Range**: [0, 1]
- **Importance**: Memory management effectiveness

### Throughput
- **Definition**: Requests processed per second
- **Formula**: `completed_requests / duration`
- **Unit**: QPS (Queries Per Second)
- **Importance**: System capacity

## Cost Metrics

### Total Cost
- **Definition**: Total deployment cost
- **Formula**: `compute_cost + idle_cost`
- **Unit**: USD
- **Importance**: Economic viability

### QPS per Dollar
- **Definition**: Throughput normalized by cost
- **Formula**: `throughput / total_cost`
- **Unit**: QPS/$
- **Importance**: Cost efficiency

### Cost per Request
- **Definition**: Average cost per request
- **Formula**: `total_cost / num_requests`
- **Unit**: USD
- **Importance**: Unit economics

## Percentiles

All latency metrics report the following percentiles:
- **P50 (Median)**: Middle value, typical experience
- **P90**: 90% of requests are faster
- **P95**: 95% of requests are faster
- **P99**: 99% of requests are faster (tail latency)

## Interpreting Results

### Good Performance Indicators
- ✓ TTFT P90 < 2 seconds
- ✓ TBT P99 < 200 milliseconds
- ✓ MFU > 50%
- ✓ Completion rate > 99%

### Red Flags
- ✗ High P99/P50 ratio (>5x): Variable performance
- ✗ MFU < 30%: Underutilized hardware
- ✗ Memory util > 95%: Risk of OOM
- ✗ Scheduling delay > latency: Overloaded system

## Example Metric Analysis
```python
results = simulator.run()

# Check SLOs
ttft_p90 = results['p90_ttft']
tbt_p99 = results['p99_tbt']

if ttft_p90 < 2000 and tbt_p99 < 200:
    print("✓ SLOs met")
else:
    print("✗ SLOs violated")

# Check efficiency
mfu = results['mean_mfu']
qps_per_dollar = results['qps_per_dollar']

print(f"Efficiency: {mfu:.1%} MFU, {qps_per_dollar:.4f} QPS/$")
```

## Metric Collection Configuration
```yaml
metrics:
  collect_interval: 1.0  # seconds
  percentiles: [50, 90, 95, 99]
```

## Visualization

Metrics can be visualized using the built-in plotting functions:
```python
from simfaasinfer.utils.visualization import plot_results

plot_results(results, output_dir="./plots")
```

This generates:
- Latency distribution plots
- Utilization over time
- Cost breakdown