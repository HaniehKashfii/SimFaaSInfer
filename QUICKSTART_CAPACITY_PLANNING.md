# Quick Start: Capacity Planning with SimFaaSInfer

This guide will help you get started with capacity planning in 5 minutes.

## Prerequisites

- FaaSInfer profiling outputs (fitted_profile.json)
- SimFaaSInfer installed
- Python 3.8+

## Step 1: Verify Profile Data

Check that you have a FaaSInfer profile:

```bash
ls -la data/profiling/compute/a100/llama-7b/
```

You should see:
- `fitted_profile.json` - Runtime statistics from profiling
- `metadata.json` - Hardware and model information (optional)

## Step 2: Run Basic Capacity Planning

```bash
python examples/run_capacity_planning.py \
  --profile data/profiling/compute/a100/llama-7b/fitted_profile.json \
  --output results/capacity_plan.json
```

This will:
- Load the FaaSInfer profile
- Search for optimal configurations
- Generate a capacity planning report

## Step 3: View Results

The script generates two files:

1. **JSON Report** (`results/capacity_plan.json`):
   - Complete results with all configurations evaluated
   - Pareto frontier points
   - Detailed metrics

2. **Text Summary** (`results/capacity_plan.txt`):
   - Human-readable recommendation
   - Performance estimates
   - Cost analysis

View the summary:
```bash
cat results/capacity_plan.txt
```

## Step 4: Customize (Optional)

### Adjust Workload Parameters

```bash
python examples/run_capacity_planning.py \
  --profile data/profiling/compute/a100/llama-7b/fitted_profile.json \
  --prompt-len 1024 \
  --output-len 512 \
  --arrival-rate 100 \
  --output results/capacity_plan.json
```

### Set Strict SLOs

```bash
python examples/run_capacity_planning.py \
  --profile data/profiling/compute/a100/llama-7b/fitted_profile.json \
  --ttft-slo 1000 \
  --tbt-slo 100 \
  --max-cost 50 \
  --output results/capacity_plan.json
```

### Compare GPU Types

```bash
python examples/run_capacity_planning.py \
  --profile data/profiling/compute/a100/llama-7b/fitted_profile.json \
  --gpu-types A100 H100 A10G \
  --output results/capacity_plan.json
```

## Step 5: Use Config Files (Advanced)

For repeatable workflows, use YAML configs:

```bash
python -m simfaasinfer.capacity_planning \
  --profile data/profiling/compute/a100/llama-7b/fitted_profile.json \
  --workload configs/workload/production.yaml \
  --constraints configs/constraints/strict_slos.yaml \
  --output results/capacity_plan.json
```

Available workload presets:
- `configs/workload/production.yaml` - General production workload
- `configs/workload/interactive_chat.yaml` - Low-latency chat
- `configs/workload/document_generation.yaml` - Long-form generation

Available constraint presets:
- `configs/constraints/strict_slos.yaml` - Interactive applications
- `configs/constraints/relaxed_slos.yaml` - Batch processing
- `configs/constraints/budget_optimized.yaml` - Cost-sensitive deployments

## Understanding the Output

### Best Configuration

The report shows the recommended deployment configuration:

```
RECOMMENDED CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hardware:
  GPU Type:         NVIDIA A100 (80GB)
  Total GPUs:       8 (2 replicas × 4 GPUs each)
  Parallelism:      TP=4, PP=1

Software:
  Scheduler:        vLLM (continuous batching)
  Max Batch Size:   128
```

This tells you:
- **GPU Type**: Which GPU to use
- **Total GPUs**: How many GPUs needed
- **Replicas**: Number of independent replicas for load balancing
- **Parallelism**: Tensor parallel (TP) and pipeline parallel (PP) configuration
- **Scheduler**: Which scheduling algorithm to use
- **Batch Size**: Maximum batch size setting

### Performance Estimates

```
PERFORMANCE ESTIMATES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Throughput:         87.5 QPS
Latency (P95):      512.7 ms
TTFT (P95):         1456.8 ms ✓
TBT (P99):          145.3 ms ✓
Resource Util:      67% MFU, 82% Memory
```

- **Throughput**: Maximum requests per second the system can handle
- **Latency**: Response time percentiles
- **TTFT**: Time to First Token (how quickly streaming starts)
- **TBT**: Time Between Tokens (streaming speed)
- **Resource Util**: GPU compute (MFU) and memory utilization

### Cost Analysis

```
COST ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hourly Cost:        $24.48/hour
Cost per 1K Req:    $0.28
QPS per Dollar:     3.58
```

- **Hourly Cost**: Total infrastructure cost per hour
- **Cost per 1K Req**: Cost per thousand requests
- **QPS per Dollar**: Throughput efficiency (higher is better)

## Next Steps

- **Fine-tune**: Adjust workload parameters to match your actual traffic
- **Calibrate**: Use production telemetry for better accuracy (see docs)
- **Compare**: Try different GPU types to find cost-optimal solution
- **Deploy**: Use the recommended configuration in your deployment

## Common Issues

### "Profile not found"

Make sure the profile path is correct. List available profiles:
```bash
find data/profiling -name "fitted_profile.json"
```

### "No configuration meets SLOs"

Your constraints may be too strict. Try:
- Relaxing SLO thresholds (`--ttft-slo`, `--tbt-slo`)
- Increasing max cost (`--max-cost`)
- Adding more powerful GPUs (`--gpu-types H100`)

### Results seem inaccurate

For better accuracy:
- Use production telemetry calibration (`--telemetry`)
- Ensure profiling was done on the same GPU type
- Verify workload parameters match your actual traffic

## Getting Help

- Full documentation: [docs/capacity_planning.md](docs/capacity_planning.md)
- Report issues: GitHub Issues
- Examples: [examples/](examples/)

## Summary

**TL;DR**: Run this command and read the output:

```bash
python examples/run_capacity_planning.py \
  --profile data/profiling/compute/a100/llama-7b/fitted_profile.json \
  --output results/capacity_plan.json && \
  cat results/capacity_plan.txt
```

The capacity planner will tell you exactly what hardware to use, how to configure it, and what performance/cost to expect.
