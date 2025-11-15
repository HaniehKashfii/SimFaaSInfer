# SimFaaSInfer: Serverless LLM Inference Simulator

A high-fidelity, event-driven simulation framework for serverless Large Language Model (LLM) inference, inspired by [Vidur](https://github.com/microsoft/vidur) from Microsoft Research.

## 🎯 Features

- **High-Fidelity Simulation**: Accurate modeling of LLM inference at millisecond granularity
- **Multiple Schedulers**: vLLM, Orca, Sarathi-Serve, FasterTransformer
- **Serverless-Native**: Cold starts, auto-scaling, cost optimization
- **Comprehensive Metrics**: TTFT, TBT, MFU, throughput, cost per request
- **Flexible Workloads**: Poisson arrivals, trace replay, burst patterns
- **Easy Configuration**: YAML-based configuration system
- **Extensible**: Plugin architecture for custom components
- **FaaSInfer Integration**: Reads profiling outputs from FaaSInfer for accurate predictions
- **Capacity Planning**: Automated configuration search and deployment recommendations

## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/simfaasinfer.git
cd simfaasinfer

# Install dependencies
pip install -e .
```

## 🚀 Quick Start
```python
from simfaasinfer import Simulator
from simfaasinfer.utils import load_config

# Load configuration
config = load_config("configs/models/llama2_7b.yaml")

# Run simulation
simulator = Simulator(config)
results = simulator.run()

# Print results
print(f"Throughput: {results['throughput']:.2f} QPS")
print(f"P95 Latency: {results['p95_latency']:.2f} ms")
print(f"QPS per Dollar: {results['qps_per_dollar']:.4f}")
```

## 📚 Documentation

- [Architecture](docs/architecture.md): System design and components
- [Metrics](docs/metrics.md): Detailed metrics guide
- [Capacity Planning](docs/capacity_planning.md): Complete capacity planning guide
- [Examples](examples/): Usage examples

## 🔧 Configuration

### Basic Configuration
```yaml
# configs/my_config.yaml
simulation:
  duration: 3600  # 1 hour
  random_seed: 42

workload:
  type: "poisson"
  arrival_rate: 10  # QPS
  
model:
  name: "llama2-7b"
  
scheduler:
  type: "vllm"
  max_batch_size: 128
```

### Supported Models

- LLaMA2 7B/70B
- Custom models via configuration

### Supported Schedulers

- **vLLM**: Continuous batching with preemption
- **Orca**: Iteration-level scheduling
- **Sarathi-Serve**: Chunked prefills
- **FasterTransformer**: Static batching

## 📊 Example Results
```
=== Simulation Results ===
Total Requests: 3600
Throughput: 10.2 QPS
Median Latency: 245 ms
P95 Latency: 512 ms
Mean MFU: 67%
QPS per Dollar: 0.0334
```

## 🎓 Examples

### Basic Simulation
```bash
python examples/basic_simulation.py
```

### Capacity Planning

SimFaaSInfer provides comprehensive capacity planning capabilities that read FaaSInfer profiling outputs and generate deployment recommendations.

#### Quick Start
```bash
# Run capacity planning with FaaSInfer profile
python examples/run_capacity_planning.py \
  --profile data/profiling/compute/a100/llama-7b/fitted_profile.json \
  --output results/capacity_plan.json
```

#### Advanced Usage
```bash
# With custom workload and strict SLOs
python examples/run_capacity_planning.py \
  --profile data/profiling/compute/a100/llama-7b/fitted_profile.json \
  --gpu-types A100 H100 \
  --max-cost 50 \
  --ttft-slo 1000 \
  --tbt-slo 100 \
  --output results/capacity_plan.json

# With production telemetry calibration
python examples/run_capacity_planning.py \
  --profile data/profiling/compute/a100/llama-7b/fitted_profile.json \
  --telemetry data/telemetry/prod_samples.json \
  --output results/capacity_plan_calibrated.json
```

#### Using Config Files
```bash
# With workload and constraint configs
python -m simfaasinfer.capacity_planning \
  --profile data/profiling/compute/a100/llama-7b/fitted_profile.json \
  --workload configs/workload/production.yaml \
  --constraints configs/constraints/strict_slos.yaml \
  --output results/capacity_plan.json
```

#### Example Output
```
===============================================
   CAPACITY PLANNING REPORT
   Model: llama-7b on NVIDIA A100
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
Throughput:         87.5 QPS
Latency (P95):      512.7 ms
TTFT (P95):         1456.8 ms ✓
TBT (P99):          145.3 ms ✓
Resource Util:      67% MFU, 82% Memory

COST ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hourly Cost:        $24.48/hour
Cost per 1K Req:    $0.28
QPS per Dollar:     3.58

SLO COMPLIANCE:     ✓ ALL SLOS MET
```

See [docs/capacity_planning.md](docs/capacity_planning.md) for complete guide.

### FaaSInfer Integration

SimFaaSInfer is designed to work seamlessly with FaaSInfer profiling outputs:

1. **Profile your model** with FaaSInfer on target hardware
2. **Export profiles** to `data/profiling/compute/{gpu}/{model}/`
3. **Run capacity planning** using the fitted profiles
4. **Optional**: Calibrate with production telemetry for improved accuracy

```bash
# Step 1: Profile with FaaSInfer (run on FaaSInfer platform)
# Generates: fitted_profile.json, metadata.json

# Step 2: Copy profiles to SimFaaSInfer
cp /path/to/faasinfer/profiles/* data/profiling/compute/a100/llama-7b/

# Step 3: Run capacity planning
python examples/run_capacity_planning.py \
  --profile data/profiling/compute/a100/llama-7b/fitted_profile.json \
  --output results/capacity_plan.json
```

## 🧪 Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_simulator.py

# Run with coverage
python -m pytest --cov=simfaasinfer tests/
```

## 📈 Performance

- Simulates 1000+ QPS workloads
- Handles 100K+ requests
- Sub-second simulation time for simple scenarios
- Minutes for complex capacity planning

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

This simulator is inspired by:
- [Vidur](https://github.com/microsoft/vidur): Microsoft Research's LLM inference simulator
- [vLLM](https://github.com/vllm-project/vllm): Efficient LLM serving
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310): Chunked prefills for LLM inference

## 📖 Citation
```bibtex
@software{simfaasinfer2024,
  title={SimFaaSInfer: A Serverless LLM Inference Simulator},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/simfaasinfer}
}
```

## 📧 Contact

For questions or issues, please open a GitHub issue or contact: your.email@example.com