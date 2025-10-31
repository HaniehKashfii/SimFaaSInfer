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
```bash
python examples/capacity_planning.py
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