# SimFaaSInfer Architecture

## Overview

SimFaaSInfer is a discrete event simulator for serverless LLM inference, designed to model the performance and cost characteristics of deploying large language models in serverless environments.

## Architecture Components

### 1. Core Simulation Engine

#### Event Queue (`core/event_queue.py`)
- **Purpose**: Manages discrete events in temporal order
- **Key Features**:
  - Priority queue implementation using heap
  - Event types for all simulation stages
  - Time-ordered event processing

#### Simulator (`core/simulator.py`)
- **Purpose**: Main orchestrator for the simulation
- **Responsibilities**:
  - Event loop management
  - Component coordination
  - Metrics aggregation
- **Key Methods**:
  - `run()`: Execute complete simulation
  - `_process_event()`: Handle individual events
  - `_finalize()`: Compute final results

#### Metrics Collector (`core/metrics_collector.py`)
- **Purpose**: Collect and aggregate performance metrics
- **Metrics Tracked**:
  - Request-level: Latency, TTFT, TBT
  - Cluster-level: MFU, memory utilization
  - Time-series data

### 2. Model Components

#### Model Config (`models/model_config.py`)
- **Purpose**: Store model architectural details
- **Contents**:
  - Layer counts, dimensions
  - Attention configuration
  - Profiled performance data

#### Execution Time Predictor (`models/execution_time_predictor.py`)
- **Purpose**: Predict operator execution times
- **Approach**:
  - Analytical models for token-level ops
  - Memory-bound modeling for attention
  - Communication overhead modeling

#### Model Profiler (`models/model_profiler.py`)
- **Purpose**: Generate profiling data
- **Features**:
  - Synthetic profile generation
  - Real profiling support (future)
  - Training data for ML predictors

### 3. Serverless Infrastructure

#### Function Instance (`serverless/function_instance.py`)
- **Purpose**: Model individual serverless instances
- **State Management**:
  - Instance lifecycle (starting, ready, busy, idle)
  - KV-cache allocation
  - Request batching

#### Cold Start Simulator (`serverless/cold_start_simulator.py`)
- **Purpose**: Model cold start delays
- **Components**:
  - Container initialization
  - Model loading
  - Memory allocation

#### Scaling Policy (`serverless/scaling_policy.py`)
- **Purpose**: Auto-scaling decisions
- **Strategy**:
  - Utilization-based thresholds
  - Cooldown periods
  - Min/max constraints

#### Cost Calculator (`serverless/cost_calculator.py`)
- **Purpose**: Calculate deployment costs
- **Cost Components**:
  - GPU compute costs
  - Idle costs
  - Per-request costs

### 4. Workload Generation

#### Request Generator (`workload/request_generator.py`)
- **Purpose**: Generate inference requests
- **Modes**:
  - Poisson arrivals
  - Trace-based replay
  - Synthetic patterns (bursts)

#### Arrival Process (`workload/arrival_process.py`)
- **Purpose**: Model request arrival patterns
- **Distributions**:
  - Poisson (memoryless)
  - Uniform (constant rate)
  - Gamma (bursty)

#### Trace Loader (`workload/trace_loader.py`)
- **Purpose**: Load real workload traces
- **Formats**:
  - JSON
  - CSV
  - Custom formats

### 5. Scheduling

#### Global Scheduler (`scheduling/scheduler.py`)
- **Purpose**: Route requests to instances
- **Policies**:
  - Round-robin
  - Least loaded
  - Random

#### Replica Scheduler
- **Purpose**: Per-instance request management
- **Responsibilities**:
  - Batch formation
  - Memory management
  - Queue management

#### Batching Strategies (`scheduling/batching_strategy.py`)
- **vLLM**: Continuous batching with preemption
- **Orca**: Iteration-level scheduling
- **Sarathi**: Chunked prefills
- **FasterTransformer**: Static batching

## Data Flow
```
Request Arrival
    ↓
Global Scheduler (routing)
    ↓
Replica Scheduler (batching)
    ↓
Function Instance (execution)
    ↓
Execution Time Predictor (runtime calculation)
    ↓
Batch Complete Event
    ↓
Metrics Collection
```

## Key Design Decisions

### 1. Discrete Event Simulation
- **Why**: Accurate temporal modeling at millisecond granularity
- **Trade-off**: Computational cost vs. accuracy

### 2. Operator-Level Modeling
- **Why**: High-fidelity predictions for various configurations
- **Trade-off**: Complexity vs. generality

### 3. Hierarchical Scheduling
- **Why**: Models real-world multi-tier architectures
- **Trade-off**: Flexibility vs. simplicity

### 4. Pluggable Components
- **Why**: Easy extensibility for new strategies
- **Trade-off**: Abstraction overhead

## Performance Considerations

### Simulation Speed
- Event queue: O(log n) per event
- Batching: O(b) where b = batch size
- Metrics: O(r) where r = requests

### Memory Usage
- Event queue: O(e) where e = pending events
- Request tracking: O(r) where r = total requests
- Time-series metrics: O(t/i) where t = duration, i = interval

### Scalability
- Can simulate 1000+ QPS workloads
- Handles 100K+ request simulations
- Multi-hour simulations practical

## Extension Points

### Adding New Models
1. Create model config in `configs/models/`
2. Add profiling data
3. Optional: Train custom predictors

### Adding New Schedulers
1. Implement `BatchingStrategy` interface
2. Define batching logic in `form_batch()`
3. Register in scheduler factory

### Adding New Workloads
1. Implement trace loader
2. Add generation logic to `RequestGenerator`
3. Configure in YAML

### Adding New Metrics
1. Add tracking in `MetricsCollector`
2. Update `compute_metrics()`
3. Add visualization in `visualization.py`

## Future Enhancements

1. **Real Profiling Integration**: Actual GPU profiling
2. **Network Modeling**: Data transfer latencies
3. **Multi-Model Support**: Multiple models per cluster
4. **Advanced Scaling**: Predictive scaling policies
5. **Cost Optimization**: Spot instances, reserved capacity