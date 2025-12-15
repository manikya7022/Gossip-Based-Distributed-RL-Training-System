# Benchmarks

## Overview

Performance benchmarks for the Gossip-Based Distributed RL Training System.

## Test Environment

- **Hardware**: Apple M-series / Intel CPU
- **OS**: macOS 12+ / Linux
- **Python**: 3.10+
- **Environment**: CartPole-v1

---

## Multi-Agent Simulation (50 Agents)

### Configuration

| Parameter | Value |
|-----------|-------|
| Agents | 50 |
| Duration | 60s |
| Privacy ε | 10.0 |
| Gossip Fanout | 3 |
| Gossip Interval | 100ms |

### Results

```
Agents completed: 50/50
Total training steps: 65
Total episodes: 5,961
Total gossip rounds: 65
Average reward: 22.39
```

### Key Findings

- **100% agent success rate**: All 50 agents completed training
- **Gossip rounds per step**: 1.0 (efficient gradient sharing)
- **Average reward**: 22.39 (CartPole max ~500)

---

## Efficiency Benchmark (10 Agents)

### Configuration

| Parameter | Value |
|-----------|-------|
| Agents | 10 |
| Duration | 60s |
| Privacy ε | 10.0 |
| Batch Size | 64 |

### Throughput Metrics

| Metric | Value |
|--------|-------|
| Training Steps | 25 |
| Episodes Completed | 2,251 |
| Steps/Second | 0.25 |
| Episodes/Second | 22.60 |
| Throughput/Agent | 2.26 eps/s |

### Reward Metrics

| Metric | Value |
|--------|-------|
| Average Reward | 22.81 |
| Max Reward | 23.54 |
| Min Reward | 22.18 |
| Std Dev | 0.45 |

### Gossip Efficiency

| Metric | Value |
|--------|-------|
| Total Gossip Rounds | 25 |
| Rounds/Agent | 2.5 |
| Rounds/Step | 1.00 |

### Privacy Metrics

| Metric | Value |
|--------|-------|
| Avg ε Spent | 2.21 |
| Max ε Spent | 2.45 |
| Privacy Efficiency | 10.30 reward/ε |

---

## Per-Agent Breakdown

| Agent | Steps | Episodes | Avg Reward | ε Spent | Step Time (ms) |
|-------|-------|----------|------------|---------|----------------|
| agent-0 | 2 | 181 | 22.63 | 1.98 | 4005 |
| agent-1 | 2 | 174 | 23.54 | 1.98 | 3976 |
| agent-2 | 2 | 176 | 23.27 | 1.98 | 4016 |
| agent-3 | 2 | 182 | 22.51 | 1.98 | 3978 |
| agent-4 | 2 | 175 | 23.41 | 1.98 | 3993 |
| agent-5 | 3 | 270 | 22.76 | 2.45 | 3981 |
| agent-6 | 3 | 277 | 22.18 | 2.45 | 3985 |
| agent-7 | 3 | 273 | 22.51 | 2.45 | 3968 |
| agent-8 | 3 | 276 | 22.26 | 2.45 | 3984 |
| agent-9 | 3 | 267 | 23.01 | 2.45 | 3974 |

---

## Running Benchmarks

### Multi-Agent Simulation

```bash
PYTHONPATH="$(pwd)/python:$PYTHONPATH" python3 scripts/launch_multi_agent.py \
    --agents 50 \
    --duration 60
```

### Efficiency Benchmark

```bash
PYTHONPATH="$(pwd)/python:$PYTHONPATH" python3 scripts/benchmark_efficiency.py \
    --agents 10 \
    --duration 120 \
    --output results/benchmark.json
```

---

## Benchmark Output Files

Results are saved to `results/` directory:

```
results/
├── benchmark_10agents_20251214_144246.json
└── ...
```

Each JSON file contains:
- Configuration parameters
- Throughput metrics
- Reward statistics
- Gossip efficiency
- Privacy consumption
- Per-agent breakdown
