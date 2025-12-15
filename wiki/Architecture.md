# System Architecture

## Overview

The Gossip-Based Distributed RL Training System consists of multiple layers working together to enable decentralized reinforcement learning with privacy guarantees.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         AGENT NODE                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Environment │───▶│  RL Trainer │───▶│   Policy    │         │
│  │ (Gymnasium) │    │    (PPO)    │    │  Network    │         │
│  └─────────────┘    └──────┬──────┘    └─────────────┘         │
│                            │                                     │
│                     ┌──────▼──────┐                             │
│                     │  Gradients  │                             │
│                     └──────┬──────┘                             │
│                            │                                     │
│  ┌─────────────────────────▼─────────────────────────┐         │
│  │              DIFFERENTIAL PRIVACY                  │         │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────────┐  │         │
│  │  │  Clipping │─▶│   Noise   │─▶│  Accountant   │  │         │
│  │  └───────────┘  └───────────┘  └───────────────┘  │         │
│  └─────────────────────────┬─────────────────────────┘         │
│                            │                                     │
│  ┌─────────────────────────▼─────────────────────────┐         │
│  │               GOSSIP PROTOCOL                      │         │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────────┐  │         │
│  │  │ Membership│  │  Push-Pull│  │  Aggregation  │  │         │
│  │  └───────────┘  └───────────┘  └───────────────┘  │         │
│  └─────────────────────────┬─────────────────────────┘         │
│                            │                                     │
│  ┌─────────────────────────▼─────────────────────────┐         │
│  │                C++ RPC LAYER                       │         │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────────┐  │         │
│  │  │Event Loop │  │  Buffers  │  │  Transport    │  │         │
│  │  │ (io_uring)│  │ (Zero-copy│  │ (TCP/SHM)     │  │         │
│  │  └───────────┘  └───────────┘  └───────────────┘  │         │
│  └───────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  OTHER AGENTS   │
                    │  (via network)  │
                    └─────────────────┘
```

## Component Layers

### 1. Environment Layer
- Uses OpenAI Gymnasium environments
- Supports CartPole, LunarLander, and custom environments
- Collects observations, actions, and rewards

### 2. Training Layer (PPO)
- Proximal Policy Optimization algorithm
- Actor-Critic neural network architecture
- GAE advantage estimation
- Gradient extraction for gossip

### 3. Privacy Layer
- Gaussian mechanism for gradient perturbation
- Per-sample gradient clipping
- Rényi Differential Privacy accounting
- Adaptive noise scheduling

### 4. Gossip Layer
- Push-pull hybrid protocol
- SWIM-based failure detection
- Secure gradient aggregation
- Locality-aware peer selection

### 5. RPC Layer (C++)
- High-performance async I/O (io_uring on Linux, kqueue on macOS)
- Zero-copy buffer management
- TCP transport with connection pooling
- Shared memory for intra-node communication

## Data Flow

1. **Collect**: Agent interacts with environment, collects trajectories
2. **Train**: PPO updates policy using collected data
3. **Extract**: Gradients extracted from neural network
4. **Privatize**: DP noise added, privacy budget tracked
5. **Gossip**: Gradients shared with random peers
6. **Aggregate**: Received gradients merged with local
7. **Apply**: Merged gradients update the policy
8. **Repeat**: Continue until convergence or budget exhausted

## File Structure

```
gossip-rl/
├── python/gossip_rl/
│   ├── agent.py           # Main orchestrator
│   ├── training/          # PPO, networks
│   ├── gossip/            # Protocol implementation
│   └── privacy/           # DP mechanisms
├── cpp/
│   ├── include/rpc/       # C++ headers
│   └── src/               # C++ implementation
├── configs/               # YAML configurations
└── scripts/               # Utility scripts
```
