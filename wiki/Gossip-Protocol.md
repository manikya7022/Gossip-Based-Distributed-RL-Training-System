# Gossip Protocol

## Overview

The gossip protocol enables decentralized gradient sharing between agents without requiring a central parameter server. This approach provides fault tolerance, scalability, and privacy benefits.

## How Gossip Works

### Push-Pull Hybrid

Each gossip round, agents:
1. **Push**: Send their gradients to randomly selected peers
2. **Pull**: Request gradients from other peers
3. **Merge**: Combine received gradients with local gradients

### Protocol Flow

```
Round 1:
  Agent-1 ──push──▶ Agent-3, Agent-5, Agent-7  (fanout=3)
  Agent-2 ──push──▶ Agent-1, Agent-4, Agent-8
  ...

Round 2:
  Agent-3 ──push──▶ Agent-2, Agent-6, Agent-9  (now has Agent-1's gradients)
  Agent-5 ──push──▶ Agent-4, Agent-8, Agent-10
  ...

Round N:
  All agents have aggregated gradients from entire network
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `interval_ms` | 100 | Time between gossip rounds |
| `fanout` | 3 | Number of peers per round |
| `push_pull` | true | Enable bidirectional exchange |

## Convergence Time

With fanout `f` and `n` agents:
- **Rounds to reach all agents**: O(log_f(n))
- **Example**: 50 agents, fanout=3 → ~4 rounds

## Code Reference

### Starting Gossip

```python
from gossip_rl.gossip import GossipProtocol, MembershipManager

# Initialize
membership = MembershipManager(local_peer=my_peer)
gossip = GossipProtocol(
    local_peer=my_peer,
    config=gossip_config,
    membership=membership,
    on_gradient_received=handle_gradients,
)

# Start
await membership.start()
await gossip.start()
```

### Sending Gradients

```python
# Set gradients for gossip
gossip.set_gradients(
    gradients=my_gradients,
    model_version=1,
    step=100,
)
```

### Receiving Aggregated Gradients

```python
# Get merged gradients from all contributors
aggregated, num_contributors = await gossip.get_aggregated_gradients()

if num_contributors > 1:
    trainer.apply_aggregated_gradients(aggregated)
```

## Membership Management

### SWIM Failure Detection

- **Probe Interval**: 500ms (configurable)
- **Probe Timeout**: 200ms
- **Indirect Probes**: 3 (ask other peers to probe)

### Peer States

| State | Description |
|-------|-------------|
| ALIVE | Peer is responsive |
| SUSPECT | Probe failed, waiting for indirect confirmation |
| DEAD | Multiple probes failed, peer removed |

## Implementation Files

- `python/gossip_rl/gossip/protocol.py` - Main gossip logic
- `python/gossip_rl/gossip/membership.py` - Peer tracking
- `python/gossip_rl/gossip/message.py` - Message formats
- `python/gossip_rl/gossip/peer.py` - Peer data structure
