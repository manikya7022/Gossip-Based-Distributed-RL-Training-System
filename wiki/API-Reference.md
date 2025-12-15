# API Reference

## Python Modules

### gossip_rl.agent

#### Agent

Main agent orchestrator combining RL training, gossip protocol, and differential privacy.

```python
from gossip_rl.agent import Agent
from gossip_rl.config import load_config

config = load_config("configs/agent.yaml")
agent = Agent(config)

# Run training
await agent.run()

# Add peer
await agent.add_peer("192.168.1.10", 5000, "peer-1")

# Get stats
stats = agent.get_stats()

# Stop
await agent.stop()
```

---

### gossip_rl.training

#### DistributedPPO

PPO trainer with gossip integration.

```python
from gossip_rl.training import DistributedPPO

trainer = DistributedPPO(
    obs_dim=4,
    action_dim=2,
    config=training_config,
    privacy_config=privacy_config,
    device="cpu",
)

# Collect experience
reward, episodes = trainer.collect_rollout(env, n_steps=2048)

# Train
stats = trainer.train_step()

# Gradients
gradients = trainer.get_gradients()
trainer.apply_aggregated_gradients(merged)

# Privacy
epsilon, delta = trainer.get_privacy_spent()

# Checkpoint
trainer.save_checkpoint("path/to/checkpoint.pt")
trainer.load_checkpoint("path/to/checkpoint.pt")
```

#### ActorCritic

Neural network for PPO.

```python
from gossip_rl.training.networks import ActorCritic

model = ActorCritic(
    obs_dim=4,
    action_dim=2,
    hidden_dims=[256, 256],
    continuous=False,
    shared_backbone=False,
)

# Forward pass
action, value, log_prob, entropy = model.get_action_and_value(obs)
```

---

### gossip_rl.gossip

#### GossipProtocol

Push-pull gossip for gradient sharing.

```python
from gossip_rl.gossip import GossipProtocol

protocol = GossipProtocol(
    local_peer=peer,
    config=gossip_config,
    membership=membership_manager,
    on_gradient_received=callback,
)

await protocol.start()

# Set gradients
protocol.set_gradients(gradients, model_version=1, step=100)

# Get aggregated
aggregated, n_contributors = await protocol.get_aggregated_gradients()

# Stats
stats = await protocol.get_stats()

await protocol.stop()
```

#### MembershipManager

Peer discovery and health monitoring.

```python
from gossip_rl.gossip import MembershipManager, Peer

peer = Peer(id="agent-1", host="localhost", port=5000)
manager = MembershipManager(
    local_peer=peer,
    probe_interval_ms=500,
    probe_timeout_ms=200,
    indirect_probes=3,
    on_peer_alive=callback,
    on_peer_dead=callback,
)

await manager.start()
await manager.add_peer(other_peer)
peers = manager.get_alive_peers()
await manager.stop()
```

---

### gossip_rl.privacy

#### GaussianMechanism

Differential privacy mechanism.

```python
from gossip_rl.privacy import GaussianMechanism, DPParams

params = DPParams(
    epsilon=1.0,
    delta=1e-5,
    clip_norm=1.0,
    noise_multiplier=1.1,
)

mechanism = GaussianMechanism(params)

# Privatize gradients
noisy, metadata = mechanism.privatize(gradients)
```

#### RDPAccountant

Privacy budget tracking.

```python
from gossip_rl.privacy import RDPAccountant

accountant = RDPAccountant()

# Track step
accountant.step(noise_multiplier=1.1, sample_rate=0.01)

# Get spent budget
epsilon, delta = accountant.get_privacy_spent(target_delta=1e-5)
```

---

### gossip_rl.config

#### Config Classes

```python
from gossip_rl.config import (
    Config,
    load_config,
    AgentConfig,
    NetworkConfig,
    GossipConfig,
    PrivacyConfig,
    TrainingConfig,
    EnvironmentConfig,
)

# Load from YAML
config = load_config("configs/agent.yaml")

# Access fields
print(config.agent.id)
print(config.privacy.epsilon)
print(config.training.learning_rate)
```

---

## C++ API

### EventLoop

```cpp
#include <rpc/event_loop.hpp>

EventLoop::Config config;
auto loop = std::make_shared<EventLoop>(config);

loop->submit_read(fd, buffer, callback);
loop->submit_write(fd, buffer, callback);
loop->run();
```

### BufferPool

```cpp
#include <rpc/buffer_pool.hpp>

BufferPoolConfig config;
auto pool = std::make_shared<BufferPool>(config);

auto buffer = pool->acquire();
buffer->write(data, size);
// Auto-returns to pool on destruction
```

### RpcServer

```cpp
#include <rpc/rpc_server.hpp>

RpcServer server(event_loop, transport, buffer_pool);
server.register_service(service);
server.start(5000);
```

### RpcClient

```cpp
RpcClient client(event_loop, transport, buffer_pool);
auto future = client.call(endpoint, method_id, payload);
auto response = future.wait(std::chrono::seconds(5));
```
