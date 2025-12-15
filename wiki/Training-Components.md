# Training Components

## Overview

The training layer implements Proximal Policy Optimization (PPO) with gossip-based distributed gradient sharing and differential privacy.

## PPO Algorithm

### Core Components

1. **Actor Network**: Outputs action probabilities
2. **Critic Network**: Estimates state values
3. **GAE**: Generalized Advantage Estimation
4. **Clipped Objective**: Prevents large policy updates

### Training Flow

```
1. Collect rollout (n_steps experiences)
2. Compute advantages using GAE
3. Update policy for n_epochs
4. Extract gradients
5. Apply DP noise (if enabled)
6. Share via gossip
7. Merge aggregated gradients
8. Repeat
```

## Configuration

```yaml
training:
  algorithm: "ppo"
  learning_rate: 0.0003
  gamma: 0.99           # Discount factor
  gae_lambda: 0.95      # GAE parameter
  clip_epsilon: 0.2     # PPO clip range
  entropy_coef: 0.01    # Entropy bonus
  value_coef: 0.5       # Value loss weight
  max_grad_norm: 0.5    # Gradient clipping
  batch_size: 64
  n_epochs: 10          # PPO epochs per update
  n_steps: 2048         # Steps per rollout
```

## Code Reference

### DistributedPPO Class

```python
from gossip_rl.training import DistributedPPO

trainer = DistributedPPO(
    obs_dim=4,
    action_dim=2,
    config=training_config,
    privacy_config=privacy_config,  # Optional
)
```

### Collect Experience

```python
reward, episodes = trainer.collect_rollout(
    env=gymnasium_env,
    n_steps=2048,
)
```

### Train Step

```python
stats = trainer.train_step()

print(f"Policy Loss: {stats.policy_loss}")
print(f"Value Loss: {stats.value_loss}")
print(f"KL Divergence: {stats.approx_kl}")
```

### Gradient Exchange

```python
# Get gradients for gossip
gradients = trainer.get_gradients()

# Apply aggregated gradients
trainer.apply_aggregated_gradients(merged_gradients)
```

### Checkpointing

```python
# Save
trainer.save_checkpoint("checkpoints/policy_step_100.pt")

# Load
trainer.load_checkpoint("checkpoints/policy_step_100.pt")
```

## Neural Network Architecture

### ActorCritic Network

```python
from gossip_rl.training.networks import ActorCritic

model = ActorCritic(
    obs_dim=4,
    action_dim=2,
    hidden_dims=[256, 256],
    continuous=False,  # Discrete actions
    shared_backbone=False,
)
```

### Network Structure

```
Input (obs_dim)
    │
    ▼
┌─────────────────┐
│  Hidden Layer 1 │ (256 units, ReLU)
└────────┬────────┘
         │
    ▼    │    ▼
┌────────┴────┐  ┌──────────┐
│   Actor     │  │  Critic  │
│ Hidden (256)│  │ Hidden(256)│
└──────┬──────┘  └─────┬────┘
       │               │
       ▼               ▼
┌──────────────┐  ┌─────────┐
│Action Probs  │  │  Value  │
│ (action_dim) │  │   (1)   │
└──────────────┘  └─────────┘
```

## Training Statistics

The `PPOStats` dataclass contains:

| Field | Description |
|-------|-------------|
| `policy_loss` | Actor loss |
| `value_loss` | Critic loss |
| `entropy_loss` | Entropy bonus |
| `approx_kl` | KL divergence estimate |
| `clip_fraction` | Ratio of clipped updates |
| `explained_variance` | Value prediction quality |
| `mean_reward` | Average reward in rollout |
| `gradient_norm` | Gradient magnitude |

## Implementation Files

| File | Description |
|------|-------------|
| `python/gossip_rl/training/ppo.py` | DistributedPPO class |
| `python/gossip_rl/training/networks.py` | Neural networks |
| `python/gossip_rl/training/replay.py` | Experience buffer |
