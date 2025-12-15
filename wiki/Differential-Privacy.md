# Differential Privacy

## Overview

Differential Privacy (DP) provides mathematical guarantees that individual training data cannot be inferred from the model or gradient updates. This system implements local DP where each agent adds noise to their gradients before sharing.

## Privacy Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Epsilon (ε) | ε | Privacy budget (lower = more private) |
| Delta (δ) | δ | Probability of privacy failure |
| Clip Norm | C | Maximum gradient norm |
| Noise Multiplier | σ | Noise scaling factor |

## Gaussian Mechanism

The system uses the Gaussian mechanism for gradient perturbation:

```
noisy_gradient = clip(gradient, C) + N(0, σ² * C²)
```

Where:
- `clip(gradient, C)` limits gradient norm to C
- `N(0, σ² * C²)` is Gaussian noise calibrated for (ε, δ)-DP

## Privacy Accounting

### Rényi Differential Privacy (RDP)

The system uses RDP for tight privacy composition:

```python
# After each training step
accountant.step(
    noise_multiplier=σ,
    sample_rate=batch_size / dataset_size,
)

# Check remaining budget
epsilon_spent, delta = accountant.get_privacy_spent(target_delta=1e-5)
```

### Budget Consumption

| Configuration | ε per step (approx) |
|---------------|---------------------|
| σ=1.0, batch=64 | 0.47 |
| σ=2.0, batch=64 | 0.12 |
| σ=4.0, batch=64 | 0.03 |

## Adaptive Noise

The system supports adaptive noise scheduling:

```python
# Noise decreases as training converges
noise_mult = scheduler.get_noise_multiplier(
    step=current_step,
    gradient_norm=grad_norm,
)
```

### Schedule Options

1. **Constant**: Fixed noise throughout training
2. **Linear Decay**: Noise decreases linearly
3. **Gradient-based**: Noise scales with gradient magnitude

## Code Reference

### Enabling DP

```yaml
# configs/agent.yaml
privacy:
  enabled: true
  epsilon: 1.0
  delta: 1e-5
  clip_norm: 1.0
  noise_multiplier: 1.1
  adaptive_noise: true
```

### Checking Privacy Budget

```python
# Get current privacy spent
epsilon, delta = trainer.get_privacy_spent()

if epsilon > target_epsilon * 0.95:
    print("Privacy budget nearly exhausted!")
```

## Privacy Efficiency Metric

From benchmarks:

```
Privacy Efficiency = Average Reward / ε Spent
                   = 22.81 / 2.21
                   = 10.30 reward/ε
```

Higher values indicate better utility-privacy tradeoff.

## Implementation Files

- `python/gossip_rl/privacy/mechanism.py` - Gaussian mechanism
- `python/gossip_rl/privacy/accountant.py` - RDP accounting
- `python/gossip_rl/privacy/adaptive.py` - Noise scheduling
