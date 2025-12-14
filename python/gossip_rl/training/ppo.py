"""Distributed PPO Implementation with Gossip-based Gradient Aggregation."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gossip_rl.config import TrainingConfig, PrivacyConfig
from gossip_rl.training.networks import (
    ActorCritic,
    get_gradient_vector,
    set_gradient_vector,
    get_parameter_vector,
    set_parameter_vector,
)
from gossip_rl.privacy.mechanism import GaussianMechanism, DPParams
from gossip_rl.privacy.accountant import RDPAccountant
from gossip_rl.privacy.adaptive import AdaptiveNoiseScheduler, NoiseScheduleConfig

logger = logging.getLogger(__name__)


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data."""
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    
    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def __len__(self) -> int:
        return len(self.observations)


@dataclass
class PPOStats:
    """PPO training statistics."""
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    total_loss: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    explained_variance: float = 0.0
    mean_reward: float = 0.0
    gradient_norm: float = 0.0


class DistributedPPO:
    """Distributed PPO with Gossip-based gradient aggregation.
    
    Features:
    - Local PPO training with clipped objective
    - Gradient extraction and gossiping
    - Optional differential privacy
    - GAE advantage estimation
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: TrainingConfig,
        privacy_config: Optional[PrivacyConfig] = None,
        device: str = "cpu",
    ):
        self.config = config
        self.privacy_config = privacy_config
        self.device = torch.device(device)
        
        # Network
        self.model = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=[256, 256],
            continuous=False,  # Adjust based on environment
            shared_backbone=False,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
        # Training state
        self.step = 0
        self.model_version = 0
        
        # Differential Privacy
        self.dp_mechanism: Optional[GaussianMechanism] = None
        self.privacy_accountant: Optional[RDPAccountant] = None
        self.noise_scheduler: Optional[AdaptiveNoiseScheduler] = None
        
        if privacy_config and privacy_config.enabled:
            self._setup_dp(privacy_config)
        
        # Gradient storage for gossip
        self.local_gradients: Optional[np.ndarray] = None
        self.aggregated_gradients: Optional[np.ndarray] = None
    
    def _setup_dp(self, config: PrivacyConfig) -> None:
        """Set up differential privacy components."""
        dp_params = DPParams(
            epsilon=config.epsilon,
            delta=config.delta,
            clip_norm=config.clip_norm,
            noise_multiplier=config.noise_multiplier if not config.adaptive_noise else None,
        )
        
        self.dp_mechanism = GaussianMechanism(dp_params)
        self.privacy_accountant = RDPAccountant()
        
        if config.adaptive_noise:
            self.noise_scheduler = AdaptiveNoiseScheduler(NoiseScheduleConfig(
                total_epsilon=config.epsilon,
                delta=config.delta,
            ))
        
        logger.info(f"DP enabled: ε={config.epsilon}, δ={config.delta}")
    
    def collect_rollout(
        self,
        env,
        n_steps: int,
    ) -> Tuple[float, int]:
        """Collect experience from environment.
        
        Returns:
            (total_reward, episode_count)
        """
        self.buffer.clear()
        total_reward = 0.0
        episode_count = 0
        
        obs = env.reset()[0] if hasattr(env.reset(), '__getitem__') else env.reset()
        
        for _ in range(n_steps):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action, value, log_prob, _ = self.model.get_action_and_value(obs_tensor)
                
                action = action.cpu().numpy()[0]
                value = value.cpu().numpy()[0]
                log_prob = log_prob.cpu().numpy()[0]
            
            next_obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            
            self.buffer.add(obs, action, reward, value, log_prob, done)
            total_reward += reward
            
            if done:
                episode_count += 1
                obs = env.reset()[0] if hasattr(env.reset(), '__getitem__') else env.reset()
            else:
                obs = next_obs
        
        return total_reward, episode_count
    
    def compute_advantages(
        self,
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns."""
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda
        
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones, dtype=np.float32)
        
        n_steps = len(rewards)
        advantages = np.zeros(n_steps)
        last_gae = 0
        
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def train_step(self) -> PPOStats:
        """Execute one PPO training update.
        
        Returns:
            Training statistics
        """
        # Get last value for GAE computation
        with torch.no_grad():
            last_obs = torch.FloatTensor(
                self.buffer.observations[-1]
            ).unsqueeze(0).to(self.device)
            _, last_value, _, _ = self.model.get_action_and_value(last_obs)
            last_value = last_value.cpu().numpy()[0]
        
        # Compute advantages
        advantages, returns = self.compute_advantages(last_value)
        
        # Convert to tensors
        obs = torch.FloatTensor(np.array(self.buffer.observations)).to(self.device)
        actions = torch.LongTensor(np.array(self.buffer.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs)).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        # PPO update
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        n_updates = 0
        
        batch_size = self.config.batch_size
        n_samples = len(obs)
        
        for epoch in range(self.config.n_epochs):
            # Shuffle indices
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                
                # Get batch
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                
                # Forward pass
                _, values, log_probs, entropy = self.model.get_action_and_value(
                    batch_obs, batch_actions
                )
                
                # Policy loss (clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon,
                )
                
                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages,
                ).mean()
                
                # Value loss
                value_loss = 0.5 * ((values - batch_returns) ** 2).mean()
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                
                # Extract gradients for gossip
                self._extract_gradients()
                
                # Apply DP noise if enabled
                if self.dp_mechanism is not None:
                    self._apply_dp_noise()
                
                self.optimizer.step()
                
                # Statistics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - ratio.log()).mean().item()
                    clip_frac = ((ratio - 1).abs() > self.config.clip_epsilon).float().mean().item()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_approx_kl += approx_kl
                total_clip_frac += clip_frac
                n_updates += 1
        
        self.step += 1
        
        # Compute explained variance
        with torch.no_grad():
            values_array = np.array(self.buffer.values)
            explained_var = 1 - np.var(returns - values_array) / (np.var(returns) + 1e-8)
        
        return PPOStats(
            policy_loss=total_policy_loss / n_updates,
            value_loss=total_value_loss / n_updates,
            entropy_loss=total_entropy_loss / n_updates,
            total_loss=(total_policy_loss + total_value_loss + total_entropy_loss) / n_updates,
            approx_kl=total_approx_kl / n_updates,
            clip_fraction=total_clip_frac / n_updates,
            explained_variance=explained_var,
            mean_reward=np.mean(self.buffer.rewards),
            gradient_norm=np.linalg.norm(self.local_gradients) if self.local_gradients is not None else 0,
        )
    
    def _extract_gradients(self) -> None:
        """Extract gradients for gossip."""
        self.local_gradients = get_gradient_vector(self.model)
    
    def _apply_dp_noise(self) -> None:
        """Apply differential privacy noise to gradients."""
        if self.local_gradients is None or self.dp_mechanism is None:
            return
        
        # Get noise multiplier (possibly adaptive)
        if self.noise_scheduler is not None:
            noise_mult = self.noise_scheduler.get_noise_multiplier(
                self.step,
                np.linalg.norm(self.local_gradients),
            )
            self.dp_mechanism.params.noise_multiplier = noise_mult
        
        # Apply DP
        noisy_gradients, metadata = self.dp_mechanism.privatize(self.local_gradients)
        
        # Update gradients in model
        set_gradient_vector(self.model, noisy_gradients)
        self.local_gradients = noisy_gradients
        
        # Track privacy
        if self.privacy_accountant is not None:
            self.privacy_accountant.step(
                noise_multiplier=self.dp_mechanism.noise_multiplier,
                sample_rate=self.config.batch_size / len(self.buffer),
            )
        
        logger.debug(
            f"DP applied: clip={metadata['clipped']}, "
            f"noise_scale={metadata['noise_scale']:.4f}"
        )
    
    def get_gradients(self) -> np.ndarray:
        """Get current gradients for gossip."""
        if self.local_gradients is None:
            return np.array([])
        return self.local_gradients.copy()
    
    def apply_aggregated_gradients(self, gradients: np.ndarray) -> None:
        """Apply aggregated gradients from gossip."""
        self.aggregated_gradients = gradients
        
        # Average local and aggregated
        if self.local_gradients is not None:
            merged = 0.5 * self.local_gradients + 0.5 * gradients
            set_gradient_vector(self.model, merged)
        
        self.model_version += 1
    
    def get_parameters(self) -> np.ndarray:
        """Get model parameters for model sync."""
        return get_parameter_vector(self.model)
    
    def set_parameters(self, parameters: np.ndarray) -> None:
        """Set model parameters from received sync."""
        set_parameter_vector(self.model, parameters)
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get (ε, δ) privacy spent."""
        if self.privacy_accountant is None:
            return 0.0, 0.0
        return self.privacy_accountant.get_privacy_spent(
            self.privacy_config.delta if self.privacy_config else 1e-5
        )
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'model_version': self.model_version,
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.model_version = checkpoint['model_version']
