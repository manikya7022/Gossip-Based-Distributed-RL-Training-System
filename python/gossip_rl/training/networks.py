"""Neural Network Architectures for Policy and Value Functions."""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class PolicyNetwork(nn.Module):
    """Actor network for policy gradient methods.
    
    Supports both discrete and continuous action spaces.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        continuous: bool = False,
        log_std_init: float = 0.0,
        activation: str = "tanh",
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.continuous = continuous
        
        # Select activation
        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU,
        }
        act_fn = activations.get(activation, nn.Tanh)
        
        # Build layers
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn())
            in_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        if continuous:
            self.mean_head = nn.Linear(in_dim, action_dim)
            self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        else:
            self.action_head = nn.Linear(in_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # Special init for output heads
        if self.continuous:
            nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
            nn.init.constant_(self.mean_head.bias, 0)
        else:
            nn.init.orthogonal_(self.action_head.weight, gain=0.01)
            nn.init.constant_(self.action_head.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Returns:
            For discrete: (action_logits, None)
            For continuous: (mean, log_std)
        """
        features = self.backbone(obs)
        
        if self.continuous:
            mean = self.mean_head(features)
            log_std = self.log_std.expand_as(mean)
            return mean, log_std
        else:
            logits = self.action_head(features)
            return logits, None
    
    def get_distribution(
        self, obs: torch.Tensor
    ) -> torch.distributions.Distribution:
        """Get action distribution."""
        if self.continuous:
            mean, log_std = self.forward(obs)
            std = log_std.exp()
            return Normal(mean, std)
        else:
            logits, _ = self.forward(obs)
            return Categorical(logits=logits)
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability.
        
        Returns:
            (action, log_prob)
        """
        dist = self.get_distribution(obs)
        
        if deterministic:
            if self.continuous:
                action = dist.mean
            else:
                action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        if self.continuous and len(action.shape) > 1:
            log_prob = log_prob.sum(dim=-1)
        
        return action, log_prob
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy of actions.
        
        Returns:
            (log_prob, entropy)
        """
        dist = self.get_distribution(obs)
        
        log_prob = dist.log_prob(actions)
        if self.continuous and len(actions.shape) > 1:
            log_prob = log_prob.sum(dim=-1)
        
        entropy = dist.entropy()
        if self.continuous and len(actions.shape) > 1:
            entropy = entropy.sum(dim=-1)
        
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Critic network for value estimation."""
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "tanh",
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        
        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU,
        }
        act_fn = activations.get(activation, nn.Tanh)
        
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # Last layer with smaller init
        final_layer = self.network[-1]
        nn.init.orthogonal_(final_layer.weight, gain=1.0)
        nn.init.constant_(final_layer.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning value estimate."""
        return self.network(obs).squeeze(-1)


class ActorCritic(nn.Module):
    """Combined Actor-Critic network with shared backbone."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        continuous: bool = False,
        shared_backbone: bool = False,
        activation: str = "tanh",
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.shared_backbone = shared_backbone
        
        if shared_backbone:
            # Shared feature extraction
            activations = {
                "tanh": nn.Tanh,
                "relu": nn.ReLU,
            }
            act_fn = activations.get(activation, nn.Tanh)
            
            layers = []
            in_dim = obs_dim
            for hidden_dim in hidden_dims[:-1]:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(act_fn())
                in_dim = hidden_dim
            
            self.backbone = nn.Sequential(*layers)
            
            # Separate heads
            self.policy = PolicyNetwork(
                in_dim, action_dim, [hidden_dims[-1]], continuous
            )
            self.value = ValueNetwork(in_dim, [hidden_dims[-1]])
        else:
            self.policy = PolicyNetwork(
                obs_dim, action_dim, hidden_dims, continuous
            )
            self.value = ValueNetwork(obs_dim, hidden_dims)
    
    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """Forward pass returning distribution and value."""
        if self.shared_backbone:
            features = self.backbone(obs)
            dist = self.policy.get_distribution(features)
            value = self.value(features)
        else:
            dist = self.policy.get_distribution(obs)
            value = self.value(obs)
        
        return dist, value
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, value, log_prob, and entropy.
        
        Returns:
            (action, value, log_prob, entropy)
        """
        if self.shared_backbone:
            features = self.backbone(obs)
            dist = self.policy.get_distribution(features)
            value = self.value(features)
        else:
            dist = self.policy.get_distribution(obs)
            value = self.value(obs)
        
        if action is None:
            if deterministic:
                if self.continuous:
                    action = dist.mean
                else:
                    action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
        
        log_prob = dist.log_prob(action)
        if self.continuous and len(action.shape) > 1:
            log_prob = log_prob.sum(dim=-1)
        
        entropy = dist.entropy()
        if self.continuous and len(action.shape) > 1:
            entropy = entropy.sum(dim=-1)
        
        return action, value, log_prob, entropy


def get_gradient_vector(model: nn.Module) -> np.ndarray:
    """Extract flattened gradient vector from model."""
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1).cpu().numpy())
    
    if not grads:
        return np.array([])
    
    return np.concatenate(grads)


def set_gradient_vector(model: nn.Module, gradient: np.ndarray) -> None:
    """Set model gradients from flattened vector."""
    offset = 0
    for param in model.parameters():
        if param.grad is not None:
            numel = param.numel()
            param.grad.data = torch.from_numpy(
                gradient[offset:offset + numel].reshape(param.shape)
            ).to(param.device)
            offset += numel


def get_parameter_vector(model: nn.Module) -> np.ndarray:
    """Extract flattened parameter vector from model."""
    params = []
    for param in model.parameters():
        params.append(param.view(-1).detach().cpu().numpy())
    
    return np.concatenate(params)


def set_parameter_vector(model: nn.Module, parameters: np.ndarray) -> None:
    """Set model parameters from flattened vector."""
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.data = torch.from_numpy(
            parameters[offset:offset + numel].reshape(param.shape)
        ).to(param.device)
        offset += numel
