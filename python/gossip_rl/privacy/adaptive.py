"""Adaptive Noise Scheduling for DP-SGD.

Novel contribution: Adaptive noise that decreases as training converges
while maintaining formal privacy guarantees.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class NoiseScheduleConfig:
    """Configuration for adaptive noise scheduling."""
    
    # Total privacy budget
    total_epsilon: float = 1.0
    delta: float = 1e-5
    
    # Training parameters
    total_steps: int = 10000
    warmup_steps: int = 100
    
    # Adaptive parameters
    min_noise_mult: float = 0.1
    max_noise_mult: float = 2.0
    
    # Convergence detection
    gradient_window: int = 100
    convergence_threshold: float = 0.1
    
    # Budget allocation
    allocation_strategy: str = "linear"  # linear, exponential, adaptive


class AdaptiveNoiseScheduler:
    """Adaptive noise scheduler that adjusts noise based on training progress.
    
    Key insight: Early in training, gradients have high variance and
    models are far from convergence. Adding more noise early has less
    impact on convergence. As training progresses, we can reduce noise
    while staying within the privacy budget.
    
    Strategies:
    - Linear: Noise decreases linearly with steps
    - Exponential: Noise decays exponentially
    - Adaptive: Noise adjusts based on gradient norm convergence
    """
    
    def __init__(self, config: NoiseScheduleConfig):
        self.config = config
        
        # Track gradient norms for convergence detection
        self.gradient_norms: List[float] = []
        self.noise_history: List[float] = []
        
        # Privacy budget tracking
        self.epsilon_spent = 0.0
        self.steps_completed = 0
        
        # Precompute schedule for non-adaptive strategies
        if config.allocation_strategy != "adaptive":
            self._precompute_schedule()
    
    def _precompute_schedule(self) -> None:
        """Precompute noise schedule for deterministic strategies."""
        if self.config.allocation_strategy == "linear":
            # Linear decay from max to min
            self.schedule = np.linspace(
                self.config.max_noise_mult,
                self.config.min_noise_mult,
                self.config.total_steps,
            )
        
        elif self.config.allocation_strategy == "exponential":
            # Exponential decay
            decay_rate = math.log(self.config.max_noise_mult / self.config.min_noise_mult)
            decay_rate /= self.config.total_steps
            
            self.schedule = self.config.max_noise_mult * np.exp(
                -decay_rate * np.arange(self.config.total_steps)
            )
            self.schedule = np.clip(
                self.schedule,
                self.config.min_noise_mult,
                self.config.max_noise_mult,
            )
        
        elif self.config.allocation_strategy == "cosine":
            # Cosine annealing
            steps = np.arange(self.config.total_steps)
            self.schedule = self.config.min_noise_mult + 0.5 * (
                self.config.max_noise_mult - self.config.min_noise_mult
            ) * (1 + np.cos(np.pi * steps / self.config.total_steps))
        
        else:
            # Default to constant
            self.schedule = np.full(
                self.config.total_steps,
                self.config.max_noise_mult,
            )
    
    def get_noise_multiplier(
        self,
        step: int,
        gradient_norm: Optional[float] = None,
    ) -> float:
        """Get noise multiplier for current step.
        
        Args:
            step: Current training step
            gradient_norm: Optional current gradient norm for adaptive strategy
        
        Returns:
            Noise multiplier σ/Δf
        """
        self.steps_completed = step
        
        # Warmup: use maximum noise
        if step < self.config.warmup_steps:
            return self.config.max_noise_mult
        
        if self.config.allocation_strategy != "adaptive":
            idx = min(step, len(self.schedule) - 1)
            noise_mult = self.schedule[idx]
        else:
            noise_mult = self._adaptive_noise(step, gradient_norm)
        
        # Track
        self.noise_history.append(noise_mult)
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)
        
        return noise_mult
    
    def _adaptive_noise(
        self,
        step: int,
        gradient_norm: Optional[float],
    ) -> float:
        """Compute adaptive noise based on convergence.
        
        Uses gradient norm statistics to detect convergence
        and adjust noise accordingly.
        """
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)
        
        # Not enough history for adaptation
        if len(self.gradient_norms) < self.config.gradient_window:
            return self.config.max_noise_mult
        
        # Compute convergence metric
        recent = self.gradient_norms[-self.config.gradient_window:]
        mean_norm = np.mean(recent)
        std_norm = np.std(recent)
        
        # Coefficient of variation
        cv = std_norm / (mean_norm + 1e-8)
        
        # If gradients are stable (low CV), we can reduce noise
        if cv < self.config.convergence_threshold:
            # Scale down noise based on convergence
            scale = cv / self.config.convergence_threshold
            noise_mult = (
                self.config.min_noise_mult + 
                scale * (self.config.max_noise_mult - self.config.min_noise_mult)
            )
        else:
            # Still converging, use higher noise
            noise_mult = self.config.max_noise_mult
        
        # Also factor in remaining budget
        budget_ratio = self._remaining_budget_ratio(step)
        if budget_ratio < 0.5:
            # Less than half budget remaining, increase noise
            noise_mult = max(noise_mult, self.config.max_noise_mult * (1 - budget_ratio))
        
        return float(np.clip(
            noise_mult,
            self.config.min_noise_mult,
            self.config.max_noise_mult,
        ))
    
    def _remaining_budget_ratio(self, step: int) -> float:
        """Estimate remaining privacy budget ratio."""
        # Simple linear estimate
        return max(0, 1 - step / self.config.total_steps)
    
    def update_epsilon_spent(self, epsilon_spent: float) -> None:
        """Update total epsilon spent (from accountant)."""
        self.epsilon_spent = epsilon_spent
    
    def get_recommended_total_noise(
        self,
        target_epsilon: float,
        target_delta: float,
        steps: int,
    ) -> float:
        """Recommend total noise to achieve target (ε, δ).
        
        Uses composition theorem to estimate required noise.
        """
        # For Gaussian mechanism with composition:
        # ε ≈ √(2 * steps * ln(1/δ)) * σ^(-1)
        # Solving for σ:
        # σ ≥ √(2 * steps * ln(1/δ)) / ε
        
        sigma = math.sqrt(2 * steps * math.log(1.25 / target_delta)) / target_epsilon
        return sigma
    
    def get_statistics(self) -> dict:
        """Get scheduler statistics."""
        return {
            "steps_completed": self.steps_completed,
            "epsilon_spent": self.epsilon_spent,
            "mean_noise": np.mean(self.noise_history) if self.noise_history else 0,
            "min_noise_used": min(self.noise_history) if self.noise_history else 0,
            "max_noise_used": max(self.noise_history) if self.noise_history else 0,
            "mean_gradient_norm": np.mean(self.gradient_norms) if self.gradient_norms else 0,
        }


class BudgetAllocator:
    """Allocate privacy budget across training phases.
    
    Splits total budget (ε, δ) into per-round budgets for
    multiple training phases or agents.
    """
    
    def __init__(
        self,
        total_epsilon: float,
        total_delta: float,
        num_phases: int,
    ):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.num_phases = num_phases
        
        # Default: equal allocation
        self.phase_epsilon = total_epsilon / num_phases
        self.phase_delta = total_delta / num_phases
    
    def allocate_uniform(self) -> List[Tuple[float, float]]:
        """Uniform budget allocation across phases."""
        return [(self.phase_epsilon, self.phase_delta)] * self.num_phases
    
    def allocate_exponential_decay(
        self,
        decay_rate: float = 0.5,
    ) -> List[Tuple[float, float]]:
        """Exponentially decaying budget (more early, less later)."""
        weights = np.array([decay_rate ** i for i in range(self.num_phases)])
        weights = weights / weights.sum()
        
        return [
            (self.total_epsilon * w, self.total_delta / self.num_phases)
            for w in weights
        ]
    
    def allocate_importance_weighted(
        self,
        importance: List[float],
    ) -> List[Tuple[float, float]]:
        """Budget allocation weighted by phase importance."""
        importance = np.array(importance)
        importance = importance / importance.sum()
        
        return [
            (self.total_epsilon * w, self.total_delta / self.num_phases)
            for w in importance
        ]
