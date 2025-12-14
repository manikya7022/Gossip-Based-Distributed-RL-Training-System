"""Differential Privacy Mechanisms for gradient perturbation."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class DPParams:
    """Differential privacy parameters."""
    epsilon: float  # Privacy budget
    delta: float    # Probability of failure
    clip_norm: float = 1.0  # Gradient clipping norm
    noise_multiplier: Optional[float] = None  # Override computed noise


class DPMechanism(ABC):
    """Abstract base class for DP mechanisms."""
    
    def __init__(self, params: DPParams):
        self.params = params
        self._noise_multiplier: Optional[float] = None
    
    @property
    def noise_multiplier(self) -> float:
        """Get the noise multiplier (sigma/sensitivity)."""
        if self.params.noise_multiplier is not None:
            return self.params.noise_multiplier
        if self._noise_multiplier is None:
            self._noise_multiplier = self._compute_noise_multiplier()
        return self._noise_multiplier
    
    @abstractmethod
    def _compute_noise_multiplier(self) -> float:
        """Compute noise multiplier from (ε, δ) parameters."""
        pass
    
    @abstractmethod
    def add_noise(self, gradients: np.ndarray) -> np.ndarray:
        """Add calibrated noise to gradients."""
        pass
    
    def clip_gradients(self, gradients: np.ndarray) -> Tuple[np.ndarray, float]:
        """Clip gradients to bound sensitivity.
        
        Returns:
            Tuple of (clipped gradients, original gradient norm)
        """
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > self.params.clip_norm:
            gradients = gradients * (self.params.clip_norm / grad_norm)
        return gradients, grad_norm
    
    def privatize(self, gradients: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Full privatization pipeline: clip + noise.
        
        Returns:
            Tuple of (privatized gradients, metadata dict)
        """
        clipped, original_norm = self.clip_gradients(gradients.copy())
        noisy = self.add_noise(clipped)
        
        metadata = {
            "original_norm": float(original_norm),
            "clipped": original_norm > self.params.clip_norm,
            "clip_norm": self.params.clip_norm,
            "noise_scale": self.noise_multiplier * self.params.clip_norm,
        }
        
        return noisy, metadata


class GaussianMechanism(DPMechanism):
    """Gaussian mechanism for (ε, δ)-differential privacy.
    
    Adds Gaussian noise calibrated to achieve (ε, δ)-DP:
        σ ≥ √(2 ln(1.25/δ)) * (Δf / ε)
    
    where Δf is the L2 sensitivity (clip_norm).
    """
    
    def _compute_noise_multiplier(self) -> float:
        """Compute σ/Δf from (ε, δ).
        
        Using the analytic Gaussian mechanism formula:
        σ ≥ √(2 ln(1.25/δ)) * Δf / ε
        """
        eps = self.params.epsilon
        delta = self.params.delta
        
        # Analytic formula for Gaussian mechanism
        # σ/Δf = √(2 ln(1.25/δ)) / ε
        noise_mult = math.sqrt(2 * math.log(1.25 / delta)) / eps
        
        return noise_mult
    
    def add_noise(self, gradients: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to gradients."""
        sigma = self.noise_multiplier * self.params.clip_norm
        noise = np.random.normal(0, sigma, gradients.shape)
        return gradients + noise.astype(gradients.dtype)


class LaplaceMechanism(DPMechanism):
    """Laplace mechanism for ε-differential privacy.
    
    Adds Laplace noise with scale b = Δf/ε
    Pure ε-DP (δ = 0).
    """
    
    def _compute_noise_multiplier(self) -> float:
        """Compute b/Δf from ε."""
        return 1.0 / self.params.epsilon
    
    def add_noise(self, gradients: np.ndarray) -> np.ndarray:
        """Add Laplace noise to gradients."""
        scale = self.noise_multiplier * self.params.clip_norm
        noise = np.random.laplace(0, scale, gradients.shape)
        return gradients + noise.astype(gradients.dtype)


class DiscretizedGaussian(GaussianMechanism):
    """Discretized Gaussian mechanism for integer-valued outputs.
    
    Useful for quantized gradients.
    """
    
    def __init__(self, params: DPParams, quantization_levels: int = 256):
        super().__init__(params)
        self.quantization_levels = quantization_levels
    
    def add_noise(self, gradients: np.ndarray) -> np.ndarray:
        """Add Gaussian noise and discretize."""
        sigma = self.noise_multiplier * self.params.clip_norm
        noise = np.random.normal(0, sigma, gradients.shape)
        noisy = gradients + noise
        
        # Discretize to fixed levels
        noisy = np.round(noisy * (self.quantization_levels / 2))
        noisy = np.clip(noisy, -self.quantization_levels // 2, self.quantization_levels // 2 - 1)
        noisy = noisy / (self.quantization_levels / 2)
        
        return noisy.astype(gradients.dtype)


def compute_sigma_for_rdp(
    q: float,  # Sampling probability
    target_epsilon: float,
    target_delta: float,
    steps: int,
    orders: Optional[list] = None,
    max_sigma: float = 1000.0,
    tolerance: float = 0.01,
) -> float:
    """Compute σ to achieve (ε, δ)-DP after `steps` iterations.
    
    Uses RDP composition and binary search.
    
    Args:
        q: Subsampling probability (batch_size / dataset_size)
        target_epsilon: Target ε
        target_delta: Target δ
        steps: Number of training steps
        orders: RDP orders to compute (default: range of useful orders)
        max_sigma: Maximum σ to search
        tolerance: Binary search tolerance
    
    Returns:
        Required noise multiplier σ
    """
    from gossip_rl.privacy.accountant import RDPAccountant
    
    if orders is None:
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    
    def check_sigma(sigma: float) -> bool:
        accountant = RDPAccountant(orders=orders)
        for _ in range(steps):
            accountant.step(noise_multiplier=sigma, sample_rate=q)
        eps = accountant.get_epsilon(target_delta)
        return eps <= target_epsilon
    
    # Binary search for minimum σ
    low, high = 0.1, max_sigma
    
    while high - low > tolerance:
        mid = (low + high) / 2
        if check_sigma(mid):
            high = mid
        else:
            low = mid
    
    return high
