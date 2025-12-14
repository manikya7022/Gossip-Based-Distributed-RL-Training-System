"""Tests for differential privacy components."""

import math
import pytest
import numpy as np

from gossip_rl.privacy.mechanism import (
    DPParams,
    GaussianMechanism,
    LaplaceMechanism,
    compute_sigma_for_rdp,
)
from gossip_rl.privacy.accountant import RDPAccountant
from gossip_rl.privacy.adaptive import AdaptiveNoiseScheduler, NoiseScheduleConfig


class TestGaussianMechanism:
    """Tests for GaussianMechanism."""
    
    def test_noise_multiplier_computation(self):
        params = DPParams(epsilon=1.0, delta=1e-5, clip_norm=1.0)
        mechanism = GaussianMechanism(params)
        
        # Expected: sqrt(2 * ln(1.25/1e-5)) / 1.0 ≈ 5.02
        assert 4.5 < mechanism.noise_multiplier < 5.5
    
    def test_gradient_clipping(self):
        params = DPParams(epsilon=1.0, delta=1e-5, clip_norm=1.0)
        mechanism = GaussianMechanism(params)
        
        # Create gradient with norm > 1
        gradients = np.ones(100) * 0.5  # norm ≈ 5.0
        
        clipped, original_norm = mechanism.clip_gradients(gradients)
        
        assert original_norm > 1.0
        assert np.linalg.norm(clipped) <= 1.0 + 1e-6
    
    def test_noise_addition(self):
        params = DPParams(epsilon=1.0, delta=1e-5, clip_norm=1.0)
        mechanism = GaussianMechanism(params)
        
        gradients = np.zeros(1000)
        noisy = mechanism.add_noise(gradients)
        
        # Noise should have expected standard deviation
        expected_std = mechanism.noise_multiplier * params.clip_norm
        actual_std = np.std(noisy)
        
        # Allow 20% tolerance
        assert abs(actual_std - expected_std) / expected_std < 0.2
    
    def test_privatize_pipeline(self):
        params = DPParams(epsilon=1.0, delta=1e-5, clip_norm=1.0)
        mechanism = GaussianMechanism(params)
        
        gradients = np.random.randn(100)
        noisy, metadata = mechanism.privatize(gradients)
        
        assert 'original_norm' in metadata
        assert 'clipped' in metadata
        assert 'noise_scale' in metadata


class TestLaplaceMechanism:
    """Tests for LaplaceMechanism."""
    
    def test_noise_multiplier(self):
        params = DPParams(epsilon=1.0, delta=0.0, clip_norm=1.0)
        mechanism = LaplaceMechanism(params)
        
        # For Laplace: b/Δf = 1/ε
        assert mechanism.noise_multiplier == 1.0
    
    def test_noise_distribution(self):
        params = DPParams(epsilon=1.0, delta=0.0, clip_norm=1.0)
        mechanism = LaplaceMechanism(params)
        
        gradients = np.zeros(10000)
        noisy = mechanism.add_noise(gradients)
        
        # Laplace noise has higher kurtosis than Gaussian
        kurtosis = np.mean((noisy - noisy.mean()) ** 4) / np.std(noisy) ** 4
        # Laplace kurtosis = 6 (excess kurtosis = 3)
        assert kurtosis > 4.0


class TestRDPAccountant:
    """Tests for RDPAccountant."""
    
    def test_single_step(self):
        accountant = RDPAccountant()
        
        accountant.step(noise_multiplier=1.0, sample_rate=0.01)
        
        eps = accountant.get_epsilon(delta=1e-5)
        assert eps > 0
        assert accountant.step_count == 1
    
    def test_composition(self):
        accountant = RDPAccountant()
        
        # Multiple steps should accumulate privacy
        for _ in range(10):
            accountant.step(noise_multiplier=1.0, sample_rate=0.01)
        
        eps_10 = accountant.get_epsilon(delta=1e-5)
        
        for _ in range(10):
            accountant.step(noise_multiplier=1.0, sample_rate=0.01)
        
        eps_20 = accountant.get_epsilon(delta=1e-5)
        
        # Privacy should increase with more steps
        assert eps_20 > eps_10
    
    def test_higher_noise_lower_epsilon(self):
        accountant1 = RDPAccountant()
        accountant2 = RDPAccountant()
        
        # Same steps, different noise
        for _ in range(100):
            accountant1.step(noise_multiplier=1.0, sample_rate=0.01)
            accountant2.step(noise_multiplier=2.0, sample_rate=0.01)
        
        eps1 = accountant1.get_epsilon(delta=1e-5)
        eps2 = accountant2.get_epsilon(delta=1e-5)
        
        # Higher noise = lower epsilon (more privacy)
        assert eps2 < eps1


class TestAdaptiveNoiseScheduler:
    """Tests for AdaptiveNoiseScheduler."""
    
    def test_warmup_phase(self):
        config = NoiseScheduleConfig(
            warmup_steps=100,
            max_noise_mult=2.0,
        )
        scheduler = AdaptiveNoiseScheduler(config)
        
        # During warmup, should use max noise
        noise = scheduler.get_noise_multiplier(50)
        assert noise == config.max_noise_mult
    
    def test_linear_schedule(self):
        config = NoiseScheduleConfig(
            warmup_steps=0,
            total_steps=100,
            min_noise_mult=0.5,
            max_noise_mult=2.0,
            allocation_strategy="linear",
        )
        scheduler = AdaptiveNoiseScheduler(config)
        
        # Start should be max
        noise_start = scheduler.get_noise_multiplier(0)
        assert abs(noise_start - 2.0) < 0.1
        
        # End should be min
        noise_end = scheduler.get_noise_multiplier(99)
        assert abs(noise_end - 0.5) < 0.1
    
    def test_exponential_schedule(self):
        config = NoiseScheduleConfig(
            warmup_steps=0,
            total_steps=100,
            min_noise_mult=0.1,
            max_noise_mult=2.0,
            allocation_strategy="exponential",
        )
        scheduler = AdaptiveNoiseScheduler(config)
        
        noise_0 = scheduler.get_noise_multiplier(0)
        noise_50 = scheduler.get_noise_multiplier(50)
        noise_99 = scheduler.get_noise_multiplier(99)
        
        # Should decay exponentially
        assert noise_0 > noise_50 > noise_99
    
    def test_adaptive_schedule(self):
        config = NoiseScheduleConfig(
            warmup_steps=0,
            total_steps=1000,
            gradient_window=10,
            convergence_threshold=0.1,
            allocation_strategy="adaptive",
        )
        scheduler = AdaptiveNoiseScheduler(config)
        
        # Initially high variance gradients
        for i in range(50):
            scheduler.get_noise_multiplier(i, gradient_norm=1.0 + np.random.randn() * 0.5)
        
        noise_high_var = scheduler.get_noise_multiplier(
            50, gradient_norm=1.0 + np.random.randn() * 0.5
        )
        
        # Now stable gradients
        for i in range(100):
            scheduler.get_noise_multiplier(100 + i, gradient_norm=1.0)
        
        noise_low_var = scheduler.get_noise_multiplier(200, gradient_norm=1.0)
        
        # Lower variance should allow lower noise
        assert noise_low_var < noise_high_var


class TestSigmaComputation:
    """Tests for compute_sigma_for_rdp helper."""
    
    @pytest.mark.slow
    def test_compute_sigma_achieves_target(self):
        target_epsilon = 1.0
        target_delta = 1e-5
        steps = 100
        sample_rate = 0.01
        
        sigma = compute_sigma_for_rdp(
            q=sample_rate,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            steps=steps,
        )
        
        # Verify with accountant
        accountant = RDPAccountant()
        for _ in range(steps):
            accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
        
        achieved_eps = accountant.get_epsilon(target_delta)
        
        # Should be close to target
        assert achieved_eps <= target_epsilon * 1.1
