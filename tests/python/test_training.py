"""Tests for RL training components."""

import pytest
import numpy as np
import torch
import gymnasium as gym

from gossip_rl.training.networks import (
    PolicyNetwork,
    ValueNetwork,
    ActorCritic,
    get_gradient_vector,
    set_gradient_vector,
    get_parameter_vector,
    set_parameter_vector,
)
from gossip_rl.training.ppo import DistributedPPO, RolloutBuffer
from gossip_rl.training.replay import ExperienceBuffer, Experience
from gossip_rl.config import TrainingConfig, PrivacyConfig


class TestPolicyNetwork:
    """Tests for PolicyNetwork."""
    
    def test_discrete_action(self):
        net = PolicyNetwork(obs_dim=4, action_dim=2, continuous=False)
        
        obs = torch.randn(10, 4)
        action, log_prob = net.get_action(obs)
        
        assert action.shape == (10,)
        assert log_prob.shape == (10,)
        assert (action >= 0).all() and (action < 2).all()
    
    def test_continuous_action(self):
        net = PolicyNetwork(obs_dim=4, action_dim=2, continuous=True)
        
        obs = torch.randn(10, 4)
        action, log_prob = net.get_action(obs)
        
        assert action.shape == (10, 2)
        assert log_prob.shape == (10,)
    
    def test_deterministic_action(self):
        net = PolicyNetwork(obs_dim=4, action_dim=2, continuous=False)
        
        obs = torch.randn(1, 4)
        action1, _ = net.get_action(obs, deterministic=True)
        action2, _ = net.get_action(obs, deterministic=True)
        
        assert action1.item() == action2.item()
    
    def test_evaluate_actions(self):
        net = PolicyNetwork(obs_dim=4, action_dim=2, continuous=False)
        
        obs = torch.randn(10, 4)
        actions = torch.randint(0, 2, (10,))
        
        log_prob, entropy = net.evaluate_actions(obs, actions)
        
        assert log_prob.shape == (10,)
        assert entropy.shape == (10,)
        assert (entropy >= 0).all()


class TestValueNetwork:
    """Tests for ValueNetwork."""
    
    def test_forward(self):
        net = ValueNetwork(obs_dim=4)
        
        obs = torch.randn(10, 4)
        values = net(obs)
        
        assert values.shape == (10,)


class TestActorCritic:
    """Tests for ActorCritic."""
    
    def test_get_action_and_value(self):
        net = ActorCritic(obs_dim=4, action_dim=2)
        
        obs = torch.randn(10, 4)
        action, value, log_prob, entropy = net.get_action_and_value(obs)
        
        assert action.shape == (10,)
        assert value.shape == (10,)
        assert log_prob.shape == (10,)
        assert entropy.shape == (10,)


class TestGradientUtilities:
    """Tests for gradient vector utilities."""
    
    def test_get_set_gradients(self):
        net = PolicyNetwork(obs_dim=4, action_dim=2)
        
        # Compute gradients
        obs = torch.randn(10, 4)
        action, log_prob = net.get_action(obs)
        loss = -log_prob.mean()
        loss.backward()
        
        # Get gradient vector
        grads = get_gradient_vector(net)
        assert len(grads) > 0
        
        # Modify and set back
        modified_grads = grads * 2
        set_gradient_vector(net, modified_grads)
        
        new_grads = get_gradient_vector(net)
        np.testing.assert_array_almost_equal(new_grads, modified_grads)
    
    def test_get_set_parameters(self):
        net = PolicyNetwork(obs_dim=4, action_dim=2)
        
        # Get parameters
        params = get_parameter_vector(net)
        assert len(params) > 0
        
        # Modify and set back
        modified_params = params * 0.5
        set_parameter_vector(net, modified_params)
        
        new_params = get_parameter_vector(net)
        np.testing.assert_array_almost_equal(new_params, modified_params)


class TestRolloutBuffer:
    """Tests for RolloutBuffer."""
    
    def test_add_and_clear(self):
        buffer = RolloutBuffer()
        
        buffer.add(
            obs=np.array([1, 2, 3, 4]),
            action=np.array([0]),
            reward=1.0,
            value=0.5,
            log_prob=-0.1,
            done=False,
        )
        
        assert len(buffer) == 1
        
        buffer.clear()
        assert len(buffer) == 0


class TestDistributedPPO:
    """Tests for DistributedPPO."""
    
    @pytest.fixture
    def ppo(self):
        config = TrainingConfig(
            learning_rate=3e-4,
            batch_size=32,
            n_steps=64,
            n_epochs=2,
        )
        return DistributedPPO(
            obs_dim=4,
            action_dim=2,
            config=config,
        )
    
    def test_collect_rollout(self, ppo):
        env = gym.make("CartPole-v1")
        
        reward, episodes = ppo.collect_rollout(env, n_steps=100)
        
        assert len(ppo.buffer) == 100
        assert reward >= 0
        
        env.close()
    
    def test_train_step(self, ppo):
        env = gym.make("CartPole-v1")
        
        ppo.collect_rollout(env, n_steps=100)
        stats = ppo.train_step()
        
        assert stats.policy_loss is not None
        assert stats.value_loss is not None
        assert ppo.step == 1
        
        env.close()
    
    def test_gradient_extraction(self, ppo):
        env = gym.make("CartPole-v1")
        
        ppo.collect_rollout(env, n_steps=100)
        ppo.train_step()
        
        gradients = ppo.get_gradients()
        assert len(gradients) > 0
        
        env.close()
    
    def test_with_privacy(self):
        config = TrainingConfig()
        privacy_config = PrivacyConfig(
            enabled=True,
            epsilon=1.0,
            delta=1e-5,
        )
        
        ppo = DistributedPPO(
            obs_dim=4,
            action_dim=2,
            config=config,
            privacy_config=privacy_config,
        )
        
        assert ppo.dp_mechanism is not None
        assert ppo.privacy_accountant is not None
        
        env = gym.make("CartPole-v1")
        ppo.collect_rollout(env, n_steps=100)
        ppo.train_step()
        
        eps, delta = ppo.get_privacy_spent()
        assert eps > 0
        
        env.close()


class TestExperienceBuffer:
    """Tests for ExperienceBuffer."""
    
    def test_add_and_sample(self):
        buffer = ExperienceBuffer(capacity=100)
        
        for i in range(50):
            exp = Experience(
                obs=np.random.randn(4).astype(np.float32),
                action=np.array([0]),
                reward=1.0,
                next_obs=np.random.randn(4).astype(np.float32),
                done=False,
            )
            buffer.add(exp)
        
        assert len(buffer) == 50
        
        samples = buffer.sample(10)
        assert len(samples) == 10
    
    def test_capacity_limit(self):
        buffer = ExperienceBuffer(capacity=10)
        
        for i in range(20):
            exp = Experience(
                obs=np.array([i], dtype=np.float32),
                action=np.array([0]),
                reward=1.0,
                next_obs=np.array([i + 1], dtype=np.float32),
                done=False,
            )
            buffer.add(exp)
        
        assert len(buffer) == 10
        
        # Oldest should be evicted
        assert buffer.buffer[0].obs[0] >= 10
