"""Gossip-RL: Gossip-Based Distributed RL Training System."""

__version__ = "1.0.0"

from gossip_rl.config import Config, load_config
from gossip_rl.agent import Agent

__all__ = ["Config", "load_config", "Agent", "__version__"]
