"""Main Agent implementation."""

import asyncio
import logging
import signal
import sys
from typing import Optional

import gymnasium as gym
import numpy as np

from gossip_rl.config import Config, load_config
from gossip_rl.gossip import GossipProtocol, MembershipManager, Peer
from gossip_rl.training import DistributedPPO
from gossip_rl.privacy import RDPAccountant

logger = logging.getLogger(__name__)


class Agent:
    """Gossip-RL Agent.
    
    Combines:
    - Local RL training (PPO)
    - Gossip protocol for gradient sharing
    - Differential privacy
    - Metrics and monitoring
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Set up logging
        log_level = getattr(logging, config.logging.level.upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        
        # Environment
        self.env = gym.make(config.environment.name)
        obs_dim = self.env.observation_space.shape[0]
        
        if hasattr(self.env.action_space, 'n'):
            action_dim = self.env.action_space.n
            continuous = False
        else:
            action_dim = self.env.action_space.shape[0]
            continuous = True
        
        # Local peer
        self.local_peer = Peer(
            id=config.agent.id,
            host="localhost",
            port=config.network.rpc_port,
        )
        
        # Membership manager
        self.membership = MembershipManager(
            local_peer=self.local_peer,
            probe_interval_ms=config.gossip.failure_detection.probe_interval_ms,
            probe_timeout_ms=config.gossip.failure_detection.probe_timeout_ms,
            indirect_probes=config.gossip.failure_detection.indirect_probes,
            on_peer_alive=self._on_peer_alive,
            on_peer_dead=self._on_peer_dead,
        )
        
        # Gossip protocol
        self.gossip = GossipProtocol(
            local_peer=self.local_peer,
            config=config.gossip,
            membership=self.membership,
            on_gradient_received=self._on_gradient_received,
        )
        
        # RL trainer
        self.trainer = DistributedPPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config.training,
            privacy_config=config.privacy if config.privacy.enabled else None,
        )
        
        self._running = False
        self._total_steps = 0
        self._total_episodes = 0
    
    def _on_peer_alive(self, peer: Peer) -> None:
        """Callback when peer becomes alive."""
        logger.info(f"Peer alive: {peer.id}")
    
    def _on_peer_dead(self, peer: Peer) -> None:
        """Callback when peer dies."""
        logger.warning(f"Peer dead: {peer.id}")
    
    def _on_gradient_received(
        self,
        gradients: np.ndarray,
        model_version: int,
        step: int,
    ) -> None:
        """Callback when gradients received from gossip."""
        logger.debug(f"Received gradients: version={model_version}, step={step}")
    
    async def add_peer(self, host: str, port: int, peer_id: Optional[str] = None) -> bool:
        """Add a peer to the gossip network."""
        peer = Peer(
            id=peer_id or f"{host}:{port}",
            host=host,
            port=port,
        )
        return await self.membership.add_peer(peer)
    
    async def run(self) -> None:
        """Main training loop."""
        self._running = True
        
        # Start gossip
        await self.membership.start()
        await self.gossip.start()
        
        logger.info(f"Agent {self.config.agent.id} started training on {self.config.environment.name}")
        
        try:
            while self._running:
                # Collect rollout
                reward, episodes = self.trainer.collect_rollout(
                    self.env,
                    self.config.training.n_steps,
                )
                self._total_steps += self.config.training.n_steps
                self._total_episodes += episodes
                
                # Train
                stats = self.trainer.train_step()
                
                # Update gossip with new gradients
                gradients = self.trainer.get_gradients()
                if len(gradients) > 0:
                    self.gossip.set_gradients(
                        gradients,
                        model_version=self.trainer.model_version,
                        step=self.trainer.step,
                    )
                
                # Apply aggregated gradients if available
                aggregated, n_contributors = await self.gossip.get_aggregated_gradients()
                if n_contributors > 1 and len(aggregated) > 0:
                    self.trainer.apply_aggregated_gradients(aggregated)
                
                # Log progress
                if self.trainer.step % 10 == 0:
                    eps, delta = self.trainer.get_privacy_spent()
                    gossip_stats = await self.gossip.get_stats()
                    
                    logger.info(
                        f"Step {self.trainer.step} | "
                        f"Episodes: {self._total_episodes} | "
                        f"Reward: {stats.mean_reward:.2f} | "
                        f"Policy Loss: {stats.policy_loss:.4f} | "
                        f"KL: {stats.approx_kl:.4f} | "
                        f"ε spent: {eps:.4f} | "
                        f"Gossip rounds: {gossip_stats['rounds_completed']}"
                    )
                
                # Save checkpoint every 50 steps
                if self.trainer.step % 50 == 0 and self.trainer.step > 0:
                    import os
                    os.makedirs("checkpoints", exist_ok=True)
                    checkpoint_path = f"checkpoints/policy_step_{self.trainer.step}.pt"
                    self.trainer.save_checkpoint(checkpoint_path)
                    logger.info(f"Saved checkpoint: {checkpoint_path}")
                
                # Check privacy budget
                if self.config.privacy.enabled:
                    eps, _ = self.trainer.get_privacy_spent()
                    if eps > self.config.privacy.epsilon * 0.95:
                        logger.warning("Privacy budget nearly exhausted!")
                
                # Small delay to allow gossip
                await asyncio.sleep(0.01)
                
        except asyncio.CancelledError:
            logger.info("Training cancelled")
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the agent."""
        self._running = False
        await self.gossip.stop()
        await self.membership.stop()
        self.env.close()
        logger.info("Agent stopped")
    
    def get_stats(self) -> dict:
        """Get agent statistics."""
        eps, delta = self.trainer.get_privacy_spent()
        return {
            "agent_id": self.config.agent.id,
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
            "model_version": self.trainer.model_version,
            "epsilon_spent": eps,
            "delta": delta,
        }


def run_agent(config_path: str = "configs/agent.yaml") -> None:
    """Run agent from configuration file."""
    config = load_config(config_path)
    agent = Agent(config)
    
    # Handle signals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    def handle_signal():
        logger.info("Received shutdown signal")
        loop.create_task(agent.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)
    
    try:
        loop.run_until_complete(agent.run())
    finally:
        loop.close()


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/agent.yaml"
    run_agent(config_path)
