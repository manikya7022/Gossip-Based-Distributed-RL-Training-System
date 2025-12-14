#!/usr/bin/env python3
"""Launch multiple gossip-rl agents for distributed training simulation.

Usage:
    python scripts/launch_multi_agent.py --agents 50 --duration 60
"""

import argparse
import asyncio
import logging
import os
import sys
import signal
import time
from typing import List, Optional

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from gossip_rl.config import Config, load_config
from gossip_rl.agent import Agent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class MultiAgentSimulator:
    """Simulate multiple gossip-rl agents in a single process."""
    
    def __init__(self, num_agents: int, base_port: int = 5000):
        self.num_agents = num_agents
        self.base_port = base_port
        self.agents: List[Agent] = []
        self.running = False
        
    def create_agent_config(self, agent_id: int) -> Config:
        """Create configuration for a specific agent."""
        # Load base config
        base_config = load_config("configs/agent.yaml")
        
        # Customize for this agent
        base_config.agent.id = f"agent-{agent_id}"
        base_config.network.rpc_port = self.base_port + agent_id
        
        # Increase privacy budget for longer simulation
        base_config.privacy.epsilon = 10.0
        
        return base_config
    
    async def run_agent(self, agent: Agent, duration: int) -> dict:
        """Run a single agent for specified duration."""
        agent_id = agent.config.agent.id
        try:
            # Start gossip components
            await agent.membership.start()
            await agent.gossip.start()
            
            start_time = time.time()
            step_count = 0
            total_episodes = 0
            total_reward = 0.0
            
            while self.running and (time.time() - start_time) < duration:
                # Collect rollout
                reward, episodes = agent.trainer.collect_rollout(
                    agent.env,
                    agent.config.training.n_steps,
                )
                total_episodes += episodes
                total_reward += reward
                
                # Train
                stats = agent.trainer.train_step()
                step_count += 1
                
                # Update gossip with gradients
                gradients = agent.trainer.get_gradients()
                if len(gradients) > 0:
                    agent.gossip.set_gradients(
                        gradients,
                        model_version=agent.trainer.model_version,
                        step=agent.trainer.step,
                    )
                
                # Get aggregated gradients
                aggregated, n_contributors = await agent.gossip.get_aggregated_gradients()
                if n_contributors > 1 and len(aggregated) > 0:
                    agent.trainer.apply_aggregated_gradients(aggregated)
                
                # Brief yield
                await asyncio.sleep(0.001)
            
            gossip_stats = await agent.gossip.get_stats()
            
            return {
                "agent_id": agent_id,
                "steps": step_count,
                "episodes": total_episodes,
                "avg_reward": total_reward / max(total_episodes, 1),
                "gossip_rounds": gossip_stats["rounds_completed"],
                "peers_contacted": gossip_stats.get("peers_contacted", 0),
            }
            
        except Exception as e:
            logger.error(f"{agent_id} error: {e}")
            return {"agent_id": agent_id, "error": str(e)}
        finally:
            await agent.gossip.stop()
            await agent.membership.stop()
            agent.env.close()
    
    async def run_simulation(self, duration: int = 60) -> List[dict]:
        """Run all agents concurrently."""
        self.running = True
        
        logger.info(f"Starting {self.num_agents} agents...")
        
        # Create agents
        for i in range(self.num_agents):
            config = self.create_agent_config(i)
            agent = Agent(config)
            self.agents.append(agent)
            if (i + 1) % 10 == 0:
                logger.info(f"  Created {i + 1}/{self.num_agents} agents")
        
        logger.info(f"All {self.num_agents} agents created. Starting training for {duration}s...")
        
        # Run all agents concurrently
        tasks = [
            self.run_agent(agent, duration)
            for agent in self.agents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r if isinstance(r, dict) else {"error": str(r)} for r in results]
    
    def stop(self):
        """Stop all agents."""
        self.running = False


def print_results(results: List[dict], duration: int):
    """Print simulation results summary."""
    print("\n" + "="*70)
    print(f"MULTI-AGENT SIMULATION RESULTS ({len(results)} agents, {duration}s)")
    print("="*70)
    
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    
    if successful:
        total_steps = sum(r["steps"] for r in successful)
        total_episodes = sum(r["episodes"] for r in successful)
        total_gossip = sum(r["gossip_rounds"] for r in successful)
        avg_reward = sum(r["avg_reward"] for r in successful) / len(successful)
        
        print(f"\nAgents completed: {len(successful)}/{len(results)}")
        print(f"Total training steps: {total_steps}")
        print(f"Total episodes: {total_episodes}")
        print(f"Total gossip rounds: {total_gossip}")
        print(f"Average reward: {avg_reward:.2f}")
        
        print("\nPer-Agent Summary (top 10):")
        print("-"*70)
        print(f"{'Agent':<12} {'Steps':<8} {'Episodes':<10} {'Reward':<10} {'Gossip':<10}")
        print("-"*70)
        
        for r in sorted(successful, key=lambda x: x["steps"], reverse=True)[:10]:
            print(f"{r['agent_id']:<12} {r['steps']:<8} {r['episodes']:<10} {r['avg_reward']:<10.2f} {r['gossip_rounds']:<10}")
    
    if failed:
        print(f"\nFailed agents: {len(failed)}")
        for r in failed[:5]:
            print(f"  - {r.get('agent_id', 'unknown')}: {r.get('error', 'unknown error')}")
    
    print("="*70)


async def main():
    parser = argparse.ArgumentParser(description='Launch multi-agent gossip-rl simulation')
    parser.add_argument('--agents', '-n', type=int, default=50,
                        help='Number of agents (default: 50)')
    parser.add_argument('--duration', '-d', type=int, default=60,
                        help='Duration in seconds (default: 60)')
    parser.add_argument('--base-port', '-p', type=int, default=5000,
                        help='Base port for agents (default: 5000)')
    
    args = parser.parse_args()
    
    simulator = MultiAgentSimulator(args.agents, args.base_port)
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        logger.info("Stopping simulation...")
        simulator.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"\nStarting {args.agents}-agent simulation for {args.duration} seconds...")
    print("Press Ctrl+C to stop early.\n")
    
    start_time = time.time()
    results = await simulator.run_simulation(args.duration)
    elapsed = time.time() - start_time
    
    print_results(results, int(elapsed))


if __name__ == '__main__':
    asyncio.run(main())
