#!/usr/bin/env python3
"""Benchmark and evaluate Gossip-RL system efficiency.

Generates efficiency metrics including:
- Training throughput (episodes/sec, steps/sec)
- Gossip communication efficiency
- Convergence analysis
- Privacy budget consumption
- Scalability analysis

Usage:
    python scripts/benchmark_efficiency.py --agents 10 --duration 120
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from gossip_rl.config import Config, load_config
from gossip_rl.agent import Agent

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Metrics for a single agent."""
    agent_id: str
    training_steps: int = 0
    episodes_completed: int = 0
    total_reward: float = 0.0
    gossip_rounds: int = 0
    privacy_epsilon_spent: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    avg_step_time_ms: float = 0.0
    rewards_history: List[float] = field(default_factory=list)


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    # Configuration
    num_agents: int = 0
    duration_seconds: float = 0.0
    environment: str = ""
    
    # Throughput metrics
    total_training_steps: int = 0
    total_episodes: int = 0
    steps_per_second: float = 0.0
    episodes_per_second: float = 0.0
    
    # Reward metrics
    avg_reward: float = 0.0
    max_reward: float = 0.0
    min_reward: float = 0.0
    reward_std: float = 0.0
    
    # Gossip metrics
    total_gossip_rounds: int = 0
    gossip_rounds_per_agent: float = 0.0
    gossip_efficiency: float = 0.0  # rounds per step
    
    # Privacy metrics
    avg_epsilon_spent: float = 0.0
    max_epsilon_spent: float = 0.0
    privacy_efficiency: float = 0.0  # reward per epsilon
    
    # Scalability
    agents_completed: int = 0
    throughput_per_agent: float = 0.0
    
    # Per-agent breakdown
    agent_metrics: List[Dict] = field(default_factory=list)
    
    # Timing
    timestamp: str = ""
    wall_clock_time: float = 0.0


class EfficiencyBenchmark:
    """Benchmark system for measuring Gossip-RL efficiency."""
    
    def __init__(self, num_agents: int, base_port: int = 6000):
        self.num_agents = num_agents
        self.base_port = base_port
        self.agents: List[Agent] = []
        self.agent_metrics: List[AgentMetrics] = []
        self.running = False
        
    def create_agent_config(self, agent_id: int) -> Config:
        """Create configuration for a specific agent."""
        base_config = load_config("configs/agent.yaml")
        base_config.agent.id = f"agent-{agent_id}"
        base_config.network.rpc_port = self.base_port + agent_id
        base_config.privacy.epsilon = 10.0  # Higher budget for benchmarking
        return base_config
    
    async def run_agent_benchmark(self, agent: Agent, metrics: AgentMetrics, duration: int) -> None:
        """Run benchmark for a single agent."""
        try:
            await agent.membership.start()
            await agent.gossip.start()
            
            metrics.start_time = time.time()
            step_times = []
            
            while self.running and (time.time() - metrics.start_time) < duration:
                step_start = time.time()
                
                # Collect rollout
                reward, episodes = agent.trainer.collect_rollout(
                    agent.env,
                    agent.config.training.n_steps,
                )
                metrics.episodes_completed += episodes
                metrics.total_reward += reward
                metrics.rewards_history.append(reward / max(episodes, 1))
                
                # Train
                agent.trainer.train_step()
                metrics.training_steps += 1
                
                # Gossip exchange
                gradients = agent.trainer.get_gradients()
                if len(gradients) > 0:
                    agent.gossip.set_gradients(
                        gradients,
                        model_version=agent.trainer.model_version,
                        step=agent.trainer.step,
                    )
                
                aggregated, _ = await agent.gossip.get_aggregated_gradients()
                if len(aggregated) > 0:
                    agent.trainer.apply_aggregated_gradients(aggregated)
                
                step_times.append((time.time() - step_start) * 1000)
                await asyncio.sleep(0.001)
            
            metrics.end_time = time.time()
            
            # Collect final metrics
            gossip_stats = await agent.gossip.get_stats()
            metrics.gossip_rounds = gossip_stats["rounds_completed"]
            
            eps, _ = agent.trainer.get_privacy_spent()
            metrics.privacy_epsilon_spent = eps
            
            if step_times:
                metrics.avg_step_time_ms = sum(step_times) / len(step_times)
            
        except Exception as e:
            logger.error(f"{metrics.agent_id} error: {e}")
        finally:
            await agent.gossip.stop()
            await agent.membership.stop()
            agent.env.close()
    
    async def run_benchmark(self, duration: int = 120) -> BenchmarkResults:
        """Run full benchmark across all agents."""
        self.running = True
        results = BenchmarkResults(
            num_agents=self.num_agents,
            duration_seconds=duration,
            timestamp=datetime.now().isoformat(),
        )
        
        print(f"\n{'='*60}")
        print(f"GOSSIP-RL EFFICIENCY BENCHMARK")
        print(f"{'='*60}")
        print(f"Agents: {self.num_agents}")
        print(f"Duration: {duration}s")
        print(f"{'='*60}\n")
        
        # Create agents
        print("Creating agents...", end=" ", flush=True)
        for i in range(self.num_agents):
            config = self.create_agent_config(i)
            results.environment = config.environment.name
            agent = Agent(config)
            self.agents.append(agent)
            
            metrics = AgentMetrics(agent_id=f"agent-{i}")
            self.agent_metrics.append(metrics)
        print(f"Done ({self.num_agents} agents)")
        
        # Run benchmark
        print(f"Running benchmark for {duration}s...\n")
        start_wall = time.time()
        
        tasks = [
            self.run_agent_benchmark(agent, metrics, duration)
            for agent, metrics in zip(self.agents, self.agent_metrics)
        ]
        await asyncio.gather(*tasks)
        
        results.wall_clock_time = time.time() - start_wall
        
        # Aggregate results
        self._aggregate_results(results)
        
        return results
    
    def _aggregate_results(self, results: BenchmarkResults) -> None:
        """Aggregate metrics from all agents."""
        import numpy as np
        
        completed = [m for m in self.agent_metrics if m.training_steps > 0]
        results.agents_completed = len(completed)
        
        if not completed:
            return
        
        # Throughput
        results.total_training_steps = sum(m.training_steps for m in completed)
        results.total_episodes = sum(m.episodes_completed for m in completed)
        results.steps_per_second = results.total_training_steps / results.wall_clock_time
        results.episodes_per_second = results.total_episodes / results.wall_clock_time
        
        # Rewards
        all_rewards = []
        for m in completed:
            if m.episodes_completed > 0:
                avg = m.total_reward / m.episodes_completed
                all_rewards.append(avg)
        
        if all_rewards:
            results.avg_reward = np.mean(all_rewards)
            results.max_reward = np.max(all_rewards)
            results.min_reward = np.min(all_rewards)
            results.reward_std = np.std(all_rewards)
        
        # Gossip
        results.total_gossip_rounds = sum(m.gossip_rounds for m in completed)
        results.gossip_rounds_per_agent = results.total_gossip_rounds / len(completed)
        if results.total_training_steps > 0:
            results.gossip_efficiency = results.total_gossip_rounds / results.total_training_steps
        
        # Privacy
        epsilons = [m.privacy_epsilon_spent for m in completed if m.privacy_epsilon_spent > 0]
        if epsilons:
            results.avg_epsilon_spent = np.mean(epsilons)
            results.max_epsilon_spent = np.max(epsilons)
            if results.avg_epsilon_spent > 0:
                results.privacy_efficiency = results.avg_reward / results.avg_epsilon_spent
        
        # Scalability
        results.throughput_per_agent = results.episodes_per_second / len(completed)
        
        # Per-agent metrics
        results.agent_metrics = [
            {
                "agent_id": m.agent_id,
                "steps": m.training_steps,
                "episodes": m.episodes_completed,
                "avg_reward": m.total_reward / max(m.episodes_completed, 1),
                "gossip_rounds": m.gossip_rounds,
                "epsilon_spent": m.privacy_epsilon_spent,
                "avg_step_time_ms": m.avg_step_time_ms,
            }
            for m in completed
        ]
    
    def stop(self):
        self.running = False


def print_results(results: BenchmarkResults):
    """Print formatted benchmark results."""
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}\n")
    
    print("CONFIGURATION")
    print("-" * 40)
    print(f"  Agents:              {results.num_agents}")
    print(f"  Environment:         {results.environment}")
    print(f"  Duration:            {results.duration_seconds:.1f}s")
    print(f"  Wall Clock Time:     {results.wall_clock_time:.1f}s")
    
    print(f"\nTHROUGHPUT METRICS")
    print("-" * 40)
    print(f"  Training Steps:      {results.total_training_steps}")
    print(f"  Episodes Completed:  {results.total_episodes}")
    print(f"  Steps/Second:        {results.steps_per_second:.2f}")
    print(f"  Episodes/Second:     {results.episodes_per_second:.2f}")
    print(f"  Throughput/Agent:    {results.throughput_per_agent:.2f} eps/s")
    
    print(f"\nREWARD METRICS")
    print("-" * 40)
    print(f"  Average Reward:      {results.avg_reward:.2f}")
    print(f"  Max Reward:          {results.max_reward:.2f}")
    print(f"  Min Reward:          {results.min_reward:.2f}")
    print(f"  Reward Std Dev:      {results.reward_std:.2f}")
    
    print(f"\nGOSSIP EFFICIENCY")
    print("-" * 40)
    print(f"  Total Gossip Rounds: {results.total_gossip_rounds}")
    print(f"  Rounds/Agent:        {results.gossip_rounds_per_agent:.1f}")
    print(f"  Rounds/Step:         {results.gossip_efficiency:.2f}")
    
    print(f"\nPRIVACY METRICS (Differential Privacy)")
    print("-" * 40)
    print(f"  Avg ε Spent:         {results.avg_epsilon_spent:.4f}")
    print(f"  Max ε Spent:         {results.max_epsilon_spent:.4f}")
    print(f"  Privacy Efficiency:  {results.privacy_efficiency:.2f} reward/ε")
    
    print(f"\nSCALABILITY")
    print("-" * 40)
    print(f"  Agents Completed:    {results.agents_completed}/{results.num_agents}")
    print(f"  Success Rate:        {100*results.agents_completed/results.num_agents:.1f}%")
    
    print(f"\n{'='*60}")
    print("TOP 5 AGENTS BY EPISODES")
    print("-" * 60)
    print(f"{'Agent':<12} {'Steps':<8} {'Episodes':<10} {'Reward':<10} {'ε Spent':<10}")
    print("-" * 60)
    
    sorted_agents = sorted(results.agent_metrics, key=lambda x: x["episodes"], reverse=True)[:5]
    for a in sorted_agents:
        print(f"{a['agent_id']:<12} {a['steps']:<8} {a['episodes']:<10} {a['avg_reward']:<10.2f} {a['epsilon_spent']:<10.4f}")
    
    print(f"{'='*60}\n")


def save_results(results: BenchmarkResults, output_path: str):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(asdict(results), f, indent=2)
    print(f"Results saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description='Benchmark Gossip-RL efficiency')
    parser.add_argument('--agents', '-n', type=int, default=10,
                        help='Number of agents (default: 10)')
    parser.add_argument('--duration', '-d', type=int, default=120,
                        help='Duration in seconds (default: 120)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file path')
    
    args = parser.parse_args()
    
    benchmark = EfficiencyBenchmark(args.agents)
    
    import signal
    def signal_handler(sig, frame):
        print("\nStopping benchmark...")
        benchmark.stop()
    signal.signal(signal.SIGINT, signal_handler)
    
    results = await benchmark.run_benchmark(args.duration)
    print_results(results)
    
    if args.output:
        save_results(results, args.output)
    else:
        # Default save location
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/benchmark_{args.agents}agents_{timestamp}.json"
        save_results(results, output_path)


if __name__ == '__main__':
    asyncio.run(main())
