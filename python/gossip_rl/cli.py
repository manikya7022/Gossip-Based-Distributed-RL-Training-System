"""CLI entry point for Gossip-RL."""

import click


@click.group()
@click.version_option(version="1.0.0")
def main():
    """Gossip-RL: Distributed RL Training System."""
    pass


@main.command()
@click.option("--config", "-c", default="configs/agent.yaml", help="Configuration file path")
def run(config: str):
    """Run a Gossip-RL agent."""
    from gossip_rl.agent import run_agent
    run_agent(config)


@main.command()
@click.option("--agents", "-n", default=10, help="Number of agents to simulate")
@click.option("--steps", "-s", default=1000, help="Number of training steps")
@click.option("--env", "-e", default="CartPole-v1", help="Environment name")
def benchmark(agents: int, steps: int, env: str):
    """Run convergence benchmark."""
    click.echo(f"Running benchmark with {agents} agents for {steps} steps on {env}")
    # TODO: Implement benchmark
    click.echo("Benchmark not yet implemented")


@main.command()
@click.option("--host", default="localhost", help="Prometheus host")
@click.option("--port", default=8000, help="Metrics port")
def metrics(host: str, port: int):
    """Start metrics server."""
    click.echo(f"Starting metrics server on {host}:{port}")
    from prometheus_client import start_http_server
    start_http_server(port, addr=host)
    click.echo("Metrics server running. Press Ctrl+C to stop.")
    import time
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
