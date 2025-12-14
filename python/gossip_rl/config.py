"""Configuration management for Gossip-RL."""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class NetworkConfig(BaseModel):
    """Network configuration."""
    rpc_port: int = 50051
    use_shared_memory: bool = True
    tcp_nodelay: bool = True
    buffer_size: int = 65536
    max_connections: int = 100


class FailureDetectionConfig(BaseModel):
    """Failure detection configuration."""
    protocol: str = "swim"
    probe_interval_ms: int = 500
    probe_timeout_ms: int = 200
    indirect_probes: int = 3


class GossipConfig(BaseModel):
    """Gossip protocol configuration."""
    interval_ms: int = 100
    fanout: int = 3
    push_pull: bool = True
    failure_detection: FailureDetectionConfig = Field(default_factory=FailureDetectionConfig)


class CompressionConfig(BaseModel):
    """Gradient compression configuration."""
    enabled: bool = True
    method: str = "topk"  # topk, random, none
    ratio: float = 0.01  # Keep top 1%


class QuantizationConfig(BaseModel):
    """Gradient quantization configuration."""
    enabled: bool = True
    bits: int = 8


class AggregationConfig(BaseModel):
    """Aggregation configuration."""
    method: str = "secure"  # simple, secure, fedavg
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)


class PrivacyConfig(BaseModel):
    """Differential privacy configuration."""
    enabled: bool = True
    epsilon: float = 1.0
    delta: float = 1e-5
    mechanism: str = "gaussian"
    clip_norm: float = 1.0
    adaptive_noise: bool = True
    noise_multiplier: float = 1.1


class TrainingConfig(BaseModel):
    """RL training configuration."""
    algorithm: str = "ppo"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 64
    n_epochs: int = 10
    n_steps: int = 2048


class EnvironmentConfig(BaseModel):
    """Environment configuration."""
    name: str = "CartPole-v1"
    n_envs: int = 4
    normalize_obs: bool = True
    normalize_reward: bool = True


class ReplayConfig(BaseModel):
    """Experience replay configuration."""
    enabled: bool = True
    kafka_bootstrap_servers: str = "localhost:19092"
    topic: str = "experiences"
    buffer_size: int = 100000
    priority: bool = False


class CheckpointConfig(BaseModel):
    """Checkpoint configuration."""
    enabled: bool = True
    directory: str = "/data/checkpoints"
    interval_steps: int = 10000
    keep_last: int = 5


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"
    file: Optional[str] = None


class MetricsConfig(BaseModel):
    """Metrics configuration."""
    enabled: bool = True
    port: int = 8000


class TracingConfig(BaseModel):
    """Tracing configuration."""
    enabled: bool = True
    jaeger_host: str = "localhost"
    jaeger_port: int = 6831
    sample_rate: float = 0.1


class SacredConfig(BaseModel):
    """Sacred experiment tracking configuration."""
    enabled: bool = True
    mongodb_uri: str = "mongodb://localhost:27017/sacred"
    experiment_name: str = "gossip_rl"


class AgentConfig(BaseModel):
    """Agent configuration."""
    id: str = "agent-1"
    name: str = "GossipAgent"


class Config(BaseSettings):
    """Main configuration."""
    agent: AgentConfig = Field(default_factory=AgentConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    gossip: GossipConfig = Field(default_factory=GossipConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    sacred: SacredConfig = Field(default_factory=SacredConfig)

    class Config:
        env_prefix = "GOSSIP_RL_"
        env_nested_delimiter = "__"


def load_config(path: str | Path) -> Config:
    """Load configuration from YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path) as f:
        data = yaml.safe_load(f)
    
    # Expand environment variables
    def expand_env(obj: Any) -> Any:
        if isinstance(obj, str):
            import os
            import re
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
            def replace(match: re.Match) -> str:
                var_name = match.group(1)
                default = match.group(2) or ""
                return os.environ.get(var_name, default)
            return re.sub(pattern, replace, obj)
        elif isinstance(obj, dict):
            return {k: expand_env(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand_env(item) for item in obj]
        return obj
    
    data = expand_env(data)
    return Config(**data)
