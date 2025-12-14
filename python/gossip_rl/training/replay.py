"""Experience Replay with Redpanda/Kafka integration."""

import asyncio
import json
import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single experience tuple."""
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
    
    # Optional priority for prioritized replay
    priority: float = 1.0
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "obs": self.obs.tolist(),
            "action": self.action.tolist() if isinstance(self.action, np.ndarray) else self.action,
            "reward": float(self.reward),
            "next_obs": self.next_obs.tolist(),
            "done": self.done,
            "info": self.info,
            "priority": self.priority,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Experience":
        """Deserialize from dictionary."""
        return cls(
            obs=np.array(data["obs"], dtype=np.float32),
            action=np.array(data["action"]),
            reward=data["reward"],
            next_obs=np.array(data["next_obs"], dtype=np.float32),
            done=data["done"],
            info=data.get("info", {}),
            priority=data.get("priority", 1.0),
        )


class ExperienceBuffer:
    """Local experience buffer with uniform sampling."""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.position = 0
    
    def add(self, experience: Experience) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add batch of experiences."""
        for i in range(len(obs)):
            self.add(Experience(
                obs=obs[i],
                action=actions[i],
                reward=rewards[i],
                next_obs=next_obs[i],
                done=dones[i],
            ))
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch."""
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        return [self.buffer[i] for i in indices]
    
    def sample_batch(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample and return as arrays."""
        experiences = self.sample(batch_size)
        
        obs = np.array([e.obs for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_obs = np.array([e.next_obs for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        return obs, actions, rewards, next_obs, dones
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self) -> None:
        self.buffer.clear()


class PrioritizedExperienceBuffer(ExperienceBuffer):
    """Prioritized experience replay buffer."""
    
    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        super().__init__(capacity)
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(self, experience: Experience) -> None:
        """Add experience with max priority."""
        experience.priority = self.max_priority
        super().add(experience)
        
        idx = (len(self.buffer) - 1) % self.capacity
        self.priorities[idx] = self.max_priority
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample with priority weighting.
        
        Returns:
            (experiences, indices, importance_weights)
        """
        n = len(self.buffer)
        priorities = self.priorities[:n] ** self.alpha
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(n, size=batch_size, p=probs, replace=False)
        experiences = [self.buffer[i] for i in indices]
        
        # Importance sampling weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


class RedpandaReplayBuffer:
    """Distributed experience replay using Redpanda (Kafka-compatible).
    
    Features:
    - Async producer for non-blocking experience storage
    - Consumer groups for distributed sampling
    - Message compression
    - Automatic topic management
    """
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:19092",
        topic: str = "experiences",
        agent_id: str = None,
        local_buffer_size: int = 10000,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        
        # Local buffer for batching
        self.local_buffer = ExperienceBuffer(local_buffer_size)
        
        # Kafka components (lazily initialized)
        self._producer = None
        self._consumer = None
        self._consumer_running = False
        self._consumer_thread: Optional[threading.Thread] = None
        
        # Received experiences from other agents
        self.received_buffer: deque = deque(maxlen=local_buffer_size)
    
    def _get_producer(self):
        """Lazy producer initialization."""
        if self._producer is None:
            try:
                from confluent_kafka import Producer
                
                self._producer = Producer({
                    'bootstrap.servers': self.bootstrap_servers,
                    'client.id': f'gossip-rl-{self.agent_id}',
                    'compression.type': 'lz4',
                    'linger.ms': 10,
                    'batch.size': 65536,
                    'acks': 'all',
                })
                logger.info(f"Kafka producer connected to {self.bootstrap_servers}")
            except ImportError:
                logger.warning("confluent-kafka not installed, using local buffer only")
            except Exception as e:
                logger.error(f"Failed to create Kafka producer: {e}")
        
        return self._producer
    
    def _get_consumer(self):
        """Lazy consumer initialization."""
        if self._consumer is None:
            try:
                from confluent_kafka import Consumer
                
                self._consumer = Consumer({
                    'bootstrap.servers': self.bootstrap_servers,
                    'group.id': f'gossip-rl-consumers-{self.agent_id}',
                    'client.id': f'gossip-rl-consumer-{self.agent_id}',
                    'auto.offset.reset': 'latest',
                    'enable.auto.commit': True,
                })
                self._consumer.subscribe([self.topic])
                logger.info(f"Kafka consumer subscribed to {self.topic}")
            except ImportError:
                logger.warning("confluent-kafka not installed")
            except Exception as e:
                logger.error(f"Failed to create Kafka consumer: {e}")
        
        return self._consumer
    
    def add(self, experience: Experience) -> None:
        """Add experience to buffer and publish to Kafka."""
        self.local_buffer.add(experience)
        
        producer = self._get_producer()
        if producer is not None:
            try:
                key = self.agent_id.encode('utf-8')
                value = json.dumps(experience.to_dict()).encode('utf-8')
                producer.produce(
                    self.topic,
                    key=key,
                    value=value,
                    callback=self._delivery_callback,
                )
                producer.poll(0)
            except Exception as e:
                logger.error(f"Failed to publish experience: {e}")
    
    def _delivery_callback(self, err, msg):
        """Kafka delivery callback."""
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
    
    def flush(self) -> None:
        """Flush pending messages."""
        if self._producer is not None:
            self._producer.flush()
    
    def start_consumer(self) -> None:
        """Start background consumer thread."""
        if self._consumer_running:
            return
        
        consumer = self._get_consumer()
        if consumer is None:
            return
        
        self._consumer_running = True
        self._consumer_thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._consumer_thread.start()
    
    def _consume_loop(self):
        """Background consumer loop."""
        consumer = self._get_consumer()
        if consumer is None:
            return
        
        while self._consumer_running:
            try:
                msg = consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    logger.error(f"Consumer error: {msg.error()}")
                    continue
                
                # Skip messages from self
                sender_id = msg.key().decode('utf-8') if msg.key() else ''
                if sender_id == self.agent_id:
                    continue
                
                # Parse experience
                data = json.loads(msg.value().decode('utf-8'))
                experience = Experience.from_dict(data)
                self.received_buffer.append(experience)
                
            except Exception as e:
                logger.error(f"Consumer error: {e}")
    
    def stop_consumer(self) -> None:
        """Stop background consumer."""
        self._consumer_running = False
        if self._consumer_thread is not None:
            self._consumer_thread.join(timeout=5.0)
        if self._consumer is not None:
            self._consumer.close()
    
    def sample(self, batch_size: int, include_remote: bool = True) -> List[Experience]:
        """Sample from local and received experiences."""
        local_samples = []
        remote_samples = []
        
        # Sample from local buffer
        local_count = batch_size // 2 if include_remote and self.received_buffer else batch_size
        if len(self.local_buffer) >= local_count:
            local_samples = self.local_buffer.sample(local_count)
        
        # Sample from received buffer
        if include_remote and self.received_buffer:
            remote_count = batch_size - len(local_samples)
            if len(self.received_buffer) >= remote_count:
                indices = np.random.randint(0, len(self.received_buffer), size=remote_count)
                remote_samples = [self.received_buffer[i] for i in indices]
        
        return local_samples + remote_samples
    
    def __len__(self) -> int:
        return len(self.local_buffer) + len(self.received_buffer)
    
    def close(self) -> None:
        """Clean up resources."""
        self.stop_consumer()
        self.flush()
        if self._producer is not None:
            self._producer = None
