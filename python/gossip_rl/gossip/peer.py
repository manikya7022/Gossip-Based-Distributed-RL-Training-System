"""Peer representation and state management."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PeerState(Enum):
    """Peer health state (SWIM-style)."""
    ALIVE = "alive"
    SUSPECT = "suspect"
    DEAD = "dead"
    LEFT = "left"


@dataclass
class Peer:
    """Represents a peer in the gossip network."""
    
    id: str
    host: str
    port: int
    state: PeerState = PeerState.ALIVE
    incarnation: int = 0
    last_seen: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    
    # Gradient aggregation state
    model_version: int = 0
    gradient_timestamp: float = 0.0
    
    # Locality information for peer selection
    datacenter: Optional[str] = None
    rack: Optional[str] = None
    latency_ms: float = float('inf')
    
    @property
    def address(self) -> str:
        """Get peer address string."""
        return f"{self.host}:{self.port}"
    
    @property
    def is_alive(self) -> bool:
        """Check if peer is considered alive."""
        return self.state == PeerState.ALIVE
    
    @property
    def is_suspect(self) -> bool:
        """Check if peer is suspected dead."""
        return self.state == PeerState.SUSPECT
    
    @property
    def is_dead(self) -> bool:
        """Check if peer is confirmed dead."""
        return self.state in (PeerState.DEAD, PeerState.LEFT)
    
    def mark_alive(self, incarnation: Optional[int] = None) -> bool:
        """Mark peer as alive.
        
        Returns True if state changed.
        """
        if incarnation is not None and incarnation <= self.incarnation:
            return False  # Stale update
        
        if incarnation is not None:
            self.incarnation = incarnation
        
        old_state = self.state
        self.state = PeerState.ALIVE
        self.last_seen = time.time()
        return old_state != PeerState.ALIVE
    
    def mark_suspect(self, incarnation: Optional[int] = None) -> bool:
        """Mark peer as suspect.
        
        Returns True if state changed.
        """
        if incarnation is not None and incarnation < self.incarnation:
            return False  # Stale update
        
        if self.state == PeerState.ALIVE:
            self.state = PeerState.SUSPECT
            return True
        return False
    
    def mark_dead(self, incarnation: Optional[int] = None) -> bool:
        """Mark peer as dead.
        
        Returns True if state changed.
        """
        if incarnation is not None and incarnation < self.incarnation:
            return False  # Stale update
        
        if self.state != PeerState.DEAD:
            self.state = PeerState.DEAD
            return True
        return False
    
    def update_latency(self, latency_ms: float, alpha: float = 0.1) -> None:
        """Update latency with exponential moving average."""
        if self.latency_ms == float('inf'):
            self.latency_ms = latency_ms
        else:
            self.latency_ms = alpha * latency_ms + (1 - alpha) * self.latency_ms
    
    def is_local(self, other: "Peer") -> bool:
        """Check if this peer is in the same datacenter as another."""
        if self.datacenter is None or other.datacenter is None:
            return False
        return self.datacenter == other.datacenter
    
    def is_same_rack(self, other: "Peer") -> bool:
        """Check if this peer is in the same rack as another."""
        return self.is_local(other) and self.rack == other.rack
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Peer):
            return False
        return self.id == other.id
    
    def to_dict(self) -> dict:
        """Serialize peer to dictionary."""
        return {
            "id": self.id,
            "host": self.host,
            "port": self.port,
            "state": self.state.value,
            "incarnation": self.incarnation,
            "last_seen": self.last_seen,
            "model_version": self.model_version,
            "datacenter": self.datacenter,
            "rack": self.rack,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Peer":
        """Deserialize peer from dictionary."""
        return cls(
            id=data["id"],
            host=data["host"],
            port=data["port"],
            state=PeerState(data.get("state", "alive")),
            incarnation=data.get("incarnation", 0),
            last_seen=data.get("last_seen", time.time()),
            model_version=data.get("model_version", 0),
            datacenter=data.get("datacenter"),
            rack=data.get("rack"),
        )
