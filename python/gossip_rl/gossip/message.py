"""Gossip message types and serialization."""

import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

import numpy as np


class MessageType(IntEnum):
    """Types of gossip messages."""
    # Membership messages
    PING = 1
    PING_REQ = 2
    ACK = 3
    SUSPECT = 4
    ALIVE = 5
    DEAD = 6
    
    # Gradient messages
    PUSH_GRADIENT = 10
    PULL_GRADIENT = 11
    PUSH_PULL_GRADIENT = 12
    
    # Aggregation messages
    AGGREGATE_REQUEST = 20
    AGGREGATE_RESPONSE = 21
    
    # Model sync
    MODEL_SYNC = 30
    MODEL_REQUEST = 31
    MODEL_RESPONSE = 32


@dataclass
class GossipMessage:
    """Base gossip message."""
    
    type: MessageType
    sender_id: str
    sequence: int
    timestamp: float = field(default_factory=time.time)
    payload: Optional[bytes] = None
    metadata: dict = field(default_factory=dict)
    
    # For gradient messages
    gradient_data: Optional[np.ndarray] = None
    indices: Optional[np.ndarray] = None  # For sparse gradients
    model_version: int = 0
    step: int = 0
    
    # For membership
    target_id: Optional[str] = None
    incarnation: int = 0
    
    def pack(self) -> bytes:
        """Serialize message to bytes."""
        # Header: type (1), sender_id len (2), sender_id, sequence (8), timestamp (8)
        sender_bytes = self.sender_id.encode('utf-8')
        header = struct.pack(
            '>BH',
            self.type.value,
            len(sender_bytes)
        )
        header += sender_bytes
        header += struct.pack('>Qd', self.sequence, self.timestamp)
        
        # Metadata
        metadata_flags = 0
        metadata_bytes = b''
        
        if self.target_id is not None:
            metadata_flags |= 0x01
            target_bytes = self.target_id.encode('utf-8')
            metadata_bytes += struct.pack('>H', len(target_bytes)) + target_bytes
        
        if self.gradient_data is not None:
            metadata_flags |= 0x02
            metadata_bytes += struct.pack('>QQ', self.model_version, self.step)
            grad_bytes = self.gradient_data.tobytes()
            metadata_bytes += struct.pack('>Q', len(grad_bytes))
            metadata_bytes += grad_bytes
            
            if self.indices is not None:
                metadata_flags |= 0x04
                idx_bytes = self.indices.tobytes()
                metadata_bytes += struct.pack('>Q', len(idx_bytes))
                metadata_bytes += idx_bytes
        
        metadata_bytes += struct.pack('>I', self.incarnation)
        
        # Payload
        payload_bytes = self.payload if self.payload else b''
        
        return (header + 
                struct.pack('>B', metadata_flags) + 
                metadata_bytes +
                struct.pack('>I', len(payload_bytes)) + 
                payload_bytes)
    
    @classmethod
    def unpack(cls, data: bytes) -> "GossipMessage":
        """Deserialize message from bytes."""
        offset = 0
        
        # Header
        msg_type, sender_len = struct.unpack_from('>BH', data, offset)
        offset += 3
        
        sender_id = data[offset:offset + sender_len].decode('utf-8')
        offset += sender_len
        
        sequence, timestamp = struct.unpack_from('>Qd', data, offset)
        offset += 16
        
        # Metadata flags
        metadata_flags = data[offset]
        offset += 1
        
        target_id = None
        gradient_data = None
        indices = None
        model_version = 0
        step = 0
        
        if metadata_flags & 0x01:
            target_len = struct.unpack_from('>H', data, offset)[0]
            offset += 2
            target_id = data[offset:offset + target_len].decode('utf-8')
            offset += target_len
        
        if metadata_flags & 0x02:
            model_version, step = struct.unpack_from('>QQ', data, offset)
            offset += 16
            
            grad_len = struct.unpack_from('>Q', data, offset)[0]
            offset += 8
            gradient_data = np.frombuffer(data[offset:offset + grad_len], dtype=np.float32)
            offset += grad_len
            
            if metadata_flags & 0x04:
                idx_len = struct.unpack_from('>Q', data, offset)[0]
                offset += 8
                indices = np.frombuffer(data[offset:offset + idx_len], dtype=np.int64)
                offset += idx_len
        
        incarnation = struct.unpack_from('>I', data, offset)[0]
        offset += 4
        
        # Payload
        payload_len = struct.unpack_from('>I', data, offset)[0]
        offset += 4
        payload = data[offset:offset + payload_len] if payload_len > 0 else None
        
        return cls(
            type=MessageType(msg_type),
            sender_id=sender_id,
            sequence=sequence,
            timestamp=timestamp,
            payload=payload,
            gradient_data=gradient_data,
            indices=indices,
            model_version=model_version,
            step=step,
            target_id=target_id,
            incarnation=incarnation,
        )
    
    @classmethod
    def create_ping(cls, sender_id: str, sequence: int, target_id: str) -> "GossipMessage":
        """Create a PING message."""
        return cls(
            type=MessageType.PING,
            sender_id=sender_id,
            sequence=sequence,
            target_id=target_id,
        )
    
    @classmethod
    def create_ack(cls, sender_id: str, sequence: int, target_id: str) -> "GossipMessage":
        """Create an ACK message."""
        return cls(
            type=MessageType.ACK,
            sender_id=sender_id,
            sequence=sequence,
            target_id=target_id,
        )
    
    @classmethod
    def create_push_gradient(
        cls,
        sender_id: str,
        sequence: int,
        gradients: np.ndarray,
        model_version: int,
        step: int,
        indices: Optional[np.ndarray] = None,
    ) -> "GossipMessage":
        """Create a PUSH_GRADIENT message."""
        return cls(
            type=MessageType.PUSH_GRADIENT,
            sender_id=sender_id,
            sequence=sequence,
            gradient_data=gradients.astype(np.float32),
            indices=indices,
            model_version=model_version,
            step=step,
        )
    
    @classmethod
    def create_push_pull_gradient(
        cls,
        sender_id: str,
        sequence: int,
        gradients: np.ndarray,
        model_version: int,
        step: int,
        indices: Optional[np.ndarray] = None,
    ) -> "GossipMessage":
        """Create a PUSH_PULL_GRADIENT message (request + send)."""
        return cls(
            type=MessageType.PUSH_PULL_GRADIENT,
            sender_id=sender_id,
            sequence=sequence,
            gradient_data=gradients.astype(np.float32),
            indices=indices,
            model_version=model_version,
            step=step,
        )
