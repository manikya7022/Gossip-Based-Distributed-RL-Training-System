"""Tests for the gossip protocol."""

import asyncio
import pytest
import numpy as np

from gossip_rl.gossip.peer import Peer, PeerState
from gossip_rl.gossip.message import GossipMessage, MessageType
from gossip_rl.gossip.membership import MembershipManager
from gossip_rl.gossip.protocol import GossipProtocol
from gossip_rl.config import GossipConfig


class TestPeer:
    """Tests for Peer class."""
    
    def test_peer_creation(self):
        peer = Peer(id="test-1", host="localhost", port=50051)
        assert peer.id == "test-1"
        assert peer.address == "localhost:50051"
        assert peer.is_alive
    
    def test_peer_state_transitions(self):
        peer = Peer(id="test-1", host="localhost", port=50051)
        
        # Alive -> Suspect
        assert peer.mark_suspect()
        assert peer.is_suspect
        
        # Suspect -> Dead
        assert peer.mark_dead()
        assert peer.is_dead
        
        # Dead -> Alive (with higher incarnation)
        assert peer.mark_alive(incarnation=1)
        assert peer.is_alive
        assert peer.incarnation == 1
    
    def test_peer_latency_update(self):
        peer = Peer(id="test-1", host="localhost", port=50051)
        
        peer.update_latency(100.0)
        assert peer.latency_ms == 100.0
        
        peer.update_latency(50.0, alpha=0.5)
        assert peer.latency_ms == 75.0


class TestGossipMessage:
    """Tests for GossipMessage serialization."""
    
    def test_ping_message(self):
        msg = GossipMessage.create_ping("sender-1", 1, "target-1")
        
        packed = msg.pack()
        unpacked = GossipMessage.unpack(packed)
        
        assert unpacked.type == MessageType.PING
        assert unpacked.sender_id == "sender-1"
        assert unpacked.sequence == 1
        assert unpacked.target_id == "target-1"
    
    def test_gradient_message(self):
        gradients = np.random.randn(100).astype(np.float32)
        
        msg = GossipMessage.create_push_gradient(
            sender_id="agent-1",
            sequence=42,
            gradients=gradients,
            model_version=5,
            step=1000,
        )
        
        packed = msg.pack()
        unpacked = GossipMessage.unpack(packed)
        
        assert unpacked.type == MessageType.PUSH_GRADIENT
        assert unpacked.model_version == 5
        assert unpacked.step == 1000
        np.testing.assert_array_almost_equal(unpacked.gradient_data, gradients)
    
    def test_sparse_gradient_message(self):
        gradients = np.random.randn(50).astype(np.float32)
        indices = np.array([0, 5, 10, 15, 20], dtype=np.int64)
        
        msg = GossipMessage.create_push_gradient(
            sender_id="agent-1",
            sequence=1,
            gradients=gradients,
            model_version=1,
            step=100,
            indices=indices,
        )
        
        packed = msg.pack()
        unpacked = GossipMessage.unpack(packed)
        
        np.testing.assert_array_equal(unpacked.indices, indices)


class TestMembershipManager:
    """Tests for MembershipManager."""
    
    @pytest.fixture
    def local_peer(self):
        return Peer(id="local", host="localhost", port=50051)
    
    @pytest.fixture
    def membership(self, local_peer):
        return MembershipManager(
            local_peer=local_peer,
            probe_interval_ms=100,
            probe_timeout_ms=50,
        )
    
    @pytest.mark.asyncio
    async def test_add_peer(self, membership):
        peer = Peer(id="peer-1", host="localhost", port=50052)
        
        result = await membership.add_peer(peer)
        assert result is True
        
        retrieved = await membership.get_peer("peer-1")
        assert retrieved is not None
        assert retrieved.id == "peer-1"
    
    @pytest.mark.asyncio
    async def test_get_alive_peers(self, membership):
        for i in range(5):
            peer = Peer(id=f"peer-{i}", host="localhost", port=50052 + i)
            await membership.add_peer(peer)
        
        alive = await membership.get_alive_peers()
        assert len(alive) == 5
    
    @pytest.mark.asyncio
    async def test_handle_ping(self, membership):
        msg = GossipMessage.create_ping("sender-1", 1, "local")
        
        response = await membership.handle_message(msg)
        
        assert response is not None
        assert response.type == MessageType.ACK
        assert response.sender_id == "local"


class TestGossipProtocol:
    """Tests for GossipProtocol."""
    
    @pytest.fixture
    def local_peer(self):
        return Peer(id="local", host="localhost", port=50051)
    
    @pytest.fixture
    def config(self):
        return GossipConfig(interval_ms=50, fanout=2, push_pull=True)
    
    @pytest.fixture
    def membership(self, local_peer):
        return MembershipManager(local_peer=local_peer)
    
    @pytest.fixture
    def protocol(self, local_peer, config, membership):
        return GossipProtocol(
            local_peer=local_peer,
            config=config,
            membership=membership,
        )
    
    def test_set_gradients(self, protocol):
        gradients = np.random.randn(100).astype(np.float32)
        
        protocol.set_gradients(gradients, model_version=1, step=100)
        
        assert protocol.current_gradients is not None
        assert protocol.model_version == 1
        assert protocol.step == 100
    
    @pytest.mark.asyncio
    async def test_handle_push_gradient(self, protocol):
        gradients = np.random.randn(100).astype(np.float32)
        protocol.set_gradients(gradients, model_version=1, step=100)
        
        incoming = np.random.randn(100).astype(np.float32)
        msg = GossipMessage.create_push_gradient(
            sender_id="peer-1",
            sequence=1,
            gradients=incoming,
            model_version=1,
            step=100,
        )
        
        await protocol.handle_message(msg)
        
        aggregated, count = await protocol.get_aggregated_gradients()
        assert count == 2  # Local + incoming
    
    @pytest.mark.asyncio
    async def test_push_pull_exchange(self, protocol):
        gradients = np.random.randn(100).astype(np.float32)
        protocol.set_gradients(gradients, model_version=1, step=100)
        
        incoming = np.random.randn(100).astype(np.float32)
        msg = GossipMessage.create_push_pull_gradient(
            sender_id="peer-1",
            sequence=1,
            gradients=incoming,
            model_version=1,
            step=100,
        )
        
        response = await protocol.handle_message(msg)
        
        assert response is not None
        assert response.type == MessageType.PUSH_GRADIENT
        assert response.gradient_data is not None
