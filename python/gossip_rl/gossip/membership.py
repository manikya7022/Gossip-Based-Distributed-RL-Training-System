"""Membership management with SWIM-style failure detection."""

import asyncio
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from gossip_rl.gossip.peer import Peer, PeerState
from gossip_rl.gossip.message import GossipMessage, MessageType

logger = logging.getLogger(__name__)


@dataclass
class ProbeState:
    """State for a pending probe."""
    target_id: str
    sequence: int
    start_time: float
    indirect_targets: List[str] = field(default_factory=list)
    acked: bool = False


class MembershipManager:
    """SWIM-style membership management with epidemic dissemination."""
    
    def __init__(
        self,
        local_peer: Peer,
        probe_interval_ms: int = 500,
        probe_timeout_ms: int = 200,
        indirect_probes: int = 3,
        suspicion_mult: int = 4,
        on_peer_alive: Optional[Callable[[Peer], None]] = None,
        on_peer_suspect: Optional[Callable[[Peer], None]] = None,
        on_peer_dead: Optional[Callable[[Peer], None]] = None,
    ):
        self.local_peer = local_peer
        self.probe_interval_ms = probe_interval_ms
        self.probe_timeout_ms = probe_timeout_ms
        self.indirect_probes = indirect_probes
        self.suspicion_mult = suspicion_mult
        
        self.on_peer_alive = on_peer_alive
        self.on_peer_suspect = on_peer_suspect
        self.on_peer_dead = on_peer_dead
        
        # Peer storage
        self.peers: Dict[str, Peer] = {}
        self.peer_lock = asyncio.Lock()
        
        # Probe state
        self.sequence_num = 0
        self.pending_probes: Dict[int, ProbeState] = {}
        self.probe_lock = asyncio.Lock()
        
        # Updates to disseminate
        self.updates_queue: List[GossipMessage] = []
        self.updates_lock = asyncio.Lock()
        
        # Suspicion timers
        self.suspicion_timers: Dict[str, asyncio.TimerHandle] = {}
        
        self._running = False
        self._probe_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the membership manager."""
        self._running = True
        self._probe_task = asyncio.create_task(self._probe_loop())
        logger.info("Membership manager started")
    
    async def stop(self) -> None:
        """Stop the membership manager."""
        self._running = False
        if self._probe_task:
            self._probe_task.cancel()
            try:
                await self._probe_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all timers
        for timer in self.suspicion_timers.values():
            timer.cancel()
        self.suspicion_timers.clear()
        
        logger.info("Membership manager stopped")
    
    async def add_peer(self, peer: Peer) -> bool:
        """Add a peer to the membership list."""
        async with self.peer_lock:
            if peer.id in self.peers:
                existing = self.peers[peer.id]
                if peer.incarnation > existing.incarnation:
                    self.peers[peer.id] = peer
                    return True
                return False
            
            self.peers[peer.id] = peer
            logger.info(f"Added peer: {peer.id} ({peer.address})")
            return True
    
    async def remove_peer(self, peer_id: str) -> Optional[Peer]:
        """Remove a peer from the membership list."""
        async with self.peer_lock:
            if peer_id in self.peers:
                peer = self.peers.pop(peer_id)
                logger.info(f"Removed peer: {peer_id}")
                return peer
            return None
    
    async def get_peer(self, peer_id: str) -> Optional[Peer]:
        """Get a peer by ID."""
        async with self.peer_lock:
            return self.peers.get(peer_id)
    
    async def get_alive_peers(self) -> List[Peer]:
        """Get all alive peers."""
        async with self.peer_lock:
            return [p for p in self.peers.values() if p.is_alive]
    
    async def get_all_peers(self) -> List[Peer]:
        """Get all peers."""
        async with self.peer_lock:
            return list(self.peers.values())
    
    async def select_random_peers(self, n: int, exclude: Optional[Set[str]] = None) -> List[Peer]:
        """Select n random alive peers."""
        exclude = exclude or set()
        async with self.peer_lock:
            candidates = [
                p for p in self.peers.values() 
                if p.is_alive and p.id not in exclude
            ]
            if len(candidates) <= n:
                return candidates
            return random.sample(candidates, n)
    
    async def handle_message(self, msg: GossipMessage) -> Optional[GossipMessage]:
        """Handle an incoming membership message."""
        if msg.type == MessageType.PING:
            return await self._handle_ping(msg)
        elif msg.type == MessageType.PING_REQ:
            return await self._handle_ping_req(msg)
        elif msg.type == MessageType.ACK:
            await self._handle_ack(msg)
        elif msg.type == MessageType.ALIVE:
            await self._handle_alive(msg)
        elif msg.type == MessageType.SUSPECT:
            await self._handle_suspect(msg)
        elif msg.type == MessageType.DEAD:
            await self._handle_dead(msg)
        return None
    
    async def _handle_ping(self, msg: GossipMessage) -> GossipMessage:
        """Handle PING message and return ACK."""
        # Update sender's last seen
        async with self.peer_lock:
            if msg.sender_id in self.peers:
                self.peers[msg.sender_id].last_seen = time.time()
                self.peers[msg.sender_id].incarnation = msg.incarnation
        
        return GossipMessage.create_ack(
            sender_id=self.local_peer.id,
            sequence=msg.sequence,
            target_id=msg.sender_id,
        )
    
    async def _handle_ping_req(self, msg: GossipMessage) -> None:
        """Handle indirect ping request."""
        if msg.target_id is None:
            return
        
        # Send PING to target on behalf of requester
        target = await self.get_peer(msg.target_id)
        if target:
            # This would send through the transport layer
            pass
    
    async def _handle_ack(self, msg: GossipMessage) -> None:
        """Handle ACK message."""
        async with self.probe_lock:
            if msg.sequence in self.pending_probes:
                probe = self.pending_probes[msg.sequence]
                probe.acked = True
                
                # Update latency
                latency_ms = (time.time() - probe.start_time) * 1000
                async with self.peer_lock:
                    if probe.target_id in self.peers:
                        self.peers[probe.target_id].update_latency(latency_ms)
                        if self.peers[probe.target_id].mark_alive(msg.incarnation):
                            # Cancel suspicion timer if any
                            if probe.target_id in self.suspicion_timers:
                                self.suspicion_timers[probe.target_id].cancel()
                                del self.suspicion_timers[probe.target_id]
                            
                            if self.on_peer_alive:
                                self.on_peer_alive(self.peers[probe.target_id])
                
                del self.pending_probes[msg.sequence]
    
    async def _handle_alive(self, msg: GossipMessage) -> None:
        """Handle ALIVE message."""
        if msg.target_id is None:
            return
        
        async with self.peer_lock:
            if msg.target_id in self.peers:
                peer = self.peers[msg.target_id]
                if peer.mark_alive(msg.incarnation):
                    logger.info(f"Peer {msg.target_id} marked alive")
                    
                    if msg.target_id in self.suspicion_timers:
                        self.suspicion_timers[msg.target_id].cancel()
                        del self.suspicion_timers[msg.target_id]
                    
                    if self.on_peer_alive:
                        self.on_peer_alive(peer)
    
    async def _handle_suspect(self, msg: GossipMessage) -> None:
        """Handle SUSPECT message."""
        if msg.target_id is None:
            return
        
        # If we're the suspect, refute with higher incarnation
        if msg.target_id == self.local_peer.id:
            self.local_peer.incarnation = max(
                self.local_peer.incarnation,
                msg.incarnation
            ) + 1
            
            # Broadcast ALIVE
            await self._broadcast_alive()
            return
        
        async with self.peer_lock:
            if msg.target_id in self.peers:
                peer = self.peers[msg.target_id]
                if peer.mark_suspect(msg.incarnation):
                    logger.warning(f"Peer {msg.target_id} suspected")
                    
                    if self.on_peer_suspect:
                        self.on_peer_suspect(peer)
                    
                    # Start suspicion timer
                    self._start_suspicion_timer(msg.target_id)
    
    async def _handle_dead(self, msg: GossipMessage) -> None:
        """Handle DEAD message."""
        if msg.target_id is None:
            return
        
        async with self.peer_lock:
            if msg.target_id in self.peers:
                peer = self.peers[msg.target_id]
                if peer.mark_dead(msg.incarnation):
                    logger.warning(f"Peer {msg.target_id} confirmed dead")
                    
                    if msg.target_id in self.suspicion_timers:
                        self.suspicion_timers[msg.target_id].cancel()
                        del self.suspicion_timers[msg.target_id]
                    
                    if self.on_peer_dead:
                        self.on_peer_dead(peer)
    
    def _start_suspicion_timer(self, peer_id: str) -> None:
        """Start a suspicion timer for a peer."""
        if peer_id in self.suspicion_timers:
            return
        
        timeout = (self.probe_interval_ms * self.suspicion_mult) / 1000.0
        loop = asyncio.get_event_loop()
        
        def on_timeout():
            asyncio.create_task(self._confirm_dead(peer_id))
        
        self.suspicion_timers[peer_id] = loop.call_later(timeout, on_timeout)
    
    async def _confirm_dead(self, peer_id: str) -> None:
        """Confirm a peer as dead after suspicion timeout."""
        if peer_id in self.suspicion_timers:
            del self.suspicion_timers[peer_id]
        
        async with self.peer_lock:
            if peer_id in self.peers:
                peer = self.peers[peer_id]
                if peer.state == PeerState.SUSPECT:
                    peer.mark_dead()
                    logger.warning(f"Peer {peer_id} confirmed dead (suspicion timeout)")
                    
                    if self.on_peer_dead:
                        self.on_peer_dead(peer)
    
    async def _broadcast_alive(self) -> None:
        """Broadcast ALIVE message to refute suspicion."""
        msg = GossipMessage(
            type=MessageType.ALIVE,
            sender_id=self.local_peer.id,
            sequence=self._next_sequence(),
            target_id=self.local_peer.id,
            incarnation=self.local_peer.incarnation,
        )
        
        async with self.updates_lock:
            self.updates_queue.append(msg)
    
    def _next_sequence(self) -> int:
        """Get next sequence number."""
        self.sequence_num += 1
        return self.sequence_num
    
    async def _probe_loop(self) -> None:
        """Main probe loop for failure detection."""
        while self._running:
            try:
                await self._probe_round()
                await asyncio.sleep(self.probe_interval_ms / 1000.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Probe loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _probe_round(self) -> None:
        """Execute a single probe round."""
        peers = await self.get_alive_peers()
        if not peers:
            return
        
        # Select random peer to probe
        target = random.choice(peers)
        await self._probe_peer(target)
    
    async def _probe_peer(self, target: Peer) -> None:
        """Probe a specific peer."""
        sequence = self._next_sequence()
        
        probe = ProbeState(
            target_id=target.id,
            sequence=sequence,
            start_time=time.time(),
        )
        
        async with self.probe_lock:
            self.pending_probes[sequence] = probe
        
        # Send PING (would go through transport)
        msg = GossipMessage.create_ping(
            sender_id=self.local_peer.id,
            sequence=sequence,
            target_id=target.id,
        )
        
        # Wait for ACK
        await asyncio.sleep(self.probe_timeout_ms / 1000.0)
        
        async with self.probe_lock:
            if sequence in self.pending_probes and not self.pending_probes[sequence].acked:
                # Direct probe failed, try indirect
                await self._indirect_probe(target, sequence)
    
    async def _indirect_probe(self, target: Peer, sequence: int) -> None:
        """Send indirect probes via other peers."""
        probes = await self.select_random_peers(
            self.indirect_probes,
            exclude={target.id, self.local_peer.id}
        )
        
        if not probes:
            # No peers for indirect probe, suspect immediately
            await self._suspect_peer(target)
            return
        
        async with self.probe_lock:
            if sequence in self.pending_probes:
                self.pending_probes[sequence].indirect_targets = [p.id for p in probes]
        
        # Send PING_REQ messages (would go through transport)
        for peer in probes:
            msg = GossipMessage(
                type=MessageType.PING_REQ,
                sender_id=self.local_peer.id,
                sequence=sequence,
                target_id=target.id,
            )
        
        # Wait for indirect ACK
        await asyncio.sleep(self.probe_timeout_ms / 1000.0)
        
        async with self.probe_lock:
            if sequence in self.pending_probes and not self.pending_probes[sequence].acked:
                await self._suspect_peer(target)
                del self.pending_probes[sequence]
    
    async def _suspect_peer(self, peer: Peer) -> None:
        """Mark a peer as suspect."""
        async with self.peer_lock:
            if peer.id in self.peers:
                if self.peers[peer.id].mark_suspect():
                    logger.warning(f"Peer {peer.id} suspected (probe timeout)")
                    
                    if self.on_peer_suspect:
                        self.on_peer_suspect(self.peers[peer.id])
                    
                    self._start_suspicion_timer(peer.id)
                    
                    # Broadcast SUSPECT
                    msg = GossipMessage(
                        type=MessageType.SUSPECT,
                        sender_id=self.local_peer.id,
                        sequence=self._next_sequence(),
                        target_id=peer.id,
                        incarnation=peer.incarnation,
                    )
                    
                    async with self.updates_lock:
                        self.updates_queue.append(msg)
    
    async def get_pending_updates(self, max_updates: int = 10) -> List[GossipMessage]:
        """Get pending membership updates to piggyback on gossip."""
        async with self.updates_lock:
            updates = self.updates_queue[:max_updates]
            self.updates_queue = self.updates_queue[max_updates:]
            return updates
