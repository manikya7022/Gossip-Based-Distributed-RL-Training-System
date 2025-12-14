"""Core Gossip Protocol Implementation."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from gossip_rl.gossip.peer import Peer, PeerState
from gossip_rl.gossip.message import GossipMessage, MessageType
from gossip_rl.gossip.membership import MembershipManager
from gossip_rl.config import GossipConfig

logger = logging.getLogger(__name__)


@dataclass
class GradientState:
    """State for tracking gradient aggregation."""
    gradients: np.ndarray
    indices: Optional[np.ndarray]
    model_version: int
    step: int
    contributors: Set[str]
    timestamp: float


class GossipProtocol:
    """Push-Pull Gossip Protocol for gradient aggregation.
    
    Implements a hybrid push-pull gossip protocol where:
    - Push: Agent sends its gradients to selected peers
    - Pull: Agent requests gradients from peers
    - Push-Pull: Combined operation for faster convergence
    
    Features:
    - Locality-aware peer selection (prefer same datacenter/rack)
    - Exponential backoff for failed connections
    - Gradient caching to avoid redundant updates
    """
    
    def __init__(
        self,
        local_peer: Peer,
        config: GossipConfig,
        membership: MembershipManager,
        on_gradient_received: Optional[Callable[[np.ndarray, int, int], None]] = None,
    ):
        self.local_peer = local_peer
        self.config = config
        self.membership = membership
        self.on_gradient_received = on_gradient_received
        
        # Gossip state
        self.sequence_num = 0
        self.current_gradients: Optional[np.ndarray] = None
        self.current_indices: Optional[np.ndarray] = None
        self.model_version = 0
        self.step = 0
        
        # Gradient cache (peer_id -> GradientState)
        self.gradient_cache: Dict[str, GradientState] = {}
        self.cache_lock = asyncio.Lock()
        
        # Aggregated gradients
        self.aggregated_gradients: Optional[np.ndarray] = None
        self.aggregation_lock = asyncio.Lock()
        
        # Backoff state for failed peers
        self.backoff: Dict[str, float] = {}
        self.max_backoff = 30.0  # seconds
        
        self._running = False
        self._gossip_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.rounds_completed = 0
        self.messages_sent = 0
        self.messages_received = 0
        self.gradients_aggregated = 0
    
    async def start(self) -> None:
        """Start the gossip protocol."""
        self._running = True
        self._gossip_task = asyncio.create_task(self._gossip_loop())
        logger.info(f"Gossip protocol started (interval={self.config.interval_ms}ms, fanout={self.config.fanout})")
    
    async def stop(self) -> None:
        """Stop the gossip protocol."""
        self._running = False
        if self._gossip_task:
            self._gossip_task.cancel()
            try:
                await self._gossip_task
            except asyncio.CancelledError:
                pass
        logger.info("Gossip protocol stopped")
    
    def set_gradients(
        self,
        gradients: np.ndarray,
        indices: Optional[np.ndarray] = None,
        model_version: int = 0,
        step: int = 0,
    ) -> None:
        """Set current gradients for gossip."""
        self.current_gradients = gradients.astype(np.float32)
        self.current_indices = indices
        self.model_version = model_version
        self.step = step
        
        # Reset aggregation
        self.aggregated_gradients = gradients.copy()
    
    async def get_aggregated_gradients(self) -> Tuple[np.ndarray, int]:
        """Get current aggregated gradients and contributor count."""
        async with self.aggregation_lock:
            if self.aggregated_gradients is None:
                return np.array([]), 0
            return self.aggregated_gradients.copy(), len(self.gradient_cache) + 1
    
    async def handle_message(self, msg: GossipMessage) -> Optional[GossipMessage]:
        """Handle an incoming gossip message."""
        self.messages_received += 1
        
        if msg.type in (MessageType.PING, MessageType.PING_REQ, MessageType.ACK,
                        MessageType.ALIVE, MessageType.SUSPECT, MessageType.DEAD):
            return await self.membership.handle_message(msg)
        
        elif msg.type == MessageType.PUSH_GRADIENT:
            await self._handle_push_gradient(msg)
            
        elif msg.type == MessageType.PULL_GRADIENT:
            return await self._handle_pull_gradient(msg)
            
        elif msg.type == MessageType.PUSH_PULL_GRADIENT:
            return await self._handle_push_pull_gradient(msg)
        
        return None
    
    async def _handle_push_gradient(self, msg: GossipMessage) -> None:
        """Handle incoming gradient push."""
        if msg.gradient_data is None:
            return
        
        await self._process_gradient(
            peer_id=msg.sender_id,
            gradients=msg.gradient_data,
            indices=msg.indices,
            model_version=msg.model_version,
            step=msg.step,
        )
    
    async def _handle_pull_gradient(self, msg: GossipMessage) -> GossipMessage:
        """Handle gradient pull request."""
        return GossipMessage.create_push_gradient(
            sender_id=self.local_peer.id,
            sequence=self._next_sequence(),
            gradients=self.current_gradients if self.current_gradients is not None else np.array([]),
            model_version=self.model_version,
            step=self.step,
            indices=self.current_indices,
        )
    
    async def _handle_push_pull_gradient(self, msg: GossipMessage) -> GossipMessage:
        """Handle push-pull gradient exchange."""
        # Process incoming gradient
        if msg.gradient_data is not None:
            await self._process_gradient(
                peer_id=msg.sender_id,
                gradients=msg.gradient_data,
                indices=msg.indices,
                model_version=msg.model_version,
                step=msg.step,
            )
        
        # Return our gradients
        return GossipMessage.create_push_gradient(
            sender_id=self.local_peer.id,
            sequence=self._next_sequence(),
            gradients=self.current_gradients if self.current_gradients is not None else np.array([]),
            model_version=self.model_version,
            step=self.step,
            indices=self.current_indices,
        )
    
    async def _process_gradient(
        self,
        peer_id: str,
        gradients: np.ndarray,
        indices: Optional[np.ndarray],
        model_version: int,
        step: int,
    ) -> None:
        """Process and aggregate received gradients."""
        async with self.cache_lock:
            # Check if we already have newer gradients from this peer
            if peer_id in self.gradient_cache:
                cached = self.gradient_cache[peer_id]
                if cached.step >= step and cached.model_version >= model_version:
                    return  # Stale update
            
            # Store in cache
            self.gradient_cache[peer_id] = GradientState(
                gradients=gradients,
                indices=indices,
                model_version=model_version,
                step=step,
                contributors={peer_id},
                timestamp=time.time(),
            )
        
        # Aggregate
        await self._aggregate_gradients()
        
        # Notify callback
        if self.on_gradient_received:
            self.on_gradient_received(gradients, model_version, step)
    
    async def _aggregate_gradients(self) -> None:
        """Aggregate all cached gradients using averaging."""
        async with self.aggregation_lock:
            async with self.cache_lock:
                if not self.gradient_cache or self.current_gradients is None:
                    return
                
                # Simple averaging (FedAvg style)
                all_gradients = [self.current_gradients]
                for state in self.gradient_cache.values():
                    if state.gradients.shape == self.current_gradients.shape:
                        all_gradients.append(state.gradients)
                
                if len(all_gradients) > 1:
                    self.aggregated_gradients = np.mean(all_gradients, axis=0).astype(np.float32)
                    self.gradients_aggregated = len(all_gradients)
    
    async def _gossip_loop(self) -> None:
        """Main gossip loop."""
        while self._running:
            try:
                await self._gossip_round()
                self.rounds_completed += 1
                await asyncio.sleep(self.config.interval_ms / 1000.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Gossip loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _gossip_round(self) -> None:
        """Execute a single gossip round."""
        if self.current_gradients is None:
            return
        
        # Select peers for gossip
        peers = await self._select_gossip_peers()
        if not peers:
            return
        
        # Create message
        if self.config.push_pull:
            msg = GossipMessage.create_push_pull_gradient(
                sender_id=self.local_peer.id,
                sequence=self._next_sequence(),
                gradients=self.current_gradients,
                model_version=self.model_version,
                step=self.step,
                indices=self.current_indices,
            )
        else:
            msg = GossipMessage.create_push_gradient(
                sender_id=self.local_peer.id,
                sequence=self._next_sequence(),
                gradients=self.current_gradients,
                model_version=self.model_version,
                step=self.step,
                indices=self.current_indices,
            )
        
        # Send to selected peers
        for peer in peers:
            try:
                # This would send through transport layer
                self.messages_sent += 1
                # Clear backoff on success
                if peer.id in self.backoff:
                    del self.backoff[peer.id]
            except Exception as e:
                logger.warning(f"Failed to gossip with {peer.id}: {e}")
                self._apply_backoff(peer.id)
    
    async def _select_gossip_peers(self) -> List[Peer]:
        """Select peers for gossip using locality-aware selection."""
        all_peers = await self.membership.get_alive_peers()
        
        # Filter out peers in backoff
        now = time.time()
        available = [
            p for p in all_peers
            if p.id not in self.backoff or now >= self.backoff[p.id]
        ]
        
        if not available:
            return []
        
        # Locality-aware selection:
        # - Prefer peers in same datacenter (lower latency)
        # - Mix with some random peers for faster global convergence
        
        same_dc = [p for p in available if p.is_local(self.local_peer)]
        other_dc = [p for p in available if not p.is_local(self.local_peer)]
        
        selected = []
        
        # Select ~2/3 from same DC if available
        local_count = min(len(same_dc), (self.config.fanout * 2) // 3)
        if local_count > 0:
            selected.extend(random.sample(same_dc, local_count))
        
        # Fill rest from other DCs
        remaining = self.config.fanout - len(selected)
        if remaining > 0 and other_dc:
            selected.extend(random.sample(other_dc, min(remaining, len(other_dc))))
        
        # If still not enough, add more from same DC
        remaining = self.config.fanout - len(selected)
        if remaining > 0:
            remaining_local = [p for p in same_dc if p not in selected]
            selected.extend(random.sample(remaining_local, min(remaining, len(remaining_local))))
        
        return selected
    
    def _apply_backoff(self, peer_id: str) -> None:
        """Apply exponential backoff for a failed peer."""
        if peer_id in self.backoff:
            delay = min(self.max_backoff, (self.backoff[peer_id] - time.time()) * 2)
        else:
            delay = 1.0
        
        self.backoff[peer_id] = time.time() + delay
    
    def _next_sequence(self) -> int:
        """Get next sequence number."""
        self.sequence_num += 1
        return self.sequence_num
    
    async def get_stats(self) -> dict:
        """Get protocol statistics."""
        return {
            "rounds_completed": self.rounds_completed,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "gradients_aggregated": self.gradients_aggregated,
            "cached_peers": len(self.gradient_cache),
            "model_version": self.model_version,
            "step": self.step,
        }
