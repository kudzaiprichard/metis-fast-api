"""
SSE Stream Manager for Simulations.

Manages asyncio.Queue instances per simulation, supports multiple
concurrent viewers per simulation, and cleans up on completion.
"""

import asyncio
import logging
from uuid import UUID
from typing import Dict, AsyncGenerator

logger = logging.getLogger(__name__)

# Sentinel value pushed to signal simulation is done
STREAM_COMPLETE = "__STREAM_COMPLETE__"
STREAM_ERROR = "__STREAM_ERROR__"


class StreamManager:
    """
    Manages SSE streams for active simulations.

    Each simulation has one source queue (written by the runner).
    Multiple viewers can subscribe — each gets their own queue
    that receives a copy of every event.
    """

    def __init__(self):
        # simulation_id -> list of subscriber queues
        self._subscribers: Dict[UUID, list[asyncio.Queue]] = {}
        # simulation_id -> True if simulation is still producing events
        self._active: Dict[UUID, bool] = {}
        self._lock = asyncio.Lock()

    async def register_simulation(self, simulation_id: UUID) -> None:
        """Called by runner before the loop starts."""
        async with self._lock:
            self._subscribers[simulation_id] = []
            self._active[simulation_id] = True
            logger.info("Stream registered: %s", simulation_id)

    async def push_event(self, simulation_id: UUID, event: dict) -> None:
        """Called by runner for each step. Fans out to all subscribers."""
        async with self._lock:
            queues = self._subscribers.get(simulation_id, [])
            for q in queues:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning(
                        "Queue full for simulation %s, dropping event", simulation_id
                    )

    async def push_complete(self, simulation_id: UUID) -> None:
        """Called by runner when simulation finishes (success or failure)."""
        async with self._lock:
            queues = self._subscribers.get(simulation_id, [])
            for q in queues:
                try:
                    q.put_nowait(STREAM_COMPLETE)
                except asyncio.QueueFull:
                    pass
            self._active[simulation_id] = False
            logger.info("Stream completed: %s", simulation_id)

    async def push_error(self, simulation_id: UUID, error: str) -> None:
        """Called by runner on failure before push_complete."""
        async with self._lock:
            queues = self._subscribers.get(simulation_id, [])
            for q in queues:
                try:
                    q.put_nowait({"type": STREAM_ERROR, "error": error})
                except asyncio.QueueFull:
                    pass

    async def subscribe(self, simulation_id: UUID) -> asyncio.Queue:
        """
        Called by SSE endpoint. Returns a queue that will receive events.
        Returns None if simulation is not registered.
        """
        async with self._lock:
            if simulation_id not in self._subscribers:
                return None

            q = asyncio.Queue(maxsize=500)
            self._subscribers[simulation_id].append(q)
            logger.info(
                "Subscriber added for %s (total: %d)",
                simulation_id,
                len(self._subscribers[simulation_id]),
            )
            return q

    async def unsubscribe(self, simulation_id: UUID, queue: asyncio.Queue) -> None:
        """Called when an SSE client disconnects."""
        async with self._lock:
            queues = self._subscribers.get(simulation_id, [])
            if queue in queues:
                queues.remove(queue)
                logger.info(
                    "Subscriber removed for %s (remaining: %d)",
                    simulation_id,
                    len(queues),
                )

    async def cleanup(self, simulation_id: UUID) -> None:
        """Remove all state for a simulation. Called after all viewers disconnect."""
        async with self._lock:
            self._subscribers.pop(simulation_id, None)
            self._active.pop(simulation_id, None)
            logger.info("Stream cleaned up: %s", simulation_id)

    def is_active(self, simulation_id: UUID) -> bool:
        """Check if a simulation is still producing events."""
        return self._active.get(simulation_id, False)

    def is_registered(self, simulation_id: UUID) -> bool:
        """Check if a simulation has an active stream."""
        return simulation_id in self._subscribers


# Global singleton
stream_manager = StreamManager()