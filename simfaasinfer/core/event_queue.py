"""Event queue implementation for discrete event simulation."""

import heapq
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional


class EventType(Enum):
    """Types of events in the simulation."""
    # Request lifecycle events
    REQUEST_ARRIVAL = "request_arrival"
    REQUEST_SCHEDULED = "request_scheduled"
    PREFILL_START = "prefill_start"
    PREFILL_COMPLETE = "prefill_complete"
    DECODE_START = "decode_start"
    DECODE_COMPLETE = "decode_complete"
    REQUEST_COMPLETE = "request_complete"

    # Batch execution events
    BATCH_START = "batch_start"
    BATCH_COMPLETE = "batch_complete"

    # Instance lifecycle events
    INSTANCE_START = "instance_start"
    INSTANCE_READY = "instance_ready"
    INSTANCE_SHUTDOWN = "instance_shutdown"

    # Scaling events
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALING_DECISION = "scaling_decision"

    # Metrics collection
    METRICS_COLLECTION = "metrics_collection"

    # Simulation control
    SIMULATION_END = "simulation_end"


@dataclass(order=True)
class Event:
    """Event in the discrete event simulation.

    Attributes:
        time: Event timestamp
        event_type: Type of event
        data: Event-specific data
        priority: Priority for tie-breaking (lower = higher priority)
    """
    time: float
    priority: int = field(default=0)
    event_type: EventType = field(compare=False)
    data: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        """Validate event after initialization."""
        if self.time < 0:
            raise ValueError("Event time cannot be negative")


class EventQueue:
    """Priority queue for managing simulation events.

    Events are ordered by time, with earlier events processed first.
    For events at the same time, priority field determines order.
    """

    def __init__(self):
        """Initialize empty event queue."""
        self._queue = []
        self._event_count = 0

    def push(self, event: Event) -> None:
        """Add event to the queue.

        Args:
            event: Event to add
        """
        heapq.heappush(self._queue, event)
        self._event_count += 1

    def pop(self) -> Event:
        """Remove and return the next event.

        Returns:
            Next event to process

        Raises:
            IndexError: If queue is empty
        """
        if self.is_empty():
            raise IndexError("Cannot pop from empty event queue")
        return heapq.heappop(self._queue)

    def peek(self) -> Optional[Event]:
        """Return the next event without removing it.

        Returns:
            Next event, or None if queue is empty
        """
        return self._queue[0] if self._queue else None

    def is_empty(self) -> bool:
        """Check if queue is empty.

        Returns:
            True if queue is empty
        """
        return len(self._queue) == 0

    def size(self) -> int:
        """Get number of events in queue.

        Returns:
            Number of events
        """
        return len(self._queue)

    def clear(self) -> None:
        """Remove all events from queue."""
        self._queue.clear()

    def __len__(self) -> int:
        """Get number of events in queue."""
        return len(self._queue)

    def __repr__(self) -> str:
        """String representation of event queue."""
        return f"EventQueue(size={len(self._queue)}, next={self.peek()})"