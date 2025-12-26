"""TraceRank: Temporal reinforcement scoring for M3"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional
import math
from vestig.core.models import EventNode


@dataclass
class TraceRankConfig:
    """TraceRank configuration"""
    enabled: bool = True
    tau_days: float = 21.0          # Recency decay time constant (3 weeks)
    cooldown_hours: float = 24.0     # Anti-burst window
    burst_discount: float = 0.2      # Weight for events in cooldown
    k: float = 0.35                  # TraceRank boost strength


def compute_tracerank_multiplier(
    events: List[EventNode],
    config: TraceRankConfig,
    query_time: Optional[datetime] = None
) -> float:
    """
    Compute TraceRank multiplier from reinforcement events.

    Algorithm:
    1. For each event, compute recency weight: exp(-Δt / τ)
    2. Apply burst discount if event within cooldown of previous
    3. Sum weighted contributions: trace = Σ(w_recency * w_burst)
    4. Convert to multiplier: 1 + k * log1p(trace)

    Args:
        events: List of EventNode objects (should be REINFORCE_* events)
        config: TraceRankConfig with algorithm parameters
        query_time: Time to compute recency from (default: now)

    Returns:
        Multiplier ∈ [1.0, ∞) to boost semantic similarity

    Examples:
        >>> config = TraceRankConfig(enabled=True, tau_days=21.0, k=0.35)
        >>> # No events → multiplier = 1.0 (no boost)
        >>> compute_tracerank_multiplier([], config)
        1.0

        >>> # One recent event → slight boost
        >>> # Multiple spaced events → higher boost
        >>> # Burst events (within 24h) → discounted
    """
    if not config.enabled or not events:
        return 1.0

    if query_time is None:
        query_time = datetime.now(timezone.utc)

    trace = 0.0
    prev_event_time = None

    # Events should be sorted newest first (get_reinforcement_events does this)
    for event in events:
        # Parse event timestamp
        event_time = datetime.fromisoformat(event.occurred_at.replace("Z", "+00:00"))

        # Recency decay: exp(-Δt / τ)
        delta_days = (query_time - event_time).total_seconds() / 86400
        w_recency = math.exp(-delta_days / config.tau_days)

        # Anti-burst: discount events within cooldown
        w_burst = 1.0
        if prev_event_time is not None:
            gap_hours = (prev_event_time - event_time).total_seconds() / 3600
            if gap_hours < config.cooldown_hours:
                w_burst = config.burst_discount

        trace += w_recency * w_burst
        prev_event_time = event_time

    # Convert trace to multiplier: 1 + k * log1p(trace)
    # log1p(x) = log(1+x) is numerically stable for small x
    multiplier = 1.0 + config.k * math.log1p(trace)

    return multiplier
