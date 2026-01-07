"""TraceRank: Temporal reinforcement scoring for M3"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from vestig.core.models import EventNode


@dataclass
class TraceRankConfig:
    """TraceRank configuration"""

    enabled: bool = True
    tau_days: float = 21.0  # Recency decay time constant (3 weeks)
    cooldown_hours: float = 24.0  # Anti-burst window
    burst_discount: float = 0.2  # Weight for events in cooldown
    k: float = 0.35  # TraceRank boost strength

    # Enhanced force multiplier parameters
    graph_connectivity_enabled: bool = True  # Boost based on inbound edges
    graph_k: float = 0.15  # Graph connectivity boost strength
    temporal_decay_enabled: bool = True  # Apply age-based decay for dynamic facts
    dynamic_tau_days: float = 90.0  # Decay time for dynamic facts (3 months)
    ephemeral_tau_days: float | None = None  # Optional override for ephemeral decay
    static_boost: float = 1.0  # Multiplier for static facts (no decay)


def compute_tracerank_multiplier(
    events: list[EventNode], config: TraceRankConfig, query_time: datetime | None = None
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


def compute_graph_connectivity_boost(inbound_edge_count: int, config: TraceRankConfig) -> float:
    """
    Compute boost multiplier based on graph connectivity.

    Memories with more inbound edges (RELATED, MENTIONS) are more central
    to the knowledge graph and may be more relevant.

    Algorithm:
    - boost = 1 + graph_k * log1p(edge_count)
    - More edges → higher boost (logarithmic growth)

    Args:
        inbound_edge_count: Number of edges pointing to this memory
        config: TraceRankConfig with graph_k parameter

    Returns:
        Multiplier ∈ [1.0, ∞) based on connectivity

    Examples:
        >>> config = TraceRankConfig(graph_connectivity_enabled=True, graph_k=0.15)
        >>> # No edges → multiplier = 1.0
        >>> compute_graph_connectivity_boost(0, config)
        1.0
        >>> # 5 edges → slight boost
        >>> compute_graph_connectivity_boost(5, config)  # ~1.27
        >>> # 20 edges → stronger boost
        >>> compute_graph_connectivity_boost(20, config)  # ~1.46
    """
    if not config.graph_connectivity_enabled or inbound_edge_count <= 0:
        return 1.0

    # Logarithmic boost: 1 + graph_k * log1p(edge_count)
    boost = 1.0 + config.graph_k * math.log1p(inbound_edge_count)
    return boost


def compute_temporal_confidence(
    t_valid: str,
    temporal_stability: str,
    config: TraceRankConfig,
    query_time: datetime | None = None,
) -> float:
    """
    Compute confidence multiplier based on temporal stability and age.

    Dynamic facts degrade over time (older = less confident).
    Static facts don't degrade (permanent truths).

    Algorithm:
    - If static: multiplier = static_boost (default 1.0, no decay)
    - If dynamic: multiplier = exp(-age_days / dynamic_tau_days)
    - If ephemeral: multiplier = exp(-age_days / ephemeral_tau_days)
    - If unknown: multiplier = 1.0 (neutral)

    Args:
        t_valid: When the fact became true (ISO 8601)
        temporal_stability: "static" | "dynamic" | "ephemeral" | "unknown"
        config: TraceRankConfig with temporal decay parameters
        query_time: Time to compute age from (default: now)

    Returns:
        Confidence multiplier ∈ (0.0, 1.0] for dynamic facts, 1.0 for static

    Examples:
        >>> config = TraceRankConfig(temporal_decay_enabled=True, dynamic_tau_days=90.0)
        >>> # Static fact, any age → no decay
        >>> compute_temporal_confidence("2020-01-01", "static", config)
        1.0
        >>> # Dynamic fact, 1 year old → significant decay
        >>> compute_temporal_confidence("2024-01-01", "dynamic", config)  # ~0.025
        >>> # Dynamic fact, 1 day old → minimal decay
        >>> compute_temporal_confidence("2025-12-26", "dynamic", config)  # ~0.99
    """
    if not config.temporal_decay_enabled:
        return 1.0

    # Static facts don't decay
    if temporal_stability == "static":
        return config.static_boost

    # Unknown stability → neutral (no decay)
    if temporal_stability == "unknown":
        return 1.0

    # Dynamic/ephemeral facts: exponential decay based on age
    if query_time is None:
        query_time = datetime.now(timezone.utc)

    try:
        t_valid_dt = datetime.fromisoformat(t_valid.replace("Z", "+00:00"))
        age_days = (query_time - t_valid_dt).total_seconds() / 86400

        # Only apply decay for positive age (past facts)
        if age_days > 0:
            tau_days = config.dynamic_tau_days
            if temporal_stability == "ephemeral":
                tau_days = config.ephemeral_tau_days or max(3.0, min(7.0, tau_days / 4))

            # Exponential decay: exp(-age / tau)
            confidence = math.exp(-age_days / tau_days)
            return max(0.01, confidence)  # Floor at 1% confidence

        # Future facts or same-day facts: no decay
        return 1.0

    except (ValueError, AttributeError):
        # Malformed t_valid → neutral confidence
        return 1.0


def compute_enhanced_multiplier(
    memory_id: str,
    temporal_stability: str,
    t_valid: str,
    inbound_edge_count: int,
    reinforcement_events: list[EventNode],
    config: TraceRankConfig,
    query_time: datetime | None = None,
) -> float:
    """
    Compute comprehensive force multiplier incorporating all factors.

    final_multiplier = reinforcement_boost × graph_boost × temporal_confidence

    Args:
        memory_id: Memory ID (for logging/debugging)
        temporal_stability: "static" | "dynamic" | "ephemeral" | "unknown"
        t_valid: When fact became true (ISO 8601)
        inbound_edge_count: Number of edges pointing to this memory
        reinforcement_events: List of reinforcement events
        config: TraceRankConfig with all parameters
        query_time: Time to compute from (default: now)

    Returns:
        Comprehensive multiplier to apply to semantic similarity

    Example:
        >>> config = TraceRankConfig()
        >>> # Dynamic fact: 30 days old, 5 edges, 2 reinforcements
        >>> compute_enhanced_multiplier(
        ...     "mem_123", "dynamic", "2025-11-27", 5, [...], config
        ... )
        # Returns ~1.5 (reinforcement × graph × temporal_decay)
    """
    # Component 1: Reinforcement boost (existing TraceRank)
    reinforcement_boost = compute_tracerank_multiplier(reinforcement_events, config, query_time)

    # Component 2: Graph connectivity boost
    graph_boost = compute_graph_connectivity_boost(inbound_edge_count, config)

    # Component 3: Temporal confidence (decay for dynamic facts)
    temporal_confidence = compute_temporal_confidence(
        t_valid, temporal_stability, config, query_time
    )

    # Combine all factors
    final_multiplier = reinforcement_boost * graph_boost * temporal_confidence

    return final_multiplier
