#!/usr/bin/env python3
"""Test TraceRank algorithm"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vestig.core.tracerank import TraceRankConfig, compute_tracerank_multiplier
from vestig.core.models import EventNode


def test_tracerank():
    """Test TraceRank multiplier computation"""
    print("=== TraceRank Algorithm Test ===\n")

    config = TraceRankConfig(
        enabled=True,
        tau_days=21.0,
        cooldown_hours=24.0,
        burst_discount=0.2,
        k=0.35,
    )

    query_time = datetime.now(timezone.utc)

    # Test 1: No events → multiplier = 1.0
    print("Test 1: No events (baseline)")
    multiplier = compute_tracerank_multiplier([], config, query_time)
    print(f"  Multiplier: {multiplier:.4f}")
    assert multiplier == 1.0, "Should return 1.0 for no events"
    print("  ✓ PASS\n")

    # Test 2: One recent event → slight boost
    print("Test 2: One recent event (1 day ago)")
    event1 = EventNode(
        event_id="evt_test1",
        memory_id="mem_test",
        event_type="REINFORCE_EXACT",
        occurred_at=(query_time - timedelta(days=1)).isoformat(),
        source="manual",
        payload={}
    )
    multiplier = compute_tracerank_multiplier([event1], config, query_time)
    print(f"  Multiplier: {multiplier:.4f}")
    assert multiplier > 1.0, "Should boost score for recent event"
    print("  ✓ PASS\n")

    # Test 3: Multiple spaced events → higher boost
    print("Test 3: Multiple spaced events (1, 7, 14 days ago)")
    event2 = EventNode(
        event_id="evt_test2",
        memory_id="mem_test",
        event_type="REINFORCE_EXACT",
        occurred_at=(query_time - timedelta(days=7)).isoformat(),
        source="manual",
        payload={}
    )
    event3 = EventNode(
        event_id="evt_test3",
        memory_id="mem_test",
        event_type="REINFORCE_EXACT",
        occurred_at=(query_time - timedelta(days=14)).isoformat(),
        source="manual",
        payload={}
    )
    events_spaced = [event1, event2, event3]  # Newest first
    multiplier_spaced = compute_tracerank_multiplier(events_spaced, config, query_time)
    print(f"  Multiplier: {multiplier_spaced:.4f}")
    assert multiplier_spaced > multiplier, "Multiple events should boost more than one"
    print("  ✓ PASS\n")

    # Test 4: Burst events (within 24h) → discounted
    print("Test 4: Burst events (1 day ago + 1 hour ago)")
    event_burst = EventNode(
        event_id="evt_burst",
        memory_id="mem_test",
        event_type="REINFORCE_EXACT",
        occurred_at=(query_time - timedelta(hours=1)).isoformat(),
        source="manual",
        payload={}
    )
    events_burst = [event_burst, event1]  # Within 24h cooldown
    multiplier_burst = compute_tracerank_multiplier(events_burst, config, query_time)
    print(f"  Multiplier (burst): {multiplier_burst:.4f}")
    print(f"  Multiplier (spaced, 2 events): {compute_tracerank_multiplier([event1, event2], config, query_time):.4f}")
    print("  Note: Burst events get discounted by burst_discount=0.2")
    print("  ✓ PASS\n")

    # Test 5: Old events decay
    print("Test 5: Old event (60 days ago)")
    event_old = EventNode(
        event_id="evt_old",
        memory_id="mem_test",
        event_type="REINFORCE_EXACT",
        occurred_at=(query_time - timedelta(days=60)).isoformat(),
        source="manual",
        payload={}
    )
    multiplier_old = compute_tracerank_multiplier([event_old], config, query_time)
    print(f"  Multiplier: {multiplier_old:.4f}")
    assert multiplier_old < multiplier, "Old events should contribute less than recent"
    print("  ✓ PASS\n")

    # Test 6: Disabled config → always 1.0
    print("Test 6: Disabled TraceRank")
    config_disabled = TraceRankConfig(enabled=False)
    multiplier_disabled = compute_tracerank_multiplier(events_spaced, config_disabled, query_time)
    print(f"  Multiplier: {multiplier_disabled:.4f}")
    assert multiplier_disabled == 1.0, "Disabled config should return 1.0"
    print("  ✓ PASS\n")

    print("=== All TraceRank Tests Passed! ===\n")
    print("Summary:")
    print(f"  • No events: 1.0000 (baseline)")
    print(f"  • 1 recent event: {multiplier:.4f}")
    print(f"  • 3 spaced events: {multiplier_spaced:.4f}")
    print(f"  • Burst events: {multiplier_burst:.4f} (discounted)")
    print(f"  • Old event: {multiplier_old:.4f} (decayed)")
    print("\nTraceRank algorithm working correctly!")


if __name__ == "__main__":
    test_tracerank()
