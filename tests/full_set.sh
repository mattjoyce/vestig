#!/bin/bash
# Full test suite - runs pytest tests + smoke tests
# Exit on first failure to catch regressions early

set -e

# Embedding provider is llm CLI (Ollama), not HuggingFace

# Require backend to prevent silent skips
export VESTIG_REQUIRE_BACKEND=1

source ~/Environments/vestig/bin/activate

echo "======================================================================"
echo "VESTIG FULL TEST SUITE"
echo "======================================================================"
echo ""

# Phase 1: Pytest tests (unit + component + integration)
echo "=== Phase 1: Pytest Tests ==="
pytest tests/ -v --tb=short
echo ""

# Phase 2: Shell-based smoke tests (CLI integration)
echo "=== Phase 2: Smoke Tests ==="
bash tests/test_m2_smoke.sh
bash tests/test_m3_smoke.sh
bash tests/test_m4_smoke.sh
echo ""

echo "======================================================================"
echo "✅ ALL TESTS PASSED"
echo "======================================================================"
