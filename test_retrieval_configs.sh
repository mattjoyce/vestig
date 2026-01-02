#!/bin/bash
# Test different TraceRank configurations on existing embedding databases
# No need to regenerate embeddings - just re-run queries with different configs

set -e

# Configuration
QA_FILE="qa_matterbase_projects.json"
RESULTS_DIR="retrieval_comparison_$(date +%s)"

# Test configurations to compare
# Format: label:db_name:config_suffix
CONFIGS=(
  "tracerank_full:matt_embeddinggemma:embeddinggemma"
  "tracerank_no_graph:matt_embeddinggemma:embeddinggemma-no-graph"
  "tracerank_off:matt_embeddinggemma:embeddinggemma-no-tracerank"
)

echo "========================================"
echo "Retrieval Configuration Comparison"
echo "========================================"
echo ""
echo "Testing model: embeddinggemma (best performer)"
echo "QA file: $QA_FILE"
echo "Results directory: $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

# Function to test a configuration
test_config() {
  local label=$1
  local db_name=$2
  local config_suffix=$3
  local config_file="config-${config_suffix}.yaml"

  echo ""
  echo "========================================"
  echo "Testing: $label"
  echo "Config: $config_file"
  echo "========================================"

  if [ ! -f "$config_file" ]; then
    echo "ERROR: Config not found: $config_file"
    return 1
  fi

  if [ ! -f "$db_name.db" ]; then
    echo "ERROR: Database not found: $db_name.db"
    return 1
  fi

  # Run search tests
  echo "Running search tests..."
  python3 ../test_qa_harness.py search "$config_file" "$QA_FILE" 2>&1 | tee "$RESULTS_DIR/${label}_search.log"

  # Copy results
  local latest_search=$(ls -t qa_results_search_*.json 2>/dev/null | head -1)
  if [ -f "$latest_search" ]; then
    cp "$latest_search" "$RESULTS_DIR/${label}_search_results.json"
  fi

  # Run recall tests
  echo "Running recall tests..."
  python3 ../test_qa_harness.py recall "$config_file" "$QA_FILE" 2>&1 | tee "$RESULTS_DIR/${label}_recall.log"

  # Copy results
  local latest_recall=$(ls -t qa_results_recall_*.json 2>/dev/null | head -1)
  if [ -f "$latest_recall" ]; then
    cp "$latest_recall" "$RESULTS_DIR/${label}_recall_results.json"
  fi

  echo "$label testing complete!"
}

# Change to test directory
cd test || exit 1

# Test each configuration
for config_spec in "${CONFIGS[@]}"; do
  IFS=':' read -r label db_name config_suffix <<< "$config_spec"
  test_config "$label" "$db_name" "$config_suffix"
done

echo ""
echo "========================================"
echo "All retrieval tests complete!"
echo "========================================"
echo "Results saved to: ./test/$RESULTS_DIR"
echo ""
echo "To analyze results, run:"
echo "  python3 analyze_embedding_results.py ./test/$RESULTS_DIR"
