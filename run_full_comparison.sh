#!/bin/bash
# Full embedding + TraceRank comparison test suite
# Tests 7 models × 3 TraceRank configs = 21 configurations
# Estimated runtime: ~35 minutes

set -e

# Configuration
QA_FILE="qa_matterbase_projects.json"
RESULTS_DIR="full_comparison_$(date +%s)"

# Models to test (model_name:db_name:base_config)
MODELS=(
  "embeddinggemma:matt_embeddinggemma:config-embeddinggemma.yaml"
  "all-minilm:matt_all_minilm:config-all-minilm.yaml"
  "bge-m3:matt_bge_m3:config-bge-m3.yaml"
  "mxbai-embed-large:matt_mxbai_embed:config-mxbai-embed.yaml"
  "granite-embedding:matt_granite_embed:config-granite-embed.yaml"
  "nomic-embed-text:matt_nomic_embed:config-nomic-embed.yaml"
  "ada-002:matt_ada_002:config-ada-002.yaml"
)

# TraceRank variations
# Format: suffix:description:k_value:graph_enabled:graph_k
TRACERANK_CONFIGS=(
  "full:TraceRank-Full:0.35:true:0.15"
  "no-graph:TraceRank-NoGraph:0.35:false:0.0"
  "off:NoTraceRank:0.0:false:0.0"
)

echo "========================================"
echo "FULL EMBEDDING + TRACERANK COMPARISON"
echo "========================================"
echo ""
echo "Models: ${#MODELS[@]}"
echo "TraceRank configs: ${#TRACERANK_CONFIGS[@]}"
echo "Total configurations: $((${#MODELS[@]} * ${#TRACERANK_CONFIGS[@]}))"
echo "QA file: $QA_FILE"
echo "Results directory: $RESULTS_DIR"
echo ""
echo "Estimated runtime: ~35 minutes"
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Change to test directory
cd test || exit 1

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to create config variant
create_config_variant() {
  local base_config=$1
  local suffix=$2
  local k_value=$3
  local graph_enabled=$4
  local graph_k=$5

  local output_config="${base_config%.yaml}-${suffix}.yaml"

  # Copy base config
  cp "$base_config" "$output_config"

  # Update TraceRank settings using sed
  # Set k value
  sed -i.bak "s/k: [0-9.]\+/k: $k_value/" "$output_config"

  # Add or update graph_connectivity_enabled
  if grep -q "graph_connectivity_enabled:" "$output_config"; then
    sed -i.bak "s/graph_connectivity_enabled: .*/graph_connectivity_enabled: $graph_enabled/" "$output_config"
  else
    # Add after k: line
    sed -i.bak "/k: $k_value/a\\
    graph_connectivity_enabled: $graph_enabled  # Auto-generated" "$output_config"
  fi

  # Add or update graph_k
  if grep -q "graph_k:" "$output_config"; then
    sed -i.bak "s/graph_k: .*/graph_k: $graph_k/" "$output_config"
  else
    # Add after graph_connectivity_enabled line
    sed -i.bak "/graph_connectivity_enabled:/a\\
    graph_k: $graph_k                       # Auto-generated" "$output_config"
  fi

  # Clean up backup files
  rm -f "${output_config}.bak"

  echo "$output_config"
}

# Function to run tests for a single configuration
run_tests() {
  local model_name=$1
  local db_name=$2
  local config_file=$3
  local test_label=$4

  echo ""
  echo "========================================"
  echo "Testing: $test_label"
  echo "Model: $model_name"
  echo "Config: $config_file"
  echo "========================================"

  if [ ! -f "$config_file" ]; then
    echo "ERROR: Config not found: $config_file"
    return 1
  fi

  if [ ! -f "${db_name}.db" ]; then
    echo "ERROR: Database not found: ${db_name}.db"
    return 1
  fi

  # Run search tests
  echo "Running search tests..."
  python3 ../test_qa_harness.py search "$config_file" "$QA_FILE" > "$RESULTS_DIR/${test_label}_search.log" 2>&1 || true

  # Copy results
  local latest_search=$(ls -t qa_results_search_*.json 2>/dev/null | head -1)
  if [ -f "$latest_search" ]; then
    cp "$latest_search" "$RESULTS_DIR/${test_label}_search_results.json"
    echo "  ✓ Search results saved"
  fi

  # Run recall tests
  echo "Running recall tests..."
  python3 ../test_qa_harness.py recall "$config_file" "$QA_FILE" > "$RESULTS_DIR/${test_label}_recall.log" 2>&1 || true

  # Copy results
  local latest_recall=$(ls -t qa_results_recall_*.json 2>/dev/null | head -1)
  if [ -f "$latest_recall" ]; then
    cp "$latest_recall" "$RESULTS_DIR/${test_label}_recall_results.json"
    echo "  ✓ Recall results saved"
  fi

  echo "  ✓ $test_label complete!"
}

# Main test loop
total_tests=$((${#MODELS[@]} * ${#TRACERANK_CONFIGS[@]}))
current_test=0

for model_spec in "${MODELS[@]}"; do
  IFS=':' read -r model_name db_name base_config <<< "$model_spec"

  echo ""
  echo "========================================"
  echo "MODEL: $model_name"
  echo "========================================"

  for tracerank_spec in "${TRACERANK_CONFIGS[@]}"; do
    IFS=':' read -r suffix description k_value graph_enabled graph_k <<< "$tracerank_spec"

    current_test=$((current_test + 1))

    echo ""
    echo "[$current_test/$total_tests] Preparing: $model_name - $description"

    # Create config variant
    config_variant=$(create_config_variant "$base_config" "$suffix" "$k_value" "$graph_enabled" "$graph_k")

    # Create test label
    test_label="${model_name}_${suffix}"

    # Run tests
    run_tests "$model_name" "$db_name" "$config_variant" "$test_label"
  done
done

echo ""
echo "========================================"
echo "ALL TESTS COMPLETE!"
echo "========================================"
echo "Total configurations tested: $total_tests"
echo "Results saved to: ./test/$RESULTS_DIR"
echo ""
echo "To analyze results, run:"
echo "  python3 analyze_embedding_results.py ./test/$RESULTS_DIR"
echo ""
echo "Detailed comparison will show:"
echo "  - Best embedding model"
echo "  - Impact of TraceRank on each model"
echo "  - Impact of graph connectivity boost"
echo "  - Commercial (ada-002) vs open-source comparison"
