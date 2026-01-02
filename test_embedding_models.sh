#!/bin/bash
# Test harness for comparing embedding model performance
# This script regenerates embeddings for each model and runs QA tests

set -e  # Exit on error

# Configuration
BASE_DB="matt_cerebras2.db"
QA_FILE="qa_matterbase_projects.json"
RESULTS_DIR="embedding_comparison_$(date +%s)"

# Embedding models to test (model_name:dimension:db_name:config_file)
# Note: paths are relative to the test directory
MODELS=(
  "nomic-embed-text:768:matt_nomic_embed.db:config-nomic-embed.yaml"
  "mxbai-embed-large:1024:matt_mxbai_embed.db:config-mxbai-embed.yaml"
  "all-minilm:384:matt_all_minilm.db:config-all-minilm.yaml"
  "bge-m3:1024:matt_bge_m3.db:config-bge-m3.yaml"
  "embeddinggemma:768:matt_embeddinggemma.db:config-embeddinggemma.yaml"
  "granite-embedding:768:matt_granite_embed.db:config-granite-embed.yaml"
)

echo "========================================"
echo "Embedding Model Comparison Test Suite"
echo "========================================"
echo ""

# Change to test directory
cd test || exit 1
echo "Working directory: $(pwd)"
echo "Base database: $BASE_DB"
echo "QA file: $QA_FILE"
echo "Results directory: $RESULTS_DIR"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check if base database exists
if [ ! -f "$BASE_DB" ]; then
  echo "ERROR: Base database not found at $BASE_DB"
  echo "Please ensure you have a populated database to test with."
  exit 1
fi

# Check how many memories are in the base database
MEMORY_COUNT=$(sqlite3 "$BASE_DB" "SELECT COUNT(*) FROM memories;" 2>/dev/null || echo "0")
echo "Base database contains $MEMORY_COUNT memories"
echo ""

if [ "$MEMORY_COUNT" -eq 0 ]; then
  echo "ERROR: Base database is empty!"
  exit 1
fi

# Function to run tests for a single model
test_model() {
  local model_name=$1
  local dimension=$2
  local db_path=$3
  local config_path=$4

  echo ""
  echo "========================================"
  echo "Testing: $model_name (dim=$dimension)"
  echo "========================================"

  # Copy base database
  echo "Copying base database to $db_path..."
  cp "$BASE_DB" "$db_path"

  # Warm up the model (load into ollama memory)
  echo "Warming up $model_name model..."
  llm embed -c "warmup" -m "$model_name" > /dev/null 2>&1 || echo "  (warmup failed, continuing anyway)"
  sleep 1
  echo "Model warmed up"

  # Regenerate embeddings
  echo "Regenerating embeddings with $model_name..."
  local regen_start=$(date +%s)
  vestig --config "$config_path" memory regen-embeddings 2>&1 | tee "$RESULTS_DIR/${model_name}_regen.log"
  local regen_end=$(date +%s)
  local regen_duration=$((regen_end - regen_start))
  echo "Embedding regeneration completed in ${regen_duration}s"

  # Run search tests
  echo "Running search tests..."
  python3 ../test_qa_harness.py search "$config_path" "$QA_FILE" 2>&1 | tee "$RESULTS_DIR/${model_name}_search.log"

  # Copy the results file to our results directory with a clear name
  local latest_search=$(ls -t qa_results_search_*.json 2>/dev/null | head -1)
  if [ -f "$latest_search" ]; then
    cp "$latest_search" "$RESULTS_DIR/${model_name}_search_results.json"
  fi

  # Run recall tests
  echo "Running recall tests..."
  python3 ../test_qa_harness.py recall "$config_path" "$QA_FILE" 2>&1 | tee "$RESULTS_DIR/${model_name}_recall.log"

  # Copy the results file to our results directory with a clear name
  local latest_recall=$(ls -t qa_results_recall_*.json 2>/dev/null | head -1)
  if [ -f "$latest_recall" ]; then
    cp "$latest_recall" "$RESULTS_DIR/${model_name}_recall_results.json"
  fi

  echo "$model_name testing complete!"
}

# Test each model
for model_spec in "${MODELS[@]}"; do
  IFS=':' read -r model_name dimension db_path config_path <<< "$model_spec"

  # Check if config exists
  if [ ! -f "$config_path" ]; then
    echo "WARNING: Config not found at $config_path, skipping $model_name"
    continue
  fi

  test_model "$model_name" "$dimension" "$db_path" "$config_path"
done

echo ""
echo "========================================"
echo "All tests complete!"
echo "========================================"
echo "Results saved to: ./test/$RESULTS_DIR"
echo ""
echo "To analyze results, run:"
echo "  python3 analyze_embedding_results.py ./test/$RESULTS_DIR"
