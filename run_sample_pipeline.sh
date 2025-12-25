#!/bin/bash
#
# Sample Pipeline Script - 使用样本数据运行完整流程
#
# 使用 data/aihirebox_company_list_sample.csv 作为输入
# 输出到 output/output_production_sample/ 目录
#
# Usage:
#   chmod +x run_sample_pipeline.sh
#   ./run_sample_pipeline.sh
#

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

INPUT_CSV="data/aihirebox_company_list_sample.csv"
OUTPUT_BASE="output/output_production_sample"

TAGGING_OUTPUT="${OUTPUT_BASE}/company_tagging"
EMBEDDING_OUTPUT="${OUTPUT_BASE}/company_embedding"
RECOMMENDER_OUTPUT="${OUTPUT_BASE}/recommender"
SIMPLE_RECALL_OUTPUT="${OUTPUT_BASE}/simple_recall"

# Model for tagging (with web search for better team_background)
TAGGING_MODEL="openai/gpt-5-mini:online"

# ============================================================================
# Pre-checks
# ============================================================================

echo "=============================================="
echo "Sample Pipeline - Full Run"
echo "=============================================="
echo ""
echo "Input:  ${INPUT_CSV}"
echo "Output: ${OUTPUT_BASE}/"
echo ""

# Check input file exists
if [ ! -f "${INPUT_CSV}" ]; then
    echo "Error: Input file not found: ${INPUT_CSV}"
    exit 1
fi

# Create output directories
mkdir -p "${TAGGING_OUTPUT}"
mkdir -p "${EMBEDDING_OUTPUT}"
mkdir -p "${RECOMMENDER_OUTPUT}"
mkdir -p "${SIMPLE_RECALL_OUTPUT}"

echo "Output directories created."
echo ""

# ============================================================================
# Step 1: Company Tagging
# ============================================================================

echo "=============================================="
echo "Step 1: Company Tagging"
echo "=============================================="
echo "Model: ${TAGGING_MODEL}"
echo "Output: ${TAGGING_OUTPUT}/"
echo ""

python run_tagging.py "${INPUT_CSV}" \
    --model "${TAGGING_MODEL}" \
    --output-dir "${TAGGING_OUTPUT}" \
    --no-reasoning

echo ""
echo "✅ Step 1 Complete: Tags saved to ${TAGGING_OUTPUT}/"
echo ""

# ============================================================================
# Step 2: Company Embedding
# ============================================================================

echo "=============================================="
echo "Step 2: Company Embedding"
echo "=============================================="
echo "Output: ${EMBEDDING_OUTPUT}/"
echo ""

python run_embedding.py "${INPUT_CSV}" \
    --output-dir "${EMBEDDING_OUTPUT}"

echo ""
echo "✅ Step 2 Complete: Embeddings saved to ${EMBEDDING_OUTPUT}/"
echo ""

# ============================================================================
# Step 3: Company Recommender
# ============================================================================

echo "=============================================="
echo "Step 3: Company Recommender"
echo "=============================================="
echo "Output: ${RECOMMENDER_OUTPUT}/"
echo ""

python run_recommender.py \
    --all \
    --tags-path "${TAGGING_OUTPUT}/company_tags.json" \
    --embeddings-dir "${EMBEDDING_OUTPUT}" \
    --output-dir "${RECOMMENDER_OUTPUT}" \
    --score-threshold 0.6

echo ""
echo "✅ Step 3 Complete: Recommendations saved to ${RECOMMENDER_OUTPUT}/"
echo ""

# ============================================================================
# Step 4: Simple Recall Recommender
# ============================================================================

echo "=============================================="
echo "Step 4: Simple Recall Recommender"
echo "=============================================="
echo "Output: ${SIMPLE_RECALL_OUTPUT}/"
echo ""

python run_simple_recommender.py \
    --all \
    --raw-csv "${INPUT_CSV}" \
    --tags-json "${TAGGING_OUTPUT}/company_tags.json" \
    --embeddings-dir "${EMBEDDING_OUTPUT}" \
    --output-dir "${SIMPLE_RECALL_OUTPUT}"

echo ""
echo "✅ Step 4 Complete: Recall results saved to ${SIMPLE_RECALL_OUTPUT}/"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Output structure:"
echo "  ${OUTPUT_BASE}/"
echo "  ├── company_tagging/"
echo "  │   ├── company_tags.json"
echo "  │   ├── company_tags.csv"
echo "  │   └── run_metadata.json"
echo "  ├── company_embedding/"
echo "  │   ├── company_embeddings.npy"
echo "  │   ├── company_embeddings.mapping.json"
echo "  │   ├── company_embeddings.json"
echo "  │   ├── company_embeddings.csv"
echo "  │   └── run_metadata.json"
echo "  ├── recommender/"
echo "  │   ├── recommendations.json"
echo "  │   └── run_metadata.json"
echo "  └── simple_recall/"
echo "      ├── recall_results.json"
echo "      └── run_metadata.json"
echo ""
echo "✅ All steps completed successfully!"
