#!/bin/bash
#
# Full Production Pipeline - AIHireBox Company Recommendation System
#
# Complete 6-stage pipeline:
#   1. Company Tagging     - LLM-based MECE tag extraction
#   2. Company Embedding   - Jina Embeddings v4 (1024-dim)
#   3. Simple Recall       - 5-rule recall system
#   4. Web Search Cache    - Company research via web search
#   5. LLM Reranker        - Select Top 5 from recall
#   6. Article Writer      - Multi-style article generation
#
# Usage:
#   ./run_full_pipeline.sh                    # Run full pipeline
#   ./run_full_pipeline.sh --skip-articles    # Skip article generation (steps 4-6)
#   ./run_full_pipeline.sh --incremental      # Only process new companies
#   ./run_full_pipeline.sh --help             # Show help
#
# Default:
#   - Input:  data/aihirebox_company_list.csv
#   - Output: output_production/
#

set -e  # Exit on error

# ==============================================================================
# Configuration
# ==============================================================================

# Default values (can be overridden by command line args)
INPUT_CSV="data/aihirebox_company_list.csv"
OUTPUT_BASE="outputs/production"

# Model configuration
DEFAULT_MODEL="openai/gpt-5-mini"
TAGGING_MODEL="${DEFAULT_MODEL}:online"  # Tagging uses :online variant
RERANK_MODEL="${DEFAULT_MODEL}"
ARTICLE_MODEL="${DEFAULT_MODEL}"
ARTICLE_STYLES="36kr"
DEFAULT_CONCURRENCY=50

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
SKIP_ARTICLES=false
INCREMENTAL=false
SHOW_HELP=false
CUSTOM_MODEL=""
TEST_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-articles)
            SKIP_ARTICLES=true
            shift
            ;;
        --incremental)
            INCREMENTAL=true
            shift
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        --input)
            INPUT_CSV="$2"
            shift 2
            ;;
        --output)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --model)
            CUSTOM_MODEL="$2"
            shift 2
            ;;
        --concurrency)
            DEFAULT_CONCURRENCY="$2"
            shift 2
            ;;
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# ==============================================================================
# Test Mode Configuration
# ==============================================================================

if [ "$TEST_MODE" = true ]; then
    INPUT_CSV="data/aihirebox_company_list_n10_sample.csv"
    OUTPUT_BASE="outputs/test"
    INCREMENTAL=true  # Use cache where possible
    echo -e "${YELLOW}Running in TEST MODE${NC}"
    echo "  Input: $INPUT_CSV"
    echo "  Output: $OUTPUT_BASE"
    echo ""
fi

# Override models if custom model specified
if [ -n "$CUSTOM_MODEL" ]; then
    TAGGING_MODEL="${CUSTOM_MODEL}:online"
    RERANK_MODEL="${CUSTOM_MODEL}"
    ARTICLE_MODEL="${CUSTOM_MODEL}"
    echo "Using custom model: $CUSTOM_MODEL"
fi

# Update output directories based on OUTPUT_BASE
TAGGING_OUTPUT="${OUTPUT_BASE}/company_tagging"
EMBEDDING_OUTPUT="${OUTPUT_BASE}/company_embedding"
SIMPLE_RECALL_OUTPUT="${OUTPUT_BASE}/simple_recall"
# Web cache is stored in a dedicated directory (persistent, not deleted with output)
WEB_CACHE_OUTPUT="cache/web_search"
RERANK_OUTPUT="${OUTPUT_BASE}/article_generator/rerank_cache"
ARTICLES_OUTPUT="${OUTPUT_BASE}/article_generator/articles"

# ==============================================================================
# Help
# ==============================================================================

if [ "$SHOW_HELP" = true ]; then
    echo "Usage: ./run_full_pipeline.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --input FILE       Input CSV file (default: data/aihirebox_company_list.csv)"
    echo "  --output DIR       Output directory (default: output_production)"
    echo "  --model MODEL      Override LLM model for all stages (e.g., openai/gpt-oss-120b)"
    echo "  --concurrency N    Set concurrency for parallel processing (default: 20)"
    echo "  --skip-articles    Skip article generation (steps 4-6)"
    echo "  --incremental      Only process new companies (merge mode)"
    echo "  --test             Run in test mode (n10 sample, output_test/, incremental)"
    echo "  --help, -h         Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_full_pipeline.sh                                    # Full pipeline with defaults"
    echo "  ./run_full_pipeline.sh --input data/sample.csv --output output_sample"
    echo "  ./run_full_pipeline.sh --model openai/gpt-oss-120b        # Use custom model"
    echo "  ./run_full_pipeline.sh --concurrency 30                   # Higher concurrency"
    echo "  ./run_full_pipeline.sh --skip-articles                    # Only stages 1-3"
    echo "  ./run_full_pipeline.sh --incremental                      # Process new companies only"
    echo "  ./run_full_pipeline.sh --test                             # Run E2E test with sample data"
    echo ""
    echo "Pipeline Stages:"
    echo "  1. Company Tagging     - LLM tag extraction"
    echo "  2. Company Embedding   - Jina Embeddings v4"
    echo "  3. Simple Recall       - 5-rule recall"
    echo "  4. Web Search Cache    - Company research"
    echo "  5. LLM Reranker        - Top 5 selection"
    echo "  6. Article Writer      - Multi-style articles"
    exit 0
fi

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}==============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}==============================================================${NC}"
}

print_step() {
    echo -e "${YELLOW}Step $1: $2${NC}"
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

# ==============================================================================
# Pre-checks
# ==============================================================================

print_header "AIHireBox Full Production Pipeline"

echo ""
echo "Configuration:"
echo "  Input:           $INPUT_CSV"
echo "  Output:          $OUTPUT_BASE/"
echo "  Model:           ${CUSTOM_MODEL:-$DEFAULT_MODEL} (default)"
echo "  Concurrency:     $DEFAULT_CONCURRENCY"
echo "  Skip Articles:   $SKIP_ARTICLES"
echo "  Incremental:     $INCREMENTAL"
echo ""

# Check input file exists
if [ ! -f "${INPUT_CSV}" ]; then
    echo -e "${RED}Error: Input file not found: ${INPUT_CSV}${NC}"
    exit 1
fi

COMPANY_COUNT=$(tail -n +2 "$INPUT_CSV" | wc -l | tr -d ' ')
echo "Companies to process: $COMPANY_COUNT"
echo ""

# Create output directories
mkdir -p "${TAGGING_OUTPUT}"
mkdir -p "${EMBEDDING_OUTPUT}"
mkdir -p "${SIMPLE_RECALL_OUTPUT}"
mkdir -p "${WEB_CACHE_OUTPUT}"
mkdir -p "${RERANK_OUTPUT}"
mkdir -p "${ARTICLES_OUTPUT}"

# ==============================================================================
# Step 1: Company Tagging
# ==============================================================================

print_header "Step 1: Company Tagging"
print_step 1 "Extracting MECE tags using LLM"
echo "Model: ${TAGGING_MODEL}"
echo "Output: ${TAGGING_OUTPUT}/"
echo ""

TAGGING_ARGS=(
    "${INPUT_CSV}"
    --model "${TAGGING_MODEL}"
    --output-dir "${TAGGING_OUTPUT}"
    --quiet
    --no-reasoning
)

if [ "$INCREMENTAL" = true ]; then
    TAGGING_ARGS+=(--merge "${TAGGING_OUTPUT}")
    echo "Mode: Incremental (merge with existing)"
fi

python run_tagging.py "${TAGGING_ARGS[@]}"

print_success "Step 1 Complete: Tags saved to ${TAGGING_OUTPUT}/"

# ==============================================================================
# Step 2: Company Embedding
# ==============================================================================

print_header "Step 2: Company Embedding"
print_step 2 "Generating Jina Embeddings v4 (1024-dim)"
echo "Output: ${EMBEDDING_OUTPUT}/"
echo ""

EMBEDDING_ARGS=(
    "${INPUT_CSV}"
    --output-dir "${EMBEDDING_OUTPUT}"
    --quiet
)

if [ "$INCREMENTAL" = true ]; then
    EMBEDDING_ARGS+=(--merge "${EMBEDDING_OUTPUT}")
    echo "Mode: Incremental (merge with existing)"
fi

python run_embedding.py "${EMBEDDING_ARGS[@]}"

print_success "Step 2 Complete: Embeddings saved to ${EMBEDDING_OUTPUT}/"

# ==============================================================================
# Step 3: Simple Recall Recommender
# ==============================================================================

print_header "Step 3: Simple Recall Recommender"
print_step 3 "Running 5-rule recall system"
echo "Output: ${SIMPLE_RECALL_OUTPUT}/"
echo ""

python run_simple_recommender.py \
    --all \
    --raw-csv "${INPUT_CSV}" \
    --tags-json "${TAGGING_OUTPUT}/company_tags.json" \
    --embeddings-dir "${EMBEDDING_OUTPUT}" \
    --output-dir "${SIMPLE_RECALL_OUTPUT}" \
    --quiet

print_success "Step 3 Complete: Recall results saved to ${SIMPLE_RECALL_OUTPUT}/"

# ==============================================================================
# Article Generation Pipeline (Steps 4-6)
# ==============================================================================

if [ "$SKIP_ARTICLES" = true ]; then
    echo ""
    echo -e "${YELLOW}Skipping article generation (--skip-articles)${NC}"
else
    # ==========================================================================
    # Step 4: Web Search Cache
    # ==========================================================================

    print_header "Step 4: Web Search Cache"
    print_step 4 "Caching web search results for all companies"
    echo "Output: ${WEB_CACHE_OUTPUT}/"
    echo ""

    WEB_SEARCH_ARGS=(
        --company-csv "${INPUT_CSV}"
        --cache-dir "${WEB_CACHE_OUTPUT}"
        --max-results 10
        --model "${RERANK_MODEL}"
        --concurrency "${DEFAULT_CONCURRENCY}"
        --skip-existing  # Always skip existing (cache is persistent)
    )

    python run_web_search_cache.py "${WEB_SEARCH_ARGS[@]}"

    print_success "Step 4 Complete: Web search cache saved to ${WEB_CACHE_OUTPUT}/"

    # ==========================================================================
    # Step 5: LLM Reranker
    # ==========================================================================

    print_header "Step 5: LLM Reranker"
    print_step 5 "Selecting Top 5 companies per rule"
    echo "Output: ${RERANK_OUTPUT}/"
    echo ""

    RERANK_ARGS=(
        --recall-results "${SIMPLE_RECALL_OUTPUT}/recall_results.json"
        --web-cache-dir "${WEB_CACHE_OUTPUT}"
        --output-dir "${RERANK_OUTPUT}"
        --model "${RERANK_MODEL}"
        --concurrency "${DEFAULT_CONCURRENCY}"
    )

    if [ "$INCREMENTAL" = true ]; then
        RERANK_ARGS+=(--skip-existing)
        echo "Mode: Incremental (skip existing)"
    fi

    python run_reranker.py "${RERANK_ARGS[@]}"

    print_success "Step 5 Complete: Rerank results saved to ${RERANK_OUTPUT}/"

    # ==========================================================================
    # Step 6: Article Writer
    # ==========================================================================

    print_header "Step 6: Article Writer"
    print_step 6 "Generating multi-style articles"
    echo "Styles: ${ARTICLE_STYLES}"
    echo "Model: ${ARTICLE_MODEL}"
    echo "Concurrency: ${DEFAULT_CONCURRENCY} workers"
    echo "Output: ${ARTICLES_OUTPUT}/"
    echo ""

    ARTICLE_ARGS=(
        --rerank-dir "${RERANK_OUTPUT}"
        --web-cache-dir "${WEB_CACHE_OUTPUT}"
        --output-dir "${ARTICLES_OUTPUT}"
        --styles ${ARTICLE_STYLES}
        --model "${ARTICLE_MODEL}"
        --concurrency ${DEFAULT_CONCURRENCY}
    )

    if [ "$INCREMENTAL" = true ]; then
        ARTICLE_ARGS+=(--skip-existing)
        echo "Mode: Incremental (skip existing)"
    fi

    python run_article_writer.py "${ARTICLE_ARGS[@]}"

    print_success "Step 6 Complete: Articles saved to ${ARTICLES_OUTPUT}/"
fi

# ==============================================================================
# Summary
# ==============================================================================

print_header "Pipeline Complete!"

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
echo "  │   └── run_metadata.json"
echo "  ├── simple_recall/"
echo "  │   ├── recall_results.json"
echo "  │   └── run_metadata.json"

if [ "$SKIP_ARTICLES" = false ]; then
    echo "  └── article_generator/"
    echo "      ├── web_search_cache/"
    echo "      ├── rerank_cache/"
    echo "      └── articles/"
    echo "          ├── index.json"
    echo "          ├── json/"
    echo "          └── markdown/"
fi

echo ""
print_success "All steps completed successfully!"
