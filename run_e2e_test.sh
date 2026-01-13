#!/bin/bash
#
# E2E Testing Script for AIHireBox Company Recommendation System
#
# This script is a lean test wrapper around the production pipeline.
# It follows the principle: test wrapper should focus on setup/verification,
# while production logic stays in one place.
#
# Usage:
#   ./run_e2e_test.sh              # Run full E2E test
#   ./run_e2e_test.sh --skip-llm   # Skip pipeline, only run import tests and verification
#   ./run_e2e_test.sh --clean      # Clean test output before running
#
# What this script does:
#   1. Run module import tests (test-specific)
#   2. Call ./run_full_pipeline.sh --test (production script with test config)
#   3. Run verification assertions (test-specific)
#   4. Report pass/fail
#

set -e  # Exit on error

# ==============================================================================
# Configuration
# ==============================================================================

TEST_OUTPUT_DIR="output_test"
SAMPLE_CSV="data/aihirebox_company_list_n10_sample.csv"
WEB_CACHE_DIR="cache/web_search"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
SKIP_LLM=false
CLEAN=false

for arg in "$@"; do
    case $arg in
        --skip-llm)
            SKIP_LLM=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        *)
            ;;
    esac
done

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}[PASSED]${NC} $1"
}

print_skip() {
    echo -e "${YELLOW}[SKIPPED]${NC} $1"
}

print_error() {
    echo -e "${RED}[FAILED]${NC} $1"
}

# ==============================================================================
# Setup
# ==============================================================================

print_header "E2E Test Setup"

# Clean test output if requested
if [ "$CLEAN" = true ]; then
    echo "Cleaning test output directory..."
    rm -rf "$TEST_OUTPUT_DIR"
    print_success "Cleaned $TEST_OUTPUT_DIR"
fi

# Check prerequisites
if [ ! -f "$SAMPLE_CSV" ]; then
    print_error "Sample CSV not found: $SAMPLE_CSV"
    exit 1
fi
COMPANY_COUNT=$(tail -n +2 "$SAMPLE_CSV" | wc -l | tr -d ' ')
print_success "Found sample CSV: $SAMPLE_CSV ($COMPANY_COUNT companies)"

# ==============================================================================
# Phase 1: Module Import Tests
# ==============================================================================

print_header "Phase 1: Module Import Tests"

python -c "
import sys

# Test core module imports
try:
    from core import (
        CompanyRecord, CompanyProfile,
        load_companies_from_csv, load_embeddings_npy, load_company_profiles,
        HEAD_COMPANY_STAGES, TAG_DIMENSIONS, DIMENSION_LABELS,
    )
    print('  [OK] core module imports')
except Exception as e:
    print(f'  [FAIL] core module: {e}')
    sys.exit(1)

# Test backward compatibility
try:
    from company_embedding import CompanyRecord, load_companies_from_csv
    print('  [OK] company_embedding backward compat')
except Exception as e:
    print(f'  [FAIL] company_embedding: {e}')
    sys.exit(1)

try:
    from company_recommender import CompanyProfile, HEAD_COMPANY_STAGES
    print('  [OK] company_recommender backward compat')
except Exception as e:
    print(f'  [FAIL] company_recommender: {e}')
    sys.exit(1)

try:
    from company_tagging import CompanyRecord, load_companies_from_csv
    print('  [OK] company_tagging backward compat')
except Exception as e:
    print(f'  [FAIL] company_tagging: {e}')
    sys.exit(1)

try:
    from simple_recommender import SimpleRecallRecommender
    print('  [OK] simple_recommender imports')
except Exception as e:
    print(f'  [FAIL] simple_recommender: {e}')
    sys.exit(1)

# Test article generator imports
try:
    from article_generator import OpenRouterWebSearcher, LLMReranker, ArticleWriter
    print('  [OK] article_generator imports')
except Exception as e:
    print(f'  [FAIL] article_generator: {e}')
    sys.exit(1)

# Test web cache functions
try:
    from article_generator.web_searcher import (
        load_web_search_cache, load_web_search_index,
        get_stale_companies, update_web_search_index,
    )
    print('  [OK] web_searcher cache functions')
except Exception as e:
    print(f'  [FAIL] web_searcher cache: {e}')
    sys.exit(1)

print('')
print('All module imports successful!')
"

print_success "All module imports passed"

# ==============================================================================
# Phase 2: Run Production Pipeline in Test Mode
# ==============================================================================

print_header "Phase 2: Run Production Pipeline (Test Mode)"

if [ "$SKIP_LLM" = true ]; then
    print_skip "Skipping pipeline run (--skip-llm)"
    echo "  Will verify existing outputs in $TEST_OUTPUT_DIR"
else
    echo "Running: ./run_full_pipeline.sh --test"
    echo ""

    # Run the production pipeline with test configuration
    ./run_full_pipeline.sh --test

    print_success "Pipeline completed successfully"
fi

# ==============================================================================
# Phase 3: Verification Assertions
# ==============================================================================

print_header "Phase 3: Verification Assertions"

# Run verification script
python -c "
import json
import sys
from pathlib import Path

test_dir = Path('$TEST_OUTPUT_DIR')
cache_dir = Path('$WEB_CACHE_DIR')
sample_csv = Path('$SAMPLE_CSV')
errors = []

print('Verifying pipeline outputs...')
print('')

# 1. Verify tagging output
tags_file = test_dir / 'company_tagging' / 'company_tags.json'
if tags_file.exists():
    with open(tags_file) as f:
        tags = json.load(f)
    print(f'  [OK] Tagging: {len(tags)} companies tagged')
else:
    errors.append('Tagging output not found')
    print(f'  [FAIL] Tagging output not found: {tags_file}')

# 2. Verify embedding output
embeddings_file = test_dir / 'company_embedding' / 'company_embeddings.npy'
if embeddings_file.exists():
    import numpy as np
    embeddings = np.load(embeddings_file)
    print(f'  [OK] Embeddings: shape {embeddings.shape}')
else:
    errors.append('Embedding output not found')
    print(f'  [FAIL] Embedding output not found: {embeddings_file}')

# 3. Verify recall output
recall_file = test_dir / 'simple_recall' / 'recall_results.json'
if recall_file.exists():
    with open(recall_file) as f:
        recall = json.load(f)
    print(f'  [OK] Recall: {len(recall)} companies processed')
else:
    errors.append('Recall output not found')
    print(f'  [FAIL] Recall output not found: {recall_file}')

# 4. Verify web search cache
if cache_dir.exists():
    cache_files = list(cache_dir.glob('cid_*.json'))
    index_file = cache_dir / 'index.json'
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        print(f'  [OK] Web Cache: {len(cache_files)} files, index has {index.get(\"total_companies\", 0)} entries')
    else:
        print(f'  [WARN] Web Cache: {len(cache_files)} files (no index)')
else:
    errors.append('Web cache directory not found')
    print(f'  [FAIL] Web cache not found: {cache_dir}')

# 5. Verify rerank output
rerank_dir = test_dir / 'article_generator' / 'rerank_cache'
if rerank_dir.exists():
    rerank_files = list(rerank_dir.glob('*.json'))
    rerank_files = [f for f in rerank_files if f.name not in ('index.json', 'run_metadata.json')]
    print(f'  [OK] Rerank: {len(rerank_files)} rerank results')
else:
    errors.append('Rerank output not found')
    print(f'  [FAIL] Rerank output not found: {rerank_dir}')

# 6. Verify article output
articles_dir = test_dir / 'article_generator' / 'articles'
if articles_dir.exists():
    json_dir = articles_dir / 'json'
    md_dir = articles_dir / 'markdown'
    json_count = len(list(json_dir.glob('*.json'))) if json_dir.exists() else 0
    md_count = len(list(md_dir.glob('*.md'))) if md_dir.exists() else 0
    if json_count > 0:
        print(f'  [OK] Articles: {json_count} JSON, {md_count} Markdown')
    else:
        errors.append('No articles generated')
        print(f'  [FAIL] No articles found in {json_dir}')
else:
    errors.append('Articles output not found')
    print(f'  [FAIL] Articles output not found: {articles_dir}')

# 7. Verify cache usage (key test!)
print('')
print('Cache usage verification:')
import csv
with open(sample_csv, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    sample_ids = [row['company_id'] for row in reader]

if cache_dir.exists() and (cache_dir / 'index.json').exists():
    with open(cache_dir / 'index.json') as f:
        index = json.load(f)
    cached_ids = set(index.get('companies', {}).keys())
    sample_set = set(sample_ids)
    cached_count = len(sample_set & cached_ids)
    coverage = cached_count / len(sample_set) * 100 if sample_set else 0
    print(f'  Sample companies: {len(sample_set)}')
    print(f'  Cached: {cached_count} ({coverage:.0f}%)')
    if coverage >= 80:
        print(f'  [OK] Cache coverage is good')
    else:
        print(f'  [WARN] Low cache coverage')

print('')
if errors:
    print(f'VERIFICATION FAILED: {len(errors)} error(s)')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
else:
    print('All verifications passed!')
"

print_success "All verification assertions passed"

# ==============================================================================
# Summary
# ==============================================================================

print_header "E2E Test Summary"

echo "Test Configuration:"
echo "  Sample CSV: $SAMPLE_CSV"
echo "  Output Dir: $TEST_OUTPUT_DIR"
echo "  Web Cache:  $WEB_CACHE_DIR"
echo ""

echo "Directory contents:"
if [ -d "$TEST_OUTPUT_DIR" ]; then
    find "$TEST_OUTPUT_DIR" -type f \( -name "*.json" -o -name "*.npy" -o -name "*.csv" -o -name "*.md" \) 2>/dev/null | wc -l | xargs -I {} echo "  {} files generated"
fi

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}E2E TESTS COMPLETED SUCCESSFULLY${NC}"
echo -e "${GREEN}============================================================${NC}"
