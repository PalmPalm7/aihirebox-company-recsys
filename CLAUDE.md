# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIHireBox Company-Side Agentic Recommendation System - a Python-based multi-stage pipeline that:
1. Extracts structured tags from company profiles using LLM
2. Generates semantic embeddings via Jina AI
3. Recommends similar companies across multiple dimensions
4. Generates multi-style articles about company relationships

## Commands

### Setup
```bash
# Create and activate virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies (uv recommended for speed)
uv pip install -r requirements.txt  # or: pip install -r requirements.txt

# Copy environment template and add API keys
cp .env.example .env
```

### Full Production Pipeline
```bash
./run_full_pipeline.sh                            # Runs all 6 stages
./run_full_pipeline.sh --skip-articles            # Only stages 1-3
./run_full_pipeline.sh --model openai/gpt-oss-120b  # Custom model
./run_full_pipeline.sh --concurrency 30           # Adjust workers (default: 20)
./run_full_pipeline.sh --incremental              # Process new companies only
```

### Individual Pipeline Stages
```bash
# Stage 1: Tag extraction (uses LLM via OpenRouter)
python run_tagging.py data/aihirebox_company_list.csv --model openai/gpt-5-mini:online --no-reasoning

# Stage 2: Generate embeddings (uses Jina API)
python run_embedding.py data/aihirebox_company_list.csv

# Stage 3: Simple recall recommender
python run_simple_recommender.py --all --output-dir outputs/production/simple_recall

# Stage 4: Web search cache (persistent, stored in cache/ directory)
python run_web_search_cache.py --company-csv data/aihirebox_company_list.csv --cache-dir cache/web_search --concurrency 20

# Stage 5: LLM reranking - quality filter with min_k/max_k range (default 1-5)
python run_reranker.py --recall-results outputs/production/simple_recall/recall_results.json --web-cache-dir cache/web_search --output-dir outputs/production/article_generator/rerank_cache --min-k 1 --max-k 5 --concurrency 20

# Stage 6: Article generation - supports --concurrency (default 20)
python run_article_writer.py --rerank-dir outputs/production/article_generator/rerank_cache --web-cache-dir cache/web_search --output-dir outputs/production/article_generator/articles --styles 36kr --concurrency 20
```

### API Server
```bash
uvicorn api.main:app --reload  # Development server on port 8000
```

### Docker
```bash
docker-compose up --build  # Build and run API container
```

## Architecture

### Core Modules (`*.py` in root)
- `company_tagging.py` - LLM-based MECE tag extraction (6 dimensions: industry, business_model, target_market, company_stage, tech_focus, team_background)
- `company_embedding.py` - Jina Embeddings v4 (1024-dim multilingual vectors)
- `company_recommender.py` - Multi-dimensional recommendation with head suppression
- `simple_recommender.py` - Lightweight 5-rule recall system (R1-R5)

### Production CLI Scripts (`run_*.py`)
Each core module has a corresponding CLI wrapper with checkpoint/resume, incremental processing, and merge capabilities.

### Article Generation (`article_generator/`)
Three-layer system with concurrent processing (default 20 workers):
- `web_searcher.py` - OpenRouter `:online` suffix for Exa.ai web search
- `reranker.py` - LLM quality filter selecting 1-5 companies per rule (宁缺毋滥 - quality over quantity)
- `article_writer.py` - Gemini-3-flash generates articles. **Recommended style: 36kr** (fact-based, web-search grounded). Other styles (xiaohongshu, etc.) are experimental - see `docs/xiaohongshu_style_retrospective.md` for why "真人视角" generation doesn't work well.

### API Layer (`api/`)
FastAPI REST server for serving generated articles with endpoints for health, metadata, companies, and articles.

## Key Technical Details

### Recommendation Scoring
- Formula: 60% tag Jaccard similarity + 40% embedding cosine similarity
- Head suppression: 50-60% penalty for public/bigtech companies to prevent domination

### 5 Recall Rules (R1-R5)
- R1: Same industry
- R2: Same tech focus
- R3: Same industry + target market
- R4: Same team background
- R5: Same industry + team background

### Reranker Quality Filter
- Selects 1-5 companies per rule (configurable via --min-k and --max-k)
- Philosophy: 宁缺毋滥 (quality over quantity) - rejects weak associations
- Acts as a quality gatekeeper, not just a ranking mechanism

### Concurrency
- Default: 20 parallel workers for web search, reranker, and article generation
- Configurable via --concurrency flag in CLI scripts and pipeline

### Environment Variables
- `OPENROUTER_API_KEY` - Primary LLM API (required)
- `OPENROUTER_FALLBACK_API_KEY` - Fallback key (optional)
- `JINA_API_KEY` - Embeddings service (required)

### Data Flow
```
CSV → Tags (JSON/CSV) → Embeddings (.npy) → Recall (JSON) → Web Search Cache → Rerank Cache → Articles (JSON/Markdown)
```

### Output Structure
**IMPORTANT**: All outputs must go under `outputs/` directory. Never create output directories in the project root.

```
outputs/
├── production/          # Production runs (default)
├── test/                # Test mode runs (--test flag)
└── {custom}/            # Custom output dirs (--output outputs/custom)

cache/
└── web_search/          # Persistent web search cache (shared across runs)
```

Production outputs (`outputs/production/`) contain: `company_tagging/`, `company_embedding/`, `simple_recall/`, `article_generator/`.

### Manual Candidate Selection
When a company fails to generate articles through the normal pipeline, use manual candidate selection:
- Script: `scripts/create_manual_rerank.py`
- Documentation: `docs/manual_candidate_selection.md`

## Validation & Troubleshooting

### Validate Production Outputs
After running the pipeline, validate that all companies have successful tagging and articles:
```bash
# Run validation (default: outputs/production)
python scripts/validate_production.py

# Specify custom production directory
python scripts/validate_production.py --production-dir outputs/production

# Output as JSON (for programmatic use)
python scripts/validate_production.py --json
```

The validation script checks:
- **Tagging issues**: Error in reasoning OR (confidence=0 with all empty fields)
- **Article issues**: Company completely missing from articles index

### Fix Failed Companies
When validation finds issues, use the incremental fix script:
```bash
# Fix specific companies (runs all 6 stages)
python scripts/fix_companies.py --company-ids cid_123 cid_124

# Dry run - see what would be executed without running
python scripts/fix_companies.py --company-ids cid_123 --dry-run

# Skip stages if they're already correct
python scripts/fix_companies.py --company-ids cid_123 --skip-tagging --skip-embedding

# Use manual candidates instead of recall/rerank
python scripts/fix_companies.py --company-ids cid_123 --skip-recall --skip-rerank

# Specify custom production directory
python scripts/fix_companies.py --company-ids cid_123 --production-dir outputs/production
```

### Typical Troubleshooting Flow
```bash
# 1. Validate production outputs
python scripts/validate_production.py

# 2. Fix detected issues (copy command from validation output)
python scripts/fix_companies.py --company-ids cid_123 cid_124 ...

# 3. Re-validate to confirm fix
python scripts/validate_production.py
```

## Git Workflow

**IMPORTANT**: Always create a new feature branch for commits. Never commit directly to `main` or `master`.

```bash
# Create a new branch for your changes
git checkout -b feature/your-feature-name

# After making changes, commit and push
git add .
git commit -m "Your commit message"
git push -u origin feature/your-feature-name

# Create a PR via GitHub CLI
gh pr create --title "Your PR title" --body "Description"
```
