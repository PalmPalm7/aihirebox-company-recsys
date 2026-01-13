# Manual Candidate Selection for Article Generation

This document describes how to manually specify candidate companies for article generation, bypassing the normal recall and rerank pipeline stages.

## Use Case

When a company fails to generate articles in the normal pipeline (e.g., due to missing tags, embedding issues, or insufficient recall candidates), you can manually specify candidates to generate articles.

## Prerequisites

- Web search cache must exist for both the query company and all candidate companies
- Company details must be available in the CSV file

## Method: Manual Rerank Cache Creation

### Step 1: Use the Manual Rerank Script

Run the script `scripts/create_manual_rerank.py` after configuring the following variables:

```python
# Configuration in scripts/create_manual_rerank.py
QUERY_COMPANY_ID = "cid_143"  # The company to generate articles for
CANDIDATE_COMPANY_IDS = [
    "cid_100",  # MiniMax
    "cid_114",  # Moonshot AI
    "cid_113",  # Baichuan
    # ... add more candidates
]
```

### Step 2: Update Selection Reasons

Edit the `SELECTION_REASONS` dictionary to provide meaningful reasons for each candidate:

```python
SELECTION_REASONS = {
    "cid_100": "MiniMax is a leading AI company with multimodal capabilities...",
    "cid_114": "Moonshot AI (Kimi) pioneered ultra-long context window...",
    # ...
}
```

### Step 3: Run the Script

```bash
source .venv/bin/activate
python scripts/create_manual_rerank.py
```

This creates rerank cache files for each rule (R1, R3, R4) in:
`outputs/output_production/article_generator/rerank_cache/cid_XXX_*.json`

### Step 4: Generate Articles

Run the article writer with the specific company ID:

```bash
python run_article_writer.py \
    --rerank-dir outputs/output_production/article_generator/rerank_cache \
    --web-cache-dir cache/web_search \
    --company-csv data/aihirebox_company_list.csv \
    --output-dir outputs/output_production/article_generator/articles \
    --company-ids cid_143 \
    --styles 36kr \
    --concurrency 1
```

## Rerank Cache File Format

The manually created rerank cache files follow this structure:

```json
{
  "query_company_id": "cid_143",
  "query_company_name": "Company Name",
  "rule_id": "R1_industry",
  "rule_name": "Same Industry",
  "narrative_angle": "A compelling story angle connecting these companies...",
  "selected_companies": [
    {
      "company_id": "cid_100",
      "company_name": "Candidate Company",
      "location": "Address",
      "company_details": "Full company description...",
      "selection_reason": "Why this company was selected..."
    }
  ],
  "reranked_at": "2026-01-13T..."
}
```

## Example: AI LLM Companies

The `scripts/create_manual_rerank.py` script was originally created for generating articles about Zhipu AI (cid_143) with the following AI LLM companies as candidates:

| CID | Company Name | Description |
|-----|--------------|-------------|
| cid_100 | MiniMax | Multimodal AGI company |
| cid_114 | Moonshot AI | Long-context LLM (Kimi) |
| cid_113 | Baichuan | Chinese LLM startup |
| cid_109 | DeepSeek | Cost-effective open-source LLM |
| cid_55 | Shanghai AI Lab | National AI research institution |
| cid_117 | 01.AI | Founded by Kai-Fu Lee |

## Notes

- This method bypasses the normal recall (20 candidates) and rerank (LLM quality filter) stages
- The narrative angle should be written to create a cohesive story across all selected companies
- Selection reasons help the article writer understand why each company is relevant
