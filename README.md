# AIHireBox Company Recsys Agentic Workflow

This repository provides a Python-based agentic workflow that turns flexible company/job inputs into XiaoHongShu-style articles recommending similar companies and roles. It stitches together:

1. A BoChaAI web search to gather recent context.
2. OpenRouter-compatible LLM calls to craft search queries and write the article.

## Prerequisites

- Python 3.10+
- API keys for:
  - BoChaAI Web Search (`BOCHAAI_API_KEY`)
  - OpenRouter (primary `OPENROUTER_API_KEY`; optional fallback `OPENROUTER_FALLBACK_API_KEY`)

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The workflow accepts a list of records. Each record can be:
- A JSON object string with `company_id`, `company_name`, `job_id`, `job_name`, `job_description` (extra fields are allowed).
- A JSON array of those values in order.
- A comma-separated string containing at least the five main fields.
- A path to a JSON file containing a list of such objects/arrays.

Example run:

```bash
python workflow.py "{\"company_id\":\"c123\",\"company_name\":\"AIHireBox\",\"job_id\":\"j001\",\"job_name\":\"Product Manager\",\"job_description\":\"Responsible for AI hiring product roadmap\"}" \
  --output-dir output
```

Outputs are Markdown articles in the specified `output` directory, named `article_<company_id>_<job_id>.md` (falling back to names if IDs are missing).

## How it works

1. **Query generation**: A system prompt guides the LLM (via OpenRouter) to craft a concise Bing-ready search query from the record.
2. **Web search**: The query is sent to BoChaAI's `/v1/web-search` endpoint with summaries enabled.
3. **Article drafting**: Another system prompt asks the LLM to produce ~300-character XiaoHongShu-style content recommending similar companies and roles, optionally embedding Markdown links from search results.

## Notes

- The workflow is defensive: it accepts varied input formats and supports an optional fallback OpenRouter API key.
- Results are written as text files so they can embed Markdown links or image references.
- Networked calls require valid API keys and internet connectivity.
