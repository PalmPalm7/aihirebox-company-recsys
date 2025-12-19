# AIHireBox Company Side Agentic Recsys

该项目构造AI职小盒旗下智能体公司咨询推荐模块，部分代码和文档由Claude Code/Codex等copilot agent修正、生成。

This repository provides a Python-based agentic workflow that turns flexible company/job inputs into XiaoHongShu-style articles recommending similar companies and roles. It stitches together:

1. Feature Engineering (Representation Learning with LLM with offline generations).
2. A BoChaAI web search to gather recent context.
3. OpenRouter-compatible LLM calls to craft search queries and write the article.

## Prerequisites

- Python 3.10+
- API keys for:
  - BoChaAI Web Search (`BOCHAAI_API_KEY`)
  - OpenRouter (primary `OPENROUTER_API_KEY`; optional fallback `OPENROUTER_FALLBACK_API_KEY`)
  - Jina AI Embeddings (`JINA_API_KEY`) - for company embeddings

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r requirements.txt
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
openai>=1.30.0
numpy>=1.24.0
python-dotenv>=1.0.1
requests>=2.32.0
tqdm>=4.66.0
```

---

# Feature Engineering

## Company Tagging (`company_tagging.py`)

Core module for extracting MECE (Mutually Exclusive, Collectively Exhaustive) tags from company details using LLM.

**[TAG_TAXONOMY](./company_tagging.py)** 由 anthropic/claude-opus-4.5 通过对[aihirebox_company_list.csv](./data/aihirebox_company_list.csv)取样直接生成。

### Tag Dimensions (6个维度)

| Dimension | 中文名 | Type | Options |
|-----------|--------|------|---------|
| **industry** | 行业领域 | Multi | `ai_llm`, `robotics`, `edtech`, `fintech`, `healthtech`, `enterprise_saas`, `ecommerce`, `gaming`, `social`, `semiconductor`, `automotive`, `consumer_hw`, `cloud_infra`, `content_media`, `biotech`, `investment`, `other` |
| **business_model** | 商业模式 | Multi | `b2b`, `b2c`, `b2b2c`, `platform`, `saas`, `hardware`, `marketplace`, `consulting` |
| **target_market** | 目标市场 | Multi | `china_domestic`, `global`, `sea`, `us`, `europe`, `japan_korea`, `latam`, `mena` |
| **company_stage** | 发展阶段 | Single | `seed`, `early`, `growth`, `pre_ipo`, `public`, `bigtech_subsidiary`, `profitable`, `unknown` |
| **tech_focus** | 技术方向 | Multi | `llm_foundation`, `computer_vision`, `speech_nlp`, `embodied_ai`, `aigc`, `3d_graphics`, `chip_hardware`, `data_infra`, `autonomous`, `blockchain`, `quantum`, `not_tech_focused` |
| **team_background** | 团队背景 | Multi | `bigtech_alumni`, `top_university`, `serial_entrepreneur`, `academic`, `industry_expert`, `international`, `unknown` |

### Basic Usage

```bash
# Process all companies (outputs to output/company_tags_<timestamp>/)
python company_tagging.py data/aihirebox_company_list.csv

# Specify output directory
python company_tagging.py data/aihirebox_company_list.csv --output-dir ./tags_output

# Process specific companies by ID
python company_tagging.py data/aihirebox_company_list.csv --company-ids cid_0 cid_1 cid_2

# Process companies from JSON file (supports {"company_ids": [...]} or [...])
python company_tagging.py data/aihirebox_company_list.csv --company-ids-json my_companies.json

# Combine both ID sources
python company_tagging.py data/aihirebox_company_list.csv --company-ids cid_0 --company-ids-json more_ids.json

# Limit for testing
python company_tagging.py data/aihirebox_company_list.csv --limit 10

# Use web search for better team_background accuracy
python company_tagging.py data/aihirebox_company_list.csv --model openai/gpt-4o-mini:online
```

### Output Format

CSV 输出使用 `|` 作为多选字段分隔符：

```csv
company_id,company_name,industry,business_model,target_market,company_stage,tech_focus,team_background,confidence_score,reasoning
cid_0,Apex Context,ai_llm|content_media,b2c|saas,global,early,llm_foundation|aigc,bigtech_alumni|top_university,0.90,该公司专注于...
```

---

## Company Embedding (`company_embedding.py`)

将公司信息（名称、地点、介绍）转化为向量表示，用于语义检索和相似度计算。使用 Jina Embeddings v4 多语言模型。

### Jina Embeddings v4 Features

- **多语言支持**：原生支持中文、英文等多种语言
- **多模态**：支持文本和图像输入
- **任务适配**：使用 LoRA 适配器针对不同任务优化（retrieval, text-matching, classification）
- **灵活维度**：支持 128-2048 维向量

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `jina-embeddings-v4` | 模型名称 |
| `dimensions` | `1024` | 向量维度（128/256/512/1024/2048）|
| `task` | `retrieval.passage` | 任务类型（用于 LoRA 适配器选择）|
| `batch_size` | `32` | 每批处理数量 |

### Basic Usage

```bash
# 处理所有公司
python run_embedding.py data/aihirebox_company_list.csv

# 指定输出目录
python run_embedding.py data/aihirebox_company_list.csv --output-dir ./output_embeddings

# 自定义维度（更大 = 更精确，更小 = 更快）
python run_embedding.py data/aihirebox_company_list.csv --dimensions 2048

# 处理特定公司
python run_embedding.py data/aihirebox_company_list.csv --company-ids cid_0 cid_1

# 从 JSON 文件读取公司 ID
python run_embedding.py data/aihirebox_company_list.csv --company-ids-json my_companies.json

# 测试模式（限制数量）
python run_embedding.py data/aihirebox_company_list.csv --limit 10

# 静默模式（生产用）
python run_embedding.py data/aihirebox_company_list.csv --quiet

# 断点续传
python run_embedding.py data/aihirebox_company_list.csv --output-dir ./embeddings --resume
```

### Output Format

输出保存到 `output/company_embeddings_<timestamp>/` 目录：

```
output/company_embeddings_20251219_120000/
├── company_embeddings.csv       # 带向量的 CSV（embedding 以 JSON 字符串存储）
├── company_embeddings.json      # JSON 格式完整数据
├── company_embeddings.npy       # NumPy 数组格式（便于计算）
├── company_embeddings.mapping.json  # company_id 到数组索引的映射
└── run_metadata.json            # 运行元数据
```

### Python API

```python
from company_embedding import CompanyEmbedder, load_companies_from_csv

# 初始化
embedder = CompanyEmbedder(
    api_key="your_jina_api_key",
    dimensions=1024,
    task="retrieval.passage",
)

# 加载公司数据
companies = load_companies_from_csv("data/aihirebox_company_list.csv")

# 生成 embedding
results = embedder.embed_companies(companies, show_progress=True)

# 使用向量
for result in results:
    print(f"{result.company_id}: {len(result.embedding)} dims")
```

### Loading Embeddings for Computation

```python
import numpy as np
from company_embedding import load_embeddings_npy

# 加载向量和映射
embeddings, mapping = load_embeddings_npy("output_embeddings/company_embeddings.npy")

# 计算相似度
from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# 获取特定公司的向量
idx = mapping["cid_0"]
vector = embeddings[idx]

# 找最相似的公司
similarities = [cosine_similarity(vector, embeddings[i]) for i in range(len(embeddings))]
```

### Cost Estimation

- 132 家公司 × ~300 tokens/公司 ≈ 40,000 tokens
- Jina 免费额度：1M tokens/月
- 预估成本：几乎免费（在免费额度内）

### Environment Variables

在 `.env` 文件中配置：

```bash
JINA_API_KEY=your_jina_api_key_here
```

---

## Production Tagging Script (`run_tagging.py`)

生产用标签提取脚本，支持 [OpenRouter Web Search](https://openrouter.ai/docs/guides/features/plugins/web-search) 增强 `team_background` 准确率。

### Features

- **Web Search 增强**：默认使用 `:online` 后缀启用实时网络搜索，提升 `team_background` 准确率
- **灵活输入**：支持任意 CSV 数据文件，支持从 JSON 文件读取 company ID 列表
- **运行元数据**：自动保存模型、时间、团队覆盖率等指标
- **时间戳输出**：默认输出到 `output/company_tags_<timestamp>/` 目录

### Available Models

| Model | Web Search | Description |
|-------|------------|-------------|
| `openai/gpt-4o-mini` | ✗ | Default baseline model |
| `openai/gpt-4o-mini:online` | ✓ | Default web search for team_background (default) |

### Other models I recommend
* "openai/gpt-oss-120b"
* "openai/gpt-oss-120b:online"
* "openai/gpt-5-mini"
* "openai/gpt-5-mini:online"
* "google/gemini-2.5-flash"
* "google/gemini-2.5-flash:online"

### Production Level Usage

```bash
# Uses the most cost-efficient method to produce company tags. Trimming post-mortem reasonings and reducing logs.
python run_tagging.py data/aihirebox_company_list.csv --model openai/gpt-5-mini:online --quiet --no-reasoning --output-dir ./output_production
```

### Usages for Human Review

```bash
# Basic usage with default model (openai/gpt-4o-mini:online)
# Outputs to output/company_tags_<timestamp>/
python run_tagging.py data/aihirebox_company_list_sample.csv

# Use baseline model without web search
python run_tagging.py data/aihirebox_company_list_sample.csv --model openai/gpt-4o-mini

# Limit companies for testing
python run_tagging.py data/aihirebox_company_list_sample.csv --limit 5

# Process specific companies by ID
python run_tagging.py data/aihirebox_company_list_sample.csv --company-ids cid_0 cid_1 cid_2

# Process companies from JSON file (supports {"company_ids": [...]} or [...])
python run_tagging.py data/aihirebox_company_list_sample.csv --company-ids-json my_companies.json

# Combine both ID sources
python run_tagging.py data/aihirebox_company_list_sample.csv --company-ids cid_0 --company-ids-json more_ids.json

# Custom output directory
python run_tagging.py data/aihirebox_company_list_sample.csv --output-dir ./my_output

# Save tag taxonomy definition
python run_tagging.py data/aihirebox_company_list_sample.csv --save-taxonomy
```

### Output

输出保存到 `output/company_tags_<timestamp>/` 目录：

```
output/company_tags_20251218_120000/
├── company_tags.csv       # 标签结果 CSV
├── company_tags.json      # 标签结果 JSON
├── run_metadata.json      # 运行元数据（模型、时间、指标等）
└── tag_taxonomy.json      # 标签体系定义（可选）
```

### JSON Input Format

支持两种 JSON 格式作为 company ID 输入：

```json
// Format 1: 对象格式
{"company_ids": ["cid_0", "cid_1", "cid_2"]}

// Format 2: 数组格式
["cid_0", "cid_1", "cid_2"]
```

### Cost & Performance

| Mode | Cost | Response Time |
|------|------|---------------|
| Baseline (无 web search) | ~$0.001/公司 | ~2-3秒/公司 |
| **Online (web search)** | ~$0.02/公司 | ~6-7秒/公司 |

---

## Model Comparison Testing (`tests/test_tagging_models.py`)

比较不同 LLM 模型在公司标签提取任务上的表现。

### Usage

```bash
python tests/test_tagging_models.py
```

输出保存到 `output/model_comparison_<timestamp>/`

---

## Web Search Enhancement Testing (`tests/test_tagging_with_websearch.py`)

测试 Web Search 对 `team_background` 标签准确率的提升效果。

### Why Web Search?

基础模型在识别团队背景时覆盖率较低（50-70%），因为公司介绍中通常缺乏详细的创始人信息。通过 `:online` 后缀启用实时网络搜索，可以显著提升准确率（+20-30%）。

### Usage

```bash
python tests/test_tagging_with_websearch.py
```

---

## Article Generation Workflow (`workflow.py`)

生成小红书风格的公司推荐文章。

### How it works

1. **Query generation**: LLM 生成搜索查询
2. **Web search**: BoChaAI 搜索相关信息
3. **Article drafting**: LLM 生成 ~300 字推荐文章

### Usage

```bash
python workflow.py "{\"company_id\":\"c123\",\"company_name\":\"AIHireBox\",\"job_id\":\"j001\",\"job_name\":\"Product Manager\",\"job_description\":\"Responsible for AI hiring product roadmap\"}" \
  --output-dir output
```

---

## Data Format

### Input CSV Specification

The input CSV file must contain the following columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `company_id` | string | ✓ | Unique identifier for the company (e.g., `cid_0`, `cid_80`) |
| `company_name` | string | ✓ | Company name, may include Chinese name in parentheses |
| `location` | string | ✓ | Company location/address |
| `company_details` | string | ✓ | Detailed company description (Chinese, typically 100-500 chars) |

### Example CSV

```csv
company_id,company_name,location,company_details
cid_80,JOJOMath（成都书声科技有限公司）,中国(四川)自由贸易试验区,JOJO 是面向全球儿童的 AI 教育独角兽...
cid_0,Apex Context,子拓国际广场,极致上下文（Apex Context）专注于AI大模型...
```

### Available Data Files

See [sample data here](https://alidocs.dingtalk.com/i/nodes/GZLxjv9VGqaDL19MCxyjj0g286EDybno?utm_scene=person_space&iframeQuery=sheet_range%3Dst-2cb3c309-95650_44_3_1_1).

Download and save it as `data/aihirebox_company_list.csv`.

| File | Companies | Description |
|------|-----------|-------------|
| `data/aihirebox_company_list.csv` | 132 | 完整公司列表 |
| `data/aihirebox_company_list_sample.csv` | 32 | 测试样本 |
| `data/aihirebox_company_list_n1_sample.csv` | 1 | 单条测试样本 |

---

## Project Structure

```
aihirebox-company-recsys/
├── company_tagging.py              # 核心标签提取模块（含 CompanyTagger 类）
├── company_embedding.py            # 核心向量嵌入模块（含 CompanyEmbedder 类）
├── run_tagging.py                  # 生产用标签提取脚本（支持 web search）
├── run_embedding.py                # 生产用向量嵌入脚本（使用 Jina v4）
├── workflow.py                     # 文章生成工作流
├── data/                           # 数据文件
│   ├── aihirebox_company_list.csv      # 完整公司列表 (132家)
│   ├── aihirebox_company_list_sample.csv # 测试样本 (32家)
│   ├── aihirebox_company_list_n1_sample.csv # 单条样本 (1家)
│   └── company_list.txt                # 原始公司数据
├── tests/                          # 测试脚本
│   ├── test_tagging_models.py          # 模型对比测试
│   ├── test_tagging_with_websearch.py  # Web search 增强测试
│   └── fixtures/                       # 测试数据
│       ├── test_apex_context.json
│       └── test_timetell.json
├── scripts/                        # 辅助脚本
│   └── merge_comparison.py             # 结果合并工具
├── output/                         # 输出目录
│   ├── model_comparison_*/             # 模型对比结果
│   ├── websearch_test_*/               # Web search 测试结果
│   ├── company_tags_*/                 # run_tagging.py 输出
│   └── company_embeddings_*/           # run_embedding.py 输出
├── .env.example                    # 环境变量示例
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Key Insights

1. **默认模型**：`openai/gpt-4o-mini:online` - 启用 web search 以提升 team_background 准确率
2. **Team Background 问题**：基础模型对团队背景的识别率较低，使用 Web Search 可提升 20-30%
3. **成本权衡**：Web Search 提升准确率但增加成本和延迟，建议仅在需要高准确率时使用
4. **向量嵌入**：使用 Jina Embeddings v4 对 company_name、location、company_details 进行语义编码，支持后续的相似度检索

## Notes

- `company_tagging.py` 提供核心 `CompanyTagger` 类和工具函数
- `company_embedding.py` 提供核心 `CompanyEmbedder` 类，用于生成向量嵌入
- `run_tagging.py` 是生产用的标签提取脚本，默认启用 web search
- `run_embedding.py` 是生产用的向量嵌入脚本，使用 Jina Embeddings v4
- 多选字段使用 `|` 分隔，便于 pandas 解析
- 网络调用需要有效的 API key 和网络连接
