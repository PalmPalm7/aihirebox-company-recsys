# Pipeline Test Run Commands

## N10 Sample Test Run

使用 n10 样本（10家公司）进行端到端测试验证。

### 完整命令

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行完整 pipeline（n10 样本，gemini 模型，36kr 风格）
./run_full_pipeline.sh \
    --input data/aihirebox_company_list_n10_sample.csv \
    --output output_test_$(date +%Y%m%d_%H%M%S) \
    --model google/gemini-3-flash-preview \
    --concurrency 10
```

### 2026-01-13 实际测试运行

```bash
source .venv/bin/activate && ./run_full_pipeline.sh \
    --input data/aihirebox_company_list_n10_sample.csv \
    --output output_test_20260113_n10 \
    --model google/gemini-3-flash-preview \
    --concurrency 10
```

**测试结果**:
- Step 1 (Tagging): 10 家公司标签提取完成
- Step 2 (Embedding): 1024-dim 向量生成完成
- Step 3 (Simple Recall): 5-rule recall 完成
- Step 4 (Web Search): 10/10 搜索完成 (9 valid, 1 empty)
- Step 5 (Reranker): 30 tasks, 29 成功, 1 empty
- Step 6 (Articles): 29/29 文章生成成功 (100%)

**总耗时**: ~2分钟

### 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--input` | `data/aihirebox_company_list_n10_sample.csv` | N10 样本 CSV |
| `--output` | `output_test_YYYYMMDD_HHMMSS` | 时间戳命名的输出目录 |
| `--model` | `google/gemini-3-flash-preview` | Gemini 模型（100%成功率） |
| `--concurrency` | `10` | 测试时用较低并发 |

### 验证检查点

1. **Web Search Cache**: 检查 `cache/web_search/` 是否被正确使用（不重新搜索已缓存的公司）
2. **Company Tags**: 检查 `{output}/company_tagging/` 输出
3. **Embeddings**: 检查 `{output}/company_embedding/*.npy` 文件
4. **36kr Articles**: 检查 `{output}/article_generator/articles/` 内容质量

---

## Full Production Run

完整数据集的生产运行。

```bash
source .venv/bin/activate

./run_full_pipeline.sh \
    --input data/aihirebox_company_list.csv \
    --output outputs/production \
    --model google/gemini-3-flash-preview \
    --concurrency 20
```

### 增量模式（只处理新公司）

```bash
./run_full_pipeline.sh \
    --input data/aihirebox_company_list.csv \
    --output outputs/production \
    --model google/gemini-3-flash-preview \
    --concurrency 20 \
    --incremental
```

---

## 单独运行各阶段

如果需要单独运行某个阶段：

```bash
# Stage 1: Tagging
python run_tagging.py data/aihirebox_company_list.csv \
    --model google/gemini-3-flash-preview:online \
    --output-dir outputs/production/company_tagging

# Stage 2: Embedding
python run_embedding.py data/aihirebox_company_list.csv \
    --output-dir outputs/production/company_embedding

# Stage 3: Simple Recall
python run_simple_recommender.py \
    --all \
    --output-dir outputs/production/simple_recall

# Stage 4: Web Search Cache (persistent cache)
python run_web_search_cache.py \
    --company-csv data/aihirebox_company_list.csv \
    --cache-dir cache/web_search \
    --concurrency 20

# Stage 5: Reranker
python run_reranker.py \
    --recall-results outputs/production/simple_recall/recall_results.json \
    --web-cache-dir cache/web_search \
    --output-dir outputs/production/article_generator/rerank_cache \
    --model google/gemini-3-flash-preview \
    --concurrency 20

# Stage 6: Article Writer (only 36kr style)
python run_article_writer.py \
    --rerank-dir outputs/production/article_generator/rerank_cache \
    --web-cache-dir cache/web_search \
    --output-dir outputs/production/article_generator/articles \
    --styles 36kr \
    --model google/gemini-3-flash-preview \
    --concurrency 20
```

---

## 模型选择

| 模型 | 成功率 | 推荐场景 |
|------|--------|----------|
| `google/gemini-3-flash-preview` | 100% | 推荐用于生产 |
| `openai/gpt-5-mini` | ~95% | 默认模型 |
| `openai/gpt-oss-120b` | ~53% | 不推荐（context length 问题） |

---

## 日期

2026-01-13
