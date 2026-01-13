#!/usr/bin/env python3
"""
Reranker CLI - LLM 精排

使用 openai/gpt-5-mini via OpenRouter 对召回结果进行精排，
从每个规则的 20 个候选中选择 1 到 max_k 个真正相关的公司。

核心原则：宁缺毋滥
- Reranker 作为质量把关者，筛选出真正有价值的关联公司
- 如果关联过于牵强，会直接排除，不强行凑数
- 最终选择数量可能少于 max_k

Usage:
    # 精排所有召回结果（默认选择 1-5 家）
    python run_reranker.py \\
        --recall-results output_production/simple_recall/recall_results.json \\
        --output-dir output_production/article_generator/rerank_cache \\
        --skip-existing

    # 自定义选择范围
    python run_reranker.py \\
        --recall-results output_production/simple_recall/recall_results.json \\
        --min-k 2 --max-k 8 \\
        --output-dir output_production/article_generator/rerank_cache

    # 精排指定公司
    python run_reranker.py \\
        --recall-results output_production/simple_recall/recall_results.json \\
        --company-ids cid_100 cid_101 \\
        --output-dir output_production/article_generator/rerank_cache

    # 使用 web search 缓存增强精排（缓存存储在 cache/web_search）
    python run_reranker.py \\
        --recall-results output_production/simple_recall/recall_results.json \\
        --web-cache-dir cache/web_search \\
        --output-dir output_production/article_generator/rerank_cache
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv

from article_generator.reranker import LLMReranker
from article_generator.web_searcher import load_web_search_cache


def load_recall_results(json_path: Path) -> List[Dict]:
    """加载召回结果"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_company_ids_from_json(json_path: Path) -> List[str]:
    """从 JSON 文件加载公司 ID 列表"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "company_ids" in data:
        return data["company_ids"]
    else:
        raise ValueError("Invalid JSON format")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank recall results using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--recall-results",
        type=Path,
        default=Path("output_production/simple_recall/recall_results.json"),
        help="Path to recall results JSON (default: output_production/simple_recall/recall_results.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_production/article_generator/rerank_cache"),
        help="Output directory for rerank cache",
    )
    parser.add_argument(
        "--web-cache-dir",
        type=Path,
        default=Path("cache/web_search"),
        help="Web search cache directory (default: cache/web_search)",
    )
    parser.add_argument(
        "--company-ids",
        nargs="+",
        help="Specific query company IDs to rerank",
    )
    parser.add_argument(
        "--company-ids-json",
        type=Path,
        help="JSON file containing company IDs to rerank",
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=1,
        help="Minimum number of companies to select per rule (default: 1)",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=5,
        help="Maximum number of companies to select per rule (default: 5)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between requests in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rules with existing cache",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-5-mini",
        help="Model to use (default: openai/gpt-5-mini)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Number of parallel workers (default: 20, use 1 for sequential mode)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    
    # 检查 API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable is required")
        return 1
    
    # 检查召回结果文件
    if not args.recall_results.exists():
        print(f"Error: Recall results file not found: {args.recall_results}")
        return 1
    
    # 加载召回结果
    print(f"Loading recall results from {args.recall_results}...")
    recall_results = load_recall_results(args.recall_results)
    print(f"Loaded {len(recall_results)} query companies")
    
    # 收集要处理的公司 ID
    target_ids: Set[str] = set()
    
    if args.company_ids:
        target_ids.update(args.company_ids)
    
    if args.company_ids_json:
        if not args.company_ids_json.exists():
            print(f"Error: JSON file not found: {args.company_ids_json}")
            return 1
        json_ids = load_company_ids_from_json(args.company_ids_json)
        target_ids.update(json_ids)
        print(f"Loaded {len(json_ids)} company IDs from {args.company_ids_json}")
    
    # 过滤召回结果
    if target_ids:
        recall_results = [
            r for r in recall_results 
            if r["query_company"]["company_id"] in target_ids
        ]
        print(f"Filtered to {len(recall_results)} query companies")
    
    if not recall_results:
        print("No recall results to process!")
        return 0
    
    # 统计总任务数
    total_tasks = sum(len(r.get("recall_groups", [])) for r in recall_results)
    print(f"Total rerank tasks: {total_tasks}")
    
    # 加载 web search 缓存（如果提供）
    web_search_cache = {}
    if args.web_cache_dir and args.web_cache_dir.exists():
        print(f"Loading web search cache from {args.web_cache_dir}...")
        web_search_cache = load_web_search_cache(args.web_cache_dir)
        print(f"Loaded {len(web_search_cache)} web search results")
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化精排器
    reranker = LLMReranker(model=args.model)
    
    print(f"\nModel: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"Selection range: {args.min_k} to {args.max_k} companies per rule")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Concurrency: {args.concurrency}")
    print(f"\nReranking {total_tasks} tasks...")

    # 执行批量精排（并发或顺序）
    if args.concurrency > 1:
        results = reranker.batch_rerank_concurrent(
            recall_results=recall_results,
            min_k=args.min_k,
            max_k=args.max_k,
            skip_existing=args.skip_existing,
            cache_dir=args.output_dir,
            web_search_cache=web_search_cache,
            show_progress=not args.quiet,
            concurrency=args.concurrency,
        )
    else:
        results = reranker.batch_rerank(
            recall_results=recall_results,
            min_k=args.min_k,
            max_k=args.max_k,
            delay_seconds=args.delay,
            skip_existing=args.skip_existing,
            cache_dir=args.output_dir,
            web_search_cache=web_search_cache,
            show_progress=not args.quiet,
        )
    
    # 统计结果
    success_count = sum(1 for r in results if r.selected_companies)
    empty_count = len(results) - success_count
    
    print(f"\n{'='*60}")
    print("RERANK SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Empty/failed: {empty_count}")
    print(f"Cache directory: {args.output_dir}")
    
    # 保存运行元数据
    metadata = {
        "run_at": datetime.now().isoformat(),
        "model": args.model,
        "min_k": args.min_k,
        "max_k": args.max_k,
        "recall_results_path": str(args.recall_results),
        "web_cache_used": bool(web_search_cache),
        "total_tasks": len(results),
        "successful": success_count,
        "empty_or_failed": empty_count,
    }
    
    metadata_file = args.output_dir / "run_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Metadata saved: {metadata_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())

