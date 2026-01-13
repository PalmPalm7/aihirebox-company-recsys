#!/usr/bin/env python3
"""
Web Search Cache CLI - 搜索公司信息并缓存

使用 OpenRouter :online (Exa.ai) 搜索公司信息，
生成长总结作为下游 RAG 语料。

缓存设计：
- 默认存储在 cache/web_search/ 目录（独立于 output，避免被误删）
- 自动维护 index.json 索引文件，记录每个公司的搜索时间
- 支持按时间过滤过期缓存

Usage:
    # 搜索所有公司（使用默认缓存目录）
    python run_web_search_cache.py \\
        --company-csv data/aihirebox_company_list.csv \\
        --skip-existing

    # 使用自定义缓存目录
    python run_web_search_cache.py \\
        --company-csv data/aihirebox_company_list.csv \\
        --cache-dir cache/web_search \\
        --max-results 10 \\
        --skip-existing

    # 搜索指定公司
    python run_web_search_cache.py \\
        --company-csv data/aihirebox_company_list.csv \\
        --company-ids cid_100 cid_101

    # 刷新超过30天的缓存
    python run_web_search_cache.py \\
        --company-csv data/aihirebox_company_list.csv \\
        --refresh-stale --max-age-days 30

    # 查看缓存索引
    python run_web_search_cache.py --show-index
"""

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from article_generator.web_searcher import (
    OpenRouterWebSearcher,
    load_web_search_index,
    get_stale_companies,
)

# 使用 core 模块的共享函数
from core import load_companies_as_dicts as load_companies_from_csv


def load_company_ids_from_json(json_path: Path) -> List[str]:
    """从 JSON 文件加载公司 ID 列表"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "company_ids" in data:
        return data["company_ids"]
    else:
        raise ValueError("Invalid JSON format. Expected list or dict with 'company_ids' key.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search company information using OpenRouter :online (Exa.ai)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--company-csv",
        type=Path,
        default=Path("data/aihirebox_company_list.csv"),
        help="Path to company CSV file (default: data/aihirebox_company_list.csv)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/web_search"),
        help="Cache directory for search results (default: cache/web_search)",
    )
    parser.add_argument(
        "--company-ids",
        nargs="+",
        help="Specific company IDs to search",
    )
    parser.add_argument(
        "--company-ids-json",
        type=Path,
        help="JSON file containing company IDs to search",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Max web search results per company (default: 10)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip companies with existing cache",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of companies to process (for testing)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-3-flash-preview",
        help="Model to use (default: google/gemini-3-flash-preview)",
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
    parser.add_argument(
        "--refresh-stale",
        action="store_true",
        help="Refresh cache entries older than --max-age-days",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=30,
        help="Max cache age in days for --refresh-stale (default: 30)",
    )
    parser.add_argument(
        "--show-index",
        action="store_true",
        help="Show cache index and exit",
    )

    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    # 处理 --show-index 选项
    if args.show_index:
        index = load_web_search_index(args.cache_dir)
        if not index:
            print(f"No cache index found at {args.cache_dir}")
            return 0
        print(f"\nWeb Search Cache Index: {args.cache_dir}")
        print(f"{'='*60}")
        print(f"Total companies: {len(index)}")
        valid_count = sum(1 for v in index.values() if v.get("is_valid", False))
        print(f"Valid results: {valid_count}")
        print(f"Invalid results: {len(index) - valid_count}")
        print(f"\nRecent entries:")
        sorted_entries = sorted(
            index.items(),
            key=lambda x: x[1].get("searched_at", ""),
            reverse=True
        )[:10]
        for cid, info in sorted_entries:
            status = "✓" if info.get("is_valid") else "✗"
            print(f"  {status} {cid}: {info.get('company_name', 'N/A')} ({info.get('searched_at', 'N/A')[:10]})")
        return 0

    # 检查 API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable is required")
        return 1

    # 加载公司数据
    print(f"Loading companies from {args.company_csv}...")
    companies = load_companies_from_csv(args.company_csv)
    print(f"Loaded {len(companies)} companies")
    
    # 收集要处理的公司 ID
    target_ids = set()
    
    if args.company_ids:
        target_ids.update(args.company_ids)
    
    if args.company_ids_json:
        if not args.company_ids_json.exists():
            print(f"Error: JSON file not found: {args.company_ids_json}")
            return 1
        json_ids = load_company_ids_from_json(args.company_ids_json)
        target_ids.update(json_ids)
        print(f"Loaded {len(json_ids)} company IDs from {args.company_ids_json}")
    
    # 处理 --refresh-stale 选项
    if args.refresh_stale:
        stale_ids = get_stale_companies(
            args.cache_dir,
            max_age_days=args.max_age_days,
            company_ids=[c["company_id"] for c in companies] if not target_ids else list(target_ids),
        )
        if stale_ids:
            target_ids.update(stale_ids)
            print(f"Found {len(stale_ids)} stale entries (older than {args.max_age_days} days)")
        else:
            print(f"No stale entries found (all within {args.max_age_days} days)")
            if not target_ids:
                return 0

    # 过滤公司
    if target_ids:
        companies = [c for c in companies if c["company_id"] in target_ids]
        print(f"Filtered to {len(companies)} companies by ID")

    # 应用 limit
    if args.limit:
        companies = companies[:args.limit]
        print(f"Limited to {len(companies)} companies")
    
    if not companies:
        print("No companies to process!")
        return 0
    
    # 创建输出目录
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化搜索器
    searcher = OpenRouterWebSearcher(model=args.model)
    
    print(f"\nModel: {args.model}")
    print(f"Output: {args.cache_dir}")
    print(f"Max results: {args.max_results}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Concurrency: {args.concurrency}")
    print(f"\nSearching {len(companies)} companies...")

    # 执行批量搜索（并发或顺序）
    if args.concurrency > 1:
        results = searcher.batch_search_concurrent(
            companies=companies,
            max_results=args.max_results,
            skip_existing=args.skip_existing,
            cache_dir=args.cache_dir,
            show_progress=not args.quiet,
            concurrency=args.concurrency,
        )
    else:
        results = searcher.batch_search(
            companies=companies,
            max_results=args.max_results,
            delay_seconds=args.delay,
            skip_existing=args.skip_existing,
            cache_dir=args.cache_dir,
            show_progress=not args.quiet,
        )
    
    # 统计结果
    valid_count = sum(1 for r in results if r.is_valid)
    invalid_count = len(results) - valid_count
    
    print(f"\n{'='*60}")
    print("SEARCH SUMMARY")
    print(f"{'='*60}")
    print(f"Total searched: {len(results)}")
    print(f"Valid results: {valid_count}")
    print(f"Invalid/empty: {invalid_count}")
    print(f"Cache directory: {args.cache_dir}")
    
    # 保存运行元数据
    metadata = {
        "run_at": datetime.now().isoformat(),
        "model": args.model,
        "max_results": args.max_results,
        "total_companies": len(results),
        "valid_results": valid_count,
        "invalid_results": invalid_count,
    }
    
    metadata_file = args.cache_dir / "run_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Metadata saved: {metadata_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())

