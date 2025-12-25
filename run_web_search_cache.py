#!/usr/bin/env python3
"""
Web Search Cache CLI - 搜索公司信息并缓存

使用 OpenRouter :online (Exa.ai) 搜索公司信息，
生成长总结作为下游 RAG 语料。

Usage:
    # 搜索所有公司
    python run_web_search_cache.py \\
        --company-csv data/aihirebox_company_list.csv \\
        --output-dir output_production/article_generator/web_search_cache \\
        --max-results 10 \\
        --skip-existing

    # 搜索指定公司
    python run_web_search_cache.py \\
        --company-csv data/aihirebox_company_list.csv \\
        --company-ids cid_100 cid_101 \\
        --output-dir output_production/article_generator/web_search_cache
        
    # 从 JSON 文件读取公司列表
    python run_web_search_cache.py \\
        --company-csv data/aihirebox_company_list.csv \\
        --company-ids-json data/target_companies.json \\
        --output-dir output_production/article_generator/web_search_cache
"""

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from article_generator.web_searcher import OpenRouterWebSearcher


def load_companies_from_csv(csv_path: Path) -> List[Dict[str, str]]:
    """从 CSV 加载公司数据"""
    companies = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            companies.append({
                "company_id": row.get("company_id", ""),
                "company_name": row.get("company_name", ""),
                "location": row.get("location", ""),
                "company_details": row.get("company_details", ""),
            })
    return companies


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
        "--output-dir",
        type=Path,
        default=Path("output_production/article_generator/web_search_cache"),
        help="Output directory for search cache (default: output_production/article_generator/web_search_cache)",
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化搜索器
    searcher = OpenRouterWebSearcher(model=args.model)
    
    print(f"\nModel: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"Max results: {args.max_results}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"\nSearching {len(companies)} companies...")
    
    # 执行批量搜索
    results = searcher.batch_search(
        companies=companies,
        max_results=args.max_results,
        delay_seconds=args.delay,
        skip_existing=args.skip_existing,
        cache_dir=args.output_dir,
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
    print(f"Cache directory: {args.output_dir}")
    
    # 保存运行元数据
    metadata = {
        "run_at": datetime.now().isoformat(),
        "model": args.model,
        "max_results": args.max_results,
        "total_companies": len(results),
        "valid_results": valid_count,
        "invalid_results": invalid_count,
    }
    
    metadata_file = args.output_dir / "run_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Metadata saved: {metadata_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())

