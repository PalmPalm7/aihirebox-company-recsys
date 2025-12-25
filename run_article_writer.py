#!/usr/bin/env python3
"""
Article Writer CLI - 批量生成文章

使用 google/gemini-3-flash-preview via OpenRouter 生成文章。
读取 rerank_cache 和 web_search_cache，按指定风格生成文章。

Usage:
    # 生成所有风格的文章
    python run_article_writer.py \\
        --rerank-dir output_production/article_generator/rerank_cache \\
        --web-cache-dir output_production/article_generator/web_search_cache \\
        --output-dir output_production/article_generator/articles \\
        --styles 36kr huxiu xiaohongshu linkedin zhihu

    # 只生成小红书风格
    python run_article_writer.py \\
        --rerank-dir output_production/article_generator/rerank_cache \\
        --output-dir output_production/article_generator/articles \\
        --styles xiaohongshu

    # 生成指定公司的文章
    python run_article_writer.py \\
        --rerank-dir output_production/article_generator/rerank_cache \\
        --company-ids cid_100 \\
        --output-dir output_production/article_generator/articles \\
        --styles 36kr
"""

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv

from article_generator.article_writer import ArticleWriter
from article_generator.reranker import load_rerank_cache
from article_generator.web_searcher import load_web_search_cache
from article_generator.models import RerankResult, SelectedCompany
from article_generator.styles import get_all_style_ids


def load_company_details_from_csv(csv_path: Path) -> Dict[str, str]:
    """从 CSV 加载公司详情映射"""
    details_map = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            company_id = row.get("company_id", "")
            company_details = row.get("company_details", "")
            if company_id:
                details_map[company_id] = company_details
    return details_map


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


def load_rerank_results_from_dir(rerank_dir: Path) -> List[RerankResult]:
    """从目录加载所有精排结果"""
    results = []
    
    for json_file in rerank_dir.glob("*.json"):
        if json_file.name == "run_metadata.json":
            continue
        
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            results.append(RerankResult(
                query_company_id=data["query_company_id"],
                query_company_name=data["query_company_name"],
                rule_id=data["rule_id"],
                rule_name=data.get("rule_name", ""),
                narrative_angle=data.get("narrative_angle", ""),
                selected_companies=[
                    SelectedCompany(**sc) for sc in data.get("selected_companies", [])
                ],
                reranked_at=data.get("reranked_at", ""),
            ))
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue
    
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate articles from reranked results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--rerank-dir",
        type=Path,
        default=Path("output_production/article_generator/rerank_cache"),
        help="Rerank cache directory",
    )
    parser.add_argument(
        "--web-cache-dir",
        type=Path,
        default=Path("output_production/article_generator/web_search_cache"),
        help="Web search cache directory",
    )
    parser.add_argument(
        "--company-csv",
        type=Path,
        default=Path("data/aihirebox_company_list.csv"),
        help="Company CSV file for details",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_production/article_generator/articles"),
        help="Output directory for articles",
    )
    parser.add_argument(
        "--company-ids",
        nargs="+",
        help="Specific query company IDs",
    )
    parser.add_argument(
        "--company-ids-json",
        type=Path,
        help="JSON file containing company IDs",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=["36kr", "xiaohongshu"],
        choices=get_all_style_ids(),
        help="Article styles to generate (default: 36kr xiaohongshu)",
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
        help="Skip articles that already exist",
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
    
    # 检查 rerank 目录
    if not args.rerank_dir.exists():
        print(f"Error: Rerank directory not found: {args.rerank_dir}")
        return 1
    
    # 加载精排结果
    print(f"Loading rerank results from {args.rerank_dir}...")
    rerank_results = load_rerank_results_from_dir(args.rerank_dir)
    print(f"Loaded {len(rerank_results)} rerank results")
    
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
    
    # 过滤精排结果
    if target_ids:
        rerank_results = [
            r for r in rerank_results 
            if r.query_company_id in target_ids
        ]
        print(f"Filtered to {len(rerank_results)} rerank results")
    
    # 过滤空的精排结果
    rerank_results = [r for r in rerank_results if r.selected_companies]
    print(f"After filtering empty results: {len(rerank_results)}")
    
    if not rerank_results:
        print("No rerank results to process!")
        return 0
    
    # 加载 web search 缓存
    web_search_cache = {}
    if args.web_cache_dir.exists():
        print(f"Loading web search cache from {args.web_cache_dir}...")
        web_search_cache = load_web_search_cache(args.web_cache_dir)
        print(f"Loaded {len(web_search_cache)} web search results")
    
    # 加载公司详情
    company_details_map = {}
    if args.company_csv.exists():
        print(f"Loading company details from {args.company_csv}...")
        company_details_map = load_company_details_from_csv(args.company_csv)
        print(f"Loaded {len(company_details_map)} company details")
    
    # 统计总任务数
    total_tasks = len(rerank_results) * len(args.styles)
    print(f"\nTotal article generation tasks: {total_tasks}")
    print(f"Styles: {', '.join(args.styles)}")
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化文章生成器
    writer = ArticleWriter(model=args.model)
    
    print(f"\nModel: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"\nGenerating {total_tasks} articles...")
    
    # 批量生成文章
    articles = writer.batch_write(
        rerank_results=rerank_results,
        style_ids=args.styles,
        web_search_cache=web_search_cache,
        company_details_map=company_details_map,
        delay_seconds=args.delay,
        skip_existing=args.skip_existing,
        output_dir=args.output_dir,
        show_progress=not args.quiet,
    )
    
    # 统计结果
    success_count = sum(1 for a in articles if a.word_count > 100)
    failed_count = len(articles) - success_count
    
    # 按风格统计
    style_stats = {}
    for a in articles:
        style_stats[a.style] = style_stats.get(a.style, 0) + 1
    
    print(f"\n{'='*60}")
    print("ARTICLE GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total articles: {len(articles)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"\nBy style:")
    for style, count in sorted(style_stats.items()):
        print(f"  {style}: {count}")
    print(f"\nOutput directory: {args.output_dir}")
    
    # 保存运行元数据
    metadata = {
        "run_at": datetime.now().isoformat(),
        "model": args.model,
        "styles": args.styles,
        "total_articles": len(articles),
        "successful": success_count,
        "failed": failed_count,
        "style_breakdown": style_stats,
    }
    
    metadata_file = args.output_dir / "run_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Metadata saved: {metadata_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())

