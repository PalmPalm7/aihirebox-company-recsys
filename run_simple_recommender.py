#!/usr/bin/env python3
"""
Simple Recall Recommender CLI - 基于规则的粗排召回脚本

该脚本提供命令行接口，用于执行基于5个规则的公司召回。

Usage:
    # 单公司召回（打印结果）
    python run_simple_recommender.py --company-id cid_100 --print-only

    # 单公司召回（保存到文件）
    python run_simple_recommender.py --company-id cid_100

    # 批量召回所有公司
    python run_simple_recommender.py --all

    # 指定输出目录
    python run_simple_recommender.py --all --output-dir output_production/simple_recall

    # 禁用头部抑制
    python run_simple_recommender.py --company-id cid_100 --no-head-suppression

    # 指定召回数量
    python run_simple_recommender.py --company-id cid_100 --top-k 30
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from simple_recommender import (
    SimpleRecallRecommender,
    RecallResult,
    RECALL_RULES,
    load_data_for_recommender,
    recall_result_to_dict,
    save_recall_results,
    print_recall_result,
)


# 默认路径
DEFAULT_RAW_CSV = Path("data/aihirebox_company_list.csv")
DEFAULT_TAGS_JSON = Path("output_production/company_tagging/company_tags.json")
DEFAULT_EMBEDDINGS_DIR = Path("output_production/company_embedding")
DEFAULT_OUTPUT_DIR = Path("output_production/simple_recall")


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Simple Recall Recommender - 基于规则的粗排召回",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 单公司召回（打印结果）
  python run_simple_recommender.py --company-id cid_100 --print-only

  # 批量召回所有公司
  python run_simple_recommender.py --all --output-dir output_production/simple_recall

  # 禁用头部抑制
  python run_simple_recommender.py --company-id cid_100 --no-head-suppression

  # 指定召回数量
  python run_simple_recommender.py --company-id cid_100 --top-k 30
        """
    )
    
    # 公司选择
    company_group = parser.add_mutually_exclusive_group(required=True)
    company_group.add_argument(
        "--company-id",
        type=str,
        help="要召回的公司ID (e.g., cid_100)",
    )
    company_group.add_argument(
        "--company-ids",
        nargs="+",
        type=str,
        help="要召回的多个公司ID",
    )
    company_group.add_argument(
        "--company-ids-json",
        type=Path,
        help="包含公司ID的JSON文件路径",
    )
    company_group.add_argument(
        "--all",
        action="store_true",
        help="召回所有公司",
    )
    
    # 数据路径
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=DEFAULT_RAW_CSV,
        help=f"原始公司数据CSV路径 (default: {DEFAULT_RAW_CSV})",
    )
    parser.add_argument(
        "--tags-json",
        type=Path,
        default=DEFAULT_TAGS_JSON,
        help=f"公司标签JSON路径 (default: {DEFAULT_TAGS_JSON})",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=DEFAULT_EMBEDDINGS_DIR,
        help=f"Embeddings目录路径 (default: {DEFAULT_EMBEDDINGS_DIR})",
    )
    
    # 输出
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录 (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="仅打印结果，不保存到文件",
    )
    
    # 召回参数
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="每个规则召回的候选数量 (default: 20)",
    )
    parser.add_argument(
        "--no-head-suppression",
        action="store_true",
        help="禁用头部抑制",
    )
    parser.add_argument(
        "--head-penalty",
        type=float,
        default=0.5,
        help="头部公司降权比例 (default: 0.5)",
    )
    
    # 其他
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，减少输出",
    )
    
    return parser.parse_args()


def load_company_ids_from_json(json_path: Path) -> List[str]:
    """从JSON文件加载公司ID列表"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "company_ids" in data:
        return data["company_ids"]
    else:
        raise ValueError(f"Invalid JSON format: expected list or dict with 'company_ids' key")


def save_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    results: List[RecallResult],
    start_time: datetime,
    end_time: datetime,
) -> None:
    """保存运行元数据"""
    metadata = {
        "run_type": "simple_recall",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "parameters": {
            "top_k": args.top_k,
            "head_suppression": not args.no_head_suppression,
            "head_penalty": args.head_penalty,
        },
        "input_files": {
            "raw_csv": str(args.raw_csv),
            "tags_json": str(args.tags_json),
            "embeddings_dir": str(args.embeddings_dir),
        },
        "results_summary": {
            "total_companies": len(results),
            "total_recall_groups": sum(len(r.recall_groups) for r in results),
        },
    }
    
    metadata_path = output_dir / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> int:
    """主入口"""
    args = parse_args()
    
    # 验证输入文件存在
    if not args.raw_csv.exists():
        print(f"Error: Raw CSV not found: {args.raw_csv}", file=sys.stderr)
        return 1
    
    if not args.tags_json.exists():
        print(f"Error: Tags JSON not found: {args.tags_json}", file=sys.stderr)
        return 1
    
    embeddings_path = args.embeddings_dir / "company_embeddings.npy"
    if not embeddings_path.exists():
        print(f"Error: Embeddings not found: {embeddings_path}", file=sys.stderr)
        return 1
    
    # 加载数据
    if not args.quiet:
        print("Loading data...")
    
    try:
        profiles, raw_companies, embeddings, mapping = load_data_for_recommender(
            args.raw_csv,
            args.tags_json,
            args.embeddings_dir,
        )
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return 1
    
    if not args.quiet:
        print(f"  Loaded {len(profiles)} company profiles")
        print(f"  Loaded {len(raw_companies)} raw company records")
        print(f"  Loaded {len(mapping)} embeddings")
    
    # 初始化推荐器
    recommender = SimpleRecallRecommender(
        profiles=profiles,
        raw_companies=raw_companies,
        embeddings=embeddings,
        embedding_mapping=mapping,
        head_suppression=not args.no_head_suppression,
        head_penalty=args.head_penalty,
    )
    
    # 确定要召回的公司ID
    company_ids: List[str] = []
    
    if args.company_id:
        company_ids = [args.company_id]
    elif args.company_ids:
        company_ids = args.company_ids
    elif args.company_ids_json:
        if not args.company_ids_json.exists():
            print(f"Error: JSON file not found: {args.company_ids_json}", file=sys.stderr)
            return 1
        company_ids = load_company_ids_from_json(args.company_ids_json)
    elif args.all:
        company_ids = list(recommender.profiles.keys())
    
    if not company_ids:
        print("No companies to process!")
        return 0
    
    if not args.quiet:
        print(f"\nProcessing {len(company_ids)} companies...")
        print(f"  Top-K: {args.top_k}")
        print(f"  Head suppression: {'enabled' if not args.no_head_suppression else 'disabled'}")
        if not args.no_head_suppression:
            print(f"  Head penalty: {args.head_penalty}")
    
    # 执行召回
    start_time = datetime.now()
    
    results: List[RecallResult] = []
    for i, cid in enumerate(company_ids, 1):
        try:
            result = recommender.recall(cid, rules=RECALL_RULES, top_k=args.top_k)
            results.append(result)
            
            if args.print_only:
                print_recall_result(result)
            elif not args.quiet and len(company_ids) <= 10:
                print(f"  [{i}/{len(company_ids)}] {cid}: {len(result.recall_groups)} rules, "
                      f"{sum(len(g.candidates) for g in result.recall_groups)} candidates")
            elif not args.quiet and i % 20 == 0:
                print(f"  Processed {i}/{len(company_ids)} companies...")
                
        except ValueError as e:
            print(f"Warning: {e}", file=sys.stderr)
            continue
    
    end_time = datetime.now()
    
    if not results:
        print("No results generated!")
        return 0
    
    # 保存结果
    if not args.print_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存召回结果
        if len(results) == 1:
            output_file = args.output_dir / f"recall_{results[0].query_company['company_id']}.json"
        else:
            output_file = args.output_dir / "recall_results.json"
        
        save_recall_results(results, output_file)
        
        if not args.quiet:
            print(f"\nSaved results to: {output_file}")
        
        # 保存元数据
        save_run_metadata(args.output_dir, args, results, start_time, end_time)
        
        if not args.quiet:
            print(f"Saved metadata to: {args.output_dir / 'run_metadata.json'}")
    
    # 打印摘要
    if not args.quiet:
        duration = (end_time - start_time).total_seconds()
        print(f"\n{'=' * 50}")
        print("SUMMARY")
        print(f"{'=' * 50}")
        print(f"Total companies processed: {len(results)}")
        print(f"Total recall groups: {sum(len(r.recall_groups) for r in results)}")
        print(f"Total candidates recalled: {sum(sum(len(g.candidates) for g in r.recall_groups) for r in results)}")
        print(f"Total time: {duration:.2f}s")
        if len(results) > 0:
            print(f"Average time per company: {duration / len(results):.3f}s")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

