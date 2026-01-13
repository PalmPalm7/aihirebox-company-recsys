#!/usr/bin/env python
"""
Incremental Fix Pipeline for Specific Companies

Runs the complete pipeline for specified company IDs:
1. Tagging (with delta merge)
2. Embedding (with merge)
3. Simple Recall (for specified companies only)
4. Web Search Cache (uses existing cache, fills gaps)
5. Rerank (for specified companies)
6. Article Generation (for specified companies)

Usage:
    python scripts/fix_companies.py --company-ids cid_143 cid_88
    python scripts/fix_companies.py --company-ids cid_143 --skip-tagging  # If tagging is OK
    python scripts/fix_companies.py --company-ids cid_143 --skip-embedding  # If embedding is OK
    python scripts/fix_companies.py --company-ids cid_143 --dry-run  # Show what would run
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str, dry_run: bool = False) -> bool:
    """Run a command and return success status."""
    cmd_str = " ".join(cmd)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}>>> {description}")
    print(f"    Command: {cmd_str}")

    if dry_run:
        return True

    try:
        result = subprocess.run(cmd, check=True)
        print(f"    [OK] Completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    [FAILED] Exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"    [FAILED] Command not found")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Incremental fix pipeline for specific companies."
    )
    parser.add_argument(
        "--company-ids",
        nargs="+",
        required=True,
        help="Company IDs to fix (e.g., cid_143 cid_88)"
    )
    parser.add_argument(
        "--production-dir",
        type=Path,
        default=Path("outputs/production"),
        help="Production output directory (default: outputs/production)"
    )
    parser.add_argument(
        "--company-csv",
        type=Path,
        default=Path("data/aihirebox_company_list.csv"),
        help="Path to company CSV file"
    )
    parser.add_argument(
        "--skip-tagging",
        action="store_true",
        help="Skip tagging stage (if tagging is already correct)"
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding stage (if embedding is already correct)"
    )
    parser.add_argument(
        "--skip-recall",
        action="store_true",
        help="Skip recall stage (if using manual candidates)"
    )
    parser.add_argument(
        "--skip-web-search",
        action="store_true",
        help="Skip web search cache update"
    )
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        help="Skip reranking stage (if using manual candidates)"
    )
    parser.add_argument(
        "--skip-articles",
        action="store_true",
        help="Skip article generation"
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=["36kr"],
        help="Article styles to generate (default: 36kr)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Concurrency for parallel operations (default: 10)"
    )

    args = parser.parse_args()

    company_ids = args.company_ids
    production_dir = args.production_dir
    company_csv = args.company_csv
    dry_run = args.dry_run

    print("=" * 60)
    print("INCREMENTAL FIX PIPELINE")
    print("=" * 60)
    print(f"\nCompanies to fix: {', '.join(company_ids)}")
    print(f"Production directory: {production_dir}")
    if dry_run:
        print("\n[DRY RUN MODE - No commands will be executed]")

    # Track success
    all_success = True

    # Stage 1: Tagging
    if not args.skip_tagging:
        cmd = [
            "python", "run_tagging.py",
            str(company_csv),
            "--company-ids", *company_ids,
            "--output", str(production_dir / "company_tagging"),
            "--merge",
            "--no-reasoning"
        ]
        if not run_command(cmd, "Stage 1: Tagging", dry_run):
            all_success = False
            print("Warning: Tagging failed, continuing with other stages...")
    else:
        print("\n[SKIP] Stage 1: Tagging (--skip-tagging)")

    # Stage 2: Embedding
    if not args.skip_embedding:
        cmd = [
            "python", "run_embedding.py",
            str(company_csv),
            "--company-ids", *company_ids,
            "--output", str(production_dir / "company_embedding"),
            "--merge"
        ]
        if not run_command(cmd, "Stage 2: Embedding", dry_run):
            all_success = False
            print("Warning: Embedding failed, continuing with other stages...")
    else:
        print("\n[SKIP] Stage 2: Embedding (--skip-embedding)")

    # Stage 3: Simple Recall
    if not args.skip_recall:
        cmd = [
            "python", "run_simple_recommender.py",
            "--company-ids", *company_ids,
            "--output-dir", str(production_dir / "simple_recall")
        ]
        if not run_command(cmd, "Stage 3: Simple Recall", dry_run):
            all_success = False
            print("Warning: Recall failed, continuing with other stages...")
    else:
        print("\n[SKIP] Stage 3: Simple Recall (--skip-recall)")

    # Stage 4: Web Search Cache
    if not args.skip_web_search:
        cmd = [
            "python", "run_web_search_cache.py",
            "--company-csv", str(company_csv),
            "--company-ids", *company_ids,
            "--cache-dir", "cache/web_search",
            "--concurrency", str(args.concurrency)
        ]
        if not run_command(cmd, "Stage 4: Web Search Cache", dry_run):
            all_success = False
            print("Warning: Web search cache failed, continuing with other stages...")
    else:
        print("\n[SKIP] Stage 4: Web Search Cache (--skip-web-search)")

    # Stage 5: Rerank
    if not args.skip_rerank:
        recall_results = production_dir / "simple_recall" / "recall_results.json"
        rerank_output = production_dir / "article_generator" / "rerank_cache"

        cmd = [
            "python", "run_reranker.py",
            "--recall-results", str(recall_results),
            "--web-cache-dir", "cache/web_search",
            "--output-dir", str(rerank_output),
            "--company-ids", *company_ids,
            "--min-k", "1",
            "--max-k", "5",
            "--concurrency", str(args.concurrency)
        ]
        if not run_command(cmd, "Stage 5: Rerank", dry_run):
            all_success = False
            print("Warning: Reranking failed, continuing with other stages...")
    else:
        print("\n[SKIP] Stage 5: Rerank (--skip-rerank)")

    # Stage 6: Article Generation
    if not args.skip_articles:
        rerank_dir = production_dir / "article_generator" / "rerank_cache"
        articles_dir = production_dir / "article_generator" / "articles"

        cmd = [
            "python", "run_article_writer.py",
            "--rerank-dir", str(rerank_dir),
            "--web-cache-dir", "cache/web_search",
            "--output-dir", str(articles_dir),
            "--company-ids", *company_ids,
            "--styles", *args.styles,
            "--concurrency", str(args.concurrency)
        ]
        if not run_command(cmd, "Stage 6: Article Generation", dry_run):
            all_success = False
            print("Warning: Article generation failed")
    else:
        print("\n[SKIP] Stage 6: Article Generation (--skip-articles)")

    # Summary
    print("\n" + "=" * 60)
    if dry_run:
        print("DRY RUN COMPLETE - No commands were executed")
    elif all_success:
        print("ALL STAGES COMPLETED SUCCESSFULLY")
        print("\nRun validation to verify:")
        print(f"  python scripts/validate_production.py --production-dir {production_dir}")
    else:
        print("SOME STAGES FAILED - Check output above")
        return 1

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
