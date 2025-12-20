#!/usr/bin/env python3
"""
Production Embedding Script - Generate company embeddings using Jina Embeddings v4.

This script provides a CLI interface for generating embeddings for company details,
with support for batch processing, checkpointing, incremental updates, and multiple output formats.

Usage:
    # Basic usage (full run)
    python run_embedding.py data/aihirebox_company_list.csv

    # Specify output directory
    python run_embedding.py data/aihirebox_company_list.csv --output-dir ./output_embeddings

    # Incremental mode - merge new embeddings with existing ones
    python run_embedding.py data/aihirebox_company_list.csv --merge output/company_embeddings_full

    # Process specific companies and merge with existing
    python run_embedding.py data/aihirebox_company_list.csv --company-ids cid_new_1 cid_new_2 --merge output/company_embeddings_full

    # Custom dimensions (128, 256, 512, 1024, 2048)
    python run_embedding.py data/aihirebox_company_list.csv --dimensions 2048

    # Production mode (quiet, minimal output)
    python run_embedding.py data/aihirebox_company_list.csv --quiet

    # Resume from checkpoint
    python run_embedding.py data/aihirebox_company_list.csv --resume
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from company_embedding import (
    CompanyEmbedder,
    CompanyEmbedding,
    CompanyRecord,
    DEFAULT_DIMENSIONS,
    DEFAULT_MODEL,
    DEFAULT_TASK,
    load_companies_from_csv,
    load_embeddings_npy,
    load_jina_api_key,
    save_embeddings_csv,
    save_embeddings_json,
    save_embeddings_npy,
    print_embedding_summary,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate company embeddings using Jina Embeddings v4.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all companies in CSV (full run)
  python run_embedding.py data/aihirebox_company_list.csv

  # Specify output directory
  python run_embedding.py data/aihirebox_company_list.csv --output-dir ./embeddings

  # Incremental mode - add new companies to existing embeddings
  python run_embedding.py data/aihirebox_company_list.csv --merge output/company_embeddings_full

  # Process specific new companies and merge with existing
  python run_embedding.py data/aihirebox_company_list.csv --company-ids cid_new_1 cid_new_2 --merge output/company_embeddings_full

  # Custom dimensions (smaller = faster, larger = more accurate)
  python run_embedding.py data/aihirebox_company_list.csv --dimensions 2048

  # Process specific companies by ID
  python run_embedding.py data/aihirebox_company_list.csv --company-ids cid_0 cid_1 cid_2

  # Production mode (quiet output)
  python run_embedding.py data/aihirebox_company_list.csv --quiet

  # Resume from checkpoint
  python run_embedding.py data/aihirebox_company_list.csv --output-dir ./embeddings --resume
        """
    )
    
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to input CSV file with company data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files (default: output/company_embeddings_<timestamp>)",
    )
    parser.add_argument(
        "--company-ids",
        nargs="+",
        help="Specific company IDs to process",
    )
    parser.add_argument(
        "--company-ids-json",
        type=Path,
        help='JSON file with company IDs (expects {"company_ids": [...]} or [...])',
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_DIMENSIONS,
        choices=[128, 256, 512, 1024, 2048],
        help=f"Embedding dimensions (default: {DEFAULT_DIMENSIONS})",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        choices=[
            "retrieval.query",
            "retrieval.passage", 
            "text-matching",
            "classification",
            "separation",
        ],
        help=f"Task type for LoRA adapter (default: {DEFAULT_TASK})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for API requests (default: 32)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of companies to process (for testing)",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "json", "npy", "all"],
        default="all",
        help="Output format (default: all)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint saving",
    )
    parser.add_argument(
        "--merge",
        type=Path,
        metavar="EXISTING_DIR",
        help="Merge with existing embeddings from specified directory (incremental mode)",
    )
    
    return parser.parse_args()


def load_company_ids_from_json(json_path: Path) -> List[str]:
    """Load company IDs from JSON file.
    
    Supports two formats:
    - {"company_ids": ["cid_0", "cid_1", ...]}
    - ["cid_0", "cid_1", ...]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "company_ids" in data:
        return data["company_ids"]
    else:
        actual_type = type(data).__name__
        extra_info = ""
        if isinstance(data, dict):
            keys = ", ".join(map(str, data.keys()))
            extra_info = f" Available keys: {keys}" if keys else " No keys present."
        raise ValueError(
            f"Invalid JSON format in '{json_path}'. Expected list or dict with 'company_ids' key, "
            f"but got {actual_type}.{extra_info}"
        )


def load_existing_embeddings(embeddings_dir: Path) -> Tuple[np.ndarray, Dict[str, int], List[CompanyEmbedding]]:
    """Load existing embeddings from a directory.
    
    Returns:
        Tuple of (numpy array, mapping dict, list of CompanyEmbedding objects)
    """
    npy_path = embeddings_dir / "company_embeddings.npy"
    json_path = embeddings_dir / "company_embeddings.json"
    
    if not npy_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {npy_path}")
    
    # Load numpy array and mapping
    embeddings, mapping = load_embeddings_npy(npy_path)
    
    # Load full CompanyEmbedding objects from JSON if available
    existing_results = []
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            existing_results.append(CompanyEmbedding(
                company_id=item["company_id"],
                company_name=item["company_name"],
                embedding=item.get("embedding"),
                token_count=item.get("token_count", 0),
                error=item.get("error"),
            ))
    else:
        # If no JSON, create minimal CompanyEmbedding objects from mapping
        # (we won't have company_name or token_count)
        for company_id, idx in mapping.items():
            existing_results.append(CompanyEmbedding(
                company_id=company_id,
                company_name="",
                embedding=embeddings[idx].tolist(),
                token_count=0,
                error=None,
            ))
    
    return embeddings, mapping, existing_results


def merge_embeddings(
    existing_results: List[CompanyEmbedding],
    new_results: List[CompanyEmbedding],
    existing_mapping: Dict[str, int],
) -> List[CompanyEmbedding]:
    """Merge new embeddings with existing ones.
    
    New results take precedence over existing ones with the same company_id.
    """
    # Create dict of existing results by company_id
    merged = {r.company_id: r for r in existing_results}
    
    # Add/update with new results
    for r in new_results:
        merged[r.company_id] = r
    
    # Return as list, maintaining a consistent order
    return list(merged.values())


def save_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    results: List[CompanyEmbedding],
    start_time: datetime,
    end_time: datetime,
) -> None:
    """Save metadata about the embedding run."""
    total = len(results)
    successful = sum(1 for r in results if r.embedding)
    total_tokens = sum(r.token_count for r in results)
    
    metadata = {
        "model": DEFAULT_MODEL,
        "dimensions": args.dimensions,
        "task": args.task,
        "batch_size": args.batch_size,
        "input_file": str(args.input_csv),
        "output_dir": str(output_dir),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "companies_total": total,
        "companies_successful": successful,
        "companies_failed": total - successful,
        "total_tokens_used": total_tokens,
        "avg_tokens_per_company": total_tokens / total if total > 0 else 0,
    }
    
    metadata_path = output_dir / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Load API key
    try:
        api_key = load_jina_api_key()
    except EnvironmentError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please set JINA_API_KEY in your .env file or environment.", file=sys.stderr)
        return 1
    
    # Load companies
    if not args.quiet:
        print(f"Loading companies from {args.input_csv}...")
    
    try:
        companies = load_companies_from_csv(args.input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input_csv}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error loading CSV: {e}", file=sys.stderr)
        return 1
    
    if not args.quiet:
        print(f"Loaded {len(companies)} companies")
    
    # Collect company IDs to filter
    company_ids_to_filter = set()
    
    if args.company_ids:
        company_ids_to_filter.update(args.company_ids)
    
    if args.company_ids_json:
        if not args.company_ids_json.exists():
            print(f"Error: JSON file not found: {args.company_ids_json}", file=sys.stderr)
            return 1
        try:
            json_ids = load_company_ids_from_json(args.company_ids_json)
            company_ids_to_filter.update(json_ids)
            if not args.quiet:
                print(f"Loaded {len(json_ids)} company IDs from {args.company_ids_json}")
        except Exception as e:
            print(f"Error loading JSON: {e}", file=sys.stderr)
            return 1
    
    # Filter companies
    if company_ids_to_filter:
        companies = [c for c in companies if c.company_id in company_ids_to_filter]
        if not args.quiet:
            print(f"Filtered to {len(companies)} companies by ID")
    
    # Apply limit
    if args.limit:
        companies = companies[:args.limit]
        if not args.quiet:
            print(f"Limited to {len(companies)} companies")
    
    if not companies:
        print("No companies to process!")
        return 0
    
    # Handle merge mode - load existing embeddings
    existing_results = []
    existing_mapping = {}
    
    if args.merge:
        if not args.merge.exists():
            print(f"Error: Merge directory not found: {args.merge}", file=sys.stderr)
            return 1
        
        try:
            _, existing_mapping, existing_results = load_existing_embeddings(args.merge)
            if not args.quiet:
                print(f"Loaded {len(existing_mapping)} existing embeddings from {args.merge}")
        except Exception as e:
            print(f"Error loading existing embeddings: {e}", file=sys.stderr)
            return 1
        
        # Filter out companies that already have embeddings (unless specifically requested via --company-ids)
        if not company_ids_to_filter:
            original_count = len(companies)
            companies = [c for c in companies if c.company_id not in existing_mapping]
            skipped = original_count - len(companies)
            if not args.quiet and skipped > 0:
                print(f"Skipping {skipped} companies with existing embeddings")
        
        if not companies:
            if not args.quiet:
                print("All companies already have embeddings. Nothing to do.")
            return 0
    
    # Set up output directory
    if args.output_dir is None:
        if args.merge:
            # Default to same directory when merging
            args.output_dir = args.merge
        else:
            # Default to output_production/company_embedding for unified structure
            args.output_dir = Path("output_production/company_embedding")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up checkpoint path
    checkpoint_path = None
    if not args.no_checkpoint:
        checkpoint_path = args.output_dir / ".checkpoint.json"
        if not args.resume and checkpoint_path.exists():
            if not args.quiet:
                print(
                    f"Warning: Existing checkpoint found at {checkpoint_path} and it will be deleted. "
                    "Use --resume to continue from it.",
                    file=sys.stderr,
                )
            checkpoint_path.unlink()  # Remove old checkpoint if not resuming
    
    # Initialize embedder
    embedder = CompanyEmbedder(
        api_key=api_key,
        dimensions=args.dimensions,
        task=args.task,
        batch_size=args.batch_size,
    )
    
    # Print config
    if not args.quiet:
        print("\nConfiguration:")
        print(f"  Model: {embedder.model}")
        print(f"  Dimensions: {embedder.dimensions}")
        print(f"  Task: {embedder.task}")
        print(f"  Batch size: {embedder.batch_size}")
        print(f"  Output: {args.output_dir}")
        if args.resume and checkpoint_path and checkpoint_path.exists():
            print("  Resuming from checkpoint")
        print()
    
    # Process companies
    start_time = datetime.now()
    
    results = embedder.embed_companies(
        companies,
        show_progress=not args.quiet,
        checkpoint_path=checkpoint_path if not args.no_checkpoint else None,
    )
    
    end_time = datetime.now()
    
    # Clean up checkpoint on success
    if checkpoint_path and checkpoint_path.exists():
        checkpoint_path.unlink()
    
    # Merge with existing results if in merge mode
    new_results = results
    if args.merge and existing_results:
        results = merge_embeddings(existing_results, new_results, existing_mapping)
        if not args.quiet:
            print(f"Merged {len(new_results)} new embeddings with {len(existing_results)} existing")
            print(f"Total embeddings: {len(results)}")
    
    # Save results
    if args.output_format in ("csv", "all"):
        csv_path = args.output_dir / "company_embeddings.csv"
        save_embeddings_csv(results, csv_path)
        if not args.quiet:
            print(f"Saved CSV: {csv_path}")
    
    if args.output_format in ("json", "all"):
        json_path = args.output_dir / "company_embeddings.json"
        save_embeddings_json(results, json_path)
        if not args.quiet:
            print(f"Saved JSON: {json_path}")
    
    if args.output_format in ("npy", "all"):
        npy_path = args.output_dir / "company_embeddings.npy"
        save_embeddings_npy(results, npy_path)
        if not args.quiet:
            print(f"Saved NPY: {npy_path}")
            print(f"Saved mapping: {npy_path.with_suffix('.mapping.json')}")
    
    # Save metadata
    save_run_metadata(args.output_dir, args, results, start_time, end_time)
    if not args.quiet:
        print(f"Saved metadata: {args.output_dir / 'run_metadata.json'}")
    
    # Print summary
    if not args.quiet:
        # Show summary of newly processed companies
        print_embedding_summary(new_results)
        duration = (end_time - start_time).total_seconds()
        print(f"\nTotal time: {duration:.1f}s")
        if new_results:
            print(f"Average: {duration / len(new_results):.2f}s per company")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
