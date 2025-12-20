#!/usr/bin/env python3
"""
Production Company Recommendation Script.

Generate multi-dimensional company recommendations with head suppression.

Usage:
    # Recommend for a single company
    python run_recommender.py --company-id cid_100

    # Recommend for multiple companies
    python run_recommender.py --company-ids cid_100 cid_109 cid_114
    
    # Recommend for all companies (batch mode)
    python run_recommender.py --all
    
    # Customize recommendation parameters
    python run_recommender.py --company-id cid_100 --num-dimensions 5 --max-per-dim 5
    
    # Adjust head suppression penalty
    python run_recommender.py --company-id cid_100 --head-penalty 0.8
    
    # Disable semantic (embedding) dimension
    python run_recommender.py --company-id cid_100 --no-semantic
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from company_recommender import (
    CompanyRecommender,
    CompanyRecommendations,
    CompanyStageHeadSuppression,
    CompositeHeadSuppression,
    IDFHeadSuppression,
    load_company_profiles,
    load_embeddings,
    print_recommendations,
    recommendations_to_dict,
    save_recommendations_json,
)


# ============================================================================
# Default Paths (unified production directory with subfolders)
# ============================================================================

DEFAULT_PRODUCTION_DIR = Path("output_production")
DEFAULT_TAGS_PATH = DEFAULT_PRODUCTION_DIR / "company_tagging" / "company_tags.json"
DEFAULT_EMBEDDINGS_DIR = DEFAULT_PRODUCTION_DIR / "company_embedding"


def find_embeddings_dir() -> Optional[Path]:
    """Find the embeddings directory.
    
    Priority:
    1. output_production/company_embedding/ (if contains company_embeddings.npy)
    2. Latest output/company_embeddings_*/ directory
    """
    # First check unified production directory
    prod_dir = DEFAULT_EMBEDDINGS_DIR
    if prod_dir.exists() and (prod_dir / "company_embeddings.npy").exists():
        return prod_dir
    
    # Fallback to legacy output/company_embeddings_*/ structure
    output_dir = Path("output")
    if not output_dir.exists():
        return None
    
    embedding_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("company_embeddings_")],
        key=lambda x: x.name,
        reverse=True,
    )
    
    return embedding_dirs[0] if embedding_dirs else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multi-dimensional company recommendations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recommend for a single company
  python run_recommender.py --company-id cid_100
  
  # Recommend for multiple companies
  python run_recommender.py --company-ids cid_100 cid_109 cid_114
  
  # Batch mode for all companies
  python run_recommender.py --all
  
  # With custom parameters
  python run_recommender.py --company-id cid_100 --num-dimensions 5 --head-penalty 0.7
  
  # Output to specific directory
  python run_recommender.py --company-id cid_100 --output-dir ./my_recommendations
        """
    )
    
    # Input selection (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--company-id",
        type=str,
        help="Single company ID to generate recommendations for",
    )
    input_group.add_argument(
        "--company-ids",
        nargs="+",
        help="Multiple company IDs to generate recommendations for",
    )
    input_group.add_argument(
        "--company-ids-json",
        type=Path,
        help='JSON file with company IDs (expects {"company_ids": [...]} or [...])',
    )
    input_group.add_argument(
        "--all",
        action="store_true",
        help="Generate recommendations for all companies",
    )
    
    # Data paths
    parser.add_argument(
        "--tags-path",
        type=Path,
        default=DEFAULT_TAGS_PATH,
        help=f"Path to company tags JSON (default: {DEFAULT_TAGS_PATH})",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help="Directory containing embeddings (default: latest in output/)",
    )
    
    # Recommendation parameters
    parser.add_argument(
        "--num-dimensions",
        type=int,
        default=5,
        help="Number of recommendation dimensions (default: 5)",
    )
    parser.add_argument(
        "--min-per-dim",
        type=int,
        default=3,
        help="Minimum companies per dimension (default: 3)",
    )
    parser.add_argument(
        "--max-per-dim",
        type=int,
        default=5,
        help="Maximum companies per dimension (default: 5)",
    )
    parser.add_argument(
        "--head-penalty",
        type=float,
        default=0.6,
        help="Head suppression penalty 0-1 (default: 0.6, higher = more suppression)",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic (embedding-based) dimension",
    )
    parser.add_argument(
        "--no-diversity",
        action="store_true",
        help="Allow companies to appear in multiple dimensions",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Minimum similarity score to include a company (default: 0.5)",
    )
    parser.add_argument(
        "--max-below-threshold",
        type=int,
        default=2,
        help="Max companies below threshold before dropping dimension (default: 2)",
    )
    parser.add_argument(
        "--no-embedding-boost",
        action="store_true",
        help="Disable embedding similarity boost for tag-based recommendations",
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: output/recommendations_<timestamp>)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print to console, don't save files",
    )
    
    return parser.parse_args()


def load_company_ids_from_json(json_path: Path) -> List[str]:
    """Load company IDs from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "company_ids" in data:
        return data["company_ids"]
    else:
        raise ValueError("Invalid JSON format. Expected list or dict with 'company_ids' key.")


def save_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    company_ids: List[str],
    results: List[CompanyRecommendations],
) -> None:
    """Save metadata about the recommendation run."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "tags_path": str(args.tags_path),
        "embeddings_dir": str(args.embeddings_dir) if args.embeddings_dir else None,
        "parameters": {
            "num_dimensions": args.num_dimensions,
            "min_per_dim": args.min_per_dim,
            "max_per_dim": args.max_per_dim,
            "head_penalty": args.head_penalty,
            "score_threshold": args.score_threshold,
            "max_below_threshold": args.max_below_threshold,
            "include_semantic": not args.no_semantic,
            "diversity_constraint": not args.no_diversity,
            "embedding_boost": not args.no_embedding_boost,
        },
        "companies_processed": len(company_ids),
        "successful_recommendations": len(results),
        "company_ids": company_ids,
    }
    
    metadata_path = output_dir / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    
    # Validate paths
    if not args.tags_path.exists():
        print(f"Error: Tags file not found: {args.tags_path}", file=sys.stderr)
        print("Run company tagging first: python run_tagging.py ...", file=sys.stderr)
        return 1
    
    # Find embeddings directory
    if args.embeddings_dir is None:
        args.embeddings_dir = find_embeddings_dir()
    
    # Load company profiles
    if not args.quiet:
        print(f"Loading company profiles from {args.tags_path}...")
    
    profiles = load_company_profiles(args.tags_path)
    
    if not args.quiet:
        print(f"Loaded {len(profiles)} company profiles")
    
    # Load embeddings if available
    embeddings = None
    embedding_mapping = None
    
    if args.embeddings_dir and not args.no_semantic:
        npy_path = args.embeddings_dir / "company_embeddings.npy"
        if npy_path.exists():
            if not args.quiet:
                print(f"Loading embeddings from {args.embeddings_dir}...")
            embeddings, embedding_mapping = load_embeddings(npy_path)
            if not args.quiet:
                print(f"Loaded {len(embedding_mapping)} embeddings")
        else:
            if not args.quiet:
                print(f"Warning: Embeddings not found at {npy_path}, semantic dimension disabled")
    
    # Initialize recommender
    head_suppression = CompositeHeadSuppression([
        (CompanyStageHeadSuppression(args.head_penalty), 0.6),
        (IDFHeadSuppression(0.3), 0.4),
    ])
    
    recommender = CompanyRecommender(
        companies=profiles,
        embeddings=embeddings,
        embedding_mapping=embedding_mapping,
        head_suppression=head_suppression,
    )
    
    # Determine company IDs to process
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
        company_ids = [p.company_id for p in profiles]
    else:
        print("Error: No companies specified", file=sys.stderr)
        return 1
    
    if not args.quiet:
        print(f"\nGenerating recommendations for {len(company_ids)} companies...")
        print(f"Parameters: {args.num_dimensions} dimensions, {args.min_per_dim}-{args.max_per_dim} companies/dim")
        print(f"Head suppression penalty: {args.head_penalty}")
        print(f"Score threshold: {args.score_threshold} (drop dimension if >{args.max_below_threshold} below)")
        print(f"Embedding boost: {'disabled' if args.no_embedding_boost else 'enabled'}")
        print(f"Semantic dimension: {'enabled' if not args.no_semantic and embeddings is not None else 'disabled'}")
        print(f"Diversity constraint: {'enabled' if not args.no_diversity else 'disabled'}")
    
    # Generate recommendations
    results = []
    for i, cid in enumerate(company_ids, 1):
        try:
            rec = recommender.recommend(
                cid,
                num_dimensions=args.num_dimensions,
                min_companies_per_dim=args.min_per_dim,
                max_companies_per_dim=args.max_per_dim,
                include_semantic=not args.no_semantic and embeddings is not None,
                diversity_constraint=not args.no_diversity,
                score_threshold=args.score_threshold,
                max_below_threshold=args.max_below_threshold,
                use_embedding_boost=not args.no_embedding_boost and embeddings is not None,
            )
            results.append(rec)
            
            if not args.quiet and not args.print_only:
                print(f"  [{i}/{len(company_ids)}] {rec.query_company_name}: {len(rec.recommendation_groups)} dimensions")
            
            if args.print_only or (not args.quiet and len(company_ids) == 1):
                print_recommendations(rec)
                
        except ValueError as e:
            print(f"Warning: {e}", file=sys.stderr)
    
    if not results:
        print("No recommendations generated!", file=sys.stderr)
        return 1
    
    # Save results
    if not args.print_only:
        if args.output_dir is None:
            # Default to output_production/recommender for unified structure
            args.output_dir = Path("output_production/recommender")
        
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save recommendations
        output_path = args.output_dir / "recommendations.json"
        save_recommendations_json(results, output_path)
        
        if not args.quiet:
            print(f"\nSaved recommendations: {output_path}")
        
        # Save metadata
        save_run_metadata(args.output_dir, args, company_ids, results)
        
        if not args.quiet:
            print(f"Saved metadata: {args.output_dir / 'run_metadata.json'}")
            print(f"\n✅ Done! Results saved to {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
