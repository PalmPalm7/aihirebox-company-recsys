#!/usr/bin/env python3
"""
Production Company Tagging Script with Web Search Support.

This script processes company data from a CSV file and extracts structured tags
using LLM models. Supports web search enhancement via OpenRouter's :online suffix
for better team_background accuracy. Also supports incremental updates via --merge.

Features:
- **Checkpoint/Resume**: Saves progress every N companies to prevent data loss
- **Merge Mode**: Incremental updates that merge with existing tags
- **Web Search**: Better team_background accuracy with :online models

Usage:
    # Basic usage with default model (openai/gpt-5-mini:online)
    python run_tagging.py data/aihirebox_company_list_sample.csv

    # Use baseline model without web search
    python run_tagging.py data/aihirebox_company_list_sample.csv --model openai/gpt-5-mini

    # Incremental mode - merge new tags with existing ones
    python run_tagging.py data/aihirebox_company_list.csv --merge output_production/company_tagging

    # Resume from interruption (auto-enabled when output dir has existing data)
    python run_tagging.py data/aihirebox_company_list.csv --resume

    # Custom checkpoint interval (save every 5 companies)
    python run_tagging.py data/aihirebox_company_list.csv --checkpoint-interval 5

    # Process specific companies and merge with existing
    python run_tagging.py data/aihirebox_company_list.csv --company-ids cid_new_1 cid_new_2 --merge output_production/

    # Limit companies for testing
    python run_tagging.py data/aihirebox_company_list_sample.csv --limit 5

    # Custom output directory
    python run_tagging.py data/aihirebox_company_list_sample.csv --output-dir ./my_output
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

from company_tagging import (
    CompanyTagger,
    CompanyTags,
    load_companies_from_csv,
    load_company_ids_from_json,
    load_api_keys,
    save_results_csv,
    save_results_json,
    save_taxonomy,
    print_summary,
    calculate_team_metrics,
)


# Default model with web search enabled for better team_background
DEFAULT_MODEL = "openai/gpt-5-mini:online"

# Default checkpoint interval (save every N companies)
DEFAULT_CHECKPOINT_INTERVAL = 10

# Available models
AVAILABLE_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-120b:online",
    "openai/gpt-5-mini",
    "openai/gpt-5-mini:online",
    "openai/gpt-4o-mini",
    "openai/gpt-4o-mini:online",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash:online",
]


class CheckpointManager:
    """Manages incremental saving of tagging results to prevent data loss."""
    
    def __init__(
        self,
        output_dir: Path,
        existing_tags: List[CompanyTags],
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        quiet: bool = False,
    ):
        self.output_dir = output_dir
        self.checkpoint_interval = checkpoint_interval
        self.quiet = quiet
        
        # Initialize with existing tags (for merge mode)
        self.all_results: Dict[str, CompanyTags] = {t.company_id: t for t in existing_tags}
        self.new_results: List[CompanyTags] = []
        self.processed_count = 0
        self.last_checkpoint_count = 0
    
    def on_result(self, result: CompanyTags, index: int, _total: int) -> None:
        """Callback for each processed company. Triggers checkpoint if needed."""
        self.new_results.append(result)
        self.all_results[result.company_id] = result
        self.processed_count = index + 1
        
        # Check if we should save a checkpoint
        if self.processed_count - self.last_checkpoint_count >= self.checkpoint_interval:
            self._save_checkpoint()
            self.last_checkpoint_count = self.processed_count
    
    def _save_checkpoint(self) -> None:
        """Save current progress to disk."""
        results_list = list(self.all_results.values())
        results_list.sort(key=lambda x: x.company_id)
        
        # Save both CSV and JSON
        csv_path = self.output_dir / "company_tags.csv"
        json_path = self.output_dir / "company_tags.json"
        
        save_results_csv(results_list, csv_path)
        save_results_json(results_list, json_path)
        
        if not self.quiet:
            print(f"  💾 Checkpoint saved: {self.processed_count} companies processed ({len(results_list)} total)")
    
    def finalize(self) -> List[CompanyTags]:
        """Save final results and return all tags."""
        # Final save (in case we didn't hit the last checkpoint)
        if self.processed_count > self.last_checkpoint_count:
            self._save_checkpoint()
        
        results_list = list(self.all_results.values())
        results_list.sort(key=lambda x: x.company_id)
        return results_list
    
    def get_new_results(self) -> List[CompanyTags]:
        """Get only the newly processed results (for summary)."""
        return self.new_results


def load_existing_tags(tags_dir: Path) -> tuple[List[CompanyTags], Set[str]]:
    """Load existing tags from a directory.
    
    Returns:
        Tuple of (list of CompanyTags, set of company_ids)
    """
    json_path = tags_dir / "company_tags.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Tags not found: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    existing_tags = []
    existing_ids = set()
    
    for item in data:
        tags = CompanyTags(
            company_id=item["company_id"],
            company_name=item["company_name"],
            industry=item.get("industry", []),
            business_model=item.get("business_model", []),
            target_market=item.get("target_market", []),
            company_stage=item.get("company_stage", "unknown"),
            tech_focus=item.get("tech_focus", []),
            team_background=item.get("team_background", []),
            confidence_score=item.get("confidence_score", 0.0),
            raw_reasoning=item.get("raw_reasoning", item.get("reasoning", "")),
        )
        existing_tags.append(tags)
        existing_ids.add(item["company_id"])
    
    return existing_tags, existing_ids


def merge_tags(
    existing_tags: List[CompanyTags],
    new_tags: List[CompanyTags],
) -> List[CompanyTags]:
    """Merge new tags with existing ones.
    
    New results take precedence over existing ones with the same company_id.
    """
    merged = {t.company_id: t for t in existing_tags}
    
    for t in new_tags:
        merged[t.company_id] = t
    
    return list(merged.values())


def save_run_metadata(
    output_dir: Path,
    model: str,
    input_file: Path,
    results: list[CompanyTags],
    duration_seconds: float,
    include_reasoning: bool = True,
) -> None:
    """Save run metadata to JSON file."""
    team_metrics = calculate_team_metrics(results)
    avg_confidence = sum(r.confidence_score for r in results) / len(results) if results else 0
    
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "web_search_enabled": ":online" in model,
        "include_reasoning": include_reasoning,
        "input_file": str(input_file),
        "companies_processed": len(results),
        "duration_seconds": round(duration_seconds, 2),
        "avg_confidence": round(avg_confidence, 3),
        "team_background_metrics": team_metrics,
    }
    
    metadata_path = output_dir / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Saved metadata: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MECE tags from company details using LLM with optional web search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Basic usage with default model (web search enabled)
  python run_tagging.py data/aihirebox_company_list_sample.csv
  
  # Use baseline model without web search
  python run_tagging.py data/aihirebox_company_list_sample.csv --model openai/gpt-5-mini
  
  # Limit companies for testing
  python run_tagging.py data/aihirebox_company_list_sample.csv --limit 5
  
  # Process specific companies by ID
  python run_tagging.py data/aihirebox_company_list_sample.csv --company-ids cid_0 cid_1
  
  # Process companies from JSON file (supports {{"company_ids": [...]}} or [...])
  python run_tagging.py data/aihirebox_company_list_sample.csv --company-ids-json my_companies.json
  
  # Resume from interruption (continues from last checkpoint)
  python run_tagging.py data/aihirebox_company_list.csv --resume
  
  # Custom checkpoint interval (save every 5 companies)
  python run_tagging.py data/aihirebox_company_list.csv --checkpoint-interval 5

Available models:
{chr(10).join(f'  - {m}' for m in AVAILABLE_MODELS)}

Note: Models with ':online' suffix enable real-time web search for better team_background accuracy.
        """
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to input CSV file with company data",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL}). Add ':online' suffix for web search.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files (default: output/company_tags_<timestamp>)",
    )
    parser.add_argument(
        "--company-ids",
        nargs="+",
        help="Specific company IDs to process (processes all if not specified)",
    )
    parser.add_argument(
        "--company-ids-json",
        type=Path,
        help="JSON file containing company IDs to process (expects {\"company_ids\": [...]} or [...])",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of companies to process (for testing)",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "json", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--save-taxonomy",
        action="store_true",
        help="Also save the tag taxonomy definition",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bar and detailed output",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable reasoning output to reduce token cost",
    )
    parser.add_argument(
        "--merge",
        type=Path,
        metavar="EXISTING_DIR",
        help="Merge with existing tags from specified directory (incremental mode)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help=f"Save progress every N companies (default: {DEFAULT_CHECKPOINT_INTERVAL})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (skips already processed companies in output dir)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = load_api_keys()
    
    # Validate input file
    if not args.input_csv.exists():
        print(f"Error: Input file not found: {args.input_csv}")
        sys.exit(1)
    
    # Load companies
    if not args.quiet:
        print(f"Loading companies from {args.input_csv}...")
    companies = load_companies_from_csv(args.input_csv)
    if not args.quiet:
        print(f"Loaded {len(companies)} companies")
    
    # Collect company IDs from both sources
    company_ids_to_filter = set()
    
    # From --company-ids argument
    if args.company_ids:
        company_ids_to_filter.update(args.company_ids)
    
    # From --company-ids-json file
    if args.company_ids_json:
        if not args.company_ids_json.exists():
            print(f"Error: JSON file not found: {args.company_ids_json}")
            sys.exit(1)
        json_ids = load_company_ids_from_json(args.company_ids_json)
        company_ids_to_filter.update(json_ids)
        if not args.quiet:
            print(f"Loaded {len(json_ids)} company IDs from {args.company_ids_json}")
    
    # Filter by company IDs if any were specified
    if company_ids_to_filter:
        companies = [c for c in companies if c.company_id in company_ids_to_filter]
        if not args.quiet:
            print(f"Filtered to {len(companies)} companies by ID")
    
    # Apply limit if specified
    if args.limit:
        companies = companies[:args.limit]
        if not args.quiet:
            print(f"Limited to {len(companies)} companies")
    
    if not companies:
        print("No companies to process!")
        sys.exit(1)
    
    # Setup output directory first (needed for resume check)
    if args.output_dir is None:
        if args.merge:
            # Default to same directory when merging
            args.output_dir = args.merge
        else:
            # Default to output_production/company_tagging for unified structure
            args.output_dir = Path("output_production/company_tagging")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle merge mode - load existing tags
    existing_tags = []
    existing_ids: Set[str] = set()
    
    if args.merge:
        if not args.merge.exists():
            print(f"Error: Merge directory not found: {args.merge}")
            sys.exit(1)
        
        try:
            existing_tags, existing_ids = load_existing_tags(args.merge)
            if not args.quiet:
                print(f"Loaded {len(existing_ids)} existing tags from {args.merge}")
        except FileNotFoundError:
            if not args.quiet:
                print(f"No existing tags found in {args.merge}, starting fresh")
        except Exception as e:
            print(f"Error loading existing tags: {e}")
            sys.exit(1)
        
        # Filter out companies that already have tags (unless specifically requested via --company-ids)
        if not company_ids_to_filter:
            original_count = len(companies)
            companies = [c for c in companies if c.company_id not in existing_ids]
            skipped = original_count - len(companies)
            if not args.quiet and skipped > 0:
                print(f"Skipping {skipped} companies with existing tags")
    
    # Handle resume mode - load from output directory
    if args.resume and not args.merge:
        try:
            existing_tags, existing_ids = load_existing_tags(args.output_dir)
            if not args.quiet:
                print(f"🔄 Resuming: Found {len(existing_ids)} already processed companies")
            
            # Filter out already processed companies
            original_count = len(companies)
            companies = [c for c in companies if c.company_id not in existing_ids]
            skipped = original_count - len(companies)
            if not args.quiet and skipped > 0:
                print(f"Skipping {skipped} already processed companies")
        except FileNotFoundError:
            if not args.quiet:
                print("No checkpoint found, starting fresh")
        except Exception as e:
            if not args.quiet:
                print(f"Warning: Could not load checkpoint: {e}, starting fresh")
    
    if not companies:
        if not args.quiet:
            print("All companies already processed. Nothing to do.")
            print(f"✅ Results available at: {args.output_dir}")
        sys.exit(0)
    
    # Initialize tagger
    is_online = ":online" in args.model
    if not args.quiet:
        print(f"\nModel: {args.model}")
        print(f"Web search: {'✓ Enabled' if is_online else '✗ Disabled'}")
        print(f"Reasoning: {'✗ Disabled' if args.no_reasoning else '✓ Enabled'}")
        print(f"Checkpoint: Every {args.checkpoint_interval} companies")
        print(f"Output: {args.output_dir}")
        print(f"Processing {len(companies)} companies...")
    
    tagger = CompanyTagger(
        openrouter_api_key=keys["openrouter"],
        fallback_api_key=keys["openrouter_fallback"],
        model=args.model,
        include_reasoning=not args.no_reasoning,
    )
    
    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(
        output_dir=args.output_dir,
        existing_tags=existing_tags,
        checkpoint_interval=args.checkpoint_interval,
        quiet=args.quiet,
    )
    
    # Process companies with incremental saving
    start_time = time.time()
    try:
        tagger.tag_companies(
            companies, 
            show_progress=not args.quiet,
            delay_seconds=args.delay,
            on_result=checkpoint_mgr.on_result,
        )
    except KeyboardInterrupt:
        if not args.quiet:
            print("\n⚠️ Interrupted! Saving checkpoint...")
        checkpoint_mgr.finalize()
        if not args.quiet:
            print("✅ Progress saved. Run with --resume to continue.")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        if not args.quiet:
            print(f"\n❌ Error: {e}")
            print("Saving checkpoint...")
        checkpoint_mgr.finalize()
        if not args.quiet:
            print("✅ Progress saved. Run with --resume to continue.")
        raise
    
    duration = time.time() - start_time
    
    # Finalize and get results
    results = checkpoint_mgr.finalize()
    new_results = checkpoint_mgr.get_new_results()
    
    if not args.quiet:
        if existing_tags:
            print(f"Merged {len(new_results)} new tags with {len(existing_tags)} existing")
            print(f"Total tags: {len(results)}")
    
    if args.save_taxonomy:
        taxonomy_path = args.output_dir / "tag_taxonomy.json"
        save_taxonomy(taxonomy_path)
        if not args.quiet:
            print(f"Saved taxonomy: {taxonomy_path}")
    
    # Save metadata
    save_run_metadata(args.output_dir, args.model, args.input_csv, results, duration, not args.no_reasoning)
    
    # Print summary
    if not args.quiet:
        # Show summary of newly processed companies
        print_summary(new_results)
        
        # Team background specific metrics for new results
        team_metrics = calculate_team_metrics(new_results)
        print("\n--- Team Background Analysis (New) ---")
        print(f"Coverage: {team_metrics['coverage_rate']*100:.1f}% ({team_metrics['known_team_background']}/{team_metrics['total_companies']})")
        print(f"Distribution: {team_metrics['tag_distribution']}")
        if new_results:
            print(f"\nDuration: {duration:.1f}s ({duration/len(new_results):.1f}s per company)")
        
        print(f"\n✅ Tagging complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
