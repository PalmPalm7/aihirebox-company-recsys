"""
Test script for comparing different LLM models on company tagging.

Tests the company_tagging.py script with multiple models:
- openai/gpt-4o-mini
- deepseek/deepseek-chat  
- google/gemini-2.5-flash

Saves results for each model separately for comparison.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from company_tagging import (
    CompanyTagger,
    CompanyRecord,
    CompanyTags,
    load_companies_from_csv,
    save_results_csv,
    save_results_json,
    print_summary,
    TAG_TAXONOMY,
)


# Models to test (OpenRouter model IDs)
MODELS_TO_TEST = [
    "openai/gpt-4o-mini",
    "deepseek/deepseek-chat",  # DeepSeek V3
    "google/gemini-2.5-flash",
]


def run_model_test(
    model: str,
    companies: List[CompanyRecord],
    api_key: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run tagging test for a single model."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model}")
    print(f"{'='*60}")
    
    # Create model-specific output directory
    model_safe_name = model.replace("/", "_").replace(".", "_")
    model_output_dir = output_dir / model_safe_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tagger with the specific model
    tagger = CompanyTagger(
        openrouter_api_key=api_key,
        model=model,
    )
    
    # Track timing
    start_time = time.time()
    
    # Run tagging (with fewer workers to avoid rate limits)
    try:
        results = tagger.tag_companies(
            companies, 
            max_workers=2,  # Lower workers for testing
            show_progress=True,
        )
        success = True
        error_msg = None
    except Exception as e:
        results = []
        success = False
        error_msg = str(e)
        print(f"Error with model {model}: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Save results if successful
    if success and results:
        save_results_csv(results, model_output_dir / "company_tags.csv")
        save_results_json(results, model_output_dir / "company_tags.json")
        print_summary(results)
    
    # Collect test metadata
    test_result = {
        "model": model,
        "success": success,
        "error": error_msg,
        "duration_seconds": round(duration, 2),
        "companies_processed": len(results) if results else 0,
        "output_dir": str(model_output_dir),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Add quality metrics if successful
    if success and results:
        # Calculate average confidence
        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        
        # Count tag coverage
        industry_coverage = sum(1 for r in results if r.industry) / len(results)
        tech_coverage = sum(1 for r in results if r.tech_focus) / len(results)
        team_coverage = sum(1 for r in results if r.team_background and "unknown" not in r.team_background) / len(results)
        
        test_result.update({
            "avg_confidence": round(avg_confidence, 3),
            "industry_coverage": round(industry_coverage, 3),
            "tech_coverage": round(tech_coverage, 3),
            "team_coverage": round(team_coverage, 3),
        })
    
    return test_result


def compare_results(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Compare and save comparison summary."""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    # Print comparison table
    print(f"\n{'Model':<30} {'Success':<10} {'Time(s)':<10} {'Confidence':<12} {'Industry%':<12}")
    print("-" * 74)
    
    for result in all_results:
        model = result["model"][:28]
        success = "✓" if result["success"] else "✗"
        duration = f"{result['duration_seconds']:.1f}"
        confidence = f"{result.get('avg_confidence', 0):.2f}" if result["success"] else "N/A"
        industry = f"{result.get('industry_coverage', 0)*100:.0f}%" if result["success"] else "N/A"
        
        print(f"{model:<30} {success:<10} {duration:<10} {confidence:<12} {industry:<12}")
    
    # Save comparison summary
    comparison_file = output_dir / "comparison_summary.json"
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_timestamp": datetime.now().isoformat(),
            "models_tested": len(all_results),
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nComparison saved to: {comparison_file}")


def main():
    # Load environment
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is required in .env file")
    
    # Load sample companies
    project_root = Path(__file__).parent.parent
    sample_csv = project_root / "data" / "aihirebox_company_list_sample.csv"
    if not sample_csv.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_csv}")
    
    print(f"Loading companies from {sample_csv}...")
    companies = load_companies_from_csv(sample_csv)
    print(f"Loaded {len(companies)} companies for testing")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "output" / f"model_comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Save taxonomy for reference
    with open(output_dir / "tag_taxonomy.json", "w", encoding="utf-8") as f:
        json.dump(TAG_TAXONOMY, f, ensure_ascii=False, indent=2)
    
    # Run tests for each model
    all_results = []
    for model in MODELS_TO_TEST:
        try:
            result = run_model_test(model, companies, api_key, output_dir)
            all_results.append(result)
        except Exception as e:
            print(f"Failed to test model {model}: {e}")
            all_results.append({
                "model": model,
                "success": False,
                "error": str(e),
                "duration_seconds": 0,
                "companies_processed": 0,
            })
        
        # Small delay between models to avoid rate limiting
        time.sleep(2)
    
    # Generate comparison report
    compare_results(all_results, output_dir)
    
    print(f"\n✅ Testing complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

