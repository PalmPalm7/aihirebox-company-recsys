#!/usr/bin/env python
"""
Production Output Validation Script

Validates production outputs by comparing against source company list:
1. Tagging: Checks for errors or completely failed tagging
2. Articles: Checks for companies missing entirely from articles index

Usage:
    python scripts/validate_production.py
    python scripts/validate_production.py --json
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class TaggingIssue:
    """Represents a tagging issue for a company."""
    company_id: str
    company_name: str
    details: str


@dataclass
class ArticleIssue:
    """Represents a company missing from articles."""
    company_id: str
    company_name: str


@dataclass
class ValidationReport:
    """Complete validation report."""
    tagging_issues: List[TaggingIssue] = field(default_factory=list)
    article_issues: List[ArticleIssue] = field(default_factory=list)
    source_company_count: int = 0
    tagged_company_count: int = 0
    article_company_count: int = 0


def load_source_companies(csv_path: Path) -> Dict[str, str]:
    """Load company_id -> company_name mapping from source CSV."""
    companies = {}
    if not csv_path.exists():
        print(f"Warning: Source CSV not found: {csv_path}")
        return companies

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            companies[row['company_id']] = row['company_name']
    return companies


def load_company_tags(tags_path: Path) -> Dict[str, dict]:
    """Load company_id -> tag_data mapping from JSON."""
    tags = {}
    if not tags_path.exists():
        print(f"Warning: Tags file not found: {tags_path}")
        return tags

    with open(tags_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for company in data:
            tags[company['company_id']] = company
    return tags


def load_article_index(index_path: Path) -> Dict[str, dict]:
    """Load company_id -> article_data mapping from index.json."""
    articles = {}
    if not index_path.exists():
        print(f"Warning: Article index not found: {index_path}")
        return articles

    with open(index_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        articles = data.get("companies", {})
    return articles


def is_tagging_failed(tag_data: dict) -> tuple[bool, str]:
    """
    Check if tagging failed for a company.
    Returns (is_failed, reason).
    """
    reasoning = tag_data.get("raw_reasoning", "")
    confidence = tag_data.get("confidence_score", 0)

    # Check for error in reasoning
    if "Error" in reasoning or "error" in reasoning:
        return True, f"Error in reasoning: {reasoning[:80]}..."

    # Check for zero confidence with all empty fields
    if confidence == 0:
        industry = tag_data.get("industry", [])
        tech_focus = tag_data.get("tech_focus", [])
        team_bg = tag_data.get("team_background", [])

        if not industry and not tech_focus and not team_bg:
            return True, "confidence=0 with all empty fields"

    return False, ""


def validate(
    source_companies: Dict[str, str],
    tags: Dict[str, dict],
    articles: Dict[str, dict]
) -> ValidationReport:
    """Run validation and return report."""
    report = ValidationReport(
        source_company_count=len(source_companies),
        tagged_company_count=len(tags),
        article_company_count=len(articles)
    )

    for cid, name in source_companies.items():
        # Check tagging
        if cid not in tags:
            report.tagging_issues.append(TaggingIssue(
                company_id=cid,
                company_name=name,
                details="Missing from company_tags.json"
            ))
        else:
            failed, reason = is_tagging_failed(tags[cid])
            if failed:
                report.tagging_issues.append(TaggingIssue(
                    company_id=cid,
                    company_name=name,
                    details=reason
                ))

        # Check articles - only flag if completely missing
        if cid not in articles:
            report.article_issues.append(ArticleIssue(
                company_id=cid,
                company_name=name
            ))

    return report


def print_report(report: ValidationReport):
    """Print validation report to console."""
    print("\n" + "=" * 60)
    print("PRODUCTION VALIDATION REPORT")
    print("=" * 60)

    print(f"\nSummary:")
    print(f"  Source companies: {report.source_company_count}")
    print(f"  Tagged companies: {report.tagged_company_count}")
    print(f"  Companies with articles: {report.article_company_count}")
    print(f"  Tagging issues: {len(report.tagging_issues)}")
    print(f"  Missing articles: {len(report.article_issues)}")

    if report.tagging_issues:
        print(f"\n--- Tagging Issues ({len(report.tagging_issues)}) ---")
        for issue in report.tagging_issues:
            print(f"  - {issue.company_id} ({issue.company_name}): {issue.details}")

    if report.article_issues:
        print(f"\n--- Missing from Articles ({len(report.article_issues)}) ---")
        for issue in report.article_issues:
            print(f"  - {issue.company_id} ({issue.company_name})")

    # Collect all company IDs with issues
    all_issue_cids = set()
    for issue in report.tagging_issues:
        all_issue_cids.add(issue.company_id)
    for issue in report.article_issues:
        all_issue_cids.add(issue.company_id)

    if all_issue_cids:
        print(f"\n--- Recommended Fix ---")
        cids_str = " ".join(sorted(all_issue_cids))
        print(f"  python scripts/fix_companies.py --company-ids {cids_str}")
    else:
        print(f"\n[OK] No issues found!")

    print("")


def report_to_json(report: ValidationReport) -> Dict:
    """Convert report to JSON-serializable dict."""
    all_issue_cids = set()
    for issue in report.tagging_issues:
        all_issue_cids.add(issue.company_id)
    for issue in report.article_issues:
        all_issue_cids.add(issue.company_id)

    return {
        "summary": {
            "source_companies": report.source_company_count,
            "tagged_companies": report.tagged_company_count,
            "companies_with_articles": report.article_company_count,
            "tagging_issues_count": len(report.tagging_issues),
            "article_issues_count": len(report.article_issues),
        },
        "tagging_issues": [
            {
                "company_id": i.company_id,
                "company_name": i.company_name,
                "details": i.details,
            }
            for i in report.tagging_issues
        ],
        "article_issues": [
            {
                "company_id": i.company_id,
                "company_name": i.company_name,
            }
            for i in report.article_issues
        ],
        "fix_command": (
            "python scripts/fix_companies.py --company-ids " +
            " ".join(sorted(all_issue_cids))
        ) if all_issue_cids else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate production outputs against source company list."
    )
    parser.add_argument(
        "--company-csv",
        type=Path,
        default=Path("data/aihirebox_company_list.csv"),
        help="Source company CSV (default: data/aihirebox_company_list.csv)"
    )
    parser.add_argument(
        "--production-dir",
        type=Path,
        default=Path("outputs/production"),
        help="Production output directory (default: outputs/production)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON"
    )

    args = parser.parse_args()

    # Paths
    tags_path = args.production_dir / "company_tagging" / "company_tags.json"
    index_path = args.production_dir / "article_generator" / "articles" / "index.json"

    # Load data
    source_companies = load_source_companies(args.company_csv)
    tags = load_company_tags(tags_path)
    articles = load_article_index(index_path)

    if not source_companies:
        print("Error: No companies found in source CSV.")
        return 1

    # Validate
    report = validate(source_companies, tags, articles)

    # Output
    if args.json:
        print(json.dumps(report_to_json(report), ensure_ascii=False, indent=2))
    else:
        print_report(report)

    # Return exit code based on issues
    if report.tagging_issues or report.article_issues:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
