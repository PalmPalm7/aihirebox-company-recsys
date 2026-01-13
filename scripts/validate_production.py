#!/usr/bin/env python
"""
Production Output Validation Script

Validates the completeness and quality of production outputs:
1. Tagging: Checks for failed tags (confidence=0, errors, empty fields)
2. Articles: Checks for missing articles compared to tagged companies

Usage:
    python scripts/validate_production.py
    python scripts/validate_production.py --production-dir outputs/production
    python scripts/validate_production.py --json  # Output as JSON
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class TaggingIssue:
    """Represents a tagging issue for a company."""
    company_id: str
    company_name: str
    issue_type: str  # "error", "low_confidence", "empty_fields"
    details: str


@dataclass
class ArticleIssue:
    """Represents an article issue for a company."""
    company_id: str
    company_name: str
    issue_type: str  # "missing_all", "missing_rules", "missing_in_index"
    expected_count: int
    actual_count: int
    missing_rules: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Complete validation report."""
    tagging_issues: List[TaggingIssue] = field(default_factory=list)
    article_issues: List[ArticleIssue] = field(default_factory=list)
    total_companies: int = 0
    tagged_companies: int = 0
    companies_with_articles: int = 0


def load_company_tags(tags_path: Path) -> List[Dict]:
    """Load company tags from JSON file."""
    if not tags_path.exists():
        print(f"Warning: Tags file not found: {tags_path}")
        return []

    with open(tags_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_article_index(index_path: Path) -> Dict:
    """Load article index from JSON file."""
    if not index_path.exists():
        print(f"Warning: Article index not found: {index_path}")
        return {"companies": {}}

    with open(index_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_tagging(tags: List[Dict]) -> List[TaggingIssue]:
    """Validate company tags for issues."""
    issues = []

    for company in tags:
        cid = company.get("company_id", "unknown")
        name = company.get("company_name", "unknown")

        # Check for error in reasoning
        reasoning = company.get("raw_reasoning", "")
        if "Error" in reasoning or "error" in reasoning.lower():
            issues.append(TaggingIssue(
                company_id=cid,
                company_name=name,
                issue_type="error",
                details=reasoning[:100] + "..." if len(reasoning) > 100 else reasoning
            ))
            continue

        # Check for zero confidence
        confidence = company.get("confidence_score", 0)
        if confidence == 0:
            issues.append(TaggingIssue(
                company_id=cid,
                company_name=name,
                issue_type="low_confidence",
                details=f"confidence_score={confidence}"
            ))
            continue

        # Check for empty required fields
        empty_fields = []
        for field_name in ["industry", "tech_focus", "team_background"]:
            value = company.get(field_name, [])
            if not value or value == []:
                empty_fields.append(field_name)

        if empty_fields:
            issues.append(TaggingIssue(
                company_id=cid,
                company_name=name,
                issue_type="empty_fields",
                details=f"Empty fields: {', '.join(empty_fields)}"
            ))

    return issues


def validate_articles(
    tags: List[Dict],
    article_index: Dict,
    expected_rules: List[str] = None
) -> List[ArticleIssue]:
    """Validate articles against tagged companies."""
    if expected_rules is None:
        expected_rules = ["R1_industry", "R3_industry_market", "R4_team_background"]

    issues = []
    companies_in_index = article_index.get("companies", {})

    # Build a map of company_id -> company_name from tags
    company_names = {c["company_id"]: c["company_name"] for c in tags}

    for company in tags:
        cid = company["company_id"]
        name = company["company_name"]

        # Check if company exists in article index
        if cid not in companies_in_index:
            issues.append(ArticleIssue(
                company_id=cid,
                company_name=name,
                issue_type="missing_all",
                expected_count=len(expected_rules) * 2,  # json + md
                actual_count=0,
                missing_rules=expected_rules
            ))
            continue

        # Check which rules are present
        company_articles = companies_in_index[cid].get("articles", [])
        present_rules = set()
        for article in company_articles:
            rule_id = article.get("rule_id")
            if rule_id:
                present_rules.add(rule_id)

        missing_rules = [r for r in expected_rules if r not in present_rules]

        if missing_rules:
            issues.append(ArticleIssue(
                company_id=cid,
                company_name=name,
                issue_type="missing_rules",
                expected_count=len(expected_rules) * 2,
                actual_count=len(company_articles),
                missing_rules=missing_rules
            ))

    return issues


def print_report(report: ValidationReport, verbose: bool = False):
    """Print validation report to console."""
    print("\n" + "=" * 60)
    print("PRODUCTION VALIDATION REPORT")
    print("=" * 60)

    print(f"\nSummary:")
    print(f"  Total tagged companies: {report.tagged_companies}")
    print(f"  Companies with articles: {report.companies_with_articles}")
    print(f"  Tagging issues: {len(report.tagging_issues)}")
    print(f"  Article issues: {len(report.article_issues)}")

    if report.tagging_issues:
        print(f"\n--- Tagging Issues ({len(report.tagging_issues)}) ---")
        for issue in report.tagging_issues:
            print(f"  - {issue.company_id} ({issue.company_name}): [{issue.issue_type}] {issue.details}")

    if report.article_issues:
        print(f"\n--- Article Issues ({len(report.article_issues)}) ---")
        for issue in report.article_issues:
            if issue.issue_type == "missing_all":
                print(f"  - {issue.company_id} ({issue.company_name}): Missing all articles")
            else:
                print(f"  - {issue.company_id} ({issue.company_name}): Missing rules: {', '.join(issue.missing_rules)}")

    # Print recommended fix command if there are issues
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
    return {
        "summary": {
            "tagged_companies": report.tagged_companies,
            "companies_with_articles": report.companies_with_articles,
            "tagging_issues_count": len(report.tagging_issues),
            "article_issues_count": len(report.article_issues),
        },
        "tagging_issues": [
            {
                "company_id": i.company_id,
                "company_name": i.company_name,
                "issue_type": i.issue_type,
                "details": i.details,
            }
            for i in report.tagging_issues
        ],
        "article_issues": [
            {
                "company_id": i.company_id,
                "company_name": i.company_name,
                "issue_type": i.issue_type,
                "expected_count": i.expected_count,
                "actual_count": i.actual_count,
                "missing_rules": i.missing_rules,
            }
            for i in report.article_issues
        ],
        "fix_command": (
            f"python scripts/fix_companies.py --company-ids " +
            " ".join(sorted(set(
                [i.company_id for i in report.tagging_issues] +
                [i.company_id for i in report.article_issues]
            )))
        ) if report.tagging_issues or report.article_issues else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate production outputs for completeness and quality."
    )
    parser.add_argument(
        "--production-dir",
        type=Path,
        default=Path("outputs/production"),
        help="Path to production output directory (default: outputs/production)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output"
    )

    args = parser.parse_args()

    # Paths
    tags_path = args.production_dir / "company_tagging" / "company_tags.json"
    articles_dir = args.production_dir / "article_generator" / "articles"
    index_path = articles_dir / "index.json"

    # Load data
    tags = load_company_tags(tags_path)
    article_index = load_article_index(index_path)

    if not tags:
        print("Error: No company tags found. Run tagging first.")
        return 1

    # Validate
    tagging_issues = validate_tagging(tags)
    article_issues = validate_articles(tags, article_index)

    # Build report
    report = ValidationReport(
        tagging_issues=tagging_issues,
        article_issues=article_issues,
        tagged_companies=len(tags),
        companies_with_articles=len(article_index.get("companies", {})),
    )

    # Output
    if args.json:
        print(json.dumps(report_to_json(report), ensure_ascii=False, indent=2))
    else:
        print_report(report, verbose=args.verbose)

    # Return exit code based on issues
    if report.tagging_issues or report.article_issues:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
