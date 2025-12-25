"""Data loader for reading and indexing articles from the filesystem."""

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .schemas import (
    ArticleResponse,
    ArticleSummary,
    CompanyInfo,
    PipelineMetadata,
)


class ArticleIndex:
    """In-memory index for fast article lookups."""

    def __init__(self, articles_dir: str | Path):
        """Initialize the article index.

        Args:
            articles_dir: Path to the articles directory
        """
        self.articles_dir = Path(articles_dir)
        self._articles: dict[tuple[str, str, str], ArticleResponse] = {}
        self._by_company: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._company_names: dict[str, str] = {}
        self._metadata: Optional[PipelineMetadata] = None
        self._loaded = False

    def load(self) -> None:
        """Load all articles from the directory into memory."""
        self._articles.clear()
        self._by_company.clear()
        self._company_names.clear()

        if not self.articles_dir.exists():
            raise FileNotFoundError(f"Articles directory not found: {self.articles_dir}")

        # Load metadata if exists
        metadata_path = self.articles_dir / "run_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._metadata = PipelineMetadata(**data)

        # Pattern: {company_id}_{rule_id}_{style}.json
        pattern = re.compile(r"^(cid_\d+)_(R\d+_[a-z_]+)_([a-z]+)\.json$")

        for file_path in self.articles_dir.glob("*.json"):
            if file_path.name == "run_metadata.json":
                continue

            match = pattern.match(file_path.name)
            if not match:
                continue

            company_id, rule_id, style = match.groups()

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    article = ArticleResponse(**data)

                    key = (company_id, rule_id, style)
                    self._articles[key] = article
                    self._by_company[company_id].append((rule_id, style))
                    self._company_names[company_id] = article.query_company_name

            except (json.JSONDecodeError, ValueError) as e:
                # Log error but continue loading other articles
                print(f"Error loading {file_path}: {e}")
                continue

        self._loaded = True

    def reload(self) -> None:
        """Reload all articles from disk."""
        self.load()

    @property
    def is_loaded(self) -> bool:
        """Check if articles have been loaded."""
        return self._loaded

    @property
    def total_articles(self) -> int:
        """Get total number of loaded articles."""
        return len(self._articles)

    @property
    def total_companies(self) -> int:
        """Get total number of unique companies."""
        return len(self._by_company)

    @property
    def metadata(self) -> Optional[PipelineMetadata]:
        """Get pipeline run metadata."""
        return self._metadata

    def get_article(
        self, company_id: str, rule_id: str, style: str
    ) -> Optional[ArticleResponse]:
        """Get a specific article.

        Args:
            company_id: Company ID (e.g., cid_100)
            rule_id: Rule ID (e.g., R1_industry)
            style: Article style (e.g., 36kr)

        Returns:
            ArticleResponse or None if not found
        """
        return self._articles.get((company_id, rule_id, style))

    def get_articles_for_company(
        self, company_id: str, rule_id: Optional[str] = None, style: Optional[str] = None
    ) -> list[ArticleResponse]:
        """Get all articles for a company with optional filtering.

        Args:
            company_id: Company ID
            rule_id: Optional rule ID filter
            style: Optional style filter

        Returns:
            List of matching articles
        """
        if company_id not in self._by_company:
            return []

        results = []
        for r_id, s in self._by_company[company_id]:
            if rule_id and r_id != rule_id:
                continue
            if style and s != style:
                continue
            article = self._articles.get((company_id, r_id, s))
            if article:
                results.append(article)

        return results

    def list_articles(
        self,
        company_id: Optional[str] = None,
        rule_id: Optional[str] = None,
        style: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[ArticleSummary], int]:
        """List articles with filtering and pagination.

        Args:
            company_id: Optional company ID filter
            rule_id: Optional rule ID filter
            style: Optional style filter
            page: Page number (1-indexed)
            page_size: Items per page

        Returns:
            Tuple of (list of article summaries, total count)
        """
        # Filter articles
        filtered = []
        for key, article in self._articles.items():
            c_id, r_id, s = key
            if company_id and c_id != company_id:
                continue
            if rule_id and r_id != rule_id:
                continue
            if style and s != style:
                continue
            filtered.append(article)

        # Sort by generated_at descending
        filtered.sort(key=lambda a: a.generated_at, reverse=True)

        total = len(filtered)

        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        page_articles = filtered[start:end]

        # Convert to summaries
        summaries = [
            ArticleSummary(
                query_company_id=a.query_company_id,
                query_company_name=a.query_company_name,
                rule_id=a.rule_id,
                style=a.style,
                title=a.title,
                word_count=a.word_count,
                generated_at=a.generated_at,
            )
            for a in page_articles
        ]

        return summaries, total

    def list_companies(self) -> list[CompanyInfo]:
        """List all companies with their article information.

        Returns:
            List of CompanyInfo objects
        """
        companies = []
        for company_id, article_keys in self._by_company.items():
            rules = set()
            styles = set()
            for rule_id, style in article_keys:
                rules.add(rule_id)
                styles.add(style)

            companies.append(
                CompanyInfo(
                    company_id=company_id,
                    company_name=self._company_names.get(company_id, ""),
                    article_count=len(article_keys),
                    rules=sorted(list(rules)),
                    styles=sorted(list(styles)),
                )
            )

        # Sort by company_id
        companies.sort(key=lambda c: c.company_id)
        return companies

    def get_available_styles(self) -> list[str]:
        """Get list of all available styles."""
        styles = set()
        for _, _, style in self._articles.keys():
            styles.add(style)
        return sorted(list(styles))

    def get_available_rules(self) -> list[str]:
        """Get list of all available rule IDs."""
        rules = set()
        for _, rule_id, _ in self._articles.keys():
            rules.add(rule_id)
        return sorted(list(rules))


# Global singleton for the article index
_article_index: Optional[ArticleIndex] = None


def get_article_index() -> ArticleIndex:
    """Get the global article index singleton.

    Returns:
        ArticleIndex instance

    Raises:
        RuntimeError: If index hasn't been initialized
    """
    if _article_index is None:
        raise RuntimeError("Article index not initialized. Call init_article_index first.")
    return _article_index


def init_article_index(articles_dir: str | Path) -> ArticleIndex:
    """Initialize the global article index.

    Args:
        articles_dir: Path to the articles directory

    Returns:
        Initialized ArticleIndex
    """
    global _article_index
    _article_index = ArticleIndex(articles_dir)
    _article_index.load()
    return _article_index


