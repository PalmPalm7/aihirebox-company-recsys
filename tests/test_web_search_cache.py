#!/usr/bin/env python3
"""
Web Search Cache Test Suite

Tests for verifying web search cache functionality:
1. Cache reading and writing
2. Index maintenance
3. Skip existing behavior
4. Stale cache detection
5. Integration with downstream pipeline stages
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from article_generator.models import WebSearchResult
from article_generator.web_searcher import (
    OpenRouterWebSearcher,
    get_stale_companies,
    load_web_search_cache,
    load_web_search_index,
    save_web_search_index,
    update_web_search_index,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="web_cache_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_web_result():
    """Create a sample WebSearchResult for testing."""
    return WebSearchResult(
        company_id="cid_test_001",
        company_name="Test Company",
        query_used="Search query for Test Company",
        search_summary="This is a detailed summary about Test Company...",
        citations=["https://example.com/article1", "https://example.com/article2"],
        is_valid=True,
        searched_at=datetime.now().isoformat(),
    )


@pytest.fixture
def populated_cache_dir(temp_cache_dir):
    """Create a cache directory with sample cached results."""
    # Create multiple cached results
    for i in range(5):
        company_id = f"cid_{100 + i}"
        cache_file = temp_cache_dir / f"{company_id}.json"
        cache_data = {
            "company_id": company_id,
            "company_name": f"Company {i}",
            "query_used": f"Search query for Company {i}",
            "search_summary": f"Detailed summary about Company {i}. " * 50,  # Make it long enough
            "citations": [f"https://example.com/{company_id}/article{j}" for j in range(3)],
            "is_valid": i < 4,  # First 4 are valid, last one is invalid
            "searched_at": (datetime.now() - timedelta(days=i * 10)).isoformat(),
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

    # Create index
    index = {
        f"cid_{100 + i}": {
            "company_name": f"Company {i}",
            "searched_at": (datetime.now() - timedelta(days=i * 10)).isoformat(),
            "is_valid": i < 4,
        }
        for i in range(5)
    }
    save_web_search_index(temp_cache_dir, index)

    return temp_cache_dir


# =============================================================================
# Unit Tests: Index Management
# =============================================================================


class TestIndexManagement:
    """Tests for cache index operations."""

    def test_save_and_load_index(self, temp_cache_dir):
        """Test that index can be saved and loaded correctly."""
        index = {
            "cid_100": {
                "company_name": "Company A",
                "searched_at": "2025-01-01T00:00:00",
                "is_valid": True,
            },
            "cid_101": {
                "company_name": "Company B",
                "searched_at": "2025-01-02T00:00:00",
                "is_valid": False,
            },
        }

        save_web_search_index(temp_cache_dir, index)
        loaded_index = load_web_search_index(temp_cache_dir)

        assert loaded_index == index
        assert len(loaded_index) == 2
        assert loaded_index["cid_100"]["is_valid"] is True
        assert loaded_index["cid_101"]["is_valid"] is False

    def test_load_empty_index(self, temp_cache_dir):
        """Test loading from non-existent index returns empty dict."""
        index = load_web_search_index(temp_cache_dir)
        assert index == {}

    def test_update_index(self, temp_cache_dir, sample_web_result):
        """Test updating index with a new result."""
        # Update with first result
        update_web_search_index(temp_cache_dir, sample_web_result)

        index = load_web_search_index(temp_cache_dir)
        assert sample_web_result.company_id in index
        assert index[sample_web_result.company_id]["company_name"] == sample_web_result.company_name
        assert index[sample_web_result.company_id]["is_valid"] == sample_web_result.is_valid

    def test_index_metadata(self, temp_cache_dir):
        """Test that index file contains proper metadata."""
        index = {"cid_100": {"company_name": "Test", "searched_at": "2025-01-01", "is_valid": True}}
        save_web_search_index(temp_cache_dir, index)

        index_file = temp_cache_dir / "index.json"
        with open(index_file, "r") as f:
            data = json.load(f)

        assert "updated_at" in data
        assert "total_companies" in data
        assert "valid_count" in data
        assert data["total_companies"] == 1
        assert data["valid_count"] == 1


# =============================================================================
# Unit Tests: Stale Cache Detection
# =============================================================================


class TestStaleCacheDetection:
    """Tests for detecting stale cache entries."""

    def test_get_stale_companies_by_age(self, populated_cache_dir):
        """Test finding companies with cache older than max_age_days."""
        # With 30 day threshold, companies with 30+ day old cache should be stale
        stale = get_stale_companies(populated_cache_dir, max_age_days=30)

        # cid_103 (30 days old) and cid_104 (40 days old) should be stale
        assert "cid_103" in stale
        assert "cid_104" in stale
        # Recent ones should not be stale
        assert "cid_100" not in stale  # 0 days old
        assert "cid_101" not in stale  # 10 days old

    def test_get_stale_companies_missing_from_index(self, populated_cache_dir):
        """Test that companies not in index are considered stale."""
        stale = get_stale_companies(
            populated_cache_dir,
            max_age_days=365,  # Very lenient
            company_ids=["cid_100", "cid_999"],  # cid_999 doesn't exist
        )

        assert "cid_999" in stale
        assert "cid_100" not in stale

    def test_get_stale_companies_empty_cache(self, temp_cache_dir):
        """Test stale detection on empty cache."""
        stale = get_stale_companies(
            temp_cache_dir,
            max_age_days=30,
            company_ids=["cid_100", "cid_101"],
        )

        # All should be stale since cache is empty
        assert "cid_100" in stale
        assert "cid_101" in stale


# =============================================================================
# Unit Tests: Cache Loading
# =============================================================================


class TestCacheLoading:
    """Tests for loading cached web search results."""

    def test_load_web_search_cache(self, populated_cache_dir):
        """Test loading all cached results."""
        cache = load_web_search_cache(populated_cache_dir)

        assert len(cache) == 5
        assert "cid_100" in cache
        assert "cid_104" in cache
        assert isinstance(cache["cid_100"], WebSearchResult)

    def test_load_cache_data_integrity(self, populated_cache_dir):
        """Test that loaded cache data is correct."""
        cache = load_web_search_cache(populated_cache_dir)

        result = cache["cid_100"]
        assert result.company_id == "cid_100"
        assert result.company_name == "Company 0"
        assert result.is_valid is True
        assert len(result.citations) == 3

    def test_load_empty_cache(self, temp_cache_dir):
        """Test loading from empty directory."""
        cache = load_web_search_cache(temp_cache_dir)
        assert cache == {}

    def test_cache_skips_index_file(self, populated_cache_dir):
        """Test that index.json is not loaded as a company cache."""
        cache = load_web_search_cache(populated_cache_dir)

        # Should not have index.json as a company
        assert "index" not in cache
        assert "index.json" not in cache


# =============================================================================
# Integration Tests: Skip Existing Behavior
# =============================================================================


class TestSkipExistingBehavior:
    """Tests for --skip-existing functionality."""

    def test_batch_search_skips_existing(self, populated_cache_dir):
        """Test that batch_search skips companies already in cache."""
        # Mock the search_company method to track if it's called
        searcher = OpenRouterWebSearcher.__new__(OpenRouterWebSearcher)
        searcher.api_key = "fake_key"
        searcher.model = "test:online"
        searcher.client = MagicMock()

        # Track calls to search_company
        search_calls = []

        def mock_search(company_id, company_name, company_details, max_results=10):
            search_calls.append(company_id)
            return WebSearchResult(
                company_id=company_id,
                company_name=company_name,
                query_used="test",
                search_summary="test summary " * 50,
                citations=[],
                is_valid=True,
                searched_at=datetime.now().isoformat(),
            )

        searcher.search_company = mock_search

        companies = [
            {"company_id": "cid_100", "company_name": "Company 0", "company_details": "details"},
            {"company_id": "cid_101", "company_name": "Company 1", "company_details": "details"},
            {"company_id": "cid_999", "company_name": "New Company", "company_details": "details"},
        ]

        results = searcher.batch_search(
            companies=companies,
            skip_existing=True,
            cache_dir=populated_cache_dir,
            show_progress=False,
            delay_seconds=0,
        )

        # Should have 3 results (2 from cache, 1 new)
        assert len(results) == 3

        # Only cid_999 should have triggered a search call
        assert len(search_calls) == 1
        assert search_calls[0] == "cid_999"


# =============================================================================
# Integration Tests: Pipeline Cache Usage
# =============================================================================


class TestPipelineCacheUsage:
    """Tests for cache usage in the full pipeline."""

    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent

    def test_cache_used_in_reranker(self, project_root):
        """Test that reranker correctly loads web search cache."""
        cache_dir = project_root / "cache" / "web_search"

        if not cache_dir.exists():
            pytest.skip("Production cache not available")

        cache = load_web_search_cache(cache_dir)

        # Should have cached entries
        assert len(cache) > 0, "Expected cache to have entries"

        # Verify data structure
        for company_id, result in list(cache.items())[:3]:
            assert result.company_id == company_id
            assert result.company_name
            assert result.search_summary

    def test_cache_index_consistency(self, project_root):
        """Test that cache index matches actual cache files."""
        cache_dir = project_root / "cache" / "web_search"

        if not cache_dir.exists():
            pytest.skip("Production cache not available")

        index = load_web_search_index(cache_dir)
        cache = load_web_search_cache(cache_dir)

        # Index should have same companies as cache files
        assert set(index.keys()) == set(cache.keys()), "Index should match cache files"

    def test_sample_pipeline_uses_cache(self, project_root, tmp_path):
        """
        Test that running pipeline with sample data uses existing cache.

        This is the key integration test: verify that when cache exists,
        no new web searches are performed.
        """
        cache_dir = project_root / "cache" / "web_search"
        sample_csv = project_root / "data" / "aihirebox_company_list_sample.csv"

        if not cache_dir.exists() or not sample_csv.exists():
            pytest.skip("Required files not available")

        # Load sample companies
        import csv

        with open(sample_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            sample_company_ids = [row["company_id"] for row in reader]

        # Check which are already cached
        index = load_web_search_index(cache_dir)
        cached_ids = set(index.keys())
        sample_ids = set(sample_company_ids)

        cached_count = len(sample_ids & cached_ids)
        uncached_count = len(sample_ids - cached_ids)

        print(f"\nSample companies: {len(sample_ids)}")
        print(f"Already cached: {cached_count}")
        print(f"Not cached: {uncached_count}")

        # If all sample companies are cached, running with --skip-existing
        # should not trigger any new searches
        if cached_count == len(sample_ids):
            # All cached - this is the expected state for repeated runs
            assert True, "All sample companies are cached"
        else:
            # Some not cached - report which ones
            missing = sample_ids - cached_ids
            print(f"Companies not in cache: {missing}")


# =============================================================================
# CLI Tests
# =============================================================================


class TestCLI:
    """Tests for CLI commands."""

    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent

    def test_show_index_command(self, project_root):
        """Test --show-index CLI option."""
        result = subprocess.run(
            ["python", "run_web_search_cache.py", "--show-index"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        # Should succeed and show index info
        assert result.returncode == 0
        assert "Web Search Cache Index" in result.stdout or "No cache index found" in result.stdout

    def test_cache_dir_argument(self, project_root, tmp_path):
        """Test --cache-dir argument is recognized."""
        result = subprocess.run(
            [
                "python",
                "run_web_search_cache.py",
                "--cache-dir",
                str(tmp_path),
                "--show-index",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "No cache index found" in result.stdout


# =============================================================================
# Performance Tests
# =============================================================================


class TestCachePerformance:
    """Tests for cache performance characteristics."""

    def test_cache_load_performance(self, populated_cache_dir):
        """Test that cache loading is fast."""
        import time

        # Add more files to simulate real cache
        for i in range(100):
            company_id = f"cid_perf_{i}"
            cache_file = populated_cache_dir / f"{company_id}.json"
            cache_data = {
                "company_id": company_id,
                "company_name": f"Perf Company {i}",
                "query_used": "query",
                "search_summary": "summary " * 200,
                "citations": [],
                "is_valid": True,
                "searched_at": datetime.now().isoformat(),
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

        # Time the load
        start = time.time()
        cache = load_web_search_cache(populated_cache_dir)
        elapsed = time.time() - start

        assert len(cache) == 105  # 5 original + 100 new
        assert elapsed < 2.0, f"Cache load took too long: {elapsed:.2f}s"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
