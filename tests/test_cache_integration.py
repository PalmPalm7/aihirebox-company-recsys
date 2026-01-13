#!/usr/bin/env python3
"""
Cache Integration Test Suite

This test verifies that the web search cache is correctly used when running
the full pipeline, avoiding unnecessary LLM calls.

Key test scenarios:
1. When cache exists for all sample companies, no new web searches are made
2. Cache data is correctly passed to reranker and article writer
3. Cache index accurately reflects cached entries
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from article_generator.web_searcher import (
    load_web_search_cache,
    load_web_search_index,
)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def get_sample_company_ids(csv_path: Path) -> List[str]:
    """Load company IDs from sample CSV."""
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [row["company_id"] for row in reader]


def check_cache_coverage(cache_dir: Path, company_ids: List[str]) -> Dict[str, any]:
    """Check how many companies are cached."""
    index = load_web_search_index(cache_dir)

    cached = set(index.keys())
    requested = set(company_ids)

    return {
        "total_requested": len(requested),
        "cached": len(cached & requested),
        "missing": list(requested - cached),
        "coverage_pct": len(cached & requested) / len(requested) * 100 if requested else 0,
    }


class TestCacheIntegration:
    """Integration tests for cache usage in pipeline."""

    @pytest.fixture
    def project_root(self):
        return get_project_root()

    @pytest.fixture
    def cache_dir(self, project_root):
        return project_root / "cache" / "web_search"

    @pytest.fixture
    def sample_csv(self, project_root):
        return project_root / "data" / "aihirebox_company_list_sample.csv"

    def test_cache_exists(self, cache_dir):
        """Verify cache directory exists and has entries."""
        assert cache_dir.exists(), f"Cache directory does not exist: {cache_dir}"

        cache = load_web_search_cache(cache_dir)
        assert len(cache) > 0, "Cache is empty"

        print(f"\nCache contains {len(cache)} entries")

    def test_cache_index_exists(self, cache_dir):
        """Verify cache index exists and is valid."""
        index_file = cache_dir / "index.json"
        assert index_file.exists(), "Index file does not exist"

        with open(index_file, "r") as f:
            data = json.load(f)

        assert "companies" in data
        assert "total_companies" in data
        assert "valid_count" in data

        print(f"\nIndex has {data['total_companies']} companies, {data['valid_count']} valid")

    def test_sample_companies_cached(self, cache_dir, sample_csv):
        """Verify all sample companies are in cache."""
        if not sample_csv.exists():
            pytest.skip(f"Sample CSV not found: {sample_csv}")

        company_ids = get_sample_company_ids(sample_csv)
        coverage = check_cache_coverage(cache_dir, company_ids)

        print(f"\nCache coverage for n20 sample:")
        print(f"  Requested: {coverage['total_requested']}")
        print(f"  Cached: {coverage['cached']}")
        print(f"  Coverage: {coverage['coverage_pct']:.1f}%")

        if coverage["missing"]:
            print(f"  Missing: {coverage['missing'][:5]}...")

        # For this test to be meaningful, we need high cache coverage
        assert coverage["coverage_pct"] >= 80, (
            f"Low cache coverage: {coverage['coverage_pct']:.1f}%. "
            f"Missing companies: {coverage['missing']}"
        )

    def test_cache_data_quality(self, cache_dir, sample_csv):
        """Verify cached data is valid and usable."""
        if not sample_csv.exists():
            pytest.skip(f"Sample CSV not found: {sample_csv}")

        company_ids = get_sample_company_ids(sample_csv)
        cache = load_web_search_cache(cache_dir)

        valid_count = 0
        invalid_count = 0
        missing_count = 0

        for cid in company_ids:
            if cid not in cache:
                missing_count += 1
                continue

            result = cache[cid]
            if result.is_valid:
                valid_count += 1
                # Verify data quality
                assert len(result.search_summary) > 100, f"Summary too short for {cid}"
            else:
                invalid_count += 1

        print(f"\nData quality for sample companies:")
        print(f"  Valid: {valid_count}")
        print(f"  Invalid: {invalid_count}")
        print(f"  Missing: {missing_count}")

        # Most should be valid
        total_found = valid_count + invalid_count
        if total_found > 0:
            valid_pct = valid_count / total_found * 100
            assert valid_pct >= 70, f"Low valid rate: {valid_pct:.1f}%"


class TestSkipExistingIntegration:
    """Test that --skip-existing properly avoids new API calls."""

    @pytest.fixture
    def project_root(self):
        return get_project_root()

    def test_dry_run_with_skip_existing(self, project_root):
        """
        Test running web search with --skip-existing on fully cached data.

        If cache works correctly, this should complete instantly with no new searches.
        """
        sample_csv = project_root / "data" / "aihirebox_company_list_sample.csv"
        cache_dir = project_root / "cache" / "web_search"

        if not sample_csv.exists():
            pytest.skip("Sample CSV not found")

        # First check cache coverage
        company_ids = get_sample_company_ids(sample_csv)
        coverage = check_cache_coverage(cache_dir, company_ids)

        if coverage["coverage_pct"] < 100:
            pytest.skip(
                f"Cache not complete ({coverage['coverage_pct']:.1f}%). "
                "Run full pipeline first to populate cache."
            )

        # Run with --skip-existing - should be very fast since all are cached
        start_time = time.time()

        result = subprocess.run(
            [
                "python",
                "run_web_search_cache.py",
                "--company-csv",
                str(sample_csv),
                "--cache-dir",
                str(cache_dir),
                "--skip-existing",
                "--quiet",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30,  # Should complete quickly if all cached
        )

        elapsed = time.time() - start_time

        print(f"\nSkip-existing run completed in {elapsed:.2f}s")
        print(f"stdout: {result.stdout[:500]}")

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Should complete very quickly (no API calls)
        assert elapsed < 10, f"Took too long ({elapsed:.2f}s) - may have made API calls"


class TestPipelineIntegration:
    """Test full pipeline integration with cache."""

    @pytest.fixture
    def project_root(self):
        return get_project_root()

    def test_reranker_loads_cache(self, project_root):
        """Test that reranker can load and use web search cache."""
        cache_dir = project_root / "cache" / "web_search"

        if not cache_dir.exists():
            pytest.skip("Cache directory not found")

        # Import reranker module
        from article_generator.web_searcher import load_web_search_cache

        cache = load_web_search_cache(cache_dir)

        assert len(cache) > 0, "Failed to load cache"

        # Verify structure matches what reranker expects
        for cid, result in list(cache.items())[:3]:
            assert hasattr(result, "company_id")
            assert hasattr(result, "company_name")
            assert hasattr(result, "search_summary")
            assert hasattr(result, "is_valid")

        print(f"\nReranker can load {len(cache)} cached entries")

    def test_article_writer_loads_cache(self, project_root):
        """Test that article writer can load and use web search cache."""
        cache_dir = project_root / "cache" / "web_search"

        if not cache_dir.exists():
            pytest.skip("Cache directory not found")

        from article_generator.web_searcher import load_web_search_cache

        cache = load_web_search_cache(cache_dir)

        # Article writer uses cache for context enrichment
        # Verify the search_summary is usable
        valid_entries = [r for r in cache.values() if r.is_valid]

        assert len(valid_entries) > 0, "No valid entries in cache"

        # Check summary quality
        for result in valid_entries[:5]:
            assert len(result.search_summary) > 200, (
                f"Summary too short for {result.company_id}: {len(result.search_summary)} chars"
            )

        print(f"\nArticle writer can use {len(valid_entries)} valid cached entries")


class TestCacheConsistency:
    """Test cache consistency and integrity."""

    @pytest.fixture
    def project_root(self):
        return get_project_root()

    @pytest.fixture
    def cache_dir(self, project_root):
        return project_root / "cache" / "web_search"

    def test_index_matches_files(self, cache_dir):
        """Verify index.json matches actual cache files."""
        if not cache_dir.exists():
            pytest.skip("Cache directory not found")

        index = load_web_search_index(cache_dir)
        cache = load_web_search_cache(cache_dir)

        index_ids = set(index.keys())
        file_ids = set(cache.keys())

        # They should match
        only_in_index = index_ids - file_ids
        only_in_files = file_ids - index_ids

        if only_in_index:
            print(f"\nWarning: In index but no file: {list(only_in_index)[:5]}")
        if only_in_files:
            print(f"\nWarning: File exists but not in index: {list(only_in_files)[:5]}")

        # Allow some tolerance but most should match
        match_rate = len(index_ids & file_ids) / max(len(index_ids), len(file_ids), 1) * 100
        assert match_rate >= 95, f"Low index-file match rate: {match_rate:.1f}%"

    def test_cache_timestamps_valid(self, cache_dir):
        """Verify all cache entries have valid timestamps."""
        if not cache_dir.exists():
            pytest.skip("Cache directory not found")

        index = load_web_search_index(cache_dir)

        invalid_timestamps = []
        for cid, info in index.items():
            searched_at = info.get("searched_at", "")
            if not searched_at:
                invalid_timestamps.append(cid)
                continue

            try:
                datetime.fromisoformat(searched_at)
            except ValueError:
                invalid_timestamps.append(cid)

        assert len(invalid_timestamps) == 0, (
            f"Invalid timestamps found: {invalid_timestamps[:5]}"
        )

        print(f"\nAll {len(index)} entries have valid timestamps")


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-s"])
