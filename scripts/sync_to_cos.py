#!/usr/bin/env python
"""
COS Sync Script - Sync production outputs to Tencent Cloud COS

Syncs local production outputs to COS bucket with the following mapping:
- outputs/production/company_tagging/ -> company_tagging/
- outputs/production/article_generator/articles/ -> articles/
- data/aihirebox_company_list.csv -> articles/aihirebox_company_list.csv

Usage:
    # Sync all production outputs
    python scripts/sync_to_cos.py

    # Dry run (show what would be synced)
    python scripts/sync_to_cos.py --dry-run

    # Sync only specific scopes
    python scripts/sync_to_cos.py --scope articles
    python scripts/sync_to_cos.py --scope tagging
    python scripts/sync_to_cos.py --scope all

    # Force overwrite all files
    python scripts/sync_to_cos.py --force
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv


# Directory mappings: (local_path, cos_path, is_directory)
SYNC_MAPPINGS: List[Tuple[str, str, bool]] = [
    # Company tagging outputs
    ("outputs/production/company_tagging/", "company_tagging/", True),
    # Article generator outputs
    ("outputs/production/article_generator/articles/json/", "articles/json/", True),
    ("outputs/production/article_generator/articles/markdown/", "articles/markdown/", True),
    ("outputs/production/article_generator/articles/index.json", "articles/index.json", False),
    # Static data files
    ("data/aihirebox_company_list.csv", "articles/aihirebox_company_list.csv", False),
]

# Scope definitions
SCOPES = {
    "all": None,  # All mappings
    "tagging": ["company_tagging/"],
    "articles": ["articles/"],
}


def check_coscmd_configured() -> bool:
    """Check if coscmd is configured."""
    config_path = Path.home() / ".cos.conf"
    return config_path.exists()


def configure_coscmd() -> bool:
    """Configure coscmd from environment variables."""
    bucket = os.getenv("COS_BUCKET")
    region = os.getenv("COS_REGION")
    secret_id = os.getenv("COS_SECRET_ID")
    secret_key = os.getenv("COS_SECRET_KEY")

    if not all([bucket, region, secret_id, secret_key]):
        print("Error: Missing COS environment variables.")
        print("Required: COS_BUCKET, COS_REGION, COS_SECRET_ID, COS_SECRET_KEY")
        return False

    cmd = [
        "coscmd", "config",
        "-a", secret_id,
        "-s", secret_key,
        "-b", bucket,
        "-r", region,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Configured coscmd for bucket: {bucket} in region: {region}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error configuring coscmd: {e}")
        return False
    except FileNotFoundError:
        print("Error: coscmd not found. Install with: pip install coscmd")
        return False


def upload_file(local_path: str, cos_path: str, dry_run: bool = False, force: bool = False) -> bool:
    """Upload a single file to COS."""
    if not Path(local_path).exists():
        print(f"  [SKIP] File not found: {local_path}")
        return True

    if dry_run:
        print(f"  [DRY-RUN] Would upload: {local_path} -> {cos_path}")
        return True

    cmd = ["coscmd", "upload"]
    if force:
        cmd.append("-f")
    cmd.extend([local_path, cos_path])

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  [OK] {local_path} -> {cos_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [FAILED] {local_path}: {e.stderr}")
        return False


def upload_directory(local_dir: str, cos_dir: str, dry_run: bool = False, force: bool = False) -> Tuple[int, int]:
    """Upload a directory to COS recursively."""
    local_path = Path(local_dir)
    if not local_path.exists():
        print(f"  [SKIP] Directory not found: {local_dir}")
        return 0, 0

    if dry_run:
        files = list(local_path.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        print(f"  [DRY-RUN] Would upload {file_count} files: {local_dir} -> {cos_dir}")
        return file_count, 0

    # Use coscmd upload -rs for recursive sync
    cmd = ["coscmd", "upload", "-rs"]
    if force:
        cmd.append("-f")
    cmd.extend([local_dir, cos_dir])

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Count uploaded files
        files = list(local_path.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        print(f"  [OK] {file_count} files: {local_dir} -> {cos_dir}")
        return file_count, 0
    except subprocess.CalledProcessError as e:
        print(f"  [FAILED] {local_dir}: {e.stderr}")
        return 0, 1


def sync_to_cos(
    scope: str = "all",
    dry_run: bool = False,
    force: bool = False,
) -> Tuple[int, int]:
    """Sync production outputs to COS.

    Returns:
        Tuple of (success_count, failure_count)
    """
    # Filter mappings by scope
    scope_filter = SCOPES.get(scope)
    mappings = SYNC_MAPPINGS
    if scope_filter:
        mappings = [m for m in mappings if any(m[1].startswith(s) for s in scope_filter)]

    print(f"\nSyncing to COS (scope: {scope}, dry_run: {dry_run}, force: {force})")
    print("=" * 60)

    total_success = 0
    total_failed = 0

    for local_path, cos_path, is_dir in mappings:
        if is_dir:
            success, failed = upload_directory(local_path, cos_path, dry_run, force)
            total_success += success
            total_failed += failed
        else:
            if upload_file(local_path, cos_path, dry_run, force):
                total_success += 1
            else:
                total_failed += 1

    return total_success, total_failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync production outputs to Tencent Cloud COS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--scope",
        choices=["all", "tagging", "articles"],
        default="all",
        help="Sync scope: all, tagging, or articles (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without uploading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing files",
    )
    parser.add_argument(
        "--skip-configure",
        action="store_true",
        help="Skip coscmd configuration (use existing config)",
    )

    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    # Configure coscmd if needed
    if not args.skip_configure:
        if not check_coscmd_configured():
            print("Configuring coscmd...")
            if not configure_coscmd():
                return 1
        else:
            # Re-configure to ensure environment variables are used
            if not configure_coscmd():
                return 1

    # Run sync
    success, failed = sync_to_cos(
        scope=args.scope,
        dry_run=args.dry_run,
        force=args.force,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SYNC SUMMARY")
    print("=" * 60)
    print(f"Successful: {success}")
    print(f"Failed: {failed}")

    if args.dry_run:
        print("\n[DRY-RUN] No files were actually uploaded.")
        print("Run without --dry-run to perform actual sync.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
