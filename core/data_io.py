"""
Shared data I/O utilities for the AIHireBox recommendation system.

This module contains common functions for loading and saving data,
extracted to reduce duplication across modules.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .models import CompanyRecord, CompanyProfile


def load_companies_from_csv(csv_path: Path) -> List[CompanyRecord]:
    """Load company records from CSV file.

    Args:
        csv_path: Path to CSV file with columns:
                  company_id, company_name, location, company_details

    Returns:
        List of CompanyRecord objects

    Example:
        companies = load_companies_from_csv(Path("data/companies.csv"))
        for c in companies:
            print(f"{c.company_name}: {c.location}")
    """
    companies = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            companies.append(CompanyRecord(
                company_id=row.get("company_id", ""),
                company_name=row.get("company_name", ""),
                location=row.get("location", ""),
                company_details=row.get("company_details", ""),
            ))
    return companies


def load_embeddings_npy(npy_path: Path) -> Tuple[np.ndarray, Dict[str, int]]:
    """Load embeddings from NumPy file with mapping.

    Expects a .mapping.json file alongside the .npy file.

    Args:
        npy_path: Path to .npy file containing embeddings array

    Returns:
        Tuple of:
        - numpy array of shape (n_companies, embedding_dim)
        - dict mapping company_id to array index

    Example:
        embeddings, mapping = load_embeddings_npy(Path("embeddings.npy"))
        idx = mapping["cid_100"]
        company_embedding = embeddings[idx]
    """
    embeddings = np.load(npy_path)
    mapping_path = npy_path.with_suffix(".mapping.json")

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    return embeddings, mapping


def load_company_profiles(json_path: Path) -> List[CompanyProfile]:
    """Load company profiles from JSON file (output of company_tagging.py).

    Args:
        json_path: Path to company_tags.json

    Returns:
        List of CompanyProfile objects with tag data

    Example:
        profiles = load_company_profiles(Path("company_tags.json"))
        for p in profiles:
            print(f"{p.company_name}: {p.industry}")
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    profiles = []
    for item in data:
        profiles.append(CompanyProfile(
            company_id=item.get("company_id", ""),
            company_name=item.get("company_name", ""),
            industry=item.get("industry", []),
            business_model=item.get("business_model", []),
            target_market=item.get("target_market", []),
            company_stage=item.get("company_stage", "unknown"),
            tech_focus=item.get("tech_focus", []),
            team_background=item.get("team_background", []),
            confidence_score=item.get("confidence_score", 0.0),
        ))

    return profiles


def load_companies_as_dicts(csv_path: Path) -> List[Dict[str, str]]:
    """Load companies from CSV and convert to dict format.

    This is a convenience function for code that expects dict format
    rather than CompanyRecord objects.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of dicts with keys: company_id, company_name, location, company_details
    """
    records = load_companies_from_csv(csv_path)
    return [
        {
            "company_id": r.company_id,
            "company_name": r.company_name,
            "location": r.location,
            "company_details": r.company_details,
        }
        for r in records
    ]
