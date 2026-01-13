"""
Core shared utilities for the AIHireBox recommendation system.

This package provides the shared data models, constants, and I/O utilities
used across the recommendation pipeline modules.

Usage:
    from core import CompanyRecord, CompanyProfile
    from core import load_companies_from_csv, load_embeddings_npy
    from core import HEAD_COMPANY_STAGES, DIMENSION_LABELS
"""

from .models import CompanyRecord, CompanyProfile
from .data_io import (
    load_companies_from_csv,
    load_embeddings_npy,
    load_company_profiles,
    load_companies_as_dicts,
)
from .constants import (
    TAG_DIMENSIONS,
    HEAD_COMPANY_STAGES,
    DEFAULT_HEAD_PENALTY,
    DEFAULT_MIN_COMPANIES_PER_DIM,
    DEFAULT_MAX_COMPANIES_PER_DIM,
    DEFAULT_NUM_DIMENSIONS,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_MAX_BELOW_THRESHOLD,
    DIMENSION_LABELS,
)

__all__ = [
    # Models
    "CompanyRecord",
    "CompanyProfile",
    # Data I/O
    "load_companies_from_csv",
    "load_embeddings_npy",
    "load_company_profiles",
    "load_companies_as_dicts",
    # Constants
    "TAG_DIMENSIONS",
    "HEAD_COMPANY_STAGES",
    "DEFAULT_HEAD_PENALTY",
    "DEFAULT_MIN_COMPANIES_PER_DIM",
    "DEFAULT_MAX_COMPANIES_PER_DIM",
    "DEFAULT_NUM_DIMENSIONS",
    "DEFAULT_SCORE_THRESHOLD",
    "DEFAULT_MAX_BELOW_THRESHOLD",
    "DIMENSION_LABELS",
]
