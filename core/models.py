"""
Shared data models for the AIHireBox recommendation system.

This module contains the core data classes used throughout the pipeline,
extracted to reduce duplication across modules.
"""

from dataclasses import dataclass, field
from typing import List

from .constants import HEAD_COMPANY_STAGES


@dataclass
class CompanyRecord:
    """Raw company record from CSV.

    This represents the basic company information as loaded from the input CSV file.
    Used by both company_tagging and company_embedding modules.
    """
    company_id: str
    company_name: str
    location: str
    company_details: str


@dataclass
class CompanyProfile:
    """Company profile with tags and metadata.

    This represents a company after tag extraction by the LLM.
    Contains the 6 MECE tag dimensions plus confidence score.
    """
    company_id: str
    company_name: str
    industry: List[str] = field(default_factory=list)
    business_model: List[str] = field(default_factory=list)
    target_market: List[str] = field(default_factory=list)
    company_stage: str = "unknown"
    tech_focus: List[str] = field(default_factory=list)
    team_background: List[str] = field(default_factory=list)
    confidence_score: float = 0.0

    def is_head_company(self) -> bool:
        """Check if this is a 'head' (big/popular) company.

        Head companies are those in stages like 'public', 'bigtech_subsidiary', etc.
        They receive a penalty in recommendations to prevent domination.
        """
        return self.company_stage in HEAD_COMPANY_STAGES
