"""
Company Recommender Module - Multi-dimensional company recommendations with head suppression.

This module provides multi-dimensional company recommendations based on:
1. Tag-based similarity (industry, business_model, target_market, etc.)
2. Embedding-based semantic similarity
3. Head suppression to prevent big companies from dominating recommendations

Design Philosophy:
- Each recommendation group explains WHY companies are similar (dimension labeling)
- Head suppression prevents popular/big companies from being over-recommended
- Controllable and interpretable recommendations
"""

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ============================================================================
# Configuration Constants (imported from core for deduplication)
# ============================================================================

# Re-export from core for backward compatibility
from core import (
    TAG_DIMENSIONS,
    HEAD_COMPANY_STAGES,
    DEFAULT_HEAD_PENALTY,
    DEFAULT_MIN_COMPANIES_PER_DIM,
    DEFAULT_MAX_COMPANIES_PER_DIM,
    DEFAULT_NUM_DIMENSIONS,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_MAX_BELOW_THRESHOLD,
    DIMENSION_LABELS,
    CompanyProfile,
    load_company_profiles as _load_profiles,
    load_embeddings_npy as _load_embeddings,
)

# ============================================================================
# Data Classes (CompanyProfile imported from core, module-specific classes below)
# ============================================================================

@dataclass
class RecommendedCompany:
    """A single recommended company with score."""
    company_id: str
    company_name: str
    similarity_score: float
    raw_score: float = 0.0  # Tag-based score before head suppression
    head_penalty_applied: bool = False
    embedding_score: Optional[float] = None  # Embedding similarity (if used)


@dataclass
class RecommendationGroup:
    """A group of recommendations for one dimension."""
    dimension_key: str
    dimension_label_zh: str
    dimension_label_en: str
    reason: str
    shared_tags: List[str]  # The specific tags that match
    companies: List[RecommendedCompany] = field(default_factory=list)


@dataclass 
class CompanyRecommendations:
    """Complete recommendations for a query company."""
    query_company_id: str
    query_company_name: str
    recommendation_groups: List[RecommendationGroup] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Head Suppression Strategies
# ============================================================================

class HeadSuppressionStrategy:
    """Base class for head suppression strategies."""
    
    def compute_penalty(self, company: CompanyProfile, global_stats: Dict) -> float:
        """Compute penalty factor (0-1). Higher = more penalty."""
        raise NotImplementedError


class CompanyStageHeadSuppression(HeadSuppressionStrategy):
    """Suppress based on company stage (public, bigtech_subsidiary, etc.)."""
    
    def __init__(self, penalty: float = DEFAULT_HEAD_PENALTY):
        self.penalty = penalty
    
    def compute_penalty(self, company: CompanyProfile, global_stats: Dict) -> float:
        if company.is_head_company():
            return self.penalty
        return 0.0


class FrequencyHeadSuppression(HeadSuppressionStrategy):
    """Suppress based on how frequently a company appears in recommendations."""
    
    def __init__(self, max_penalty: float = 0.5):
        self.max_penalty = max_penalty
    
    def compute_penalty(self, company: CompanyProfile, global_stats: Dict) -> float:
        freq = global_stats.get("company_frequencies", {}).get(company.company_id, 0)
        total = global_stats.get("total_recommendations", 1)
        # Normalize frequency to 0-1 range
        normalized_freq = min(freq / max(total * 0.1, 1), 1.0)
        return normalized_freq * self.max_penalty


class IDFHeadSuppression(HeadSuppressionStrategy):
    """IDF-like suppression: companies appearing in many tag combinations get penalized."""
    
    def __init__(self, max_penalty: float = 0.4):
        self.max_penalty = max_penalty
    
    def compute_penalty(self, company: CompanyProfile, global_stats: Dict) -> float:
        # Count how many tags this company has (more tags = more "common")
        tag_count = (
            len(company.industry) + 
            len(company.business_model) + 
            len(company.target_market) +
            len(company.tech_focus) +
            len(company.team_background)
        )
        # Companies with more tags are more "generic" - penalize slightly
        max_tags = global_stats.get("max_tags_per_company", 20)
        normalized = min(tag_count / max_tags, 1.0)
        return normalized * self.max_penalty


class CompositeHeadSuppression(HeadSuppressionStrategy):
    """Combine multiple suppression strategies."""
    
    def __init__(self, strategies: List[Tuple[HeadSuppressionStrategy, float]]):
        """
        Args:
            strategies: List of (strategy, weight) tuples
        """
        self.strategies = strategies
    
    def compute_penalty(self, company: CompanyProfile, global_stats: Dict) -> float:
        total_penalty = 0.0
        total_weight = sum(w for _, w in self.strategies)
        
        for strategy, weight in self.strategies:
            penalty = strategy.compute_penalty(company, global_stats)
            total_penalty += penalty * (weight / total_weight)
        
        return min(total_penalty, 0.9)  # Cap at 90% penalty


# ============================================================================
# Company Recommender Class
# ============================================================================

class CompanyRecommender:
    """Multi-dimensional company recommender with head suppression.
    
    Features:
    - Tag-based similarity across multiple dimensions
    - Embedding-based semantic similarity (optional)
    - Configurable head suppression to prevent big companies from dominating
    - Dimension labeling for interpretability
    
    Example:
        recommender = CompanyRecommender(
            companies=companies,
            embeddings=embeddings,
            embedding_mapping=mapping,
        )
        recs = recommender.recommend("cid_100", num_dimensions=5)
    """
    
    def __init__(
        self,
        companies: List[CompanyProfile],
        embeddings: Optional[np.ndarray] = None,
        embedding_mapping: Optional[Dict[str, int]] = None,
        head_suppression: Optional[HeadSuppressionStrategy] = None,
        head_penalty: float = DEFAULT_HEAD_PENALTY,
    ):
        """Initialize the recommender.
        
        Args:
            companies: List of CompanyProfile objects
            embeddings: Optional numpy array of company embeddings
            embedding_mapping: Optional dict mapping company_id to embedding index
            head_suppression: Optional custom head suppression strategy
            head_penalty: Default penalty for head companies (if no custom strategy)
        """
        self.companies = {c.company_id: c for c in companies}
        self.company_list = companies
        self.embeddings = embeddings
        self.embedding_mapping = embedding_mapping or {}
        
        # Build inverted index for tag-based lookup
        self._build_inverted_index()
        
        # Setup head suppression strategy
        if head_suppression is None:
            self.head_suppression = CompositeHeadSuppression([
                (CompanyStageHeadSuppression(head_penalty), 0.6),
                (IDFHeadSuppression(0.3), 0.4),
            ])
        else:
            self.head_suppression = head_suppression
        
        # Compute global stats for head suppression
        self._compute_global_stats()
    
    def _build_inverted_index(self) -> None:
        """Build inverted index for efficient tag-based lookup."""
        self.tag_index: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        
        for company in self.company_list:
            # Index multi-value dimensions
            for tag in company.industry:
                self.tag_index["industry"][tag].add(company.company_id)
            for tag in company.business_model:
                self.tag_index["business_model"][tag].add(company.company_id)
            for tag in company.target_market:
                self.tag_index["target_market"][tag].add(company.company_id)
            for tag in company.tech_focus:
                self.tag_index["tech_focus"][tag].add(company.company_id)
            for tag in company.team_background:
                self.tag_index["team_background"][tag].add(company.company_id)
            
            # Index single-value dimensions
            self.tag_index["company_stage"][company.company_stage].add(company.company_id)
    
    def _compute_global_stats(self) -> None:
        """Compute global statistics for head suppression."""
        self.global_stats = {
            "total_companies": len(self.companies),
            "company_frequencies": {},  # Will be updated during recommendations
            "total_recommendations": 0,
            "max_tags_per_company": max(
                len(c.industry) + len(c.business_model) + len(c.target_market) +
                len(c.tech_focus) + len(c.team_background)
                for c in self.company_list
            ) if self.company_list else 10,
        }
    
    def _get_company_tags(self, company: CompanyProfile, dimension: str) -> List[str]:
        """Get tags for a specific dimension."""
        if dimension == "company_stage":
            return [company.company_stage] if company.company_stage != "unknown" else []
        return getattr(company, dimension, [])
    
    def _compute_tag_similarity(
        self, 
        query_company: CompanyProfile,
        candidate: CompanyProfile,
        dimension: str,
    ) -> Tuple[float, List[str]]:
        """Compute Jaccard similarity for a tag dimension.
        
        Returns:
            Tuple of (similarity score, list of shared tags)
        """
        query_tags = set(self._get_company_tags(query_company, dimension))
        candidate_tags = set(self._get_company_tags(candidate, dimension))
        
        if not query_tags or not candidate_tags:
            return 0.0, []
        
        shared = query_tags & candidate_tags
        union = query_tags | candidate_tags
        
        similarity = len(shared) / len(union) if union else 0.0
        return similarity, list(shared)
    
    def _compute_embedding_similarity(
        self,
        query_id: str,
        candidate_id: str,
    ) -> float:
        """Compute cosine similarity between embeddings."""
        if self.embeddings is None:
            return 0.0
        
        query_idx = self.embedding_mapping.get(query_id)
        candidate_idx = self.embedding_mapping.get(candidate_id)
        
        if query_idx is None or candidate_idx is None:
            return 0.0
        
        query_vec = self.embeddings[query_idx]
        candidate_vec = self.embeddings[candidate_idx]
        
        # Cosine similarity
        dot_product = np.dot(query_vec, candidate_vec)
        norm_product = np.linalg.norm(query_vec) * np.linalg.norm(candidate_vec)
        
        if norm_product == 0:
            return 0.0
        
        return float(dot_product / norm_product)
    
    def _apply_head_suppression(
        self,
        candidate: CompanyProfile,
        raw_score: float,
    ) -> Tuple[float, bool]:
        """Apply head suppression to a similarity score.
        
        Returns:
            Tuple of (adjusted_score, was_penalty_applied)
        """
        penalty = self.head_suppression.compute_penalty(candidate, self.global_stats)
        
        if penalty > 0:
            adjusted_score = raw_score * (1 - penalty)
            return adjusted_score, True
        
        return raw_score, False
    
    def _get_dimension_label(self, dimension: str, tag: str) -> Tuple[str, str]:
        """Get human-readable label for a dimension-tag combination."""
        key = f"{dimension}_{tag}"
        
        if key in DIMENSION_LABELS:
            return DIMENSION_LABELS[key]["zh"], DIMENSION_LABELS[key]["en"]
        
        # Fallback: generate from dimension and tag
        zh = f"同为{tag}类型"
        en = f"Same {tag} type"
        return zh, en
    
    def _find_candidates_for_dimension(
        self,
        query_company: CompanyProfile,
        dimension: str,
        exclude_ids: Set[str],
        min_companies: int,
        max_companies: int,
        use_embedding_boost: bool = True,
        embedding_weight: float = 0.4,
    ) -> List[Tuple[str, float, List[str], float, bool, float]]:
        """Find candidate companies for a specific dimension.
        
        Args:
            query_company: Query company profile
            dimension: Tag dimension to search
            exclude_ids: Company IDs to exclude
            min_companies: Minimum companies to return
            max_companies: Maximum companies to return
            use_embedding_boost: Whether to boost scores with embedding similarity
            embedding_weight: Weight for embedding similarity (0-1, default 0.4)
        
        Returns:
            List of (company_id, final_score, shared_tags, tag_score, head_penalty_applied, embedding_score)
        """
        query_tags = self._get_company_tags(query_company, dimension)
        
        if not query_tags:
            return []
        
        # Find all companies that share at least one tag
        candidate_ids: Set[str] = set()
        for tag in query_tags:
            if tag in self.tag_index[dimension]:
                candidate_ids |= self.tag_index[dimension][tag]
        
        # Remove query company and already recommended companies
        candidate_ids -= {query_company.company_id}
        candidate_ids -= exclude_ids
        
        # Score candidates
        scored_candidates = []
        for cid in candidate_ids:
            candidate = self.companies.get(cid)
            if not candidate:
                continue
            
            tag_score, shared_tags = self._compute_tag_similarity(
                query_company, candidate, dimension
            )
            
            if tag_score > 0 and shared_tags:
                # Compute embedding similarity if available and requested
                embedding_score = 0.0
                if use_embedding_boost and self.embeddings is not None:
                    embedding_score = self._compute_embedding_similarity(
                        query_company.company_id, cid
                    )
                
                # Combine tag and embedding scores
                # Formula: (1 - weight) * tag_score + weight * embedding_score
                if embedding_score > 0:
                    combined_score = (1 - embedding_weight) * tag_score + embedding_weight * embedding_score
                else:
                    combined_score = tag_score
                
                # Apply head suppression
                adjusted_score, penalty_applied = self._apply_head_suppression(
                    candidate, combined_score
                )
                
                scored_candidates.append((
                    cid, adjusted_score, shared_tags, tag_score, penalty_applied, embedding_score
                ))
        
        # Sort by adjusted score
        scored_candidates.sort(key=lambda x: -x[1])
        
        return scored_candidates[:max_companies]
    
    def _find_semantic_candidates(
        self,
        query_company: CompanyProfile,
        exclude_ids: Set[str],
        max_companies: int,
    ) -> List[Tuple[str, float, float, bool]]:
        """Find semantically similar companies using embeddings.
        
        Returns:
            List of (company_id, score, raw_score, head_penalty_applied)
        """
        if self.embeddings is None:
            return []
        
        query_idx = self.embedding_mapping.get(query_company.company_id)
        if query_idx is None:
            return []
        
        query_vec = self.embeddings[query_idx]
        
        # Compute similarities with all other companies
        similarities = []
        for cid, idx in self.embedding_mapping.items():
            if cid == query_company.company_id or cid in exclude_ids:
                continue
            
            candidate = self.companies.get(cid)
            if not candidate:
                continue
            
            candidate_vec = self.embeddings[idx]
            
            # Cosine similarity
            dot_product = np.dot(query_vec, candidate_vec)
            norm_product = np.linalg.norm(query_vec) * np.linalg.norm(candidate_vec)
            
            if norm_product > 0:
                raw_score = float(dot_product / norm_product)
                adjusted_score, penalty_applied = self._apply_head_suppression(
                    candidate, raw_score
                )
                similarities.append((cid, adjusted_score, raw_score, penalty_applied))
        
        # Sort by score
        similarities.sort(key=lambda x: -x[1])
        
        return similarities[:max_companies]
    
    def _prioritize_dimensions(
        self,
        query_company: CompanyProfile,
        min_companies_per_dim: int,
    ) -> List[Tuple[str, str, int, List[str]]]:
        """Prioritize dimensions based on query company's tags.

        Args:
            query_company: Query company profile
            min_companies_per_dim: Minimum companies required per dimension

        Returns:
            List of (dimension, primary_tag, count, all_tags) sorted by priority
        """
        dimension_scores: List[Tuple[str, str, int, List[str]]] = []

        for dimension in TAG_DIMENSIONS:
            tags = self._get_company_tags(query_company, dimension)
            if tags:
                for tag in tags:
                    # Skip "unknown" and "other" tags
                    if tag in {"unknown", "other", "not_tech_focused"}:
                        continue

                    # Count how many companies share this tag
                    count = len(self.tag_index[dimension].get(tag, set())) - 1  # -1 for query
                    if count >= min_companies_per_dim:
                        dimension_scores.append((dimension, tag, count, tags))

        # Sort by count (prefer dimensions with more candidates, but not too many)
        # Sweet spot: enough candidates but not too generic
        dimension_scores.sort(key=lambda x: -min(x[2], 50))

        return dimension_scores

    def _filter_candidates_by_threshold(
        self,
        candidates: List[Tuple[str, float, List[str], float, bool, float]],
        score_threshold: float,
        max_below_threshold: int,
        max_companies_per_dim: int,
    ) -> Optional[List[Tuple[str, float, List[str], float, bool, float]]]:
        """Filter candidates by score threshold.

        Args:
            candidates: List of (company_id, score, shared_tags, tag_score, penalty_applied, emb_score)
            score_threshold: Minimum score threshold
            max_below_threshold: Max companies below threshold before rejecting dimension
            max_companies_per_dim: Maximum companies to return

        Returns:
            Filtered candidates list, or None if dimension should be skipped
        """
        filtered_candidates = []
        below_threshold_count = 0

        for cid, score, shared_tags, tag_score, penalty_applied, emb_score in candidates:
            if score >= score_threshold:
                filtered_candidates.append((cid, score, shared_tags, tag_score, penalty_applied, emb_score))
            else:
                below_threshold_count += 1

        # Skip this dimension if too many are below threshold
        if below_threshold_count > max_below_threshold:
            return None

        # Limit to max companies
        return filtered_candidates[:max_companies_per_dim]

    def _create_tag_dimension_group(
        self,
        query_company: CompanyProfile,
        dimension: str,
        primary_tag: str,
        filtered_candidates: List[Tuple[str, float, List[str], float, bool, float]],
        diversity_constraint: bool,
        used_company_ids: Set[str],
    ) -> RecommendationGroup:
        """Create a recommendation group for a tag-based dimension.

        Args:
            query_company: Query company profile
            dimension: Dimension name
            primary_tag: Primary tag for this dimension
            filtered_candidates: Filtered candidate list
            diversity_constraint: Whether to enforce diversity
            used_company_ids: Set to track used company IDs (modified in-place)

        Returns:
            RecommendationGroup object
        """
        label_zh, label_en = self._get_dimension_label(dimension, primary_tag)

        companies = []
        shared_tags_for_group = set()

        for cid, score, shared_tags, tag_score, penalty_applied, emb_score in filtered_candidates:
            company = self.companies[cid]
            companies.append(RecommendedCompany(
                company_id=cid,
                company_name=company.company_name,
                similarity_score=round(score, 3),
                raw_score=round(tag_score, 3),
                head_penalty_applied=penalty_applied,
                embedding_score=round(emb_score, 3) if emb_score > 0 else None,
            ))
            shared_tags_for_group.update(shared_tags)

            if diversity_constraint:
                used_company_ids.add(cid)

        return RecommendationGroup(
            dimension_key=f"{dimension}_{primary_tag}",
            dimension_label_zh=label_zh,
            dimension_label_en=label_en,
            reason=f"这些公司与{query_company.company_name}在{label_zh}方面相似",
            shared_tags=list(shared_tags_for_group),
            companies=companies,
        )

    def _create_semantic_dimension_group(
        self,
        query_company: CompanyProfile,
        semantic_candidates: List[Tuple[str, float, float, bool]],
        score_threshold: float,
        max_below_threshold: int,
        min_companies_per_dim: int,
        max_companies_per_dim: int,
    ) -> Optional[RecommendationGroup]:
        """Create a recommendation group for semantic similarity dimension.

        Args:
            query_company: Query company profile
            semantic_candidates: List of (company_id, score, raw_score, penalty_applied)
            score_threshold: Minimum score threshold
            max_below_threshold: Max companies below threshold
            min_companies_per_dim: Minimum companies required
            max_companies_per_dim: Maximum companies to return

        Returns:
            RecommendationGroup object, or None if dimension should be skipped
        """
        filtered_semantic = []
        below_threshold_count = 0

        for cid, score, raw_score, penalty_applied in semantic_candidates:
            if score >= score_threshold:
                filtered_semantic.append((cid, score, raw_score, penalty_applied))
            else:
                below_threshold_count += 1

        # Only add if not too many below threshold
        if below_threshold_count > max_below_threshold or len(filtered_semantic) < min_companies_per_dim:
            return None

        filtered_semantic = filtered_semantic[:max_companies_per_dim]
        companies = []
        for cid, score, raw_score, penalty_applied in filtered_semantic:
            company = self.companies[cid]
            companies.append(RecommendedCompany(
                company_id=cid,
                company_name=company.company_name,
                similarity_score=round(score, 3),
                raw_score=round(raw_score, 3),
                head_penalty_applied=penalty_applied,
                embedding_score=round(raw_score, 3),
            ))

        return RecommendationGroup(
            dimension_key="semantic_similar",
            dimension_label_zh="业务描述相似",
            dimension_label_en="Semantically Similar",
            reason=f"这些公司的业务描述与{query_company.company_name}语义相似",
            shared_tags=["semantic"],
            companies=companies,
        )

    def recommend(
        self,
        query_company_id: str,
        num_dimensions: int = DEFAULT_NUM_DIMENSIONS,
        min_companies_per_dim: int = DEFAULT_MIN_COMPANIES_PER_DIM,
        max_companies_per_dim: int = DEFAULT_MAX_COMPANIES_PER_DIM,
        include_semantic: bool = True,
        diversity_constraint: bool = True,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        max_below_threshold: int = DEFAULT_MAX_BELOW_THRESHOLD,
        use_embedding_boost: bool = True,
    ) -> CompanyRecommendations:
        """Generate multi-dimensional recommendations for a company.

        Args:
            query_company_id: ID of the query company
            num_dimensions: Number of recommendation dimensions (3-5)
            min_companies_per_dim: Minimum companies per dimension
            max_companies_per_dim: Maximum companies per dimension
            include_semantic: Include embedding-based semantic dimension
            diversity_constraint: If True, each company appears in at most 1 dimension
            score_threshold: Minimum similarity score to include a company (default: 0.5)
            max_below_threshold: Max companies below threshold before dropping dimension (default: 2)
            use_embedding_boost: Boost tag-based scores with embedding similarity (default: True)

        Returns:
            CompanyRecommendations object
        """
        query_company = self.companies.get(query_company_id)
        if not query_company:
            raise ValueError(f"Company not found: {query_company_id}")

        recommendation_groups: List[RecommendationGroup] = []
        used_company_ids: Set[str] = set()  # For diversity constraint

        # Prioritize dimensions based on query company's tags
        dimension_scores = self._prioritize_dimensions(query_company, min_companies_per_dim)

        # Generate recommendations for top dimensions
        seen_dimension_tags: Set[Tuple[str, str]] = set()

        for dimension, primary_tag, count, all_tags in dimension_scores:
            if len(recommendation_groups) >= num_dimensions:
                break

            # Skip if we already have this dimension-tag combo
            if (dimension, primary_tag) in seen_dimension_tags:
                continue
            seen_dimension_tags.add((dimension, primary_tag))

            # Find candidates with embedding boost
            exclude = used_company_ids if diversity_constraint else set()
            candidates = self._find_candidates_for_dimension(
                query_company, dimension, exclude,
                min_companies_per_dim, max_companies_per_dim * 2,  # Get more candidates for filtering
                use_embedding_boost=use_embedding_boost,
            )

            if len(candidates) < min_companies_per_dim:
                continue

            # Apply score threshold filtering
            filtered_candidates = self._filter_candidates_by_threshold(
                candidates, score_threshold, max_below_threshold, max_companies_per_dim
            )

            if filtered_candidates is None or len(filtered_candidates) < min_companies_per_dim:
                continue

            # Build recommendation group
            group = self._create_tag_dimension_group(
                query_company, dimension, primary_tag, filtered_candidates,
                diversity_constraint, used_company_ids
            )
            recommendation_groups.append(group)

        # Add semantic dimension if requested and we have embeddings
        if include_semantic and self.embeddings is not None and len(recommendation_groups) < num_dimensions:
            exclude = used_company_ids if diversity_constraint else set()
            semantic_candidates = self._find_semantic_candidates(
                query_company, exclude, max_companies_per_dim * 2
            )

            semantic_group = self._create_semantic_dimension_group(
                query_company, semantic_candidates, score_threshold,
                max_below_threshold, min_companies_per_dim, max_companies_per_dim
            )

            if semantic_group is not None:
                recommendation_groups.append(semantic_group)

        # Build result
        return CompanyRecommendations(
            query_company_id=query_company_id,
            query_company_name=query_company.company_name,
            recommendation_groups=recommendation_groups,
            metadata={
                "num_dimensions": len(recommendation_groups),
                "head_suppression_applied": True,
                "diversity_constraint": diversity_constraint,
                "include_semantic": include_semantic,
                "score_threshold": score_threshold,
                "max_below_threshold": max_below_threshold,
                "use_embedding_boost": use_embedding_boost,
            },
        )
    
    def batch_recommend(
        self,
        company_ids: List[str],
        **kwargs,
    ) -> List[CompanyRecommendations]:
        """Generate recommendations for multiple companies."""
        results = []
        for cid in company_ids:
            try:
                rec = self.recommend(cid, **kwargs)
                results.append(rec)
            except ValueError as e:
                print(f"Warning: {e}")
                continue
        return results


# ============================================================================
# Utility Functions
# ============================================================================

def load_company_profiles(json_path: Path) -> List[CompanyProfile]:
    """Load company profiles from JSON file (output of company_tagging.py).

    Note: This function delegates to core.load_company_profiles for consistency.
    Kept here for backward compatibility.
    """
    return _load_profiles(json_path)


def load_embeddings(npy_path: Path) -> Tuple[np.ndarray, Dict[str, int]]:
    """Load embeddings from numpy file with mapping.

    Note: This function delegates to core.load_embeddings_npy for consistency.
    Kept here for backward compatibility.
    """
    return _load_embeddings(npy_path)


def recommendations_to_dict(recs: CompanyRecommendations) -> Dict[str, Any]:
    """Convert CompanyRecommendations to dictionary."""
    return {
        "query_company_id": recs.query_company_id,
        "query_company_name": recs.query_company_name,
        "recommendation_groups": [
            {
                "dimension_key": g.dimension_key,
                "dimension_label_zh": g.dimension_label_zh,
                "dimension_label_en": g.dimension_label_en,
                "reason": g.reason,
                "shared_tags": g.shared_tags,
                "companies": [
                    {
                        "company_id": c.company_id,
                        "company_name": c.company_name,
                        "similarity_score": c.similarity_score,
                        "raw_score": c.raw_score,
                        "head_penalty_applied": c.head_penalty_applied,
                        "embedding_score": c.embedding_score,
                    }
                    for c in g.companies
                ],
            }
            for g in recs.recommendation_groups
        ],
        "metadata": recs.metadata,
    }


def save_recommendations_json(
    recommendations: List[CompanyRecommendations],
    output_path: Path,
) -> None:
    """Save recommendations to JSON file."""
    data = [recommendations_to_dict(r) for r in recommendations]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_recommendations(recs: CompanyRecommendations) -> None:
    """Print recommendations in a human-readable format."""
    print("\n" + "=" * 70)
    print(f"推荐结果: {recs.query_company_name} ({recs.query_company_id})")
    print("=" * 70)
    
    for i, group in enumerate(recs.recommendation_groups, 1):
        print(f"\n【维度 {i}】{group.dimension_label_zh}")
        print(f"  原因: {group.reason}")
        print(f"  共同标签: {', '.join(group.shared_tags)}")
        print("  推荐公司:")
        
        for c in group.companies:
            penalty_marker = "⬇" if c.head_penalty_applied else ""
            emb_info = f", emb={c.embedding_score:.2f}" if c.embedding_score else ""
            print(f"    - {c.company_name} (相似度: {c.similarity_score:.2f}{penalty_marker}{emb_info})")
    
    print("\n" + "-" * 70)
    print(f"总维度数: {recs.metadata.get('num_dimensions', 0)}")
    print(f"头部抑制: {'已启用' if recs.metadata.get('head_suppression_applied') else '未启用'}")
    threshold = recs.metadata.get('score_threshold', 0.5)
    print(f"分数阈值: {threshold}")
    print(f"Embedding加成: {'已启用' if recs.metadata.get('use_embedding_boost') else '未启用'}")
