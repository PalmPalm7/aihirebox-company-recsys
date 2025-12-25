"""Pydantic models for API request/response schemas."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ArticleResponse(BaseModel):
    """Response model for a single article."""

    query_company_id: str = Field(..., description="Company ID of the query company")
    query_company_name: str = Field(..., description="Name of the query company")
    rule_id: str = Field(..., description="Recall rule ID (e.g., R1_industry)")
    style: str = Field(..., description="Article style (e.g., 36kr, xiaohongshu)")
    title: str = Field(..., description="Article title")
    content: str = Field(..., description="Article content in markdown format")
    word_count: int = Field(..., description="Word count of the article")
    candidate_company_ids: list[str] = Field(
        default_factory=list, description="IDs of companies mentioned in the article"
    )
    key_takeaways: list[str] = Field(
        default_factory=list, description="Key takeaways from the article"
    )
    citations: list[str] = Field(
        default_factory=list, description="Source citations/URLs"
    )
    generated_at: datetime = Field(..., description="When the article was generated")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query_company_id": "cid_100",
                "query_company_name": "MiniMax",
                "rule_id": "R1_industry",
                "style": "36kr",
                "title": "大模型进入交卷时刻：当技术不再是唯一的护城河",
                "content": "通用大模型的军备竞赛已经进入了下半场...",
                "word_count": 1455,
                "candidate_company_ids": ["cid_118", "cid_40", "cid_66"],
                "key_takeaways": ["大模型竞争重心从单纯的参数规模转向商业闭环..."],
                "citations": [],
                "generated_at": "2025-12-22T09:56:33.107284",
            }
        }
    }


class ArticleSummary(BaseModel):
    """Summary model for article listing (without full content)."""

    query_company_id: str
    query_company_name: str
    rule_id: str
    style: str
    title: str
    word_count: int
    generated_at: datetime


class ArticleListResponse(BaseModel):
    """Response model for paginated article list."""

    total: int = Field(..., description="Total number of articles matching the query")
    page: int = Field(..., description="Current page number (1-indexed)")
    page_size: int = Field(..., description="Number of items per page")
    articles: list[ArticleSummary] = Field(..., description="List of article summaries")


class CompanyInfo(BaseModel):
    """Basic company information."""

    company_id: str
    company_name: str
    article_count: int = Field(..., description="Number of articles for this company")
    rules: list[str] = Field(..., description="Available rule IDs for this company")
    styles: list[str] = Field(..., description="Available styles for this company")


class CompanyListResponse(BaseModel):
    """Response model for company list."""

    total: int = Field(..., description="Total number of companies with articles")
    companies: list[CompanyInfo]


class PipelineMetadata(BaseModel):
    """Metadata about the pipeline run that generated the articles."""

    run_at: datetime = Field(..., description="When the pipeline was run")
    model: str = Field(..., description="LLM model used for generation")
    styles: list[str] = Field(..., description="Styles generated")
    total_articles: int = Field(..., description="Total articles generated")
    successful: int = Field(..., description="Successfully generated articles")
    failed: int = Field(..., description="Failed article generations")
    style_breakdown: dict[str, int] = Field(
        default_factory=dict, description="Count of articles per style"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    articles_loaded: int = Field(..., description="Number of articles loaded")
    companies_count: int = Field(..., description="Number of unique companies")


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str = Field(..., description="Error message")

