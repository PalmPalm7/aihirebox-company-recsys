"""
Article Generator Package

LLM 精排与文章生成系统，包含:
- OpenRouter :online Web Search (Layer 0)
- GPT-5-mini Reranker (Layer 1)  
- Gemini Article Writer (Layer 2)
"""

from .models import (
    WebSearchResult,
    SelectedCompany,
    RerankResult,
    Article,
    ArticleStyle,
)
from .web_searcher import OpenRouterWebSearcher
from .reranker import LLMReranker
from .styles import ARTICLE_STYLES, get_style
from .article_writer import ArticleWriter

__all__ = [
    # Models
    "WebSearchResult",
    "SelectedCompany", 
    "RerankResult",
    "Article",
    "ArticleStyle",
    # Core classes
    "OpenRouterWebSearcher",
    "LLMReranker",
    "ArticleWriter",
    # Styles
    "ARTICLE_STYLES",
    "get_style",
]

