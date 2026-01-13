"""
Article Generator Data Models

定义文章生成系统的所有数据结构。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class WebSearchResult:
    """OpenRouter :online Web 搜索结果
    
    包含公司的搜索摘要和引用来源，作为下游 RAG 语料。
    """
    company_id: str
    company_name: str
    query_used: str                     # 搜索 prompt（包含 company_details 上下文）
    search_summary: str                 # 长总结（800-1500字，作为 RAG 语料）
    citations: List[str] = field(default_factory=list)  # 引用的来源（markdown links）
    is_valid: bool = True               # 是否找到有效信息
    searched_at: str = ""               # 搜索时间戳
    
    def __post_init__(self):
        if not self.searched_at:
            self.searched_at = datetime.now().isoformat()


@dataclass
class SelectedCompany:
    """精排选中的公司"""
    company_id: str
    company_name: str
    location: str
    company_details: str
    selection_reason: str               # 选择理由


@dataclass
class RerankResult:
    """LLM 精排结果
    
    从 20 个 candidates 中选择 top_k 个最相关的公司。
    """
    query_company_id: str
    query_company_name: str
    rule_id: str
    rule_name: str
    narrative_angle: str                # 叙事角度/故事线
    selected_companies: List[SelectedCompany] = field(default_factory=list)
    reranked_at: str = ""
    
    def __post_init__(self):
        if not self.reranked_at:
            self.reranked_at = datetime.now().isoformat()


@dataclass
class Article:
    """生成的文章"""
    query_company_id: str
    query_company_name: str
    rule_id: str
    style: str                          # 风格: 36kr, huxiu, xiaohongshu, linkedin, zhihu
    title: str
    content: str
    word_count: int
    candidate_company_ids: List[str] = field(default_factory=list)  # 所有被选中的候选公司 ID
    key_takeaways: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    generated_at: str = ""
    # 小红书 config-based 生成的配置元数据
    style_config: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()
        if not self.word_count:
            self.word_count = len(self.content)


@dataclass
class ArticleStyle:
    """文章风格配置"""
    style_id: str
    name_zh: str
    name_en: str
    word_count_min: int
    word_count_max: int
    use_emoji: bool
    tone: str                           # 语气描述
    structure: str                      # 结构说明
    example_intro: str                  # 开头示例

