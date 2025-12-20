"""
Simple Recall Recommender - 基于规则的粗排模块

该模块提供一个简化版的公司推荐召回系统，基于5个规则召回候选公司：
1. R1_industry: 核心行业匹配
2. R2_tech_focus: 技术路线匹配
3. R3_industry_market: 行业+市场组合匹配
4. R4_team_background: 团队画像匹配
5. R5_industry_team: 行业+团队组合匹配

特点：
- 每个规则召回 Top 20 候选公司
- 使用轻量级头部抑制（只用 CompanyStageHeadSuppression）
- 输出包含原始数据（company_name, location, company_details）
- 设计用于后续 LLM 精排
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

# 复用现有模块
from company_embedding import (
    CompanyRecord, 
    load_companies_from_csv, 
    load_embeddings_npy,
)
from company_recommender import (
    CompanyProfile, 
    load_company_profiles,
    CompanyStageHeadSuppression,
)


# ============================================================================
# 规则定义
# ============================================================================

RECALL_RULES = [
    {
        "rule_id": "R1_industry",
        "name": "同行业公司",
        "name_en": "Same Industry",
        "dimensions": ["industry"],
        "story_template": "同为{tags}领域的公司",
    },
    {
        "rule_id": "R2_tech_focus",
        "name": "同技术方向",
        "name_en": "Same Tech Focus",
        "dimensions": ["tech_focus"],
        "story_template": "同为{tags}技术方向的公司",
    },
    {
        "rule_id": "R3_industry_market",
        "name": "同行业同市场",
        "name_en": "Same Industry & Market",
        "dimensions": ["industry", "target_market"],
        "story_template": "同为面向{target_market}市场的{industry}公司",
    },
    {
        "rule_id": "R4_team_background",
        "name": "同团队背景",
        "name_en": "Same Team Background",
        "dimensions": ["team_background"],
        "story_template": "同为{tags}背景的创业公司",
    },
    {
        "rule_id": "R5_industry_team",
        "name": "同行业同背景",
        "name_en": "Same Industry & Team",
        "dimensions": ["industry", "team_background"],
        "story_template": "同为{industry}领域的{team_background}创业公司",
    },
]


# ============================================================================
# 标签中文映射
# ============================================================================

TAG_LABELS_ZH = {
    # Industry
    "ai_llm": "AI大模型",
    "robotics": "机器人",
    "edtech": "教育科技",
    "fintech": "金融科技",
    "healthtech": "医疗健康",
    "enterprise_saas": "企业服务",
    "ecommerce": "电商",
    "gaming": "游戏",
    "social": "社交",
    "semiconductor": "半导体",
    "automotive": "汽车出行",
    "consumer_hw": "消费电子",
    "cloud_infra": "云基础设施",
    "content_media": "内容媒体",
    "biotech": "生物科技",
    "investment": "投资",
    "other": "其他",
    
    # Target Market
    "china_domestic": "国内",
    "global": "全球",
    "sea": "东南亚",
    "us": "北美",
    "europe": "欧洲",
    "japan_korea": "日韩",
    "latam": "拉美",
    "mena": "中东北非",
    
    # Tech Focus
    "llm_foundation": "大模型底层",
    "computer_vision": "计算机视觉",
    "speech_nlp": "语音/NLP",
    "embodied_ai": "具身智能",
    "aigc": "AIGC",
    "3d_graphics": "3D图形",
    "chip_hardware": "芯片硬件",
    "data_infra": "数据基础设施",
    "autonomous": "自动驾驶",
    "blockchain": "区块链",
    "quantum": "量子计算",
    "not_tech_focused": "非技术驱动",
    
    # Team Background
    "bigtech_alumni": "大厂背景",
    "top_university": "名校背景",
    "serial_entrepreneur": "连续创业者",
    "academic": "学术背景",
    "industry_expert": "行业专家",
    "international": "国际化团队",
    "unknown": "未知",
}


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class RecallCandidate:
    """召回的候选公司"""
    company_id: str
    company_name: str
    location: str
    company_details: str
    final_score: float
    tag_score: float
    embedding_score: float
    head_penalty_applied: bool
    shared_tags: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class RecallGroup:
    """一个规则的召回结果"""
    rule_id: str
    rule_name: str
    rule_name_en: str
    rule_story: str
    matched_tags: Dict[str, List[str]]
    candidates: List[RecallCandidate] = field(default_factory=list)


@dataclass
class RecallResult:
    """完整的召回结果"""
    query_company: Dict[str, str]
    recall_groups: List[RecallGroup] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 核心类
# ============================================================================

class SimpleRecallRecommender:
    """基于规则的简单召回推荐器
    
    特点：
    - 5 个召回规则
    - 轻量级头部抑制（只用 CompanyStageHeadSuppression）
    - 输出包含原始数据
    
    Example:
        recommender = SimpleRecallRecommender(profiles, raw_companies, embeddings, mapping)
        result = recommender.recall("cid_100", top_k=20)
    """
    
    def __init__(
        self,
        profiles: List[CompanyProfile],
        raw_companies: List[CompanyRecord],
        embeddings: np.ndarray,
        embedding_mapping: Dict[str, int],
        head_suppression: bool = True,
        head_penalty: float = 0.5,
    ):
        """初始化推荐器
        
        Args:
            profiles: 公司标签数据列表
            raw_companies: 原始公司数据列表
            embeddings: 公司向量矩阵
            embedding_mapping: company_id -> 向量索引映射
            head_suppression: 是否启用头部抑制
            head_penalty: 头部公司降权比例 (0-1)
        """
        self.profiles = {p.company_id: p for p in profiles}
        self.raw_companies = {c.company_id: c for c in raw_companies}
        self.embeddings = embeddings
        self.embedding_mapping = embedding_mapping
        
        # 只用 CompanyStageHeadSuppression，移除 IDFHeadSuppression
        self.head_suppression_enabled = head_suppression
        self.head_suppression = CompanyStageHeadSuppression(head_penalty) if head_suppression else None
        
        # 全局统计（用于头部抑制计算）
        self.global_stats = self._compute_global_stats()
    
    def _compute_global_stats(self) -> Dict:
        """计算全局统计信息"""
        return {
            "total_companies": len(self.profiles),
            "max_tags_per_company": max(
                len(p.industry) + len(p.tech_focus) + len(p.target_market) + len(p.team_background)
                for p in self.profiles.values()
            ) if self.profiles else 10,
        }
    
    def _get_company_tags(self, profile: CompanyProfile, dimension: str) -> Set[str]:
        """获取公司在指定维度的 tags"""
        tags = getattr(profile, dimension, [])
        if isinstance(tags, str):
            filtered = {tags} if tags and tags != "unknown" else set()
            if not filtered and tags:
                logging.warning(
                    f"All tags filtered out for company {profile.company_id} in dimension '{dimension}': "
                    f"original tag was '{tags}'"
                )
            return filtered
        
        filtered = set(t for t in tags if t and t not in {"unknown", "other", "not_tech_focused"})
        if not filtered and tags:
            logging.warning(
                f"All tags filtered out for company {profile.company_id} in dimension '{dimension}': "
                f"original tags were {tags}"
            )
        return filtered
    
    def _match_rule(self, query: CompanyProfile, rule: Dict) -> Set[str]:
        """匹配规则，返回候选公司ID集合
        
        Args:
            query: 查询公司的 profile
            rule: 规则配置
            
        Returns:
            匹配的候选公司 ID 集合
        """
        dimensions = rule["dimensions"]
        candidate_ids: Set[str] = None
        
        for dim in dimensions:
            query_tags = self._get_company_tags(query, dim)
            if not query_tags:
                return set()  # 查询公司在该维度没有 tag，无法匹配
            
            # 找出所有在该维度有交集的公司
            dim_matches = set()
            for cid, profile in self.profiles.items():
                if cid == query.company_id:
                    continue
                candidate_tags = self._get_company_tags(profile, dim)
                if query_tags & candidate_tags:  # 有交集
                    dim_matches.add(cid)
            
            # 多维度取交集
            if candidate_ids is None:
                candidate_ids = dim_matches
            else:
                candidate_ids &= dim_matches
        
        return candidate_ids or set()
    
    def _compute_tag_similarity(
        self, 
        query: CompanyProfile, 
        candidate: CompanyProfile, 
        dimensions: List[str]
    ) -> Tuple[float, Dict[str, List[str]]]:
        """计算 Tag Jaccard 相似度
        
        Args:
            query: 查询公司
            candidate: 候选公司
            dimensions: 要计算的维度列表
            
        Returns:
            (相似度分数, 共同tags字典)
        """
        all_query_tags = set()
        all_candidate_tags = set()
        shared_tags: Dict[str, List[str]] = {}
        
        for dim in dimensions:
            query_tags = self._get_company_tags(query, dim)
            candidate_tags = self._get_company_tags(candidate, dim)
            
            all_query_tags |= query_tags
            all_candidate_tags |= candidate_tags
            
            overlap = query_tags & candidate_tags
            if overlap:
                shared_tags[dim] = list(overlap)
        
        # Jaccard 相似度
        union = all_query_tags | all_candidate_tags
        if not union:
            return 0.0, {}
        
        intersection = all_query_tags & all_candidate_tags
        score = len(intersection) / len(union)
        
        return score, shared_tags
    
    def _compute_embedding_similarity(self, query_id: str, candidate_id: str) -> float:
        """计算 Embedding 余弦相似度"""
        query_idx = self.embedding_mapping.get(query_id)
        candidate_idx = self.embedding_mapping.get(candidate_id)
        
        if query_idx is None or candidate_idx is None:
            return 0.0
        
        query_vec = self.embeddings[query_idx]
        candidate_vec = self.embeddings[candidate_idx]
        
        # 余弦相似度
        dot_product = np.dot(query_vec, candidate_vec)
        norm_product = np.linalg.norm(query_vec) * np.linalg.norm(candidate_vec)
        
        if norm_product == 0:
            return 0.0
        
        return float(dot_product / norm_product)
    
    def _apply_head_suppression(self, candidate: CompanyProfile, raw_score: float) -> Tuple[float, bool]:
        """应用头部抑制
        
        Returns:
            (调整后分数, 是否应用了惩罚)
        """
        if not self.head_suppression_enabled or self.head_suppression is None:
            return raw_score, False
        
        penalty = self.head_suppression.compute_penalty(candidate, self.global_stats)
        
        if penalty > 0:
            adjusted_score = raw_score * (1 - penalty)
            return adjusted_score, True
        
        return raw_score, False
    
    def _generate_story(self, rule: Dict, matched_tags: Dict[str, List[str]]) -> str:
        """生成规则故事描述"""
        template = rule.get("story_template", "相似公司")
        
        # 将 tags 转换为中文
        def tags_to_zh(tags: List[str]) -> str:
            zh_tags = [TAG_LABELS_ZH.get(t, t) for t in tags]
            return "、".join(zh_tags[:3])  # 最多显示3个
        
        # 替换模板中的占位符
        if "{tags}" in template:
            all_tags = []
            for dim_tags in matched_tags.values():
                all_tags.extend(dim_tags)
            template = template.replace("{tags}", tags_to_zh(all_tags))
        
        for dim, tags in matched_tags.items():
            placeholder = "{" + dim + "}"
            if placeholder in template:
                template = template.replace(placeholder, tags_to_zh(tags))
        
        return template
    
    def _score_candidates(
        self,
        query_id: str,
        candidate_ids: Set[str],
        rule: Dict,
    ) -> List[RecallCandidate]:
        """计算候选公司分数并排序
        
        Args:
            query_id: 查询公司ID
            candidate_ids: 候选公司ID集合
            rule: 规则配置
            
        Returns:
            排序后的候选公司列表
        """
        query = self.profiles.get(query_id)
        if not query:
            return []
        
        dimensions = rule["dimensions"]
        scored_candidates = []
        
        for cid in candidate_ids:
            candidate = self.profiles.get(cid)
            raw_company = self.raw_companies.get(cid)
            
            if not candidate or not raw_company:
                continue
            
            # 计算 Tag 相似度
            tag_score, shared_tags = self._compute_tag_similarity(query, candidate, dimensions)
            
            # 计算 Embedding 相似度
            embedding_score = self._compute_embedding_similarity(query_id, cid)
            
            # 融合分数 (6:4 权重)
            combined_score = 0.6 * tag_score + 0.4 * embedding_score
            
            # 应用头部抑制
            final_score, penalty_applied = self._apply_head_suppression(candidate, combined_score)
            
            scored_candidates.append(RecallCandidate(
                company_id=cid,
                company_name=raw_company.company_name,
                location=raw_company.location,
                company_details=raw_company.company_details,
                final_score=round(final_score, 4),
                tag_score=round(tag_score, 4),
                embedding_score=round(embedding_score, 4),
                head_penalty_applied=penalty_applied,
                shared_tags=shared_tags,
            ))
        
        # 按 final_score 降序排序
        scored_candidates.sort(key=lambda x: -x.final_score)
        
        return scored_candidates
    
    def recall(
        self,
        query_id: str,
        rules: List[Dict] = None,
        top_k: int = 20,
    ) -> RecallResult:
        """执行召回
        
        Args:
            query_id: 查询公司ID
            rules: 规则列表，默认使用全部5个规则
            top_k: 每个规则召回的候选数量
            
        Returns:
            RecallResult 包含所有规则的召回结果
        """
        if rules is None:
            rules = RECALL_RULES
        
        query = self.profiles.get(query_id)
        raw_query = self.raw_companies.get(query_id)
        
        if not query or not raw_query:
            raise ValueError(f"Company not found: {query_id}")
        
        recall_groups = []
        
        for rule in rules:
            # 1. 匹配规则
            candidate_ids = self._match_rule(query, rule)
            
            if not candidate_ids:
                continue
            
            # 2. 计算分数并排序
            candidates = self._score_candidates(query_id, candidate_ids, rule)
            
            # 3. 取 Top K
            top_candidates = candidates[:top_k]
            
            if not top_candidates:
                continue
            
            # 4. 获取查询公司在该规则维度的 tags
            matched_tags: Dict[str, List[str]] = {}
            for dim in rule["dimensions"]:
                query_tags = self._get_company_tags(query, dim)
                if query_tags:
                    matched_tags[dim] = list(query_tags)
            
            # 5. 生成故事描述
            story = self._generate_story(rule, matched_tags)
            
            recall_groups.append(RecallGroup(
                rule_id=rule["rule_id"],
                rule_name=rule["name"],
                rule_name_en=rule["name_en"],
                rule_story=story,
                matched_tags=matched_tags,
                candidates=top_candidates,
            ))
        
        return RecallResult(
            query_company={
                "company_id": query_id,
                "company_name": raw_query.company_name,
                "location": raw_query.location,
                "company_details": raw_query.company_details,
            },
            recall_groups=recall_groups,
            metadata={
                "total_rules": len(recall_groups),
                "candidates_per_rule": top_k,
                "head_suppression": "CompanyStageHeadSuppression" if self.head_suppression_enabled else "disabled",
                "score_formula": "0.6 * tag_score + 0.4 * embedding_score",
            },
        )
    
    def batch_recall(
        self,
        company_ids: List[str],
        rules: List[Dict] = None,
        top_k: int = 20,
    ) -> List[RecallResult]:
        """批量召回
        
        Args:
            company_ids: 查询公司ID列表
            rules: 规则列表
            top_k: 每个规则召回的候选数量
            
        Returns:
            RecallResult 列表
        """
        results = []
        for cid in company_ids:
            try:
                result = self.recall(cid, rules, top_k)
                results.append(result)
            except ValueError as e:
                print(f"Warning: {e}")
                continue
        return results


# ============================================================================
# 工具函数
# ============================================================================

def recall_result_to_dict(result: RecallResult) -> Dict[str, Any]:
    """将 RecallResult 转换为字典"""
    return {
        "query_company": result.query_company,
        "recall_groups": [
            {
                "rule_id": g.rule_id,
                "rule_name": g.rule_name,
                "rule_name_en": g.rule_name_en,
                "rule_story": g.rule_story,
                "matched_tags": g.matched_tags,
                "candidates": [
                    {
                        "company_id": c.company_id,
                        "company_name": c.company_name,
                        "location": c.location,
                        "company_details": c.company_details,
                        "final_score": c.final_score,
                        "tag_score": c.tag_score,
                        "embedding_score": c.embedding_score,
                        "head_penalty_applied": c.head_penalty_applied,
                    }
                    for c in g.candidates
                ],
            }
            for g in result.recall_groups
        ],
        "metadata": result.metadata,
    }


def save_recall_results(
    results: List[RecallResult],
    output_path: Path,
) -> None:
    """保存召回结果到 JSON 文件"""
    data = [recall_result_to_dict(r) for r in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_recall_result(result: RecallResult) -> None:
    """打印召回结果"""
    print("\n" + "=" * 70)
    print(f"召回结果: {result.query_company['company_name']} ({result.query_company['company_id']})")
    print("=" * 70)
    
    for group in result.recall_groups:
        print(f"\n【{group.rule_id}】{group.rule_name}")
        print(f"  故事: {group.rule_story}")
        print(f"  匹配标签: {group.matched_tags}")
        print(f"  候选公司 ({len(group.candidates)} 家):")
        
        for i, c in enumerate(group.candidates[:10], 1):  # 只显示前10个
            penalty_marker = " ⬇头部" if c.head_penalty_applied else ""
            print(f"    {i}. {c.company_name} (score={c.final_score:.3f}, tag={c.tag_score:.3f}, emb={c.embedding_score:.3f}){penalty_marker}")
        
        if len(group.candidates) > 10:
            print(f"    ... 还有 {len(group.candidates) - 10} 家")
    
    print("\n" + "-" * 70)
    print(f"总规则数: {result.metadata.get('total_rules', 0)}")
    print(f"每规则候选数: {result.metadata.get('candidates_per_rule', 20)}")
    print(f"头部抑制: {result.metadata.get('head_suppression', 'unknown')}")


def load_data_for_recommender(
    raw_csv_path: Path,
    tags_json_path: Path,
    embeddings_dir: Path,
) -> Tuple[List[CompanyProfile], List[CompanyRecord], np.ndarray, Dict[str, int]]:
    """加载推荐器所需的所有数据
    
    Args:
        raw_csv_path: 原始公司数据 CSV 路径
        tags_json_path: 公司标签 JSON 路径
        embeddings_dir: Embeddings 目录路径
        
    Returns:
        (profiles, raw_companies, embeddings, mapping)
    """
    # 加载原始数据
    raw_companies = load_companies_from_csv(raw_csv_path)
    
    # 加载标签数据
    profiles = load_company_profiles(tags_json_path)
    
    # 加载 Embeddings
    embeddings_path = embeddings_dir / "company_embeddings.npy"
    embeddings, mapping = load_embeddings_npy(embeddings_path)
    
    return profiles, raw_companies, embeddings, mapping

