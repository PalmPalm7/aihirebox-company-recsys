"""
LLM Reranker - Layer 1

使用 openai/gpt-5-mini via OpenRouter 进行精排，
从 20 个 candidates 中选择 top_k 个最相关的公司。

输入: recall_results.json (来自 simple_recommender)
输出: rerank_cache/{query_company_id}_{rule_id}.json
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .models import RerankResult, SelectedCompany, WebSearchResult


class LLMReranker:
    """LLM 精排器
    
    使用 GPT-5-mini 对召回的候选公司进行精排，
    选择最适合生成文章的公司组合。
    
    Example:
        reranker = LLMReranker()
        result = reranker.rerank(query_company, candidates, "R1_industry", top_k=5)
    """
    
    DEFAULT_MODEL = "openai/gpt-5-mini"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """初始化精排器
        
        Args:
            api_key: OpenRouter API key
            model: 模型名称（默认 gpt-5-mini）
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.model = model or self.DEFAULT_MODEL
    
    def _build_rerank_prompt(
        self,
        query_company: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        rule_id: str,
        rule_story: str,
        min_k: int,
        max_k: int,
        web_search_results: Optional[Dict[str, WebSearchResult]] = None,
    ) -> str:
        """构建精排 prompt

        Args:
            query_company: 查询公司信息
            candidates: 候选公司列表
            rule_id: 规则ID
            rule_story: 规则故事描述
            min_k: 最少选择数量
            max_k: 最多选择数量
            web_search_results: Web 搜索结果缓存（可选）
        """
        # 构建候选公司列表
        candidate_sections = []
        for i, c in enumerate(candidates, 1):
            section = f"""
## 候选公司 {i}: {c['company_name']} ({c['company_id']})
- 地点: {c.get('location', '未知')}
- 公司介绍: {c.get('company_details', '无')[:500]}
- 召回分数: {c.get('final_score', 0):.3f}
"""
            # 添加 web search 结果（如果有）
            if web_search_results and c['company_id'] in web_search_results:
                ws = web_search_results[c['company_id']]
                if ws.is_valid:
                    # 截取摘要前500字
                    summary_preview = ws.search_summary[:500] + "..." if len(ws.search_summary) > 500 else ws.search_summary
                    section += f"- 网络搜索摘要: {summary_preview}\n"

            candidate_sections.append(section)

        candidates_text = "\n".join(candidate_sections)

        # 构建选择数量说明
        if min_k == max_k:
            selection_count = f"**{max_k} 家**"
        else:
            selection_count = f"**{min_k} 到 {max_k} 家**"

        return f"""你是一位专业的科技行业分析师，需要从候选公司中筛选真正适合推荐的公司。

## 任务背景
用户正在查看公司 **{query_company['company_name']}**，我们需要推荐与之相关的公司。

推荐维度: {rule_story}
规则ID: {rule_id}

## 查询公司信息
- 公司名称: {query_company['company_name']}
- 公司ID: {query_company['company_id']}
- 地点: {query_company.get('location', '未知')}
- 公司介绍: {query_company.get('company_details', '无')[:500]}

## 候选公司列表（共 {len(candidates)} 家）
{candidates_text}

## 任务要求
请从以上候选公司中选择 {selection_count} 真正适合推荐的公司。

**重要原则：宁缺毋滥**
- 你的任务是作为质量把关者，筛选出真正有价值的关联公司
- 如果某个候选公司与查询公司的关联过于牵强，请直接排除，不要强行关联
- 只选择那些能够讲出有意义故事的公司，而不是凑数
- 如果只有 1-2 家公司真正相关，那就只选 1-2 家，不需要凑满 {max_k} 家

选择标准:
1. **真实相关性**: 与查询公司在"{rule_story}"维度是否有真实、有意义的关联（而非表面相似）
2. **故事价值**: 能否与查询公司一起讲出有洞察的行业故事
3. **信息充分度**: 是否有足够信息支撑一篇有质量的文章
4. **避免强行关联**: 如果关联点太弱或太牵强，宁可不选

## 输出格式
请以 JSON 格式输出，包含:
1. selected_company_ids: 选中的公司ID列表（按推荐优先级排序，可以少于 {max_k} 家）
2. narrative_angle: 用一句话描述这组公司的故事角度/叙事线
3. selection_reasons: 每个公司的选择理由（简短）
4. rejected_reason: （可选）如果选择数量少于 {max_k}，简要说明为什么其他候选不够格

示例（选了3家而非5家）:
```json
{{
  "selected_company_ids": ["cid_1", "cid_5", "cid_8"],
  "narrative_angle": "这三家公司都在用AI重构传统教育场景，各有独特切入点",
  "selection_reasons": {{
    "cid_1": "核心业务与查询公司高度互补，可形成对比分析",
    "cid_5": "代表另一种技术路线，增加多样性",
    "cid_8": "有明确的商业化成果，增加文章可信度"
  }},
  "rejected_reason": "其他候选公司虽然同属教育行业，但业务模式或技术路线差异过大，难以形成有意义的对比分析"
}}
```

请直接输出 JSON，不需要其他解释。
"""
    
    def _parse_rerank_response(
        self,
        response_text: str,
        query_company: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        rule_id: str,
        rule_name: str,
    ) -> RerankResult:
        """解析 LLM 精排响应"""
        try:
            # 尝试提取 JSON
            json_text = response_text
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_text = response_text.split("```")[1].split("```")[0]
            
            data = json.loads(json_text.strip())
        except (json.JSONDecodeError, IndexError):
            # 解析失败，返回默认结果
            return RerankResult(
                query_company_id=query_company["company_id"],
                query_company_name=query_company["company_name"],
                rule_id=rule_id,
                rule_name=rule_name,
                narrative_angle="解析失败",
                selected_companies=[],
            )
        
        # 构建选中公司列表
        selected_ids = data.get("selected_company_ids", [])
        selection_reasons = data.get("selection_reasons", {})
        
        # 创建 candidates 索引
        candidates_map = {c["company_id"]: c for c in candidates}
        
        selected_companies = []
        for cid in selected_ids:
            if cid in candidates_map:
                c = candidates_map[cid]
                selected_companies.append(SelectedCompany(
                    company_id=cid,
                    company_name=c["company_name"],
                    location=c.get("location", ""),
                    company_details=c.get("company_details", ""),
                    selection_reason=selection_reasons.get(cid, ""),
                ))
        
        return RerankResult(
            query_company_id=query_company["company_id"],
            query_company_name=query_company["company_name"],
            rule_id=rule_id,
            rule_name=rule_name,
            narrative_angle=data.get("narrative_angle", ""),
            selected_companies=selected_companies,
        )
    
    def rerank(
        self,
        query_company: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        rule_id: str,
        rule_name: str = "",
        rule_story: str = "",
        min_k: int = 1,
        max_k: int = 5,
        web_search_results: Optional[Dict[str, WebSearchResult]] = None,
    ) -> RerankResult:
        """对候选公司进行精排

        Args:
            query_company: 查询公司信息
            candidates: 候选公司列表
            rule_id: 规则ID
            rule_name: 规则名称
            rule_story: 规则故事描述
            min_k: 最少选择数量（默认1）
            max_k: 最多选择数量（默认5）
            web_search_results: Web 搜索结果缓存

        Returns:
            RerankResult 精排结果
        """
        prompt = self._build_rerank_prompt(
            query_company=query_company,
            candidates=candidates,
            rule_id=rule_id,
            rule_story=rule_story or rule_name,
            min_k=min_k,
            max_k=max_k,
            web_search_results=web_search_results,
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            
            response_text = response.choices[0].message.content or "{}"
            
            return self._parse_rerank_response(
                response_text=response_text,
                query_company=query_company,
                candidates=candidates,
                rule_id=rule_id,
                rule_name=rule_name,
            )
            
        except Exception as e:
            return RerankResult(
                query_company_id=query_company["company_id"],
                query_company_name=query_company["company_name"],
                rule_id=rule_id,
                rule_name=rule_name,
                narrative_angle=f"Error: {str(e)}",
                selected_companies=[],
            )
    
    def batch_rerank(
        self,
        recall_results: List[Dict[str, Any]],
        min_k: int = 1,
        max_k: int = 5,
        delay_seconds: float = 0.5,
        skip_existing: bool = True,
        cache_dir: Optional[Path] = None,
        web_search_cache: Optional[Dict[str, WebSearchResult]] = None,
        show_progress: bool = True,
    ) -> List[RerankResult]:
        """批量精排

        Args:
            recall_results: 召回结果列表（来自 simple_recommender）
            min_k: 每个规则最少选择的公司数量（默认1）
            max_k: 每个规则最多选择的公司数量（默认5）
            delay_seconds: 请求间隔
            skip_existing: 是否跳过已有缓存
            cache_dir: 缓存目录
            web_search_cache: Web 搜索缓存
            show_progress: 是否显示进度

        Returns:
            RerankResult 列表
        """
        from tqdm import tqdm
        
        results = []
        
        # 展开所有 (query_company, recall_group) 对
        tasks = []
        for recall_result in recall_results:
            query_company = recall_result["query_company"]
            for group in recall_result.get("recall_groups", []):
                tasks.append({
                    "query_company": query_company,
                    "rule_id": group["rule_id"],
                    "rule_name": group["rule_name"],
                    "rule_story": group.get("rule_story", ""),
                    "candidates": group["candidates"],
                })
        
        iterator = tasks
        if show_progress:
            iterator = tqdm(tasks, desc="Reranking")
        
        for task in iterator:
            query_company = task["query_company"]
            rule_id = task["rule_id"]
            cache_key = f"{query_company['company_id']}_{rule_id}"
            
            # 检查缓存
            if skip_existing and cache_dir:
                cache_file = cache_dir / f"{cache_key}.json"
                if cache_file.exists():
                    try:
                        with open(cache_file, "r", encoding="utf-8") as f:
                            cached = json.load(f)
                        results.append(RerankResult(
                            query_company_id=cached["query_company_id"],
                            query_company_name=cached["query_company_name"],
                            rule_id=cached["rule_id"],
                            rule_name=cached.get("rule_name", ""),
                            narrative_angle=cached.get("narrative_angle", ""),
                            selected_companies=[
                                SelectedCompany(**sc) for sc in cached.get("selected_companies", [])
                            ],
                            reranked_at=cached.get("reranked_at", ""),
                        ))
                        continue
                    except Exception:
                        pass
            
            # 执行精排
            result = self.rerank(
                query_company=query_company,
                candidates=task["candidates"],
                rule_id=rule_id,
                rule_name=task["rule_name"],
                rule_story=task["rule_story"],
                min_k=min_k,
                max_k=max_k,
                web_search_results=web_search_cache,
            )
            results.append(result)
            
            # 保存缓存
            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = cache_dir / f"{cache_key}.json"
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "query_company_id": result.query_company_id,
                        "query_company_name": result.query_company_name,
                        "rule_id": result.rule_id,
                        "rule_name": result.rule_name,
                        "narrative_angle": result.narrative_angle,
                        "selected_companies": [
                            {
                                "company_id": sc.company_id,
                                "company_name": sc.company_name,
                                "location": sc.location,
                                "company_details": sc.company_details,
                                "selection_reason": sc.selection_reason,
                            }
                            for sc in result.selected_companies
                        ],
                        "reranked_at": result.reranked_at,
                    }, f, ensure_ascii=False, indent=2)
            
            # 添加延迟
            if delay_seconds > 0:
                time.sleep(delay_seconds)

        return results

    def batch_rerank_concurrent(
        self,
        recall_results: List[Dict[str, Any]],
        min_k: int = 1,
        max_k: int = 5,
        skip_existing: bool = True,
        cache_dir: Optional[Path] = None,
        web_search_cache: Optional[Dict[str, WebSearchResult]] = None,
        show_progress: bool = True,
        concurrency: int = 20,
    ) -> List[RerankResult]:
        """并发批量精排

        Args:
            recall_results: 召回结果列表（来自 simple_recommender）
            min_k: 每个规则最少选择的公司数量（默认1）
            max_k: 每个规则最多选择的公司数量（默认5）
            skip_existing: 是否跳过已有缓存
            cache_dir: 缓存目录
            web_search_cache: Web 搜索缓存
            show_progress: 是否显示进度
            concurrency: 并发数（默认20）

        Returns:
            RerankResult 列表
        """
        from tqdm import tqdm

        results = []
        tasks = []

        # 创建缓存目录
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        # 展开所有 (query_company, recall_group) 对，检查缓存
        for recall_result in recall_results:
            query_company = recall_result["query_company"]
            for group in recall_result.get("recall_groups", []):
                rule_id = group["rule_id"]
                cache_key = f"{query_company['company_id']}_{rule_id}"

                # 检查缓存
                if skip_existing and cache_dir:
                    cache_file = cache_dir / f"{cache_key}.json"
                    if cache_file.exists():
                        try:
                            with open(cache_file, "r", encoding="utf-8") as f:
                                cached = json.load(f)
                            results.append(RerankResult(
                                query_company_id=cached["query_company_id"],
                                query_company_name=cached["query_company_name"],
                                rule_id=cached["rule_id"],
                                rule_name=cached.get("rule_name", ""),
                                narrative_angle=cached.get("narrative_angle", ""),
                                selected_companies=[
                                    SelectedCompany(**sc) for sc in cached.get("selected_companies", [])
                                ],
                                reranked_at=cached.get("reranked_at", ""),
                            ))
                            continue
                        except Exception:
                            pass

                tasks.append({
                    "query_company": query_company,
                    "rule_id": rule_id,
                    "rule_name": group["rule_name"],
                    "rule_story": group.get("rule_story", ""),
                    "candidates": group["candidates"],
                    "cache_key": cache_key,
                })

        if not tasks:
            return results

        def rerank_and_cache(task: Dict[str, Any]) -> RerankResult:
            """精排单个任务并保存缓存"""
            result = self.rerank(
                query_company=task["query_company"],
                candidates=task["candidates"],
                rule_id=task["rule_id"],
                rule_name=task["rule_name"],
                rule_story=task["rule_story"],
                min_k=min_k,
                max_k=max_k,
                web_search_results=web_search_cache,
            )

            # 保存缓存
            if cache_dir:
                cache_file = cache_dir / f"{task['cache_key']}.json"
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "query_company_id": result.query_company_id,
                        "query_company_name": result.query_company_name,
                        "rule_id": result.rule_id,
                        "rule_name": result.rule_name,
                        "narrative_angle": result.narrative_angle,
                        "selected_companies": [
                            {
                                "company_id": sc.company_id,
                                "company_name": sc.company_name,
                                "location": sc.location,
                                "company_details": sc.company_details,
                                "selection_reason": sc.selection_reason,
                            }
                            for sc in result.selected_companies
                        ],
                        "reranked_at": result.reranked_at,
                    }, f, ensure_ascii=False, indent=2)

            return result

        # 并发执行
        if show_progress:
            print(f"Reranking {len(tasks)} tasks with {concurrency} workers...")
            print(f"(Skipped {len(results)} cached results)")

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(rerank_and_cache, t): t for t in tasks}

            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(tasks), desc="Reranking (parallel)")

            for future in iterator:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    task = futures[future]
                    print(f"Error reranking {task.get('cache_key')}: {e}")

        return results


def load_rerank_cache(cache_dir: Path) -> Dict[str, RerankResult]:
    """加载精排缓存
    
    Args:
        cache_dir: 缓存目录
        
    Returns:
        {"{company_id}_{rule_id}": RerankResult}
    """
    results = {}
    
    if not cache_dir.exists():
        return results
    
    for cache_file in cache_dir.glob("*.json"):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            key = f"{data['query_company_id']}_{data['rule_id']}"
            results[key] = RerankResult(
                query_company_id=data["query_company_id"],
                query_company_name=data["query_company_name"],
                rule_id=data["rule_id"],
                rule_name=data.get("rule_name", ""),
                narrative_angle=data.get("narrative_angle", ""),
                selected_companies=[
                    SelectedCompany(**sc) for sc in data.get("selected_companies", [])
                ],
                reranked_at=data.get("reranked_at", ""),
            )
        except Exception:
            continue
    
    return results

