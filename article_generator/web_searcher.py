"""
OpenRouter Web Searcher - Layer 0

使用 OpenRouter 的 :online 后缀启用 Exa.ai 搜索，
带 company_details 上下文搜索，自动剔除无关文章，输出长总结作为 RAG 语料。

API 调用方式:
    model = "openai/gpt-5-mini:online"
    # 使用 plugins 参数可自定义 max_results

定价:
    $4 / 1,000 web searches
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

from .models import WebSearchResult


class OpenRouterWebSearcher:
    """OpenRouter :online Web 搜索器
    
    使用 Exa.ai 搜索公司信息，自动验证并生成长总结。
    一次调用完成：搜索 -> 验证相关性 -> 剔除无关文章 -> 生成总结
    
    Example:
        searcher = OpenRouterWebSearcher()
        result = searcher.search_company("cid_100", "MiniMax", "MiniMax是一家...")
    """
    
    DEFAULT_MODEL = "openai/gpt-5-mini:online"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """初始化搜索器
        
        Args:
            api_key: OpenRouter API key，默认从环境变量读取
            model: 模型名称，默认使用 gpt-5-mini:online
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        
        # 确保模型带有 :online 后缀
        self.model = model or self.DEFAULT_MODEL
        if ":online" not in self.model:
            self.model = f"{self.model}:online"
    
    def _build_search_prompt(
        self,
        company_name: str,
        company_details: str,
    ) -> str:
        """构建带上下文的搜索 prompt
        
        让模型带着 company_details 搜索，自动剔除不相关的文章。
        """
        return f"""请搜索并总结关于以下公司的最新信息。

## 目标公司信息（来自我们的数据库）
公司名称：{company_name}
公司简介：{company_details}

## 搜索要求
1. 请搜索 10 篇以上关于这家公司的文章进行 cross-reference
2. **重要**：如果搜索到的文章与上述公司简介描述的公司不是同一家（如同名但不同的公司、算法名称等），请直接剔除该文章，不要纳入总结
3. 只保留确实是关于这家公司的相关文章

## 输出要求
请基于相关文章，用中文撰写一份详细的公司研究报告，包含：

1. **公司概况**：核心业务、产品线、技术方向
2. **最新动态**：近期融资、产品发布、战略合作、人事变动等
3. **行业地位**：市场份额、竞争优势、主要竞争对手
4. **技术实力**：核心技术、专利、研发团队背景
5. **商业模式**：盈利模式、客户群体、合作伙伴
6. **发展前景**：行业趋势、公司战略、潜在风险

请尽量详细（800-1500字），这份报告将作为下游文章生成的 RAG 语料。
每个信息点请用 markdown 链接引用来源。
如果没有找到相关的有效文章，请明确说明"未找到关于该公司的相关信息"。
"""
    
    def _extract_citations(self, text: str) -> List[str]:
        """从 markdown 文本中提取引用链接
        
        匹配格式: [text](url)
        """
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(pattern, text)
        return list(set(url for _, url in matches))  # 去重
    
    def _is_valid_result(self, search_summary: str) -> bool:
        """判断搜索结果是否有效
        
        检查是否包含"未找到"等无效标记。
        """
        invalid_markers = [
            "未找到",
            "没有找到",
            "无法找到",
            "找不到相关",
            "暂无相关",
            "无相关信息",
        ]
        
        # 检查无效标记
        for marker in invalid_markers:
            if marker in search_summary:
                return False
        
        # 检查内容长度
        if len(search_summary) < 200:
            return False
        
        return True
    
    def search_company(
        self,
        company_id: str,
        company_name: str,
        company_details: str,
        max_results: int = 10,
    ) -> WebSearchResult:
        """搜索公司信息，自动验证并生成长总结
        
        一次调用完成：搜索 -> 验证相关性 -> 剔除无关文章 -> 生成总结
        
        Args:
            company_id: 公司ID
            company_name: 公司名称
            company_details: 公司详情（用于上下文验证）
            max_results: 搜索结果数量（默认10条）
            
        Returns:
            WebSearchResult 包含搜索摘要和引用
        """
        search_prompt = self._build_search_prompt(company_name, company_details)
        
        try:
            # 使用 :online 模型搜索
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": search_prompt}],
                extra_body={
                    "plugins": [{"id": "web", "max_results": max_results}]
                },
            )
            
            search_summary = response.choices[0].message.content or ""
            citations = self._extract_citations(search_summary)
            is_valid = self._is_valid_result(search_summary)
            
            return WebSearchResult(
                company_id=company_id,
                company_name=company_name,
                query_used=search_prompt,
                search_summary=search_summary,
                citations=citations,
                is_valid=is_valid,
                searched_at=datetime.now().isoformat(),
            )
            
        except Exception as e:
            # 返回错误结果
            return WebSearchResult(
                company_id=company_id,
                company_name=company_name,
                query_used=search_prompt,
                search_summary=f"搜索出错: {str(e)}",
                citations=[],
                is_valid=False,
                searched_at=datetime.now().isoformat(),
            )
    
    def batch_search(
        self,
        companies: List[Dict[str, str]],
        max_results: int = 10,
        delay_seconds: float = 1.0,
        skip_existing: bool = True,
        cache_dir: Optional[Path] = None,
        show_progress: bool = True,
    ) -> List[WebSearchResult]:
        """批量搜索公司信息
        
        Args:
            companies: 公司列表 [{"company_id": "...", "company_name": "...", "company_details": "..."}]
            max_results: 每次搜索的结果数量
            delay_seconds: 请求间隔（避免 rate limit）
            skip_existing: 是否跳过已有缓存
            cache_dir: 缓存目录
            show_progress: 是否显示进度
            
        Returns:
            WebSearchResult 列表
        """
        from tqdm import tqdm
        
        results = []
        
        iterator = companies
        if show_progress:
            iterator = tqdm(companies, desc="Web searching")
        
        for company in iterator:
            company_id = company["company_id"]
            company_name = company["company_name"]
            company_details = company.get("company_details", "")
            
            # 检查缓存
            if skip_existing and cache_dir:
                cache_file = cache_dir / f"{company_id}.json"
                if cache_file.exists():
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    results.append(WebSearchResult(
                        company_id=cached["company_id"],
                        company_name=cached["company_name"],
                        query_used=cached.get("query_used", ""),
                        search_summary=cached["search_summary"],
                        citations=cached.get("citations", []),
                        is_valid=cached.get("is_valid", True),
                        searched_at=cached.get("searched_at", ""),
                    ))
                    continue
            
            # 执行搜索
            result = self.search_company(
                company_id=company_id,
                company_name=company_name,
                company_details=company_details,
                max_results=max_results,
            )
            results.append(result)
            
            # 保存缓存
            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = cache_dir / f"{company_id}.json"
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "company_id": result.company_id,
                        "company_name": result.company_name,
                        "query_used": result.query_used,
                        "search_summary": result.search_summary,
                        "citations": result.citations,
                        "is_valid": result.is_valid,
                        "searched_at": result.searched_at,
                    }, f, ensure_ascii=False, indent=2)
                # Update index
                update_web_search_index(cache_dir, result)

            # 添加延迟
            if delay_seconds > 0:
                time.sleep(delay_seconds)

        return results

    def batch_search_concurrent(
        self,
        companies: List[Dict[str, str]],
        max_results: int = 10,
        skip_existing: bool = True,
        cache_dir: Optional[Path] = None,
        show_progress: bool = True,
        concurrency: int = 20,
    ) -> List[WebSearchResult]:
        """并发批量搜索公司信息

        Args:
            companies: 公司列表 [{"company_id": "...", "company_name": "...", "company_details": "..."}]
            max_results: 每次搜索的结果数量
            skip_existing: 是否跳过已有缓存
            cache_dir: 缓存目录
            show_progress: 是否显示进度
            concurrency: 并发数（默认20）

        Returns:
            WebSearchResult 列表
        """
        from tqdm import tqdm

        results = []
        tasks = []

        # 创建缓存目录
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        # 筛选需要处理的公司
        for company in companies:
            company_id = company["company_id"]

            # 检查缓存
            if skip_existing and cache_dir:
                cache_file = cache_dir / f"{company_id}.json"
                if cache_file.exists():
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    results.append(WebSearchResult(
                        company_id=cached["company_id"],
                        company_name=cached["company_name"],
                        query_used=cached.get("query_used", ""),
                        search_summary=cached["search_summary"],
                        citations=cached.get("citations", []),
                        is_valid=cached.get("is_valid", True),
                        searched_at=cached.get("searched_at", ""),
                    ))
                    continue

            tasks.append(company)

        if not tasks:
            return results

        def search_and_cache(company: Dict[str, str]) -> WebSearchResult:
            """搜索单个公司并保存缓存"""
            company_id = company["company_id"]
            company_name = company["company_name"]
            company_details = company.get("company_details", "")

            result = self.search_company(
                company_id=company_id,
                company_name=company_name,
                company_details=company_details,
                max_results=max_results,
            )

            # 保存缓存
            if cache_dir:
                cache_file = cache_dir / f"{company_id}.json"
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "company_id": result.company_id,
                        "company_name": result.company_name,
                        "query_used": result.query_used,
                        "search_summary": result.search_summary,
                        "citations": result.citations,
                        "is_valid": result.is_valid,
                        "searched_at": result.searched_at,
                    }, f, ensure_ascii=False, indent=2)
                # Update index (thread-safe via file locking in OS)
                update_web_search_index(cache_dir, result)

            return result

        # 并发执行
        if show_progress:
            print(f"Searching {len(tasks)} companies with {concurrency} workers...")
            print(f"(Skipped {len(results)} cached results)")

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(search_and_cache, c): c for c in tasks}

            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(tasks), desc="Web searching (parallel)")

            for future in iterator:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    company = futures[future]
                    print(f"Error searching {company.get('company_id')}: {e}")

        return results


def load_web_search_cache(cache_dir: Path) -> Dict[str, WebSearchResult]:
    """加载 web search 缓存

    Args:
        cache_dir: 缓存目录

    Returns:
        {company_id: WebSearchResult}
    """
    results = {}

    if not cache_dir.exists():
        return results

    for cache_file in cache_dir.glob("*.json"):
        if cache_file.name == "index.json":
            continue  # Skip index file
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            results[data["company_id"]] = WebSearchResult(
                company_id=data["company_id"],
                company_name=data["company_name"],
                query_used=data.get("query_used", ""),
                search_summary=data["search_summary"],
                citations=data.get("citations", []),
                is_valid=data.get("is_valid", True),
                searched_at=data.get("searched_at", ""),
            )
        except Exception:
            continue

    return results


def load_web_search_index(cache_dir: Path) -> Dict[str, dict]:
    """加载 web search 缓存索引

    Args:
        cache_dir: 缓存目录

    Returns:
        {company_id: {"company_name": str, "searched_at": str, "is_valid": bool}}
    """
    index_file = cache_dir / "index.json"
    if not index_file.exists():
        return {}

    try:
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("companies", {})
    except Exception:
        return {}


def save_web_search_index(cache_dir: Path, index: Dict[str, dict]) -> None:
    """保存 web search 缓存索引

    Args:
        cache_dir: 缓存目录
        index: {company_id: {"company_name": str, "searched_at": str, "is_valid": bool}}
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_file = cache_dir / "index.json"

    with open(index_file, "w", encoding="utf-8") as f:
        json.dump({
            "updated_at": datetime.now().isoformat(),
            "total_companies": len(index),
            "valid_count": sum(1 for v in index.values() if v.get("is_valid", False)),
            "companies": index,
        }, f, ensure_ascii=False, indent=2)


def update_web_search_index(cache_dir: Path, result: WebSearchResult) -> None:
    """更新单个公司的缓存索引

    Args:
        cache_dir: 缓存目录
        result: WebSearchResult
    """
    index = load_web_search_index(cache_dir)
    index[result.company_id] = {
        "company_name": result.company_name,
        "searched_at": result.searched_at,
        "is_valid": result.is_valid,
    }
    save_web_search_index(cache_dir, index)


def get_stale_companies(
    cache_dir: Path,
    max_age_days: int = 30,
    company_ids: Optional[List[str]] = None,
) -> List[str]:
    """获取需要刷新的公司列表

    Args:
        cache_dir: 缓存目录
        max_age_days: 最大缓存天数（默认30天）
        company_ids: 可选，只检查指定公司

    Returns:
        需要刷新的 company_id 列表
    """
    from datetime import timedelta

    index = load_web_search_index(cache_dir)
    cutoff = datetime.now() - timedelta(days=max_age_days)
    stale = []

    check_ids = company_ids if company_ids else list(index.keys())

    for cid in check_ids:
        if cid not in index:
            stale.append(cid)
            continue

        entry = index[cid]
        searched_at = entry.get("searched_at", "")
        if not searched_at:
            stale.append(cid)
            continue

        try:
            search_time = datetime.fromisoformat(searched_at)
            if search_time < cutoff:
                stale.append(cid)
        except ValueError:
            stale.append(cid)

    return stale

