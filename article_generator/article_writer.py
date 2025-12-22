"""
Article Writer - Layer 2

使用 google/gemini-3-flash-preview via OpenRouter 生成文章。
读取 rerank_cache 和 web_search_cache，根据风格模板生成文章。

支持 5 种风格:
- 36kr: 专业、数据驱动、行业分析
- huxiu: 犀利、有态度、深度评论
- xiaohongshu: 轻松、口语化、分点列举
- linkedin: 职场视角、强调机会
- zhihu: 知识分享、逻辑清晰
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .models import Article, ArticleStyle, RerankResult, WebSearchResult
from .styles import ARTICLE_STYLES, get_style


class ArticleWriter:
    """文章生成器
    
    基于精排结果和 Web 搜索缓存，按指定风格生成文章。
    
    Example:
        writer = ArticleWriter()
        article = writer.write_article(rerank_result, style="36kr", web_search_cache={})
    """
    
    DEFAULT_MODEL = "google/gemini-3-flash-preview"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """初始化文章生成器
        
        Args:
            api_key: OpenRouter API key
            model: 模型名称（默认 gemini-3-flash-preview）
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.model = model or self.DEFAULT_MODEL
    
    def _build_article_prompt(
        self,
        rerank_result: RerankResult,
        style: ArticleStyle,
        web_search_cache: Dict[str, WebSearchResult],
        query_company_details: str = "",
    ) -> str:
        """构建文章生成 prompt
        
        Args:
            rerank_result: 精排结果
            style: 文章风格配置
            web_search_cache: Web 搜索缓存
            query_company_details: 查询公司的详细信息
        """
        # 构建推荐公司信息
        company_sections = []
        for i, sc in enumerate(rerank_result.selected_companies, 1):
            section = f"""
### 推荐公司 {i}: {sc.company_name}
- 公司ID: {sc.company_id}
- 地点: {sc.location}
- 公司介绍: {sc.company_details[:800] if sc.company_details else '无详细信息'}
- 选择理由: {sc.selection_reason}
"""
            # 添加 web search 结果
            if sc.company_id in web_search_cache:
                ws = web_search_cache[sc.company_id]
                if ws.is_valid:
                    section += f"""
**网络搜索补充信息:**
{ws.search_summary[:1500]}

引用来源: {', '.join(ws.citations[:5]) if ws.citations else '无'}
"""
            company_sections.append(section)
        
        companies_text = "\n".join(company_sections)
        
        # 查询公司的 web search 信息
        query_web_info = ""
        if rerank_result.query_company_id in web_search_cache:
            ws = web_search_cache[rerank_result.query_company_id]
            if ws.is_valid:
                query_web_info = f"""
**网络搜索补充信息:**
{ws.search_summary[:1500]}
"""
        
        # Emoji 指示
        emoji_instruction = ""
        if style.use_emoji:
            emoji_instruction = """
**重要**: 请在标题和正文中适当使用 emoji 增加趣味性和可读性。
常用 emoji: 🔥 ✨ 💡 📊 🚀 💰 🎯 👀 📈 ⭐ 🏆 💪
"""
        else:
            emoji_instruction = """
**重要**: 请勿使用任何 emoji，保持专业严肃的风格。
"""
        
        return f"""你是{style.name_zh}的资深撰稿人。现在手里有一组公司素材，写一篇深度稿。

## 你的写作人设
{style.tone}
{emoji_instruction}

## 写法指导
{style.structure}

## 开头可以参考这个感觉（不要照抄）
"{style.example_intro}"

---

## 素材

**你要写的主线/角度**: {rerank_result.narrative_angle}

**主角公司（重点写）**:
{rerank_result.query_company_name}
{query_company_details[:800] if query_company_details else '暂无详情'}
{query_web_info}

**相关公司（配角，灵活使用）**:
{companies_text}

---

## 注意事项
- 字数 {style.word_count_min}-{style.word_count_max} 字
- 不要写成百科介绍，要像记者写报道一样有节奏有观点
- 配角公司不需要每家都写，挑有对比价值或能推进叙事的写
- 公司之间的过渡要自然，别用"接下来介绍"这种话
- 标题要抓人，别太学术腔

## 输出
JSON格式，三个字段：
```json
{{
  "title": "标题",
  "content": "正文（markdown）",
  "key_takeaways": ["要点1", "要点2", "要点3"]
}}
```
"""
    
    def _parse_article_response(
        self,
        response_text: str,
        rerank_result: RerankResult,
        style_id: str,
    ) -> Article:
        """解析文章生成响应"""
        # 提取所有候选公司 ID
        candidate_company_ids = [
            sc.company_id for sc in rerank_result.selected_companies
        ]
        
        try:
            # 尝试提取 JSON
            json_text = response_text
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_text = response_text.split("```")[1].split("```")[0]
            
            data = json.loads(json_text.strip())
        except (json.JSONDecodeError, IndexError):
            # 解析失败，尝试直接使用响应文本
            return Article(
                query_company_id=rerank_result.query_company_id,
                query_company_name=rerank_result.query_company_name,
                rule_id=rerank_result.rule_id,
                style=style_id,
                title="生成失败",
                content=response_text,
                word_count=len(response_text),
                candidate_company_ids=candidate_company_ids,
                key_takeaways=[],
                citations=[],
            )
        
        content = data.get("content", "")
        
        return Article(
            query_company_id=rerank_result.query_company_id,
            query_company_name=rerank_result.query_company_name,
            rule_id=rerank_result.rule_id,
            style=style_id,
            title=data.get("title", "无标题"),
            content=content,
            word_count=len(content),
            candidate_company_ids=candidate_company_ids,
            key_takeaways=data.get("key_takeaways", []),
            citations=self._extract_citations(content),
        )
    
    def _extract_citations(self, text: str) -> List[str]:
        """从 markdown 文本中提取引用链接"""
        import re
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(pattern, text)
        return list(set(url for _, url in matches))
    
    def write_article(
        self,
        rerank_result: RerankResult,
        style_id: str,
        web_search_cache: Dict[str, WebSearchResult],
        query_company_details: str = "",
    ) -> Article:
        """生成单篇文章
        
        Args:
            rerank_result: 精排结果
            style_id: 风格ID (36kr, huxiu, xiaohongshu, linkedin, zhihu)
            web_search_cache: Web 搜索缓存
            query_company_details: 查询公司详情
            
        Returns:
            Article 生成的文章
        """
        style = get_style(style_id)
        if not style:
            raise ValueError(f"Unknown style: {style_id}")
        
        prompt = self._build_article_prompt(
            rerank_result=rerank_result,
            style=style,
            web_search_cache=web_search_cache,
            query_company_details=query_company_details,
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7,  # 稍高温度增加创意
            )
            
            response_text = response.choices[0].message.content or "{}"
            
            return self._parse_article_response(
                response_text=response_text,
                rerank_result=rerank_result,
                style_id=style_id,
            )
            
        except Exception as e:
            return Article(
                query_company_id=rerank_result.query_company_id,
                query_company_name=rerank_result.query_company_name,
                rule_id=rerank_result.rule_id,
                style=style_id,
                title="生成失败",
                content=f"Error: {str(e)}",
                word_count=0,
                candidate_company_ids=[sc.company_id for sc in rerank_result.selected_companies],
                key_takeaways=[],
                citations=[],
            )
    
    def batch_write(
        self,
        rerank_results: List[RerankResult],
        style_ids: List[str],
        web_search_cache: Dict[str, WebSearchResult],
        company_details_map: Dict[str, str],
        delay_seconds: float = 0.5,
        skip_existing: bool = True,
        output_dir: Optional[Path] = None,
        show_progress: bool = True,
    ) -> List[Article]:
        """批量生成文章
        
        Args:
            rerank_results: 精排结果列表
            style_ids: 风格ID列表
            web_search_cache: Web 搜索缓存
            company_details_map: {company_id: company_details}
            delay_seconds: 请求间隔
            skip_existing: 是否跳过已有文章
            output_dir: 输出目录
            show_progress: 是否显示进度
            
        Returns:
            Article 列表
        """
        from tqdm import tqdm
        
        articles = []
        
        # 展开所有 (rerank_result, style) 对
        tasks = []
        for rr in rerank_results:
            for style_id in style_ids:
                tasks.append({
                    "rerank_result": rr,
                    "style_id": style_id,
                })
        
        iterator = tasks
        if show_progress:
            iterator = tqdm(tasks, desc="Writing articles")
        
        for task in iterator:
            rr = task["rerank_result"]
            style_id = task["style_id"]
            
            # 文件名: {query_company_id}_{rule_id}_{style}.json
            filename = f"{rr.query_company_id}_{rr.rule_id}_{style_id}.json"
            
            # 检查是否已存在
            if skip_existing and output_dir:
                output_file = output_dir / filename
                if output_file.exists():
                    try:
                        with open(output_file, "r", encoding="utf-8") as f:
                            cached = json.load(f)
                        articles.append(Article(
                            query_company_id=cached["query_company_id"],
                            query_company_name=cached["query_company_name"],
                            rule_id=cached["rule_id"],
                            style=cached["style"],
                            title=cached["title"],
                            content=cached["content"],
                            word_count=cached.get("word_count", 0),
                            candidate_company_ids=cached.get("candidate_company_ids", []),
                            key_takeaways=cached.get("key_takeaways", []),
                            citations=cached.get("citations", []),
                            generated_at=cached.get("generated_at", ""),
                        ))
                        continue
                    except Exception:
                        pass
            
            # 生成文章
            article = self.write_article(
                rerank_result=rr,
                style_id=style_id,
                web_search_cache=web_search_cache,
                query_company_details=company_details_map.get(rr.query_company_id, ""),
            )
            articles.append(article)
            
            # 保存文章
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / filename
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "query_company_id": article.query_company_id,
                        "query_company_name": article.query_company_name,
                        "rule_id": article.rule_id,
                        "style": article.style,
                        "title": article.title,
                        "content": article.content,
                        "word_count": article.word_count,
                        "candidate_company_ids": article.candidate_company_ids,
                        "key_takeaways": article.key_takeaways,
                        "citations": article.citations,
                        "generated_at": article.generated_at,
                    }, f, ensure_ascii=False, indent=2)
                
                # 同时保存 markdown 版本
                md_file = output_dir / filename.replace(".json", ".md")
                with open(md_file, "w", encoding="utf-8") as f:
                    f.write(f"# {article.title}\n\n")
                    f.write(f"> 风格: {article.style} | 规则: {article.rule_id}\n")
                    f.write(f"> 推荐公司: {', '.join(article.candidate_company_ids)}\n\n")
                    f.write(article.content)
                    if article.key_takeaways:
                        f.write("\n\n---\n\n## 核心要点\n\n")
                        for takeaway in article.key_takeaways:
                            f.write(f"- {takeaway}\n")
            
            # 添加延迟
            if delay_seconds > 0:
                time.sleep(delay_seconds)
        
        return articles


def load_articles(articles_dir: Path) -> Dict[str, Article]:
    """加载已生成的文章
    
    Args:
        articles_dir: 文章目录
        
    Returns:
        {"{company_id}_{rule_id}_{style}": Article}
    """
    articles = {}
    
    if not articles_dir.exists():
        return articles
    
    for json_file in articles_dir.glob("*.json"):
        if json_file.name == "run_metadata.json":
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            key = f"{data['query_company_id']}_{data['rule_id']}_{data['style']}"
            articles[key] = Article(
                query_company_id=data["query_company_id"],
                query_company_name=data["query_company_name"],
                rule_id=data["rule_id"],
                style=data["style"],
                title=data["title"],
                content=data["content"],
                word_count=data.get("word_count", 0),
                candidate_company_ids=data.get("candidate_company_ids", []),
                key_takeaways=data.get("key_takeaways", []),
                citations=data.get("citations", []),
                generated_at=data.get("generated_at", ""),
            )
        except Exception:
            continue
    
    return articles

