"""
Company Tagging Script - Extract MECE tags from company details for clustering.

This script processes company information and extracts structured tags across
multiple dimensions, enabling downstream clustering and analysis tasks.

Tag Dimensions (MECE):
1. Industry/Domain (行业领域)
2. Business Model (商业模式)
3. Target Market (目标市场)
4. Company Stage (发展阶段)
5. Technology Focus (技术方向)
6. Team Background (团队背景)
"""

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


# ============================================================================
# Tag Taxonomy Definition (MECE)
# ============================================================================

TAG_TAXONOMY = {
    "industry": {
        "name_zh": "行业领域",
        "name_en": "Industry/Domain",
        "options": [
            "ai_llm",           # AI/大模型
            "robotics",         # 机器人/具身智能
            "edtech",           # 教育科技
            "fintech",          # 金融科技
            "healthtech",       # 医疗健康
            "enterprise_saas",  # 企业服务/SaaS
            "ecommerce",        # 电商/零售
            "gaming",           # 游戏/娱乐
            "social",           # 社交/社区
            "semiconductor",    # 半导体/芯片
            "automotive",       # 汽车/出行
            "consumer_hw",      # 消费电子/硬件
            "cloud_infra",      # 云计算/基础设施
            "content_media",    # 内容/媒体
            "biotech",          # 生物科技
            "investment",       # 投资/金融服务
            "other",            # 其他
        ],
        "multi_select": True,
    },
    "business_model": {
        "name_zh": "商业模式",
        "name_en": "Business Model",
        "options": [
            "b2b",              # 企业服务
            "b2c",              # 消费者服务
            "b2b2c",            # 混合模式
            "platform",         # 平台模式
            "saas",             # SaaS订阅
            "hardware",         # 硬件产品
            "marketplace",      # 市场平台
            "consulting",       # 咨询服务
        ],
        "multi_select": True,
    },
    "target_market": {
        "name_zh": "目标市场",
        "name_en": "Target Market",
        "options": [
            "china_domestic",   # 国内市场
            "global",           # 全球市场
            "sea",              # 东南亚
            "us",               # 北美
            "europe",           # 欧洲
            "japan_korea",      # 日韩
            "latam",            # 拉美
            "mena",             # 中东北非
        ],
        "multi_select": True,
    },
    "company_stage": {
        "name_zh": "发展阶段",
        "name_en": "Company Stage",
        "options": [
            "seed",             # 种子轮/天使轮
            "early",            # 早期(A/B轮)
            "growth",           # 成长期(C轮+)
            "pre_ipo",          # Pre-IPO
            "public",           # 已上市
            "bigtech_subsidiary", # 大厂子公司
            "profitable",       # 已盈利
            "unknown",          # 未知
        ],
        "multi_select": False,
    },
    "tech_focus": {
        "name_zh": "技术方向",
        "name_en": "Technology Focus",
        "options": [
            "llm_foundation",   # 大语言模型/基础模型
            "computer_vision",  # 计算机视觉
            "speech_nlp",       # 语音/NLP
            "embodied_ai",      # 具身智能
            "aigc",             # AIGC/内容生成
            "3d_graphics",      # 3D/图形学
            "chip_hardware",    # 芯片/硬件
            "data_infra",       # 数据/基础设施
            "autonomous",       # 自动驾驶/自主系统
            "blockchain",       # 区块链/Web3
            "quantum",          # 量子计算
            "not_tech_focused", # 非技术驱动
        ],
        "multi_select": True,
    },
    "team_background": {
        "name_zh": "团队背景",
        "name_en": "Team Background",
        "options": [
            "bigtech_alumni",   # 大厂背景(BAT/TMD/Google等)
            "top_university",   # 顶尖高校(清北/藤校等)
            "serial_entrepreneur", # 连续创业者
            "academic",         # 学术背景/教授创业
            "industry_expert",  # 行业专家
            "international",    # 海归/国际化背景
            "unknown",          # 未知
        ],
        "multi_select": True,
    },
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CompanyTags:
    """Structured tags for a single company."""
    company_id: str
    company_name: str
    industry: List[str] = field(default_factory=list)
    business_model: List[str] = field(default_factory=list)
    target_market: List[str] = field(default_factory=list)
    company_stage: str = "unknown"
    tech_focus: List[str] = field(default_factory=list)
    team_background: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    raw_reasoning: str = ""


@dataclass 
class CompanyRecord:
    """Raw company record from CSV."""
    company_id: str
    company_name: str
    location: str
    company_details: str


# ============================================================================
# Company Tagger Class
# ============================================================================

class CompanyTagger:
    """Extract MECE tags from company details using LLM.
    
    Supports web search enhancement via OpenRouter's :online suffix for better
    team_background accuracy.
    """
    
    DEFAULT_MODEL = "openai/gpt-5-mini"
    
    def __init__(
        self,
        openrouter_api_key: str,
        fallback_api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: Optional[str] = None,
        include_reasoning: bool = True,
    ):
        self.client = OpenAI(api_key=openrouter_api_key, base_url=base_url)
        self.fallback_api_key = fallback_api_key
        self.base_url = base_url
        self.model = model or self.DEFAULT_MODEL
        self.is_online = ":online" in self.model
        self.include_reasoning = include_reasoning
    
    def _build_taxonomy_description(self) -> str:
        """Build taxonomy description for prompts."""
        taxonomy_desc = []
        for dim_key, dim_info in TAG_TAXONOMY.items():
            options_str = ", ".join(dim_info["options"])
            multi = "可多选" if dim_info["multi_select"] else "单选"
            taxonomy_desc.append(
                f"- {dim_key} ({dim_info['name_zh']}): [{options_str}] ({multi})"
            )
        return "\n".join(taxonomy_desc)
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with tag taxonomy."""
        taxonomy_desc = self._build_taxonomy_description()
        
        # Additional instructions for web search mode
        web_search_note = ""
        team_background_emphasis = ""
        
        if self.is_online:
            web_search_note = """
特别说明 - 团队背景研究：
你可以使用网络搜索来查找关于公司创始人和核心团队的信息。
请特别关注：
1. 创始人的教育背景（是否来自清华、北大、斯坦福等顶尖高校）
2. 创始人的工作经历（是否来自BAT、字节、Google等大厂）
3. 是否有连续创业经历
4. 是否有学术背景（教授、博士等）

如果你找到了相关信息，请在reasoning中引用来源。
"""
            team_background_emphasis = """
team_background 标签说明：
- bigtech_alumni: 创始人/核心团队来自大厂（BAT、字节、Google、Meta、微软等）
- top_university: 创始人/核心团队毕业于顶尖高校（清华、北大、斯坦福、MIT、哈佛等）
- serial_entrepreneur: 创始人有多次创业经历
- academic: 学术背景（教授、博士、研究员创业）
- industry_expert: 行业专家背景
- international: 海归或国际化背景
- unknown: 只有在完全无法获取信息时才使用
"""
        
        # Reasoning instruction based on include_reasoning flag
        if self.include_reasoning:
            reasoning_instruction = "5. 输出JSON格式，包含所有维度的标签和简短推理"
            reasoning_example = ',\n  "reasoning": "该公司专注于AI大模型开发，提供企业级SaaS服务，创始团队来自BAT..."'
        else:
            reasoning_instruction = "5. 输出JSON格式，只包含标签，不需要reasoning字段"
            reasoning_example = ""
        
        return f"""你是一个专业的公司分析师，擅长从公司介绍中提取结构化标签。

请根据以下维度对公司进行分类标注，每个维度从给定选项中选择最合适的标签：

{taxonomy_desc}
{web_search_note}
输出要求：
1. 严格使用给定的标签选项，不要创造新标签
2. 对于多选维度，选择所有适用的标签
3. 对于单选维度，只选择最主要的一个
4. {"【重点】team_background 维度要尽量准确，如果能从公司介绍或搜索结果中找到创始人背景信息，请不要使用 \"unknown\"" if self.is_online else "如果信息不足以判断，使用 \"unknown\" 或 \"other\""}
{reasoning_instruction}
{team_background_emphasis}
JSON格式示例：
{{
  "industry": ["ai_llm", "enterprise_saas"],
  "business_model": ["b2b", "saas"],
  "target_market": ["china_domestic", "global"],
  "company_stage": "early",
  "tech_focus": ["llm_foundation", "aigc"],
  "team_background": ["bigtech_alumni", "top_university"],
  "confidence": 0.85{reasoning_example}
}}"""

    def _build_user_prompt(self, company: CompanyRecord) -> str:
        """Build user prompt for a specific company."""
        search_hint = ""
        if self.is_online:
            search_hint = f"\n\n请搜索 \"{company.company_name} 创始人 团队\" 来获取更多关于创始团队的信息。"
        
        return f"""请分析以下公司并提取标签：

公司ID: {company.company_id}
公司名称: {company.company_name}
所在地: {company.location}
公司介绍:
{company.company_details}
{search_hint}
请输出JSON格式的标签结果。"""

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call LLM with fallback support."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,  # Lower temperature for more consistent tagging
            )
            return response.choices[0].message.content or "{}"
        except Exception:
            if not self.fallback_api_key:
                raise
            backup_client = OpenAI(api_key=self.fallback_api_key, base_url=self.base_url)
            response = backup_client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            return response.choices[0].message.content or "{}"

    def _parse_llm_response(self, company: CompanyRecord, response: str) -> CompanyTags:
        """Parse LLM JSON response into CompanyTags."""
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return CompanyTags(
                company_id=company.company_id,
                company_name=company.company_name,
                raw_reasoning=f"JSON parse error: {response[:200]}",
            )
        
        def validate_tags(tags: Any, dimension: str) -> List[str]:
            """Validate tags against taxonomy."""
            if not tags:
                return []
            if isinstance(tags, str):
                tags = [tags]
            valid_options = set(TAG_TAXONOMY[dimension]["options"])
            return [t for t in tags if t in valid_options]
        
        return CompanyTags(
            company_id=company.company_id,
            company_name=company.company_name,
            industry=validate_tags(data.get("industry"), "industry"),
            business_model=validate_tags(data.get("business_model"), "business_model"),
            target_market=validate_tags(data.get("target_market"), "target_market"),
            company_stage=data.get("company_stage", "unknown") 
                if data.get("company_stage") in TAG_TAXONOMY["company_stage"]["options"] 
                else "unknown",
            tech_focus=validate_tags(data.get("tech_focus"), "tech_focus"),
            team_background=validate_tags(data.get("team_background"), "team_background"),
            confidence_score=float(data.get("confidence", 0.0)),
            raw_reasoning=data.get("reasoning", ""),
        )

    def tag_company(self, company: CompanyRecord) -> CompanyTags:
        """Extract tags for a single company."""
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_user_prompt(company)},
        ]
        try:
            response = self._call_llm(messages)
            return self._parse_llm_response(company, response)
        except Exception as e:
            return CompanyTags(
                company_id=company.company_id,
                company_name=company.company_name,
                raw_reasoning=f"Error: {str(e)}",
            )

    def tag_companies(
        self, 
        companies: List[CompanyRecord], 
        max_workers: int = 5,
        show_progress: bool = True,
        delay_seconds: float = 0.0,
    ) -> List[CompanyTags]:
        """Extract tags for multiple companies.
        
        Args:
            companies: List of company records to process
            max_workers: Number of parallel workers (ignored if delay_seconds > 0)
            show_progress: Whether to show progress bar
            delay_seconds: Delay between requests (enables sequential mode, recommended for :online)
        
        Returns:
            List of CompanyTags results
        """
        results = []
        
        # Use sequential processing if delay is specified (for web search rate limiting)
        if delay_seconds > 0 or self.is_online:
            actual_delay = delay_seconds if delay_seconds > 0 else 0.5
            iterator = companies
            if show_progress:
                iterator = tqdm(companies, desc=f"Tagging with {self.model}")
            
            for company in iterator:
                result = self.tag_company(company)
                results.append(result)
                if actual_delay > 0:
                    time.sleep(actual_delay)
        else:
            # Use parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_company = {
                    executor.submit(self.tag_company, company): company 
                    for company in companies
                }
                
                iterator = as_completed(future_to_company)
                if show_progress:
                    iterator = tqdm(iterator, total=len(companies), desc="Tagging companies")
                
                for future in iterator:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        company = future_to_company[future]
                        results.append(CompanyTags(
                            company_id=company.company_id,
                            company_name=company.company_name,
                            raw_reasoning=f"Error: {str(e)}",
                        ))
        
        # Sort by company_id to maintain order
        results.sort(key=lambda x: x.company_id)
        return results


# ============================================================================
# Utility Functions
# ============================================================================

def load_companies_from_csv(csv_path: Path) -> List[CompanyRecord]:
    """Load company records from CSV file."""
    companies = []
    # Use utf-8-sig to handle BOM (byte order mark) if present
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            companies.append(CompanyRecord(
                company_id=row.get("company_id", ""),
                company_name=row.get("company_name", ""),
                location=row.get("location", ""),
                company_details=row.get("company_details", ""),
            ))
    return companies


def save_results_csv(results: List[CompanyTags], output_path: Path) -> None:
    """Save tagging results to CSV."""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            "company_id", "company_name",
            "industry", "business_model", "target_market",
            "company_stage", "tech_focus", "team_background",
            "confidence_score", "reasoning"
        ])
        # Data
        for result in results:
            writer.writerow([
                result.company_id,
                result.company_name,
                "|".join(result.industry),
                "|".join(result.business_model),
                "|".join(result.target_market),
                result.company_stage,
                "|".join(result.tech_focus),
                "|".join(result.team_background),
                f"{result.confidence_score:.2f}",
                result.raw_reasoning,
            ])


def save_results_json(results: List[CompanyTags], output_path: Path) -> None:
    """Save tagging results to JSON."""
    data = [asdict(r) for r in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_taxonomy(output_path: Path) -> None:
    """Save the tag taxonomy for reference."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(TAG_TAXONOMY, f, ensure_ascii=False, indent=2)


def print_summary(results: List[CompanyTags]) -> None:
    """Print summary statistics of tagging results."""
    print("\n" + "=" * 60)
    print("TAGGING SUMMARY")
    print("=" * 60)
    
    total = len(results)
    print(f"Total companies processed: {total}")
    
    # Industry distribution
    industry_counts: Dict[str, int] = {}
    for r in results:
        for tag in r.industry:
            industry_counts[tag] = industry_counts.get(tag, 0) + 1
    
    print("\nIndustry Distribution:")
    for tag, count in sorted(industry_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {tag}: {count} ({count/total*100:.1f}%)")
    
    # Stage distribution
    stage_counts: Dict[str, int] = {}
    for r in results:
        stage_counts[r.company_stage] = stage_counts.get(r.company_stage, 0) + 1
    
    print("\nCompany Stage Distribution:")
    for tag, count in sorted(stage_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count} ({count/total*100:.1f}%)")
    
    # Average confidence
    avg_conf = sum(r.confidence_score for r in results) / total if total > 0 else 0
    print(f"\nAverage confidence score: {avg_conf:.2f}")


def calculate_team_metrics(results: List[CompanyTags]) -> Dict[str, Any]:
    """Calculate detailed metrics for team_background coverage."""
    total = len(results)
    
    # Count companies with non-unknown team background
    known_count = sum(1 for r in results if r.team_background and "unknown" not in r.team_background)
    
    # Distribution of team_background tags
    tag_counts: Dict[str, int] = {}
    for r in results:
        for tag in r.team_background:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    return {
        "total_companies": total,
        "known_team_background": known_count,
        "unknown_team_background": total - known_count,
        "coverage_rate": round(known_count / total, 3) if total > 0 else 0,
        "tag_distribution": dict(sorted(tag_counts.items(), key=lambda x: -x[1])),
    }


def load_api_keys() -> Dict[str, Optional[str]]:
    """Load API keys from environment."""
    load_dotenv()
    primary_key = os.getenv("OPENROUTER_API_KEY")
    fallback_key = os.getenv("OPENROUTER_FALLBACK_API_KEY")
    if not primary_key:
        raise EnvironmentError("OPENROUTER_API_KEY is required.")
    return {
        "openrouter": primary_key,
        "openrouter_fallback": fallback_key,
    }


# ============================================================================
# CLI Entry Point
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MECE tags from company details for clustering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all companies in CSV
  python company_tagging.py data/aihirebox_company_list.csv
  
  # Specify output directory
  python company_tagging.py data/aihirebox_company_list.csv --output-dir ./tags_output
  
  # Process specific companies by ID
  python company_tagging.py data/aihirebox_company_list.csv --company-ids cid_0 cid_1 cid_2
  
  # Process companies from JSON file (supports {"company_ids": [...]} or [...])
  python company_tagging.py data/aihirebox_company_list.csv --company-ids-json my_companies.json
  
  # Combine both ID sources
  python company_tagging.py data/aihirebox_company_list.csv --company-ids cid_0 --company-ids-json more_ids.json
  
  # Limit number of companies (for testing)
  python company_tagging.py data/aihirebox_company_list.csv --limit 10
  
  # Adjust parallel workers
  python company_tagging.py data/aihirebox_company_list.csv --workers 3
  
  # Use web search for better team_background accuracy
  python company_tagging.py data/aihirebox_company_list.csv --model openai/gpt-5-mini:online
        """
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to input CSV file with company data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files (default: output/company_tags_<timestamp>)",
    )
    parser.add_argument(
        "--company-ids-json",
        type=Path,
        help="JSON file containing company IDs to process (expects {\"company_ids\": [...]} or [...])",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=CompanyTagger.DEFAULT_MODEL,
        help=f"LLM model to use (default: {CompanyTagger.DEFAULT_MODEL}). Add ':online' for web search.",
    )
    parser.add_argument(
        "--company-ids",
        nargs="+",
        help="Specific company IDs to process (processes all if not specified)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of companies to process (for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5, ignored for :online models)",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "json", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--save-taxonomy",
        action="store_true",
        help="Also save the tag taxonomy definition",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable reasoning output to reduce token cost",
    )
    return parser.parse_args()


def load_company_ids_from_json(json_path: Path) -> List[str]:
    """Load company IDs from a JSON file.
    
    Supports two formats:
    - {"company_ids": ["cid_0", "cid_1", ...]}
    - ["cid_0", "cid_1", ...]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "company_ids" in data:
        return data["company_ids"]
    else:
        raise ValueError(f"Invalid JSON format. Expected list or dict with 'company_ids' key.")


def main() -> None:
    args = parse_args()
    keys = load_api_keys()
    
    # Load companies
    print(f"Loading companies from {args.input_csv}...")
    companies = load_companies_from_csv(args.input_csv)
    print(f"Loaded {len(companies)} companies")
    
    # Collect company IDs from both sources
    company_ids_to_filter = set()
    
    # From --company-ids argument
    if args.company_ids:
        company_ids_to_filter.update(args.company_ids)
    
    # From --company-ids-json file
    if args.company_ids_json:
        if not args.company_ids_json.exists():
            print(f"Error: JSON file not found: {args.company_ids_json}")
            return
        json_ids = load_company_ids_from_json(args.company_ids_json)
        company_ids_to_filter.update(json_ids)
        print(f"Loaded {len(json_ids)} company IDs from {args.company_ids_json}")
    
    # Filter by company IDs if any were specified
    if company_ids_to_filter:
        companies = [c for c in companies if c.company_id in company_ids_to_filter]
        print(f"Filtered to {len(companies)} companies by ID")
    
    # Apply limit if specified
    if args.limit:
        companies = companies[:args.limit]
        print(f"Limited to {len(companies)} companies")
    
    if not companies:
        print("No companies to process!")
        return
    
    # Initialize tagger
    tagger = CompanyTagger(
        openrouter_api_key=keys["openrouter"],
        fallback_api_key=keys["openrouter_fallback"],
        model=args.model,
        include_reasoning=not args.no_reasoning,
    )
    
    # Process companies
    is_online = ":online" in args.model
    print(f"\nModel: {args.model}")
    print(f"Web search: {'✓ Enabled' if is_online else '✗ Disabled'}")
    print(f"Reasoning: {'✗ Disabled' if args.no_reasoning else '✓ Enabled'}")
    print(f"Tagging {len(companies)} companies...")
    
    results = tagger.tag_companies(companies, max_workers=args.workers)
    
    # Create output directory with timestamp if not specified
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("output") / f"company_tags_{timestamp}"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    if args.output_format in ("csv", "both"):
        csv_path = args.output_dir / "company_tags.csv"
        save_results_csv(results, csv_path)
        print(f"Saved CSV: {csv_path}")
    
    if args.output_format in ("json", "both"):
        json_path = args.output_dir / "company_tags.json"
        save_results_json(results, json_path)
        print(f"Saved JSON: {json_path}")
    
    if args.save_taxonomy:
        taxonomy_path = args.output_dir / "tag_taxonomy.json"
        save_taxonomy(taxonomy_path)
        print(f"Saved taxonomy: {taxonomy_path}")
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
