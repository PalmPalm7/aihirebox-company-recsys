"""
Test script for comparing LLM models WITH web search enabled for better team_background grounding.

Uses the :online suffix to enable real-time web search via OpenRouter.
See: https://openrouter.ai/docs/guides/features/plugins/web-search

Models tested:
- openai/gpt-5-mini:online
- google/gemini-2.5-flash:online
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from company_tagging import (
    CompanyRecord,
    CompanyTags,
    load_companies_from_csv,
    save_results_csv,
    save_results_json,
    print_summary,
    TAG_TAXONOMY,
)


# Models to test - with :online suffix for web search
MODELS_TO_TEST = [
    # With web search enabled
    ("openai/gpt-5-mini:online", "gpt4o-mini-online"),
    ("google/gemini-2.5-flash:online", "gemini-flash-online"),
    # Without web search (baseline)
    ("openai/gpt-5-mini", "gpt4o-mini-baseline"),
    ("google/gemini-2.5-flash", "gemini-flash-baseline"),
]


class WebSearchTagger:
    """Company tagger with web search capability for better grounding."""
    
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.is_online = ":online" in model
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with emphasis on team_background research."""
        taxonomy_desc = []
        for dim_key, dim_info in TAG_TAXONOMY.items():
            options_str = ", ".join(dim_info["options"])
            multi = "可多选" if dim_info["multi_select"] else "单选"
            taxonomy_desc.append(
                f"- {dim_key} ({dim_info['name_zh']}): [{options_str}] ({multi})"
            )
        
        web_search_note = ""
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

        return f"""你是一个专业的公司分析师，擅长从公司介绍中提取结构化标签。

请根据以下维度对公司进行分类标注，每个维度从给定选项中选择最合适的标签：

{chr(10).join(taxonomy_desc)}
{web_search_note}
输出要求：
1. 严格使用给定的标签选项，不要创造新标签
2. 对于多选维度，选择所有适用的标签
3. 对于单选维度，只选择最主要的一个
4. 【重点】team_background 维度要尽量准确，如果能从公司介绍或搜索结果中找到创始人背景信息，请不要使用 "unknown"
5. 输出JSON格式，包含所有维度的标签和简短推理

team_background 标签说明：
- bigtech_alumni: 创始人/核心团队来自大厂（BAT、字节、Google、Meta、微软等）
- top_university: 创始人/核心团队毕业于顶尖高校（清华、北大、斯坦福、MIT、哈佛等）
- serial_entrepreneur: 创始人有多次创业经历
- academic: 学术背景（教授、博士、研究员创业）
- industry_expert: 行业专家背景
- international: 海归或国际化背景
- unknown: 只有在完全无法获取信息时才使用

JSON格式示例：
{{
  "industry": ["ai_llm", "enterprise_saas"],
  "business_model": ["b2b", "saas"],
  "target_market": ["china_domestic", "global"],
  "company_stage": "early",
  "tech_focus": ["llm_foundation", "aigc"],
  "team_background": ["bigtech_alumni", "top_university"],
  "confidence": 0.85,
  "reasoning": "该公司专注于AI大模型开发，创始人来自字节跳动，毕业于清华大学..."
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
请输出JSON格式的标签结果，特别注意准确识别team_background。"""

    def tag_company(self, company: CompanyRecord) -> CompanyTags:
        """Tag a single company with optional web search."""
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_user_prompt(company)},
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            content = response.choices[0].message.content or "{}"
            
            # Parse response
            data = json.loads(content)
            
            def validate_tags(tags: Any, dimension: str) -> List[str]:
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
        except Exception as e:
            return CompanyTags(
                company_id=company.company_id,
                company_name=company.company_name,
                raw_reasoning=f"Error: {str(e)}",
            )

    def tag_companies(
        self, 
        companies: List[CompanyRecord], 
        max_workers: int = 2,
        show_progress: bool = True,
    ) -> List[CompanyTags]:
        """Tag multiple companies sequentially (to avoid rate limits with web search)."""
        results = []
        iterator = companies
        if show_progress:
            iterator = tqdm(companies, desc=f"Tagging with {self.model}")
        
        for company in iterator:
            result = self.tag_company(company)
            results.append(result)
            # Small delay to avoid rate limiting, especially for web search
            if self.is_online:
                time.sleep(0.5)
        
        return results


def calculate_team_metrics(results: List[CompanyTags]) -> Dict[str, Any]:
    """Calculate detailed metrics for team_background."""
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


def run_model_test(
    model: str,
    model_name: str,
    companies: List[CompanyRecord],
    api_key: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run tagging test for a single model."""
    is_online = ":online" in model
    print(f"\n{'='*60}")
    print(f"Testing: {model_name} {'(with web search)' if is_online else '(baseline)'}")
    print(f"{'='*60}")
    
    # Create model-specific output directory
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tagger
    tagger = WebSearchTagger(api_key=api_key, model=model)
    
    # Track timing
    start_time = time.time()
    
    try:
        results = tagger.tag_companies(companies, show_progress=True)
        success = True
        error_msg = None
    except Exception as e:
        results = []
        success = False
        error_msg = str(e)
        print(f"Error: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Save results
    if success and results:
        save_results_csv(results, model_output_dir / "company_tags.csv")
        save_results_json(results, model_output_dir / "company_tags.json")
        
        # Calculate team metrics
        team_metrics = calculate_team_metrics(results)
        
        print(f"\n--- Team Background Analysis ---")
        print(f"Coverage: {team_metrics['coverage_rate']*100:.1f}% ({team_metrics['known_team_background']}/{team_metrics['total_companies']})")
        print(f"Distribution: {team_metrics['tag_distribution']}")
    
    # Build result
    test_result = {
        "model": model,
        "model_name": model_name,
        "web_search_enabled": is_online,
        "success": success,
        "error": error_msg,
        "duration_seconds": round(duration, 2),
        "companies_processed": len(results) if results else 0,
        "output_dir": str(model_output_dir),
        "timestamp": datetime.now().isoformat(),
    }
    
    if success and results:
        team_metrics = calculate_team_metrics(results)
        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        
        test_result.update({
            "avg_confidence": round(avg_confidence, 3),
            "team_coverage": team_metrics["coverage_rate"],
            "team_distribution": team_metrics["tag_distribution"],
        })
    
    return test_result


def compare_results(all_results: List[Dict[str, Any]], output_dir: Path) -> None:
    """Generate comparison report with focus on team_background."""
    print("\n" + "="*70)
    print("COMPARISON SUMMARY - TEAM BACKGROUND FOCUS")
    print("="*70)
    
    # Header
    print(f"\n{'Model':<25} {'Web Search':<12} {'Time(s)':<10} {'Team Coverage':<15} {'Confidence':<12}")
    print("-" * 74)
    
    for result in all_results:
        model = result["model_name"][:23]
        web = "✓" if result["web_search_enabled"] else "✗"
        duration = f"{result['duration_seconds']:.1f}"
        team_cov = f"{result.get('team_coverage', 0)*100:.1f}%" if result["success"] else "N/A"
        conf = f"{result.get('avg_confidence', 0):.2f}" if result["success"] else "N/A"
        
        print(f"{model:<25} {web:<12} {duration:<10} {team_cov:<15} {conf:<12}")
    
    # Save comparison
    comparison_file = output_dir / "websearch_comparison.json"
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_timestamp": datetime.now().isoformat(),
            "focus": "team_background improvement with web search",
            "models_tested": len(all_results),
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nComparison saved to: {comparison_file}")


def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is required")
    
    # Load sample companies
    sample_csv = Path("aihirebox_company_list_sample.csv")
    if not sample_csv.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_csv}")
    
    print(f"Loading companies from {sample_csv}...")
    companies = load_companies_from_csv(sample_csv)
    
    # Limit to fewer companies for web search test (to control costs)
    companies = companies[:10]  # Test with 10 companies
    print(f"Testing with {len(companies)} companies (limited for web search cost control)")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / f"websearch_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Run tests
    all_results = []
    for model, model_name in MODELS_TO_TEST:
        try:
            result = run_model_test(model, model_name, companies, api_key, output_dir)
            all_results.append(result)
        except Exception as e:
            print(f"Failed to test {model_name}: {e}")
            all_results.append({
                "model": model,
                "model_name": model_name,
                "web_search_enabled": ":online" in model,
                "success": False,
                "error": str(e),
                "duration_seconds": 0,
                "companies_processed": 0,
            })
        
        time.sleep(3)  # Delay between models
    
    # Generate comparison
    compare_results(all_results, output_dir)
    
    print(f"\n✅ Web search testing complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

