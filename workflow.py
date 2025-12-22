import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class JobRecord:
    company_id: str
    company_name: str
    job_id: str
    job_name: str
    job_description: str
    extras: Dict[str, Any]


@dataclass
class CompanyInfo:
    name: str
    scale: str
    location: str
    details: str


class AgenticWorkflow:
    def __init__(self, bocha_api_key: str, openrouter_api_key: str, fallback_openrouter_api_key: Optional[str] = None,
                 openrouter_base_url: str = "https://openrouter.ai/api/v1", search_count: int = 5,
                 company_list_path: Optional[Path] = None):
        self.bocha_api_key = bocha_api_key
        self.search_count = search_count
        self.openrouter_client = OpenAI(
            api_key=openrouter_api_key,
            base_url=openrouter_base_url,
        )
        self.fallback_openrouter_api_key = fallback_openrouter_api_key
        self.openrouter_base_url = openrouter_base_url
        self.company_list = self._load_company_list(company_list_path) if company_list_path else []

    def parse_record(self, raw: str) -> JobRecord:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return self._record_from_dict(parsed)
            if isinstance(parsed, list):
                return self._record_from_list(parsed)
        except json.JSONDecodeError:
            pass

        parts = [segment.strip() for segment in raw.split(",") if segment.strip()]
        if len(parts) < 5:
            raise ValueError(
                "Expected at least five fields (company_id, company_name, job_id, job_name, job_description)."
            )

        company_id, company_name, job_id, job_name, *rest = parts
        job_description = rest[0] if rest else ""
        extras = {f"extra_{i}": value for i, value in enumerate(rest[1:], start=1)}
        return JobRecord(
            company_id=company_id,
            company_name=company_name,
            job_id=job_id,
            job_name=job_name,
            job_description=job_description,
            extras=extras,
        )

    @staticmethod
    def _record_from_dict(data: Dict[str, Any]) -> JobRecord:
        company_id = str(data.get("company_id", ""))
        company_name = str(data.get("company_name", ""))
        job_id = str(data.get("job_id", ""))
        job_name = str(data.get("job_name", ""))
        job_description = str(data.get("job_description", data.get("description", "")))
        extras = {k: v for k, v in data.items() if k not in {"company_id", "company_name", "job_id", "job_name", "job_description", "description"}}
        return JobRecord(
            company_id=company_id,
            company_name=company_name,
            job_id=job_id,
            job_name=job_name,
            job_description=job_description,
            extras=extras,
        )

    @staticmethod
    def _record_from_list(data: List[Any]) -> JobRecord:
        if len(data) < 5:
            raise ValueError("List input requires at least five items.")
        company_id, company_name, job_id, job_name, job_description, *rest = data
        extras = {f"extra_{i}": value for i, value in enumerate(rest, start=1)}
        return JobRecord(
            company_id=str(company_id),
            company_name=str(company_name),
            job_id=str(job_id),
            job_name=str(job_name),
            job_description=str(job_description),
            extras=extras,
        )

    def _load_company_list(self, company_list_path: Path) -> List[CompanyInfo]:
        """Load and parse company list from txt file."""
        if not company_list_path.exists():
            return []
        
        companies = []
        content = company_list_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        
        # Skip header line
        current_company = []
        for line in lines[1:]:
            if line.startswith("logo\t"):
                # Process previous company if exists
                if current_company:
                    parts = current_company[0].split("\t")
                    if len(parts) >= 4:
                        name = parts[1].strip()
                        scale = parts[2].strip()
                        location = parts[3].strip()
                        details = "\n".join(current_company[1:]).strip()
                        if name:  # Only add if company has a name
                            companies.append(CompanyInfo(
                                name=name,
                                scale=scale,
                                location=location,
                                details=details
                            ))
                # Start new company
                current_company = [line]
            elif current_company:
                # Continue collecting lines for current company
                current_company.append(line)
        
        # Process last company
        if current_company:
            parts = current_company[0].split("\t")
            if len(parts) >= 4:
                name = parts[1].strip()
                scale = parts[2].strip()
                location = parts[3].strip()
                details = "\n".join(current_company[1:]).strip()
                if name:
                    companies.append(CompanyInfo(
                        name=name,
                        scale=scale,
                        location=location,
                        details=details
                    ))
        
        return companies

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = self.openrouter_client.chat.completions.create(
                model="gpt-5-mini",
                messages=messages,
            )
            return response.choices[0].message.content or ""
        except Exception as first_error:  # noqa: BLE001
            if not self.fallback_openrouter_api_key:
                raise
            backup_client = OpenAI(api_key=self.fallback_openrouter_api_key, base_url=self.openrouter_base_url)
            response = backup_client.chat.completions.create(
                model="gpt-5-mini",
                messages=messages,
            )
            return response.choices[0].message.content or ""

    def generate_search_query(self, record: JobRecord) -> str:
        # Detect if this is a company research query
        is_company_research = "公司咨询" in record.job_name or "公司调研" in record.job_name or "company research" in record.job_name.lower()
        
        if is_company_research:
            system_prompt = (
                "You craft precise Chinese web search queries to gather comprehensive information about a specific company. "
                "Focus on company background, business model, products, market position, funding, team, and recent news."
            )
            user_prompt = (
                "Given the company information, propose a single Bing-friendly search query to find detailed company information. "
                "Keep it focused and specific to the company name.\n"
                f"company_name: {record.company_name}\n"
                "Generate a query that will find: company overview, business details, news, and market insights."
            )
        else:
            system_prompt = (
                "You craft concise Chinese web search queries to gather public insights about companies and jobs. "
                "Focus on discovering similar companies, adjacent job titles, and talent movements."
            )
            
            company_context = ""
            if self.company_list:
                company_names = [c.name for c in self.company_list[:10]]  # Limit to first 10
                company_context = f"\n\nPrioritize finding information about these companies: {', '.join(company_names)}"
            
            user_prompt = (
                "Given the following structured data, propose a single Bing-friendly query. "
                "Keep it under 18 Chinese characters if possible, avoid quotes, and prefer niche keywords.\n"
                f"company_id: {record.company_id}\n"
                f"company_name: {record.company_name}\n"
                f"job_id: {record.job_id}\n"
                f"job_name: {record.job_name}\n"
                f"job_description: {record.job_description}\n"
                f"extras: {json.dumps(record.extras, ensure_ascii=False)}"
                f"{company_context}"
            )
        
        return self._call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]).strip()

    def search_web(self, query: str) -> List[Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.bocha_api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "query": query,
            "count": self.search_count,
            "freshness": "noLimit",
            "summary": True,
        }
        response = requests.post("https://api.bocha.cn/v1/web-search", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        pages = data.get("webPages", {}).get("value", [])
        return [
            {
                "title": page.get("name"),
                "url": page.get("url"),
                "snippet": page.get("snippet") or page.get("summary"),
                "site": page.get("siteName"),
            }
            for page in pages
        ]

    def generate_article(self, record: JobRecord, query: str, search_results: List[Dict[str, Any]]) -> str:
        # Detect if this is a company research query
        is_company_research = "公司咨询" in record.job_name or "公司调研" in record.job_name or "company research" in record.job_name.lower()
        
        if is_company_research:
            # Generate professional company research report
            system_prompt = (
                "你是一名专业的商业分析师和咨询顾问。"
                "请基于搜索结果，撰写一份详实、专业的公司咨询报告，约500-800字。"
                "报告应包含：公司背景、业务模式、产品服务、市场定位、竞争优势、发展动态等方面。"
                "使用专业、客观的语言，提供有价值的商业洞察。可以包含 Markdown 链接作为信息来源。"
            )
            context_lines = [
                f"研究对象：{record.company_name}",
                f"查询关键词：{query}",
                "\n信息来源：",
            ]
            for idx, item in enumerate(search_results, start=1):
                line = f"{idx}. {item.get('title')} - {item.get('snippet')} ({item.get('url')})"
                context_lines.append(line)
            
            context = "\n".join(context_lines)
            
            user_prompt = (
                f"基于以上搜索结果，撰写一份关于{record.company_name}的专业公司咨询报告。"
                "\n报告结构建议："
                "\n1. 公司概况（成立时间、总部位置、规模等）"
                "\n2. 业务与产品（核心业务、主要产品/服务）"
                "\n3. 市场定位（目标市场、行业地位）"
                "\n4. 竞争优势（技术、团队、资源等）"
                "\n5. 最新动态（融资、产品、合作等）"
                "\n6. 发展前景（机遇与挑战）"
                "\n\n请确保信息准确、分析深入，避免空泛描述。"
            )
        else:
            # Generate job recommendation article in Xiaohongshu style
            system_prompt = (
                "你是一名擅长写小红书风格笔记的内容创作者。"
                "请用友好、轻松的语气写一段约300字的中文介绍，"
                "帮求职者发现与给定公司和岗位相似、值得关注的其他公司和岗位。"
                "输出可以包含可点击的 Markdown 链接。"
            )
            context_lines = [
                f"查询词：{query}",
                f"公司：{record.company_name} ({record.company_id})，岗位：{record.job_name} ({record.job_id})",
                f"岗位描述：{record.job_description}",
                "搜索发现：",
            ]
            for idx, item in enumerate(search_results, start=1):
                line = f"{idx}. {item.get('title')} - {item.get('snippet')} ({item.get('url')})"
                context_lines.append(line)
            
            # Add company list context
            if self.company_list:
                context_lines.append("\n重点推荐的公司（优先提及这些公司）：")
                for idx, company in enumerate(self.company_list[:15], start=1):  # Top 15 companies
                    context_lines.append(f"{idx}. {company.name}")
                    if company.location:
                        context_lines.append(f"   位置：{company.location}")
                    if company.details:
                        # Truncate details to first 200 chars for context
                        details_preview = company.details[:200] + "..." if len(company.details) > 200 else company.details
                        context_lines.append(f"   简介：{details_preview}")
            
            context = "\n".join(context_lines)

            user_prompt = (
                "基于以上素材，写出一篇小红书风格的文章。"
                "优先推荐和介绍'重点推荐的公司'列表中的公司，这些公司与目标岗位高度相关。"
                "多给出同行业、相似岗位的灵感和建议，鼓励读者探索。"
                "不要重复原文，避免空话，直接给干货。"
            )

        return self._call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context + "\n\n" + user_prompt},
        ]).strip()

    def run(self, raw_records: List[str], output_dir: Path) -> List[Path]:
        written_files: List[Path] = []
        for raw in raw_records:
            record = self.parse_record(raw)
            query = self.generate_search_query(record)
            search_results = self.search_web(query)
            article = self.generate_article(record, query, search_results)

            safe_company = record.company_id or record.company_name.replace(" ", "_")
            safe_job = record.job_id or record.job_name.replace(" ", "_")
            filename = f"article_{safe_company}_{safe_job}.md"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / filename
            output_path.write_text(article, encoding="utf-8")
            written_files.append(output_path)
        return written_files


def load_keys() -> Dict[str, Optional[str]]:
    load_dotenv()
    bocha_key = os.getenv("BOCHAAI_API_KEY")
    primary_openrouter_key = os.getenv("OPENROUTER_API_KEY")
    fallback_openrouter_key = os.getenv("OPENROUTER_FALLBACK_API_KEY")
    if not bocha_key:
        raise EnvironmentError("BOCHAAI_API_KEY is required.")
    if not primary_openrouter_key:
        raise EnvironmentError("OPENROUTER_API_KEY is required.")
    return {
        "bocha": bocha_key,
        "openrouter": primary_openrouter_key,
        "openrouter_fallback": fallback_openrouter_key,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agentic workflow to generate company/job articles.")
    parser.add_argument(
        "records",
        nargs="+",
        help="Input records describing company and job data. Accepts JSON strings, comma-separated values, or paths to JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to write generated articles.",
    )
    return parser.parse_args()


def collect_inputs(args: argparse.Namespace) -> List[str]:
    records: List[str] = []
    for entry in args.records:
        potential_path = Path(entry)
        if potential_path.exists() and potential_path.is_file():
            file_content = potential_path.read_text(encoding="utf-8")
            try:
                loaded = json.loads(file_content)
                if isinstance(loaded, list):
                    records.extend([json.dumps(item, ensure_ascii=False) for item in loaded])
                    continue
            except json.JSONDecodeError:
                pass
            records.append(file_content)
        else:
            records.append(entry)
    return records


def main() -> None:
    args = parse_args()
    env_keys = load_keys()
    
    # Look for company_list.txt in current directory
    company_list_path = Path("company_list.txt")
    if not company_list_path.exists():
        company_list_path = None
    
    workflow = AgenticWorkflow(
        bocha_api_key=env_keys["bocha"],
        openrouter_api_key=env_keys["openrouter"],
        fallback_openrouter_api_key=env_keys["openrouter_fallback"],
        company_list_path=company_list_path,
    )
    
    if company_list_path and workflow.company_list:
        print(f"Loaded {len(workflow.company_list)} companies from company_list.txt")
    
    records = collect_inputs(args)
    paths = workflow.run(records, args.output_dir)
    for path in paths:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
