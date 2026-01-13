#!/usr/bin/env python
"""
为cid_143（智谱清言）手动创建rerank_cache文件

手动指定6个AI大模型领域的候选公司作为candidates，
绕过正常的召回和精排流程。
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path


# 配置
QUERY_COMPANY_ID = "cid_143"
CANDIDATE_COMPANY_IDS = [
    "cid_100",  # MiniMax
    "cid_114",  # 月之暗面
    "cid_113",  # 百川智能
    "cid_109",  # DeepSeek
    "cid_55",   # 上海人工智能实验室
    "cid_117",  # 零一万物
]

# 规则配置
RULES = [
    {
        "rule_id": "R1_industry",
        "rule_name": "同行业公司",
        "narrative_angle": "中国AI大模型六大独角兽：智谱清言与MiniMax、月之暗面、百川智能、DeepSeek、上海AI Lab、零一万物的技术路径与商业化探索"
    },
    {
        "rule_id": "R3_industry_market",
        "rule_name": "同行业同市场",
        "narrative_angle": "中国通用大模型市场的激烈竞争：智谱清言如何在六大竞争对手中构建差异化优势"
    },
    {
        "rule_id": "R4_team_background",
        "rule_name": "同团队背景",
        "narrative_angle": "清华系与顶尖高校AI人才的创业浪潮：从智谱清言看中国AI创业者的学术背景与技术积累"
    },
]

# 为每个候选公司的选择理由
SELECTION_REASONS = {
    "cid_100": "MiniMax是中国AI大模型六小龙之一，拥有文本、语音、视觉多模态融合能力，是智谱清言在通用AGI领域的直接竞争对手。",
    "cid_114": "月之暗面（Kimi）凭借超长上下文窗口技术在C端市场快速崛起，与智谱清言形成差异化竞争格局。",
    "cid_113": "百川智能由搜狗创始人王小川创立，专注于中文大模型，与智谱清言在技术路线和商业化方向上有诸多可比性。",
    "cid_109": "DeepSeek以开源策略和极致性价比著称，其技术创新（如MoE架构）对智谱清言形成差异化竞争压力。",
    "cid_55": "上海人工智能实验室是国家级AI研究机构，书生系列大模型代表了学术界的技术实力，与智谱清言的清华背景形成有趣对照。",
    "cid_117": "零一万物由李开复创立，是中国AI创业的标杆企业之一，与智谱清言同属通用大模型赛道的头部玩家。",
}


def load_company_data(csv_path: Path) -> dict:
    """加载公司数据"""
    df = pd.read_csv(csv_path)
    company_data = {}
    for _, row in df.iterrows():
        company_data[row['company_id']] = {
            'company_name': row['company_name'],
            'location': row['location'],
            'company_details': row['company_details']
        }
    return company_data


def create_rerank_result(
    query_company_id: str,
    query_company_name: str,
    rule_id: str,
    rule_name: str,
    narrative_angle: str,
    selected_companies: list
) -> dict:
    """创建rerank结果"""
    return {
        "query_company_id": query_company_id,
        "query_company_name": query_company_name,
        "rule_id": rule_id,
        "rule_name": rule_name,
        "narrative_angle": narrative_angle,
        "selected_companies": selected_companies,
        "reranked_at": datetime.now().isoformat()
    }


def main():
    # 路径配置
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "aihirebox_company_list.csv"
    output_dir = project_root / "outputs" / "output_production" / "article_generator" / "rerank_cache"

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载公司数据
    print(f"Loading company data from {csv_path}...")
    company_data = load_company_data(csv_path)

    # 获取查询公司信息
    if QUERY_COMPANY_ID not in company_data:
        print(f"Error: Query company {QUERY_COMPANY_ID} not found in CSV")
        return

    query_company = company_data[QUERY_COMPANY_ID]
    query_company_name = query_company['company_name']
    print(f"Query company: {query_company_name} ({QUERY_COMPANY_ID})")

    # 构建候选公司列表
    selected_companies = []
    for cid in CANDIDATE_COMPANY_IDS:
        if cid not in company_data:
            print(f"Warning: Candidate company {cid} not found in CSV, skipping")
            continue

        company = company_data[cid]
        selected_companies.append({
            "company_id": cid,
            "company_name": company['company_name'],
            "location": company['location'],
            "company_details": company['company_details'],
            "selection_reason": SELECTION_REASONS.get(cid, "AI大模型领域的重要竞争对手")
        })
        print(f"  - {company['company_name']} ({cid})")

    print(f"\nTotal candidates: {len(selected_companies)}")

    # 为每个规则创建rerank_cache文件
    for rule in RULES:
        rerank_result = create_rerank_result(
            query_company_id=QUERY_COMPANY_ID,
            query_company_name=query_company_name,
            rule_id=rule["rule_id"],
            rule_name=rule["rule_name"],
            narrative_angle=rule["narrative_angle"],
            selected_companies=selected_companies
        )

        output_file = output_dir / f"{QUERY_COMPANY_ID}_{rule['rule_id']}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(rerank_result, f, ensure_ascii=False, indent=2)

        print(f"Created: {output_file}")

    print(f"\nDone! Created {len(RULES)} rerank cache files for {query_company_name}")
    print(f"\nNext step: Run article writer with:")
    print(f"  python run_article_writer.py \\")
    print(f"    --rerank-dir {output_dir} \\")
    print(f"    --web-cache-dir cache/web_search \\")
    print(f"    --company-csv data/aihirebox_company_list.csv \\")
    print(f"    --output-dir outputs/output_production/article_generator/articles \\")
    print(f"    --company-ids {QUERY_COMPANY_ID} \\")
    print(f"    --styles 36kr \\")
    print(f"    --concurrency 1")


if __name__ == "__main__":
    main()
