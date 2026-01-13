#!/usr/bin/env python
"""
Platform Analytics Generator - 平台数据分析脚本

Generates analytics JSON with:
1. 标签分布 - Distribution of company tags across 6 dimensions
2. 推荐生态健康度 - Recommendation health metrics (Gini index, overlap)
3. 文章关联公司统计 - Article candidate statistics by rule

Usage:
    python scripts/generate_analytics.py
    python scripts/generate_analytics.py --pretty
    python scripts/generate_analytics.py --output outputs/production/analytics.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Set, Tuple


# Dimension field names and Chinese labels
DIMENSION_LABELS = {
    "industry": "行业",
    "business_model": "商业模式",
    "target_market": "目标市场",
    "company_stage": "公司阶段",
    "tech_focus": "技术方向",
    "team_background": "团队背景",
}


def calculate_gini(values: List[int]) -> float:
    """
    计算基尼系数 (0-1)
    0 = 完全平等, 1 = 完全不平等
    """
    if not values or all(v == 0 for v in values):
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)
    cumsum = 0
    for i, v in enumerate(sorted_values):
        cumsum += (2 * (i + 1) - n - 1) * v

    total = sum(sorted_values)
    if total == 0:
        return 0.0

    return cumsum / (n * total)


def calculate_jaccard(set1: Set[str], set2: Set[str]) -> float:
    """计算两个集合的Jaccard相似度"""
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def load_company_tags(tags_path: Path) -> List[Dict]:
    """加载公司标签数据"""
    if not tags_path.exists():
        print(f"Warning: Tags file not found: {tags_path}")
        return []

    with open(tags_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rerank_cache(rerank_dir: Path) -> List[Dict]:
    """加载精排缓存数据"""
    results = []
    if not rerank_dir.exists():
        print(f"Warning: Rerank cache not found: {rerank_dir}")
        return results

    for json_file in rerank_dir.glob("*.json"):
        if json_file.name == "run_metadata.json":
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return results


def load_articles(articles_dir: Path) -> List[Dict]:
    """加载文章JSON数据"""
    results = []
    json_dir = articles_dir / "json"
    if not json_dir.exists():
        print(f"Warning: Articles JSON dir not found: {json_dir}")
        return results

    for json_file in json_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return results


def compute_tag_distributions(company_tags: List[Dict]) -> Dict[str, Any]:
    """
    指标1: 计算6个维度的标签分布
    """
    total_companies = len(company_tags)
    distributions = {}

    for field, label in DIMENSION_LABELS.items():
        counter = Counter()

        for company in company_tags:
            value = company.get(field, [])
            # Handle both array and single value fields
            if isinstance(value, list):
                for v in value:
                    counter[v] += 1
            elif value:
                counter[value] += 1

        # Convert to distribution with count and percentage
        dist = {}
        for tag, count in counter.most_common():
            dist[tag] = {
                "数量": count,
                "占比": round(count / total_companies * 100, 1) if total_companies > 0 else 0
            }

        distributions[label] = dist

    return {
        "公司总数": total_companies,
        "各维度分布": distributions
    }


def compute_recommendation_health(
    rerank_results: List[Dict],
    articles: List[Dict],
    company_tags: List[Dict]
) -> Dict[str, Any]:
    """
    指标2: 推荐生态健康度
    - 被推荐频率分布 (Gini)
    - 候选重叠度
    - 按规则细分
    """
    # 2a. 被推荐频率统计
    candidate_counts = Counter()  # company_id -> count
    rule_candidate_counts = defaultdict(Counter)  # rule_id -> {company_id: count}

    for result in rerank_results:
        rule_id = result.get("rule_id", "unknown")
        selected = result.get("selected_companies", [])
        for company in selected:
            cid = company.get("company_id")
            if cid:
                candidate_counts[cid] += 1
                rule_candidate_counts[rule_id][cid] += 1

    # Create set of all company IDs
    all_company_ids = {c.get("company_id") for c in company_tags}
    recommended_ids = set(candidate_counts.keys())
    never_recommended = all_company_ids - recommended_ids

    # Frequency distribution buckets
    freq_distribution = {
        "1-5次": 0,
        "6-10次": 0,
        "11-20次": 0,
        "20次以上": 0
    }
    for count in candidate_counts.values():
        if count <= 5:
            freq_distribution["1-5次"] += 1
        elif count <= 10:
            freq_distribution["6-10次"] += 1
        elif count <= 20:
            freq_distribution["11-20次"] += 1
        else:
            freq_distribution["20次以上"] += 1

    # Gini coefficient for recommendation frequency
    all_counts = [candidate_counts.get(cid, 0) for cid in all_company_ids]
    overall_gini = calculate_gini(all_counts)

    # Top 10 most recommended
    top_10 = [
        {"公司ID": cid, "被推荐次数": count}
        for cid, count in candidate_counts.most_common(10)
    ]

    # 2b. 候选重叠度 - 同一公司不同规则的candidates是否相似
    company_rule_candidates = defaultdict(dict)  # query_company_id -> {rule_id: set(candidate_ids)}

    for result in rerank_results:
        query_cid = result.get("query_company_id")
        rule_id = result.get("rule_id")
        candidates = {c.get("company_id") for c in result.get("selected_companies", [])}
        if query_cid and rule_id:
            company_rule_candidates[query_cid][rule_id] = candidates

    # Calculate pairwise Jaccard for each company's rules
    overlap_scores = []
    high_overlap_count = 0

    for query_cid, rules in company_rule_candidates.items():
        rule_list = list(rules.keys())
        if len(rule_list) < 2:
            continue

        company_overlaps = []
        for i in range(len(rule_list)):
            for j in range(i + 1, len(rule_list)):
                jaccard = calculate_jaccard(rules[rule_list[i]], rules[rule_list[j]])
                company_overlaps.append(jaccard)

        if company_overlaps:
            avg_overlap = mean(company_overlaps)
            overlap_scores.append(avg_overlap)
            if avg_overlap > 0.5:  # High overlap threshold
                high_overlap_count += 1

    avg_jaccard = round(mean(overlap_scores), 3) if overlap_scores else 0.0

    # 2c. 按规则细分统计
    by_rule = {}
    for rule_id, counts in rule_candidate_counts.items():
        rule_counts = list(counts.values())
        by_rule[rule_id] = {
            "基尼系数": round(calculate_gini(rule_counts), 3),
            "被推荐公司数": len(counts),
            "平均被推荐次数": round(mean(rule_counts), 1) if rule_counts else 0
        }

    return {
        "被推荐频率": {
            "基尼系数": round(overall_gini, 3),
            "前10被推荐公司": top_10,
            "从未被推荐公司数": len(never_recommended),
            "频率分布": freq_distribution
        },
        "候选重叠度": {
            "平均Jaccard相似度": avg_jaccard,
            "高重叠公司数": high_overlap_count,
            "分析公司数": len(overlap_scores)
        },
        "按规则细分": by_rule
    }


def compute_article_candidates(articles: List[Dict]) -> Dict[str, Any]:
    """
    指标3: 文章关联公司数量统计
    """
    # Overall statistics
    all_counts = []
    by_rule = defaultdict(list)

    for article in articles:
        candidates = article.get("candidate_company_ids", [])
        count = len(candidates)
        all_counts.append(count)

        rule_id = article.get("rule_id", "unknown")
        by_rule[rule_id].append(count)

    # Distribution (1-5)
    distribution = Counter(all_counts)
    dist_formatted = {
        f"{i}家": distribution.get(i, 0)
        for i in range(1, 6)
    }

    # Overall stats
    overall = {
        "平均关联公司数": round(mean(all_counts), 2) if all_counts else 0,
        "中位数": int(median(all_counts)) if all_counts else 0,
        "分布": dist_formatted
    }

    # By rule stats
    rule_stats = {}
    for rule_id, counts in sorted(by_rule.items()):
        rule_stats[rule_id] = {
            "平均": round(mean(counts), 2) if counts else 0,
            "中位数": int(median(counts)) if counts else 0,
            "文章数": len(counts)
        }

    return {
        "总体": overall,
        "按规则": rule_stats
    }


def generate_analytics(production_dir: Path) -> Dict[str, Any]:
    """生成完整的分析报告"""
    # Load data
    tags_path = production_dir / "company_tagging" / "company_tags.json"
    rerank_dir = production_dir / "article_generator" / "rerank_cache"
    articles_dir = production_dir / "article_generator" / "articles"

    company_tags = load_company_tags(tags_path)
    rerank_results = load_rerank_cache(rerank_dir)
    articles = load_articles(articles_dir)

    print(f"Loaded {len(company_tags)} companies")
    print(f"Loaded {len(rerank_results)} rerank results")
    print(f"Loaded {len(articles)} articles")

    # Compute metrics
    tag_distributions = compute_tag_distributions(company_tags)
    recommendation_health = compute_recommendation_health(rerank_results, articles, company_tags)
    article_candidates = compute_article_candidates(articles)

    return {
        "生成时间": datetime.now().isoformat(),
        "数据概览": {
            "公司总数": len(company_tags),
            "精排结果数": len(rerank_results),
            "文章总数": len(articles)
        },
        "标签分布": tag_distributions,
        "推荐生态健康度": recommendation_health,
        "文章关联公司统计": article_candidates
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成平台数据分析报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--production-dir",
        type=Path,
        default=Path("outputs/production"),
        help="生产数据目录 (default: outputs/production)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出JSON路径 (default: {production-dir}/analytics.json)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="格式化JSON输出",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Generate analytics
    print("=" * 60)
    print("平台数据分析")
    print("=" * 60)

    analytics = generate_analytics(args.production_dir)

    # Determine output path
    output_path = args.output or (args.production_dir / "analytics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analytics, f, ensure_ascii=False, indent=2 if args.pretty else None)

    print(f"\n分析报告已保存: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("分析摘要")
    print("=" * 60)
    print(f"公司总数: {analytics['数据概览']['公司总数']}")
    print(f"文章总数: {analytics['数据概览']['文章总数']}")

    gini = analytics['推荐生态健康度']['被推荐频率']['基尼系数']
    print(f"推荐频率基尼系数: {gini} ({'较均匀' if gini < 0.4 else '较集中'})")

    avg_candidates = analytics['文章关联公司统计']['总体']['平均关联公司数']
    print(f"平均关联公司数: {avg_candidates}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
