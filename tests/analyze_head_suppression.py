#!/usr/bin/env python3
"""
头部公司抑制策略分析
分析 IDFHeadSuppression 的假设是否合理：标签越多的公司是否越应该被抑制？

核心问题：
1. 标签多 = 更通用/更大？还是只是文档更全？
2. 标签数量与公司阶段的关系
3. 标签数量与置信度的关系
4. 什么样的公司标签最多？
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    --company-tags-path: company_tags.csv 的路径，默认使用项目相对路径：
        <repo_root>/output_production/company_tagging/company_tags.csv
    --output-path: 输出图表的路径，默认使用项目相对路径：
        <repo_root>/output_production/head_suppression_analysis.png
    """
    default_tags_path = Path(__file__).resolve().parents[1] / 'output_production' / 'company_tagging' / 'company_tags.csv'
    default_output_path = Path(__file__).resolve().parents[1] / 'output_production' / 'head_suppression_analysis.png'
    parser = argparse.ArgumentParser(description="头部公司抑制策略分析")
    parser.add_argument(
        "--company-tags-path",
        type=str,
        default=str(default_tags_path),
        help=f"company_tags.csv 路径，默认: {default_tags_path}",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(default_output_path),
        help=f"输出图表路径，默认: {default_output_path}",
    )
    return parser.parse_args()


args = _parse_args()

# 读取数据
df = pd.read_csv(args.company_tags_path)
df = df.dropna(subset=['company_id'])

print("=" * 70)
print("头部公司抑制策略分析 (Head Suppression Analysis)")
print("=" * 70)

# 计算每个公司的标签数量
def count_tags(row):
    """计算单个公司的标签总数"""
    count = 0
    for col in ['industry', 'business_model', 'target_market', 'tech_focus', 'team_background']:
        if pd.notna(row[col]) and str(row[col]).strip():
            count += len(str(row[col]).split('|'))
    return count

def count_tags_by_dim(row, dim):
    """计算单个维度的标签数"""
    if pd.notna(row[dim]) and str(row[dim]).strip():
        return len(str(row[dim]).split('|'))
    return 0

df['total_tags'] = df.apply(count_tags, axis=1)
df['industry_tags'] = df.apply(lambda r: count_tags_by_dim(r, 'industry'), axis=1)
df['business_tags'] = df.apply(lambda r: count_tags_by_dim(r, 'business_model'), axis=1)
df['market_tags'] = df.apply(lambda r: count_tags_by_dim(r, 'target_market'), axis=1)
df['tech_tags'] = df.apply(lambda r: count_tags_by_dim(r, 'tech_focus'), axis=1)
df['team_tags'] = df.apply(lambda r: count_tags_by_dim(r, 'team_background'), axis=1)

# 定义头部公司
HEAD_STAGES = {'public', 'bigtech_subsidiary', 'profitable', 'pre_ipo'}
df['is_head'] = df['company_stage'].isin(HEAD_STAGES)

print(f"\n总公司数: {len(df)}")
print(f"头部公司数 (public/bigtech/profitable/pre_ipo): {df['is_head'].sum()}")
print(f"非头部公司数: {(~df['is_head']).sum()}")

# ============================================================================
# 分析 1: 标签数量分布
# ============================================================================
print("\n" + "=" * 50)
print("分析 1: 标签数量分布")
print("=" * 50)

print(f"\n标签数量统计:")
print(f"  最小: {df['total_tags'].min()}")
print(f"  最大: {df['total_tags'].max()}")
print(f"  均值: {df['total_tags'].mean():.2f}")
print(f"  中位数: {df['total_tags'].median():.2f}")
print(f"  标准差: {df['total_tags'].std():.2f}")

# ============================================================================
# 分析 2: 标签数量 vs 公司阶段 (关键问题!)
# ============================================================================
print("\n" + "=" * 50)
print("分析 2: 标签数量 vs 公司阶段 (关键问题!)")
print("=" * 50)

stage_tag_stats = df.groupby('company_stage').agg({
    'total_tags': ['mean', 'median', 'std', 'count'],
    'confidence_score': 'mean'
}).round(2)
print(stage_tag_stats)

print("\n头部 vs 非头部公司标签对比:")
head_stats = df[df['is_head']]['total_tags']
non_head_stats = df[~df['is_head']]['total_tags']
print(f"  头部公司标签均值: {head_stats.mean():.2f} (n={len(head_stats)})")
print(f"  非头部公司标签均值: {non_head_stats.mean():.2f} (n={len(non_head_stats)})")
print(f"  差异: {head_stats.mean() - non_head_stats.mean():.2f}")

# 简单的效果大小计算 (Cohen's d)
pooled_std = np.sqrt((head_stats.std()**2 + non_head_stats.std()**2) / 2)
cohens_d = (head_stats.mean() - non_head_stats.mean()) / pooled_std if pooled_std > 0 else 0
print(f"\n  效果大小 (Cohen's d): {cohens_d:.3f}")
# 使用简单的置换检验估计显著性
diff_observed = head_stats.mean() - non_head_stats.mean()
combined = np.concatenate([head_stats.values, non_head_stats.values])
np.random.seed(42)
n_permutations = 10000
count_extreme = 0
for _ in range(n_permutations):
    np.random.shuffle(combined)
    diff_perm = combined[:len(head_stats)].mean() - combined[len(head_stats):].mean()
    if diff_perm >= diff_observed:
        count_extreme += 1
p_value = count_extreme / n_permutations
print(f"  置换检验 p-value: {p_value:.4f}")
if p_value < 0.05:
    print("  ✅ 差异显著 (p < 0.05)")
else:
    print("  ⚠️ 差异不显著 (p >= 0.05)")

# ============================================================================
# 分析 3: 标签数量 vs 置信度
# ============================================================================
print("\n" + "=" * 50)
print("分析 3: 标签数量 vs 置信度")
print("=" * 50)

corr_tags_conf = df['total_tags'].corr(df['confidence_score'])
print(f"标签数量与置信度相关系数: {corr_tags_conf:.3f}")

if corr_tags_conf > 0.3:
    print("  ⚠️ 正相关: 标签多可能是因为信息更完整，不是因为公司更'通用'")
elif corr_tags_conf < -0.3:
    print("  标签多的公司置信度反而低")
else:
    print("  相关性较弱")

# ============================================================================
# 分析 4: 具体案例 - 标签最多/最少的公司
# ============================================================================
print("\n" + "=" * 50)
print("分析 4: 具体案例分析")
print("=" * 50)

print("\n标签最多的10家公司:")
top_tags = df.nlargest(10, 'total_tags')[['company_name', 'company_stage', 'total_tags', 'confidence_score', 'industry']]
for _, row in top_tags.iterrows():
    head_marker = "🏢 HEAD" if row['company_stage'] in HEAD_STAGES else ""
    print(f"  {row['company_name']}: {row['total_tags']}个标签, 阶段={row['company_stage']} {head_marker}")

print("\n标签最少的10家公司:")
bottom_tags = df.nsmallest(10, 'total_tags')[['company_name', 'company_stage', 'total_tags', 'confidence_score', 'industry']]
for _, row in bottom_tags.iterrows():
    head_marker = "🏢 HEAD" if row['company_stage'] in HEAD_STAGES else ""
    print(f"  {row['company_name']}: {row['total_tags']}个标签, 阶段={row['company_stage']} {head_marker}")

# ============================================================================
# 分析 5: IDFHeadSuppression 惩罚计算模拟
# ============================================================================
print("\n" + "=" * 50)
print("分析 5: IDFHeadSuppression 惩罚模拟")
print("=" * 50)

max_tags = df['total_tags'].max()
max_penalty = 0.4  # 默认值

df['idf_penalty'] = (df['total_tags'] / max_tags).clip(0, 1) * max_penalty

print(f"\nIDF惩罚分布 (max_penalty={max_penalty}):")
print(f"  最小惩罚: {df['idf_penalty'].min():.3f}")
print(f"  最大惩罚: {df['idf_penalty'].max():.3f}")
print(f"  均值惩罚: {df['idf_penalty'].mean():.3f}")

# 比较头部和非头部公司受到的IDF惩罚
print(f"\n头部 vs 非头部公司受到的IDF惩罚:")
print(f"  头部公司IDF惩罚均值: {df[df['is_head']]['idf_penalty'].mean():.3f}")
print(f"  非头部公司IDF惩罚均值: {df[~df['is_head']]['idf_penalty'].mean():.3f}")

# ============================================================================
# 分析 6: 问题检测 - 高标签非头部公司
# ============================================================================
print("\n" + "=" * 50)
print("分析 6: 问题检测 - 被误伤的公司")
print("=" * 50)

# 找出高标签但非头部的公司（被IDF惩罚但不应该被惩罚）
median_tags = df['total_tags'].median()
high_tag_non_head = df[(df['total_tags'] > median_tags) & (~df['is_head'])]
print(f"\n高标签(>{median_tags:.0f})的非头部公司 (可能被误伤):")
print(f"数量: {len(high_tag_non_head)}")
for _, row in high_tag_non_head.nlargest(8, 'total_tags').iterrows():
    print(f"  {row['company_name']}: {row['total_tags']}个标签, 阶段={row['company_stage']}, IDF惩罚={row['idf_penalty']:.3f}")

# 找出低标签的头部公司（应该被抑制但IDF惩罚很小）
low_tag_head = df[(df['total_tags'] <= median_tags) & (df['is_head'])]
print(f"\n低标签(<={median_tags:.0f})的头部公司 (可能被漏掉):")
print(f"数量: {len(low_tag_head)}")
for _, row in low_tag_head.nsmallest(8, 'total_tags').iterrows():
    print(f"  {row['company_name']}: {row['total_tags']}个标签, 阶段={row['company_stage']}, IDF惩罚={row['idf_penalty']:.3f}")

# ============================================================================
# 绘图
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('头部公司抑制策略分析\n(Head Suppression Strategy Analysis)', fontsize=16, fontweight='bold')

# 图1: 标签数量分布
ax1 = axes[0, 0]
ax1.hist(df['total_tags'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(df['total_tags'].mean(), color='red', linestyle='--', label=f'Mean: {df["total_tags"].mean():.1f}')
ax1.axvline(df['total_tags'].median(), color='orange', linestyle='--', label=f'Median: {df["total_tags"].median():.1f}')
ax1.set_xlabel('标签数量 (Total Tags)')
ax1.set_ylabel('频数 (Frequency)')
ax1.set_title('标签数量分布')
ax1.legend()

# 图2: 标签数量 vs 公司阶段 (箱线图)
ax2 = axes[0, 1]
stage_order = ['seed', 'early', 'growth', 'profitable', 'pre_ipo', 'public', 'bigtech_subsidiary', 'unknown']
stage_order = [s for s in stage_order if s in df['company_stage'].values]
df_plot = df[df['company_stage'].isin(stage_order)]
stage_positions = range(len(stage_order))
bp_data = [df_plot[df_plot['company_stage'] == stage]['total_tags'].values for stage in stage_order]
bp = ax2.boxplot(bp_data, positions=stage_positions, patch_artist=True)
colors = ['lightgreen' if s not in HEAD_STAGES else 'salmon' for s in stage_order]
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax2.set_xticks(stage_positions)
ax2.set_xticklabels(stage_order, rotation=45, ha='right')
ax2.set_ylabel('标签数量 (Total Tags)')
ax2.set_title('公司阶段 vs 标签数量\n(红色=头部公司)')

# 图3: 头部 vs 非头部 标签分布
ax3 = axes[0, 2]
ax3.hist(df[df['is_head']]['total_tags'], bins=12, alpha=0.7, label='头部公司 (Head)', color='salmon')
ax3.hist(df[~df['is_head']]['total_tags'], bins=12, alpha=0.7, label='非头部公司 (Non-Head)', color='lightgreen')
ax3.axvline(head_stats.mean(), color='red', linestyle='--', linewidth=2)
ax3.axvline(non_head_stats.mean(), color='green', linestyle='--', linewidth=2)
ax3.set_xlabel('标签数量 (Total Tags)')
ax3.set_ylabel('频数 (Frequency)')
ax3.set_title(f'头部 vs 非头部公司标签分布\n(p-value={p_value:.4f})')
ax3.legend()

# 图4: 标签数量 vs 置信度 散点图
ax4 = axes[1, 0]
colors_scatter = ['salmon' if is_head else 'steelblue' for is_head in df['is_head']]
ax4.scatter(df['total_tags'], df['confidence_score'], c=colors_scatter, alpha=0.6, s=50)
z = np.polyfit(df['total_tags'], df['confidence_score'], 1)
p = np.poly1d(z)
ax4.plot(df['total_tags'].sort_values(), p(df['total_tags'].sort_values()), "r--", alpha=0.8, 
         label=f'趋势线 (r={corr_tags_conf:.3f})')
ax4.set_xlabel('标签数量 (Total Tags)')
ax4.set_ylabel('置信度 (Confidence Score)')
ax4.set_title(f'标签数量 vs 置信度\n(相关系数 r={corr_tags_conf:.3f})')
ax4.legend()

# 图5: IDF惩罚分布
ax5 = axes[1, 1]
ax5.hist(df[df['is_head']]['idf_penalty'], bins=10, alpha=0.7, label='头部公司', color='salmon')
ax5.hist(df[~df['is_head']]['idf_penalty'], bins=10, alpha=0.7, label='非头部公司', color='lightgreen')
ax5.set_xlabel('IDF惩罚值 (IDF Penalty)')
ax5.set_ylabel('频数 (Frequency)')
ax5.set_title('IDF惩罚分布\n(理想情况: 红色应该偏右)')
ax5.legend()

# 图6: 各维度标签数分布
ax6 = axes[1, 2]
dims = ['industry_tags', 'business_tags', 'market_tags', 'tech_tags', 'team_tags']
dim_names = ['Industry', 'Business', 'Market', 'Tech', 'Team']
dim_means = [df[d].mean() for d in dims]
dim_stds = [df[d].std() for d in dims]
bars = ax6.bar(dim_names, dim_means, yerr=dim_stds, capsize=5, color='teal', alpha=0.7)
ax6.set_ylabel('平均标签数 (Mean Tags)')
ax6.set_title('各维度平均标签数')
for bar, mean in zip(bars, dim_means):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{mean:.1f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(args.output_path, dpi=150, bbox_inches='tight')
plt.close()  # 不显示，直接关闭

# ============================================================================
# 结论和建议
# ============================================================================
print("\n" + "=" * 70)
print("📊 分析结论与建议")
print("=" * 70)

print("\n关键发现:")
print(f"  1. 头部公司平均标签数: {head_stats.mean():.2f}")
print(f"  2. 非头部公司平均标签数: {non_head_stats.mean():.2f}")
print(f"  3. 差异显著性 p-value: {p_value:.4f}")
print(f"  4. 标签数与置信度相关系数: {corr_tags_conf:.3f}")

if head_stats.mean() > non_head_stats.mean() and p_value < 0.05:
    print("\n✅ IDFHeadSuppression 逻辑有一定合理性:")
    print("   头部公司确实标签更多，IDF惩罚能起到一定抑制作用")
elif head_stats.mean() <= non_head_stats.mean():
    print("\n⚠️ IDFHeadSuppression 逻辑可能有问题:")
    print("   头部公司标签并不比非头部公司多，IDF惩罚可能误伤创业公司")
else:
    print("\n⚠️ 差异不显著，IDFHeadSuppression 效果有限")

if corr_tags_conf > 0.3:
    print("\n⚠️ 额外警告: 标签数与置信度正相关")
    print("   这意味着标签多的公司可能只是信息更完整，不是更'通用'")
    print("   IDFHeadSuppression 可能在惩罚高质量的数据")

print(f"\n被误伤的公司数量: {len(high_tag_non_head)} (高标签非头部)")
print(f"被漏掉的公司数量: {len(low_tag_head)} (低标签头部)")

print("\n建议:")
if p_value >= 0.05 or len(high_tag_non_head) > len(low_tag_head):
    print("  🔴 考虑移除 IDFHeadSuppression 或降低其权重")
    print("     因为标签数量不能有效区分头部/非头部公司")
else:
    print("  🟢 IDFHeadSuppression 可以保留")
    print("     但建议降低 max_penalty 或权重")

print("\n图表已保存到: head_suppression_analysis.png")
