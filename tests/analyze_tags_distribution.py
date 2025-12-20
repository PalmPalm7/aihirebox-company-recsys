#!/usr/bin/env python3
"""
公司标签分布分析脚本
分析 company_tags.csv 中各标签的分布情况
"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('/Users/anxie/Documents/aihirebox/aihirebox-company-recsys/output_production/company_tags.csv')

# 过滤掉空行
df = df.dropna(subset=['company_id'])

print(f"总公司数: {len(df)}")
print(f"列名: {df.columns.tolist()}")

# 创建一个大图，包含多个子图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('公司标签分布分析 (Company Tags Distribution)', fontsize=16, fontweight='bold')

# 1. 行业分布 (Industry Distribution)
industry_counts = Counter()
for industries in df['industry'].dropna():
    for ind in str(industries).split('|'):
        if ind.strip():
            industry_counts[ind.strip()] += 1

# 取前15个行业
top_industries = dict(sorted(industry_counts.items(), key=lambda x: x[1], reverse=True)[:15])
ax1 = axes[0, 0]
bars1 = ax1.barh(list(top_industries.keys()), list(top_industries.values()), color='steelblue')
ax1.set_xlabel('数量 (Count)')
ax1.set_title('行业分布 Top 15 (Industry)', fontsize=12)
ax1.invert_yaxis()
for bar, count in zip(bars1, top_industries.values()):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, str(count), va='center', fontsize=9)

# 2. 公司阶段分布 (Company Stage Distribution)
stage_counts = df['company_stage'].value_counts()
ax2 = axes[0, 1]
colors = plt.cm.Set3(range(len(stage_counts)))
wedges, texts, autotexts = ax2.pie(stage_counts.values, labels=stage_counts.index, autopct='%1.1f%%', 
                                    colors=colors, startangle=90)
ax2.set_title('公司阶段分布 (Company Stage)', fontsize=12)

# 3. 商业模式分布 (Business Model Distribution)
biz_model_counts = Counter()
for models in df['business_model'].dropna():
    for model in str(models).split('|'):
        if model.strip():
            biz_model_counts[model.strip()] += 1

top_biz = dict(sorted(biz_model_counts.items(), key=lambda x: x[1], reverse=True)[:10])
ax3 = axes[0, 2]
bars3 = ax3.bar(top_biz.keys(), top_biz.values(), color='coral')
ax3.set_xlabel('商业模式 (Business Model)')
ax3.set_ylabel('数量 (Count)')
ax3.set_title('商业模式分布 Top 10', fontsize=12)
ax3.tick_params(axis='x', rotation=45)
for bar in bars3:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             str(int(bar.get_height())), ha='center', fontsize=9)

# 4. 技术焦点分布 (Tech Focus Distribution)
tech_counts = Counter()
for techs in df['tech_focus'].dropna():
    for tech in str(techs).split('|'):
        if tech.strip():
            tech_counts[tech.strip()] += 1

top_tech = dict(sorted(tech_counts.items(), key=lambda x: x[1], reverse=True)[:15])
ax4 = axes[1, 0]
bars4 = ax4.barh(list(top_tech.keys()), list(top_tech.values()), color='seagreen')
ax4.set_xlabel('数量 (Count)')
ax4.set_title('技术焦点分布 Top 15 (Tech Focus)', fontsize=12)
ax4.invert_yaxis()
for bar, count in zip(bars4, top_tech.values()):
    ax4.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, str(count), va='center', fontsize=9)

# 5. 目标市场分布 (Target Market Distribution)
market_counts = Counter()
for markets in df['target_market'].dropna():
    for market in str(markets).split('|'):
        if market.strip():
            market_counts[market.strip()] += 1

top_markets = dict(sorted(market_counts.items(), key=lambda x: x[1], reverse=True)[:10])
ax5 = axes[1, 1]
bars5 = ax5.bar(top_markets.keys(), top_markets.values(), color='orchid')
ax5.set_xlabel('目标市场 (Target Market)')
ax5.set_ylabel('数量 (Count)')
ax5.set_title('目标市场分布 Top 10', fontsize=12)
ax5.tick_params(axis='x', rotation=45)
for bar in bars5:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             str(int(bar.get_height())), ha='center', fontsize=9)

# 6. 置信度分数分布 (Confidence Score Distribution)
ax6 = axes[1, 2]
confidence_scores = df['confidence_score'].dropna()
ax6.hist(confidence_scores, bins=15, color='goldenrod', edgecolor='black', alpha=0.7)
ax6.set_xlabel('置信度分数 (Confidence Score)')
ax6.set_ylabel('频数 (Frequency)')
ax6.set_title(f'置信度分数分布\n均值={confidence_scores.mean():.2f}, 中位数={confidence_scores.median():.2f}', fontsize=12)
ax6.axvline(confidence_scores.mean(), color='red', linestyle='--', label=f'Mean: {confidence_scores.mean():.2f}')
ax6.axvline(confidence_scores.median(), color='blue', linestyle='--', label=f'Median: {confidence_scores.median():.2f}')
ax6.legend()

plt.tight_layout()
plt.savefig('/Users/anxie/Documents/aihirebox/aihirebox-company-recsys/output_production/tags_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n图表已保存到: tags_distribution.png")

# 打印一些统计摘要
print("\n" + "="*50)
print("统计摘要 (Summary Statistics)")
print("="*50)
print(f"\n行业数量: {len(industry_counts)} 种")
print(f"技术焦点数量: {len(tech_counts)} 种")
print(f"商业模式数量: {len(biz_model_counts)} 种")
print(f"目标市场数量: {len(market_counts)} 种")
print(f"\n公司阶段分布:")
for stage, count in stage_counts.items():
    print(f"  {stage}: {count} ({count/len(df)*100:.1f}%)")
print(f"\n置信度分数: 范围 [{confidence_scores.min():.2f}, {confidence_scores.max():.2f}]")
