import pandas as pd

# 读取三个文件
company_list = pd.read_csv('aihirebox_company_list_sample.csv')
baseline_tags = pd.read_csv('output/websearch_test_20251217_184647/gpt4o-mini-baseline/company_tags.csv')
online_tags = pd.read_csv('output/websearch_test_20251217_184647/gpt4o-mini-online/company_tags.csv')

# 获取前10个公司
company_list_top10 = company_list.head(10)

print("Company list (前10个):")
print(company_list_top10[['company_id', 'company_name']].to_string())
print(f"\nBaseline tags 数量: {len(baseline_tags)}")
print(f"Online tags 数量: {len(online_tags)}")

# 重命名 baseline 和 online 的列名（除了 company_id 和 company_name）
baseline_rename = {col: f"{col}_baseline" for col in baseline_tags.columns if col not in ['company_id', 'company_name']}
online_rename = {col: f"{col}_online" for col in online_tags.columns if col not in ['company_id', 'company_name']}

baseline_tags_renamed = baseline_tags.rename(columns=baseline_rename)
online_tags_renamed = online_tags.rename(columns=online_rename)

# 先合并 company_list_top10 和 baseline
merged = company_list_top10.merge(baseline_tags_renamed, on=['company_id', 'company_name'], how='left')

# 再合并 online
merged = merged.merge(online_tags_renamed, on=['company_id', 'company_name'], how='left')

# 保存合并结果
output_path = 'output/websearch_test_20251217_184647/merged_comparison.csv'
merged.to_csv(output_path, index=False)

print(f"\n合并完成！保存到: {output_path}")
print(f"\n合并后的列名:")
for col in merged.columns:
    print(f"  - {col}")

print(f"\n合并后数据行数: {len(merged)}")
