"""
Shared constants for the AIHireBox recommendation system.

This module contains all shared constants used across the pipeline,
extracted to reduce duplication and improve maintainability.
"""

# Dimensions available for tag-based similarity
TAG_DIMENSIONS = [
    "industry",
    "business_model",
    "target_market",
    "company_stage",
    "tech_focus",
    "team_background",
]

# Company stages considered "big/popular" - apply head suppression
HEAD_COMPANY_STAGES = {
    "public",
    "bigtech_subsidiary",
    "profitable",
    "pre_ipo",
}

# Default head suppression penalty (0-1, higher = more suppression)
DEFAULT_HEAD_PENALTY = 0.6

# Default recommender settings
DEFAULT_MIN_COMPANIES_PER_DIM = 3
DEFAULT_MAX_COMPANIES_PER_DIM = 5
DEFAULT_NUM_DIMENSIONS = 5

# Score threshold settings
DEFAULT_SCORE_THRESHOLD = 0.5  # Minimum score to include a company
DEFAULT_MAX_BELOW_THRESHOLD = 2  # Max companies below threshold before dropping dimension


# ============================================================================
# Dimension Label Definitions (Chinese + English)
# ============================================================================

DIMENSION_LABELS = {
    # Industry labels
    "industry_ai_llm": {"zh": "AI大模型公司", "en": "AI/LLM Companies"},
    "industry_robotics": {"zh": "机器人/具身智能公司", "en": "Robotics Companies"},
    "industry_edtech": {"zh": "教育科技公司", "en": "EdTech Companies"},
    "industry_fintech": {"zh": "金融科技公司", "en": "FinTech Companies"},
    "industry_healthtech": {"zh": "医疗健康公司", "en": "HealthTech Companies"},
    "industry_enterprise_saas": {"zh": "企业服务/SaaS公司", "en": "Enterprise SaaS Companies"},
    "industry_ecommerce": {"zh": "电商/零售公司", "en": "E-commerce Companies"},
    "industry_gaming": {"zh": "游戏/娱乐公司", "en": "Gaming Companies"},
    "industry_social": {"zh": "社交/社区平台", "en": "Social Platforms"},
    "industry_semiconductor": {"zh": "半导体/芯片公司", "en": "Semiconductor Companies"},
    "industry_automotive": {"zh": "汽车/出行公司", "en": "Automotive Companies"},
    "industry_consumer_hw": {"zh": "消费电子/硬件公司", "en": "Consumer Hardware"},
    "industry_cloud_infra": {"zh": "云计算/基础设施", "en": "Cloud Infrastructure"},
    "industry_content_media": {"zh": "内容/媒体公司", "en": "Content/Media Companies"},
    "industry_biotech": {"zh": "生物科技公司", "en": "Biotech Companies"},
    "industry_investment": {"zh": "投资/金融服务", "en": "Investment/Finance"},

    # Business model labels
    "business_model_b2b": {"zh": "同为B2B企业服务", "en": "B2B Services"},
    "business_model_b2c": {"zh": "同为C端消费产品", "en": "B2C Products"},
    "business_model_platform": {"zh": "同为平台模式", "en": "Platform Business"},
    "business_model_saas": {"zh": "同为SaaS订阅模式", "en": "SaaS Model"},
    "business_model_hardware": {"zh": "同为硬件产品", "en": "Hardware Products"},
    "business_model_marketplace": {"zh": "同为市场平台", "en": "Marketplace"},

    # Target market labels
    "target_market_global": {"zh": "同为全球化公司", "en": "Global Companies"},
    "target_market_china_domestic": {"zh": "同为国内市场公司", "en": "China Domestic"},
    "target_market_sea": {"zh": "同为出海东南亚", "en": "SEA Market"},
    "target_market_us": {"zh": "同为出海北美", "en": "US Market"},

    # Company stage labels
    "company_stage_early": {"zh": "同为早期创业公司", "en": "Early Stage Startups"},
    "company_stage_growth": {"zh": "同为成长期公司", "en": "Growth Stage Companies"},
    "company_stage_seed": {"zh": "同为种子期公司", "en": "Seed Stage Startups"},

    # Tech focus labels
    "tech_focus_llm_foundation": {"zh": "同为大模型技术", "en": "LLM/Foundation Models"},
    "tech_focus_computer_vision": {"zh": "同为计算机视觉", "en": "Computer Vision"},
    "tech_focus_embodied_ai": {"zh": "同为具身智能", "en": "Embodied AI"},
    "tech_focus_aigc": {"zh": "同为AIGC内容生成", "en": "AIGC"},
    "tech_focus_chip_hardware": {"zh": "同为芯片/硬件技术", "en": "Chip/Hardware"},
    "tech_focus_data_infra": {"zh": "同为数据基础设施", "en": "Data Infrastructure"},
    "tech_focus_autonomous": {"zh": "同为自动驾驶/自主系统", "en": "Autonomous Systems"},
    "tech_focus_speech_nlp": {"zh": "同为语音/NLP技术", "en": "Speech/NLP"},

    # Team background labels
    "team_background_bigtech_alumni": {"zh": "同为大厂背景团队", "en": "Big Tech Alumni"},
    "team_background_top_university": {"zh": "同为顶尖高校背景", "en": "Top University Background"},
    "team_background_serial_entrepreneur": {"zh": "同为连续创业者", "en": "Serial Entrepreneurs"},
    "team_background_academic": {"zh": "同为学术背景创业", "en": "Academic Background"},
    "team_background_international": {"zh": "同为海归/国际化团队", "en": "International Team"},

    # Embedding-based (semantic) dimension
    "semantic_similar": {"zh": "业务描述相似", "en": "Semantically Similar"},
}
