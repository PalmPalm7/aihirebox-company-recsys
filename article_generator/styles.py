"""
Article Styles - 5种文章风格定义

| Style       | 字数      | Emoji | 特点                     |
|-------------|-----------|-------|--------------------------|
| 36kr        | 800-1200  | No    | 专业、数据驱动、行业分析 |
| huxiu       | 1000-1500 | No    | 犀利、有态度、深度评论   |
| xiaohongshu | 500-800   | Yes   | 轻松、口语化、分点列举   |
| linkedin    | 600-1000  | No    | 职场视角、强调机会       |
| zhihu       | 1000-1500 | No    | 知识分享、逻辑清晰       |
"""

from dataclasses import dataclass
from typing import Dict, Optional

from .models import ArticleStyle


# 风格定义
ARTICLE_STYLES: Dict[str, ArticleStyle] = {
    "36kr": ArticleStyle(
        style_id="36kr",
        name_zh="36氪深度稿",
        name_en="36Kr Feature",
        word_count_min=800,
        word_count_max=1200,
        use_emoji=False,
        tone="专业但不端着，有观点，敢下判断",
        structure="""
- 开头抛出一个有冲击力的观点或行业洞察（不要"本文将介绍"这种开法）
- 中段围绕核心公司展开，相关公司用对比/递进/转折自然引入，不要平均用力
- 不需要每家公司都用小标题，可以在行文中自然切换
- 结尾给一个判断或留一个开放性问题，不要写"综上所述"
""",
        example_intro="大模型创业已经死了一批，但真正值得关注的，反而是现在才冒出来的这群人。",
    ),
    
    "huxiu": ArticleStyle(
        style_id="huxiu",
        name_zh="虎嗅风格",
        name_en="Huxiu Style",
        word_count_min=1000,
        word_count_max=1500,
        use_emoji=False,
        tone="犀利、有态度、深度洞察",
        structure="""
1. 抛出争议性观点或反直觉洞察
2. 现象描述与问题提出
3. 深度分析（多角度论证）
4. 行业内幕或独家观点
5. 犀利总结（可带批判性）
""",
        example_intro="当所有人都在谈论'AGI'时，真正赚到钱的AI公司在做什么？",
    ),
    
    "xiaohongshu": ArticleStyle(
        style_id="xiaohongshu",
        name_zh="小红书风格",
        name_en="Xiaohongshu Style",
        word_count_min=500,
        word_count_max=800,
        use_emoji=True,
        tone="轻松、亲切、口语化、分享感",
        structure="""
1. 吸睛标题（带emoji）
2. 开篇hook（1-2句吸引眼球）
3. 核心内容分点列举（3-5点）
4. 个人感受/推荐理由
5. 互动引导（提问、话题）
""",
        example_intro="姐妹们！！最近发现了几家超牛的AI公司🔥 做求职的一定要看！",
    ),
    
    "linkedin": ArticleStyle(
        style_id="linkedin",
        name_zh="LinkedIn风格",
        name_en="LinkedIn Style",
        word_count_min=600,
        word_count_max=1000,
        use_emoji=False,
        tone="职业、专业、机会导向、激励性",
        structure="""
1. 职业洞察开篇
2. 行业机会分析
3. 公司/岗位推荐（突出发展前景）
4. 职业建议（技能、方向）
5. 行动号召（鼓励尝试）
""",
        example_intro="As the AI industry continues to evolve, new career opportunities are emerging in unexpected places.",
    ),
    
    "zhihu": ArticleStyle(
        style_id="zhihu",
        name_zh="知乎风格",
        name_en="Zhihu Style",
        word_count_min=1000,
        word_count_max=1500,
        use_emoji=False,
        tone="理性、严谨、知识分享、逻辑清晰",
        structure="""
1. 问题定义与背景说明
2. 核心概念解释
3. 分析框架（清晰的逻辑结构）
4. 案例解读（具体公司分析）
5. 总结与延伸思考
""",
        example_intro="这个问题涉及到AI行业的几个核心议题，我从技术、商业和人才三个维度来分析。",
    ),
}


def get_style(style_id: str) -> Optional[ArticleStyle]:
    """获取指定风格配置
    
    Args:
        style_id: 风格ID (36kr, huxiu, xiaohongshu, linkedin, zhihu)
        
    Returns:
        ArticleStyle 或 None
    """
    return ARTICLE_STYLES.get(style_id)


def get_all_style_ids() -> list:
    """获取所有风格ID列表"""
    return list(ARTICLE_STYLES.keys())

