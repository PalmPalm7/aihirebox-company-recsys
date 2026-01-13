# 小红书风格生成的反思与结论

## 背景

我们尝试为文章生成系统增加"小红书风格"——模拟真人视角的社交媒体帖子。

## 尝试过程

### 第一版：简单身份设定
- 设定为"转行AI的打工人"视角
- **问题**：29篇文章100%都是"转行AI"主题，开头相同，情绪弧线相同，标签相同

### 第二版：Config-based 采样系统
设计了7个维度的可采样变量空间：
- persona (10种身份：求职者、在职员工、产品用户、HR、行业观察者等)
- scenario (11种场景：面试复盘、入职第一周、offer对比等)
- topic_angle (10种主题切面)
- emotion (9种情绪基调)
- narrative_device (8种叙事手法)
- ending (6种结尾类型)
- hashtag_pack (7种标签包)

修改了代码：
- `article_generator/models.py`: 增加 `style_config` 字段存储采样配置
- `article_generator/article_writer.py`: 解析嵌套JSON格式，保存配置元数据

### 第三版：强制多样性规则
在prompt中添加"禁止默认选择"规则，禁止使用 `industry_observer`、`research_deep_dive` 等"懒惰选项"。

**结果**：多样性有所改善（从93%单一persona降到6种persona），但仍存在聚类。

## 核心问题

即使技术上实现了config多样性，根本问题依然存在：

### 1. LLM无法生成真实的"真人感"
- **资讯类文章**（如36kr风格）：用web search做grounding，只展示事实(Fact)，不展示观点(Opinion)，与RAG逻辑一致
- **真人视角文章**（如小红书风格）：要求LLM凭空生成"个人体验"和"情感表达"，这些内容无法与现实对齐(Align)

### 2. 生成的"观点"没有事实支撑
真人帖子的可信度来自于：
- 真实的工作经历
- 真实的面试体验
- 真实的产品使用感受

LLM没有这些真实经历，生成的内容本质上是"编造"的，无论config如何采样，都无法解决这个根本问题。

### 3. 趋同性是LLM的固有特性
即使添加采样规则，LLM仍然会：
- 选择"看起来合理"的组合
- 生成相似的叙事结构
- 使用相似的语言风格

## 结论

**小红书风格（真人视角）的生成方案不可行**，原因：
1. LLM生成的"真人感"无法对齐真实人类体验
2. 观点型内容缺乏事实grounding
3. 多样性问题无法从根本上解决

**当前策略**：优先使用36kr等资讯类prompt格式，因为：
1. 基于web search的事实grounding
2. 只展示Fact，不展示Opinion
3. 与RAG逻辑一致，输出质量可控

## 文件变更记录

以下文件在此次尝试中被修改，但功能目前不推荐使用：
- `prompts/article_styles/xiaohongshu.md` - config采样系统prompt
- `article_generator/article_writer.py` - 支持xiaohongshu嵌套JSON解析
- `article_generator/models.py` - Article增加style_config字段

代码改动保留但不删除，作为技术探索的记录。

## 日期

2026-01-13
