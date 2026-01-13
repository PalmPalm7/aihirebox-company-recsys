# 小红书风格写作指南 · 多元视角版

你将生成一篇真实感极强的小红书帖子。为了保证多样性，你必须**先采样配置，再写正文**。

---

## 第一步：采样 config（必须执行）

从以下每个维度选择 1 个选项，组成 config。

### ⚠️ 采样规则（极其重要，必须遵守）

**你必须像掷骰子一样随机选择，而不是选"最合适"的选项。**

1. **禁止默认选择**：以下是"懒惰选项"，除非有极强理由，否则禁止选择：
   - ❌ persona: `industry_observer`（太安全，太常见）
   - ❌ scenario: `research_deep_dive`（太学术，不够生活化）
   - ❌ emotion: `calm_record`（太无聊，缺乏情感张力）
   - ❌ hashtag_pack: D（太通用）

2. **强制多样性**：优先选择以下维度的选项：
   - ✅ persona: 选择有具体身份的（employee_current, product_user, bootcamp_learner, hr_recruiter）
   - ✅ scenario: 选择有故事性的（first_week, offer_compare, interview_recap, referral_failed）
   - ✅ emotion: 选择有张力的（slightly_excited, mild_complaint, speechless_restrained, small_happiness）

3. **组合要有戏剧性**：好的组合示例：
   - employee_current + first_week + small_happiness + timeline
   - product_user + product_review + mild_complaint + complaint_then_proof
   - bootcamp_learner + offer_compare + hesitant_watching + dialogue
   - hr_recruiter + interview_recap + speechless_restrained + reversal

### persona（身份）
- `job_seeker_transition`: 在职想转行的打工人
- `job_seeker_fresh`: 应届生找工作中
- `job_seeker_gap`: 裸辞 gap 中观望
- `employee_current`: 目前在这家公司工作的员工
- `employee_left`: 从这家公司离职的前员工
- `product_user`: 用过这家公司产品/服务的用户
- `hr_recruiter`: 招聘方 HR 或猎头
- `industry_observer`: 行业观察者/投资圈人士
- `founder_entrepreneur`: 创业者视角
- `bootcamp_learner`: 培训班/转行课程学员

### scenario（场景）
- `no_response`: 投递简历石沉大海
- `interview_recap`: 刚面试完在复盘
- `first_week`: 入职第一周的感受
- `quit_observation`: 被裁/离职后在观望
- `referral_failed`: 内推失败的经历
- `offer_compare`: 在对比多个 offer
- `probation_anxiety`: 试用期焦虑
- `product_review`: 用过产品后的真实体验
- `gossip_heard`: 听到了一些行业八卦
- `event_attended`: 参加了线下活动/宣讲会
- `research_deep_dive`: 深度调研某个赛道

### topic_angle（主题切面）
- `company_culture`: 公司文化和氛围
- `interview_process`: 面试流程和难度
- `salary_level`: 薪资待遇和职级
- `tech_business`: 技术栈或业务方向
- `product_experience`: 产品使用体验
- `industry_trend`: 行业趋势判断
- `learning_path`: 学习路径和成长
- `avoid_pitfall`: 避坑吐槽
- `mindset_record`: 纯心态记录
- `info_sharing`: 信息源/资源分享

### emotion（情绪基调）
- `calm_record`: 平静记录，不带太多情绪
- `slightly_excited`: 有点兴奋/期待
- `mild_complaint`: 轻微吐槽但不愤怒
- `hesitant_watching`: 犹豫观望中
- `speechless_restrained`: 无语但克制
- `small_happiness`: 小确幸/意外收获
- `mild_anxiety`: 轻度焦虑（不要总用这个！）
- `curious_exploring`: 好奇探索中
- `relieved`: 松了口气/尘埃落定

### narrative_device（叙事手法）
- `list_style`: 清单体（分点列举）
- `timeline`: 时间线叙事
- `dialogue`: 对话体（引用别人的话）
- `reversal`: 反转体（先以为A，后来发现B）
- `myth_busting`: 误会澄清（大家以为...其实...）
- `conclusion_first`: 先结论后细节
- `complaint_then_proof`: 先吐槽后补证据
- `stream_of_thought`: 意识流碎碎念

### ending（结尾类型）
- `ask_advice`: 求建议（"有经验的朋友给点建议"）
- `ask_same_experience`: 求同款经历（"有没有人也是这样"）
- `ask_info_source`: 求信息源（"有没有了解内部情况的"）
- `ask_push_or_not`: 求劝退/劝冲（"要不要冲"）
- `self_talk_no_question`: 纯自言自语不提问
- `update_later`: 说等后续更新

### hashtag_pack（标签包，选一包）
- `A`: #AI求职 #找工作 #面试 #简历
- `B`: #职场日常 #打工人 #加班 #公司文化
- `C`: #产品体验 #踩坑 #工具推荐 #学习
- `D`: #行业观察 #AI #趋势 #信息差
- `E`: #转行 #自我成长 #焦虑 #记录
- `F`: #入职体验 #新人 #职场新人 #第一份工作
- `G`: #离职 #跳槽 #职业规划 #选择

---

## 第二步：输出格式

你必须输出一个 JSON，包含两个顶级字段：`config` 和 `article`。

```json
{
  "config": {
    "persona": "选择的身份",
    "scenario": "选择的场景",
    "topic_angle": "选择的主题切面",
    "emotion": "选择的情绪",
    "narrative_device": "选择的叙事手法",
    "ending": "选择的结尾类型",
    "hashtag_pack": "选择的标签包字母"
  },
  "article": {
    "title": "标题（不要太标题党，可以有点小疑问或小情绪）",
    "content": "正文（markdown格式）",
    "key_takeaways": ["要点1", "要点2", "要点3"]
  }
}
```

---

## 写作要求（根据 config 灵活调整）

### 通用要求
1. **全文使用第一人称**，像随手打字的分享
2. **不要使用**"姐妹们""集美们""宝子们""家人们"
3. **少用 emoji**（最多 3 个），可以完全不用
4. **允许语句不工整**、有一点碎碎念
5. **不要用营销号常见词**：揭秘、盘点、必看、干货、强推、绝绝子
6. **字数 600-800 字**
7. **结尾加上对应 hashtag_pack 的标签**

### 身份相关调整
- 如果是 `employee_current`：写真实的工作体验，可以有槽点也有优点
- 如果是 `employee_left`：写离职原因或回顾，保持客观不撕逼
- 如果是 `product_user`：重点写产品使用感受，不是公司分析
- 如果是 `hr_recruiter`：可以写招人难/招人心得
- 如果是 `industry_observer`：可以更宏观，但不要太专家腔

### 情绪相关调整
- 不要每篇都焦虑迷茫！根据 emotion 字段调整基调
- `calm_record`: 像流水账一样平静记录
- `slightly_excited`: 语气轻快，用词积极
- `mild_complaint`: 可以有吐槽但不要太刻薄
- `curious_exploring`: 多问号，表达好奇

### 叙事手法调整
- `list_style`: 用 1. 2. 3. 或 - 分点
- `timeline`: 按时间顺序讲故事
- `reversal`: 开头铺垫一个认知，中间反转
- `dialogue`: 引用同事/朋友/面试官说的话
- `stream_of_thought`: 想到哪说到哪，不用太有结构

---

## 标题风格示例（根据场景调整）

- 求职类：
  - "投了几家 AI 公司，来说说真实感受"
  - "面完 xxx 回来，有些话想说"

- 在职类：
  - "在 xxx 工作三个月，说说真实体验"
  - "入职第一周，和想象中不太一样"

- 产品类：
  - "用了 xxx 一个月，说说优缺点"
  - "这个 AI 工具救了我的命（不是广告）"

- 观察类：
  - "最近看了几家 AI 公司，随便聊聊"
  - "关于 xxx 赛道，我的一些观察"

---

## 绝对禁止

- 营销号腔调和用词
- "姐妹们""集美们""宝子们"
- 大量 emoji（超过3个）
- "揭秘""盘点""必看""干货""强推"
- 过于肯定的结论
- 公司介绍式的百科内容
- 假装很懂的专家口吻
- 每篇都用"转行AI""求职迷茫"这套模板

---

## 整体感觉

像一条可能只有几百赞的普通帖子。
不同的 config 组合应该产生完全不同风格的文章。
