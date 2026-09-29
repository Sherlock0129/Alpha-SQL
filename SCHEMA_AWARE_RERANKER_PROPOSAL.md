# 面向 Text-to-SQL 的 Schema-aware SQL Selector

## 1. 项目概述

本项目拟训练一个独立的监督式 Schema-aware SQL Selector，提高从候选 SQL 集合中选出执行正确 SQL 的概率。选择器不负责生成 SQL，也不替换或重新训练上游 Schema Linker；它将问题、数据库 schema、候选 SQL、候选使用的表列、Schema Linker 输出及执行反馈作为判别证据，对候选进行排序。

项目只研究固定候选池上的 SQL 选择问题。候选池由上游生成器离线提供，选择器的训练和评估均不依赖特定搜索算法。该设计将“候选生成能力”和“候选选择能力”分离，能够直接测量 selector 对 `Oracle Accuracy - Final Accuracy` 差距的贡献。

## 2. 研究背景与动机

Text-to-SQL 系统通常能够生成多个语法正确、可执行但语义不同的 SQL。常见候选选择方法依据执行结果自一致性，优先选择属于最大执行结果簇的候选。这种方法具有较强稳定性，但隐含假设是“多数候选对应的执行结果更可能正确”。当生成器系统性误解问题、选择相似但错误的列、遗漏过滤条件或构造错误连接时，多数票也可能集中在错误结果上。

MSc-SQL 等工作表明，多候选联合判别有机会超过单纯执行结果投票。初步实验同样观察到：固定候选池上的自一致性 EX 为 62%，而 oracle EX 为 70%。这意味着在 50 个评估问题中，候选池已经包含更多正确答案，但当前选择规则未能将其置顶。因此，训练更准确的 SQL Selector 具有明确空间。

本研究的核心观点是：候选 SQL 是否正确，不仅取决于其能否执行或是否属于多数结果簇，还取决于候选使用的表、列、连接路径与问题语义及原 Schema Linker 输出是否一致。将这些信号联合建模，有望更准确地区分结构相似的候选 SQL。

## 3. 研究问题

本项目拟回答以下问题：

1. 在固定候选池下，神经重排序器能否超过执行结果自一致性基线？
2. 候选实际使用的表列及其与原 Schema Linker 输出的一致性，能否提供独立于 SQL 结构和执行反馈的增益？
3. Pointwise、Pairwise 和 Groupwise 候选建模中，哪种方式更适合 Text-to-SQL 候选选择？
4. 如何防止 reranker 在训练数据库上获得提升、但在未见数据库上发生性能下降？
5. reranker 的主要上限来自模型能力、有效正负训练对数量，还是候选池本身的 oracle 上界？

## 4. 研究假设

主要假设为：

> 在不修改原 Schema Linker 的前提下，将问题—SQL 语义匹配、候选表列使用情况、Schema Linker 一致性、连接完整性和执行反馈联合输入 pairwise reranker，可以提高未见数据库上的候选选择准确率，并关闭部分 oracle gap。

对应的可证伪条件为：如果加入 schema 信号后，在多随机种子、数据库隔离验证和独立测试集上均不能显著超过 SQL + execution reranker，则不能宣称 Schema-aware 设计有效。

## 5. 系统方案

整体流程如下：

```text
问题、evidence、完整数据库 schema、候选 SQL 集合 {s1...sk}
                              │
                    原 Schema Linker 输出（可选）
                              │
                     SQL 执行、解析与特征构造
                              │
                    Schema-aware SQL Selector
                              │
               自一致性残差融合 / 置信门控
                              │
                           最终 SQL
```

候选池在训练前离线生成并固定。所有选择方法必须使用完全相同的候选集合，从而保证性能差异只来自选择器，而不是候选生成预算、采样温度或搜索过程。候选生成器不属于本研究模型，其作用仅是提供具有一定多样性和 oracle 上界的训练、验证与测试数据。

## 6. 候选表示

每个候选 SQL 的输入由四类信息组成。

### 6.1 问题语义

- 自然语言问题；
- BIRD evidence/hint；
- 问题与候选 SQL 的 cross-encoder 表示。

### 6.2 SQL 信息

- SQL 文本；
- 表、列、聚合、过滤、排序、分组、子查询和集合操作；
- SQL 长度与结构复杂度；
- SQL 能否成功解析和执行。

### 6.3 Schema-aware 信息

- 候选 SQL 实际使用的表和列；
- 候选表列对原 Schema Linker 输出的 precision、recall 和 coverage；
- 是否使用未被 linker 选择的表列；
- 外键连接覆盖、连接图连通性及 join key 完整性；
- 问题、SQL 和候选 schema 的联合语义表示。

由于本研究不修改原 Schema Linker，若其只输出硬选择结果而没有逐表、逐列概率，则不伪造置信度。当前阶段使用集合一致性和语义编码作为 schema 信号；未来只有在 linker 原生提供可信概率时，才增加概率校准特征。

### 6.4 执行与生成信息

- 执行成功、报错类型、行列数量和空值比例；
- 执行结果簇大小及其占候选池比例；
- 候选来源、采样轮次和上游生成器分数。

上游生成轨迹不是选择器的必要输入，避免模型依赖某一种候选生成或搜索实现。

## 7. 模型与训练目标

### 7.1 基线模型

首先训练 Logistic Regression、LightGBM 或小型 MLP，以验证 schema 特征是否存在可学习信号。基线至少包括：

- Execution self-consistency；
- SQL-only；
- SQL + execution；
- SQL + schema + execution。

### 7.2 Pairwise reranker

对于同一问题中的正确候选 \(s^+\) 和错误候选 \(s^-\)，模型学习：

\[
f(q,s^+) > f(q,s^-)
\]

使用 RankNet 损失：

\[
\mathcal{L}_{rank}=-\log \sigma(f(q,s^+)-f(q,s^-))
\]

并加入较小权重的类别平衡 pointwise BCE，以稳定训练。可执行但语义错误、属于多数执行结果簇、表列高度相似或仅缺少一个条件的候选获得更高难例权重。

### 7.3 语义编码器

当前实现使用预训练 MiniLM cross-encoder 对以下文本对编码：

```text
Text A: question + evidence
Text B: candidate SQL + candidate-used schema + linker-selected schema
```

在训练数据较少时冻结编码器，仅训练轻量排序头，降低过拟合风险。后续数据规模达到要求后，再比较冻结编码器、部分解冻和全量微调。

### 7.4 保守决策策略

reranker 不直接无条件替换自一致性，而作为其残差：

\[
S_{final}(s)=S_{consistency}(s)+\alpha f(q,s)
\]

其中 \(\alpha\) 只在数据库隔离验证集上选择。另一方案是置信门控：只有神经模型相对基线候选的分差超过阈值时才改变选择，否则保留原结果。零权重或高阈值能够精确退化为原自一致性基线。

## 8. 数据构建

数据采用 BIRD train/dev，并严格遵循以下划分原则：

- 只使用 BIRD train 的 gold 执行标签训练；
- BIRD dev 只用于最终评估，不参与训练和阈值选择；
- 内部验证按数据库划分，而不是随机按问题划分；
- 每个问题保留所有去重后的真实生成候选；
- 训练集可进行 gold-positive 消融，但必须单独报告，并防止模型学习规范化 SQL 风格捷径；
- dev 候选池禁止注入 gold SQL。

有效训练问题必须同时包含正确和错误候选。仅增加全部错误的候选不能形成 pairwise 监督，因此后续候选生成应优先提高“同题正负共存率”，而不只是增加候选总数。

## 9. 当前实现与预实验

项目已经实现：

- 独立候选池的加载、去重、执行与 gold 标签构建；
- SQL 多格式解析及候选级特征构造；
- SQL、schema、join、execution 特征提取；
- GPU Pairwise MLP reranker；
- 冻结 MiniLM cross-encoder 的语义 reranker；
- 自一致性残差融合与置信门控；
- 按数据库隔离的验证方式；
- 训练集 gold-positive 消融。

当前 300 题训练池的统计为：

| 项目 | 数值 |
|---|---:|
| 请求生成的训练问题 | 300 |
| 成功生成候选的问题 | 286 |
| 去重候选 SQL | 956 |
| 无 Gold 注入的真实 pairwise 对 | 111 |
| 加 Gold 正例后的 pairwise 对 | 361 |
| Dev 问题 | 50 |
| Dev 候选 | 186 |

独立 dev 初步结果为：

| 方法 | EX |
|---|---:|
| 执行结果自一致性 | **62%** |
| 候选池 Oracle | 70% |
| 数值 Schema-aware reranker | 58% |
| 语义 Schema-aware reranker | 60% |
| Gold-positive SQL-only reranker | **62%** |
| Gold-positive Schema-aware reranker | 60% |

当前结果尚未证明 reranker 有效。内部验证提升但未迁移到未见数据库，说明主要问题是有效正负对不足、训练数据库覆盖有限，以及 gold-positive 带来的候选风格偏差。这一负结果将作为下一阶段数据构建和消融设计的依据，而不是被隐藏或解释为性能提升。

## 10. 实验设计

### 10.1 主要指标

- Execution Accuracy（EX）；
- Oracle Accuracy；
- Selection Accuracy on Solvable Questions；
- Gap Closure：

\[
\text{GapClosure}=\frac{EX_{reranker}-EX_{baseline}}{EX_{oracle}-EX_{baseline}}
\]

- reranker 改票次数、正确改票数和错误改票数；
- 不同数据库上的宏平均 EX；
- 多随机种子的均值和标准差。

### 10.2 核心消融

| 实验 | 目的 |
|---|---|
| Self-consistency | 原始选择基线 |
| SQL-only | 测量 SQL 结构信号 |
| SQL + execution | 测量执行反馈增益 |
| SQL + schema | 测量 schema 独立贡献 |
| SQL + schema + execution | 完整主模型 |
| 移除 linker-selected schema | 验证原 Schema Linker 输出价值 |
| 移除 candidate-used schema | 验证候选表列解析价值 |
| 随机问题验证 vs 数据库隔离验证 | 测量数据库泄漏影响 |
| 真实正例 vs Gold 注入 | 测量 SQL 风格捷径 |
| 裸神经选择 vs 残差融合 vs 置信门控 | 测量决策稳定性 |

### 10.3 成功标准

主张 Schema-aware reranker 有效需要同时满足：

1. 在相同候选池上显著超过 self-consistency；
2. 显著超过 SQL + execution reranker；
3. 至少三个随机种子方向一致；
4. 提升出现在未见数据库，而不只是在随机问题验证集；
5. 正确改票数高于错误改票数；
6. 不依赖 dev gold 调参或选择 checkpoint。

建议阶段目标为：首先将 50 题 pilot dev EX 从 62% 提升到至少 66%，即正确恢复当前四个 oracle-gap 问题中的两个；随后在更大 dev 子集和完整 BIRD dev 上复验。

## 11. 后续工作计划

### 阶段一：改进监督数据（第 1–3 周）

- 将训练候选覆盖扩展到至少 1,000 个问题和 40 个数据库；
- 统计每题正例数、负例数和候选去重率；
- 对没有正例的问题进行针对性补采样或使用更强生成配置；
- 构造错误连接、错误列、漏条件、错误聚合和错误排序等可执行难负例；
- 避免只用规范化 gold SQL 作为正例，优先保留真实生成的执行等价正例。

### 阶段二：稳定的 pairwise reranker（第 4–6 周）

- 比较 Logistic、LightGBM、MLP 和 cross-encoder；
- 采用数据库隔离的交叉验证；
- 对概率和 pairwise margin 进行校准；
- 分析每类错误的正确改票率和错误改票率。

### 阶段三：Groupwise 建模（第 7–9 周）

- 将同一问题的 Top-k 候选联合输入 critic；
- 显式比较候选间表列差异、谓词差异和执行结果差异；
- 比较 pointwise 初筛 + pairwise tournament 与完整 groupwise critic。

### 阶段四：选择器封装与完整评估（第 10–12 周）

- 将训练完成的 selector 封装为独立推理模块；
- 在完全相同的固定候选池上进行公平比较；
- 测量推理时延、GPU 显存、API 调用量和端到端 EX；
- 完成消融、失败案例分析和论文写作。

## 12. 风险与应对

| 风险 | 影响 | 应对措施 |
|---|---|---|
| 候选池没有正确 SQL | reranker 无法恢复 | 同时报告 oracle；优化生成覆盖率 |
| 同题正负对不足 | pairwise 训练无效 | 定向补采样，统计正负共存率 |
| Gold SQL 风格泄漏 | 内部验证虚高 | Gold 注入只做消融；优先使用生成正例 |
| Schema Linker 漏选 | 错误压低正确 SQL | schema 仅作软特征，不做硬过滤 |
| 数据库域迁移 | dev 性能下降 | 按数据库划分验证和交叉验证 |
| 执行等价标签噪声 | 错误监督 | 规范化结果、超时控制、人工抽检 |
| 候选池构建成本高 | 实验周期过长 | 离线生成、逐题缓存、并行与可恢复执行 |

## 13. 预期贡献

本项目预期贡献包括：

1. 一个不修改上游生成器和 Schema Linker 的独立候选 SQL 选择器；
2. 一套将候选实际表列、linker 选择和 join 完整性联合建模的 Schema-aware 表示；
3. 一套与具体生成算法解耦、支持可恢复处理的固定候选池构建流程；
4. 针对 Text-to-SQL 候选选择的数据库隔离评估与 Gap Closure 指标；
5. 对真实候选监督、Gold 正例注入、执行自一致性及跨数据库泛化的系统消融分析。

## 14. 项目定位

本方法属于监督式 SQL 候选选择，不应描述为严格的端到端 zero-shot 系统。上游候选生成器和 Schema Linker 保持固定，selector 使用 BIRD train 的执行标签训练。论文主张应集中在“Schema-aware 信息能否改善 SQL 选择”以及“如何稳定关闭 oracle gap”，而不是笼统宣称神经网络本身带来提升。

## 参考材料

- MSc-SQL：多候选联合判别与候选选择方法。
- `2025.naacl-long.107.pdf`。
- `2410.01943v1.pdf`。
- `2026.acl-long.313.pdf`。
- BIRD Text-to-SQL benchmark。
