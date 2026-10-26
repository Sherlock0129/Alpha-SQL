# NL2SQL Handbook 学习笔记：从模块优化到可验证、可学习的 SQL 选择

日期：2026-10-19。面向当前 Schema Linking、候选 SQL 选择和固定训练集课程学习项目。

仓库：[HKUSTDial/NL2SQL_Handbook](https://github.com/HKUSTDial/NL2SQL_Handbook)。本次查询到的 main 提交为 `bf4f2e0174c79d9a1d68dc02b761c522630f49ce`，提交时间 2026-08-31，提交说明为 “Update VLDB 2026”。关联综述使用 arXiv:2408.05109v7，2026-08-25 修订；其期刊出版信息为 TKDE 2025。

## 摘要

本次学习以 Handbook 的模块划分为入口，阅读 README、六个主题章节和关联综述相关部分，并对 29 篇代表性论文核验一手来源，补充到 2026 年 9 月的候选选择研究。目的不是复制论文列表，而是解释当前 Text-to-SQL 系统在何处获得收益、有哪些不能混淆的实验变量，以及这些证据如何约束我们的研究。文献显示，schema grounding、候选覆盖、语义验证、候选选择和监督质量应分别测量：强模型不一定从 schema 裁剪受益，可执行 SQL 不一定回答问题，而执行结果一致也不一定意味着普适语义等价。监督 pairwise、联合 critic、pointwise ORM 和中间监督已有直接先例，因此当前研究不宜只声称“训练一个 schema-aware reranker”。更具体的问题是，在冻结候选、标签、模型与推理规则后，可靠的 SQL 语义错误标签能否改善训练调度，减少达标时间且不增加普通题回归。本文给出技术地图、重点论文对比、错误标注设计、评测口径核验和建议阅读顺序；没有运行外部项目、训练模型或产生新的准确率结果。

## 1. 学习范围与研究问题

Handbook 是综述、论文目录和实践资源，不是一个统一实现全部方法的 Text-to-SQL 引擎。仓库的 chapter 提供模块说明，slides/report 提供讲义与分析，`src/dataset_analyze` 提供数据分析入口；不同论文代码位于各自项目。学习它不等于复现了其中所有系统。[仓库说明](https://github.com/HKUSTDial/NL2SQL_Handbook)

本次聚焦三个问题：

- **RQ1：当前技术的主要控制点是什么？** 如何区分 grounding、生成、约束、验证、选择和训练？
- **RQ2：哪些论文最接近我们的 SQL 选择器？** 原有想法中哪些部分已有先例，哪些比较才公平？
- **RQ3：错误感知课程如何接入？** 怎样从错误分析得到可靠训练信号，而不把新数据、额外监督或评测变化当作课程收益？

关联综述按 Model、Data、Evaluation、Error Analysis 组织研究；这四条线需要一起看，单独提升某个模块的分数不能推出端到端收益。[Survey, v7](https://arxiv.org/abs/2408.05109v7) [1]

## 2. 阅读方法与证据边界

使用 deep-research 的多视角流程，并行核验 grounding/生成、selector/训练、评价/反证三条线，再统一综合。主要检索词为论文全名，以及 `text-to-SQL verifier reranking`、`schema subsetting evaluation`、`semantic errors`、`curriculum Text-to-SQL 2026`。在目录之外补检索近期论文和原论文相互引用的近邻。

阅读的仓库章节为 [Pre-Processing](https://github.com/HKUSTDial/NL2SQL_Handbook/blob/main/chapter/Pre_Processing.md)、[Translation](https://github.com/HKUSTDial/NL2SQL_Handbook/blob/main/chapter/Translation_method.md)、[Post-Processing](https://github.com/HKUSTDial/NL2SQL_Handbook/blob/main/chapter/Post_Processing.md)、[Benchmark](https://github.com/HKUSTDial/NL2SQL_Handbook/blob/main/chapter/Benchmark.md)、[Evaluation](https://github.com/HKUSTDial/NL2SQL_Handbook/blob/main/chapter/Evaluation.md)、[Error Analysis](https://github.com/HKUSTDial/NL2SQL_Handbook/blob/main/chapter/Error_Analysis.md)。

29 篇代表性论文核验题名、作者和出版/预印本版本；涉及具体机制、消融或评测时查相关正文。不是全部论文逐页精读，不是穷尽所有新论文，也未复现作者结果。Schema Subsetting 的 VLDB PDF 获取失败，该条只依据官方会议摘要与作者元数据，不引用未读的实现细节。MaP-SQL 的方法说明主要依据摘要及方法段落，不据此宣布其全面优于训练方法。

Handbook 用于发现线索，论文结论以一手来源为准。本次发现需要注意的入口问题包括：

- README 的 “Beyond Static Pipelines” 链接指向 `2504.15077`，该编号实际是 Think2SQL；正确的 SquRL 论文为 [2602.15564](https://arxiv.org/abs/2602.15564)。
- 部分链接存在嵌套 Markdown、重复条目和旧题名；不能由目录先后顺序推断当前排名。
- Benchmark 章节本身仍以较早基准为主，不能代替 README 更新及 Spider 2.0 等原文。
- Translation 章对 RAT-SQL 的 beam/rerank 描述不宜直接采信；原文核心是 relation-aware encoder 和结构化 decoder，不应把它引为独立 SQL reranker。

这些核验问题不否定 Handbook 的导航价值，但说明“读目录”和“形成可引用证据”是两步。

## 3. 技术地图：六个控制点与三种效率

| 控制点 | 输入与输出 | 主要解决什么 | 不能保证什么 |
|---|---|---|---|
| Grounding | 问题、schema、值、证据 → 相关上下文 | 表列定位、值匹配、业务知识 | 召回完整不等于 SQL 逻辑正确 |
| Generation / Search | 上下文 → 一个或多个 SQL | 逻辑组合、候选多样性与覆盖 | 更多候选不保证存在正确答案 |
| Constraints / Compiler | token/typed plan → 合法 SQL | 语法、表列、作用域、连接规则 | 合法性不等于回答用户问题 |
| Verification / Revision | 问题与 SQL → 判断、反馈或修订 | 检测可执行但语义错误的 SQL | 检测器和修订本身也会出错 |
| Selection | 固定候选 → 最终 SQL / 拒答 | 缩小 oracle 与最终正确率差距 | 无法选出池中不存在的正确 SQL |
| Training / Data | 数据与反馈 → 参数或记忆 | 学习 grounding、生成或判断能力 | 更多监督不等于同数据下训练更高效 |

这些是分析轴，不是互斥的系统标签；一个方法可同时改变多个位置。端到端论文的总提升不能未经消融分配给其中某一个模块。

还必须区分三种效率：训练效率是达标所需 updates、tokens、GPU 时间；推理效率是每题模型调用、tokens、延迟和执行次数；SQL 执行效率是数据库运行生成 SQL 的成本。复杂度路由属于推理分配，VES 属于 SQL 执行评价，都不能直接证明课程学习减少训练时间。

## 4. Schema Linking：从“分类准不准”转向保留了什么信息

### 4.1 联合关系编码与显式筛选不是同一种方法

[RAT-SQL](https://aclanthology.org/2020.acl-main.677/) [2] 把问题词、表列和 PK/FK、名称/值匹配等关系纳入 relation-aware 编码；[RESDSQL](https://arxiv.org/abs/2302.05965) [3] 则显式训练表列分类并排序、筛选，再进行 skeleton-aware SQL 生成。前者说明 schema-aware 可以是表示层的关系建模，后者说明筛选和呈现顺序也是干预，不能把二者都简化成“表列二分类”。

RESDSQL 的整个系统同时涉及筛选、排序和 skeleton；其效果不能直接归因于 classifier 更准。对我们的比较，应分别看原始输出、共同 PK/FK 补全后的输入、字段呈现顺序，以及生成器实际使用情况 [2][3]。

### 4.2 强模型是否仍需要裁剪？答案依赖规模

[The Death of Schema Linking?](https://arxiv.org/abs/2408.07702v2) [4] 在 schema 能放入上下文等条件下揭示删掉必要信息的风险；[VLDB 2026 Schema Subsetting 比较](https://vldb.org/2026/program.html) [5] 则进一步区分小 schema 与大 schema：一些方法在大 schema 上有准确率或 token 收益，小 schema 裁剪可能损失更多。两者并未证明 grounding 无用，而是提示硬裁剪必须与 full-schema 基线、模型能力和上下文预算一起评价。

这与 [3] 不矛盾：微调 parser 与较强 LLM、输入预算与 schema 规模不同。应区分“识别相关字段”与“提前永久删除其余字段”；也要区分字段平均 recall 与整题必要信息完整召回。

### 4.3 当前 287 题结果应怎样放入文献背景？

[项目受控报告](D:/University/Alpha-SQL/LINKER_ABLATION_287_RESULTS.md) 中，共同 PK/FK 补全后完整召回为 LLM 248/287、Jev 246/287；最终 local EX 为 209/287 和 199/287，差值区间跨 0。它不支持 Jev 替换收益，也不支持“schema linking 整体无用”。

更窄的解释是：原始 recall 的差异未保留到实际下游输入，且生成器仍会犯投影、过滤、聚合等错误。两臂都裁剪了 schema，因此还没有回答 full-schema 是否更好。104 题表列集合相同但表顺序不同，也提醒我们把排序与内容变化分开。当前研究主线已转向训练效率，不必为补充所有 linker 对照立即重新生成候选。

## 5. 生成与约束：减少机械错误，不能替代语义判别

[PICARD](https://aclanthology.org/2021.emnlp-main.779/) [6] 在生成 token 时增量解析并拒绝非法续接；[Alpha-SQL](https://arxiv.org/abs/2502.17248v2) [7] 则搜索 LLM 语义动作轨迹，以执行一致性提供无 gold 的代理奖励。一个主要约束可走的路径，另一个主要探索更多路径；二者均不能让“执行成功”成为语义正确证明。

[DeepEye-SQL](https://arxiv.org/html/2510.17586v3) [8] 在系统层结合多路 grounding、关系闭包、多版本生成、工具检查和置信度门控选择。与 [7] 相比，它更强调可检查的编排；但确定性 checker 只意味着给定规则可重复执行，不意味着规则对自然语言语义完备。

对当前项目的启发是分层定位错误：你的 Jev typed decision → Python AST → SQLite 思路属于外部受约束构造，与 PICARD 的 token 增量解析不是同一实现。局部 compiler 能力不足会造成 coverage miss；字段齐全却聚合错误属于生成语义；正确候选被 SC 丢掉才是 selection error。不能用后两者的改进掩盖 compiler 支持范围，也不能把空结果直接判作错误。

## 6. 候选选择：现有研究已覆盖 pointwise、pairwise 和 listwise

### 6.1 从生成概率转向问题—SQL 的整体判断

[MetaSQL](https://arxiv.org/abs/2402.17144) [9] 用 metadata 控制候选生成并两阶段排名；[CHASE-SQL](https://arxiv.org/abs/2410.01943) [10] 用多路径生成和偏好优化的候选对选择。二者共同说明生成和选择可以分开，但同时也改变了候选池，不能把整体结果当成固定池 selector 的单独收益。

CHASE-SQL 对候选对使用两条 SQL 所涉及 schema 的并集，训练中交换正负位置，推理时比较并累计胜场 [10]。因此，“schema-aware pairwise selector”本身已有直接先例；本项目的区别需要落在可靠错误信号和训练调度，而不是首次把 schema 放进 reranker。

### 6.2 独立评分与联合比较解决不同问题

[MSc-SQL](https://aclanthology.org/2025.naacl-long.107/) [11] 让 critic 联合查看多个 SQL 及执行反馈；[GradeSQL](https://arxiv.org/abs/2509.01308v2) [12] 训练 outcome verifier，为候选提供独立分数。前者利用候选间对比，后者推理更容易扩展；不能只凭架构名称认定 listwise 或 pairwise 总是更优。

MSc-SQL 的固定候选分析适合借鉴：同时给出 oracle、SC、逐条 critic 与联合 critic。GradeSQL 的自动执行标签适合组织数据，但其 balanced 训练用下采样改变样本量，不是同一多重集的纯课程实验 [11][12]。

名称也需精确：`2025.naacl-long.107` 是 **MSc-SQL: Multi-Sample Critiquing Small Language Models For Text-To-SQL Translation**；[MCS-SQL](https://aclanthology.org/2025.coling-main.24/) 是另一篇关于 multiple prompts / multiple-choice selection 的论文，不能混用实验数字。

### 6.3 中间监督、独立验证和免微调记忆是不同路线

[OpenSQL](https://www.vldb.org/pvldb/vol19/p1628-li.pdf) [13] 用合成的中间监督训练 grounding、生成和 clause/semantic-level selector；[VLDB 2026 Verification](https://www.vldb.org/pvldb/vol19/p3732-alrashed.pdf) [14] 则把无需 gold 的候选语义验证作为独立任务，使用 SQL→NL 的 round-trip critique 和合成小数据库的 execution consistency。前者增加监督内容，后者增加诊断机制，都不能直接作为 fixed-set schedule 的效果证据。

9 月的新论文 [MaP-SQL](https://arxiv.org/abs/2609.00834v1) [15] 进一步以从训练数据提炼的记忆辅助 listwise selection，并通过多排列排名缓解位置偏差。与 [12][13] 相比，它把部分参数学习换成显式记忆和检索。免微调不等于无训练数据依赖，也不等于无成本；记忆构建和推理 token 必须计入。它适合作为未来替代路线，不需要现在把课程研究改成 memory 系统。

| 路线 | 判断粒度 | 主要信号 | 对当前项目的直接用途 |
|---|---|---|---|
| MetaSQL [9] | 两阶段排名 | metadata、全局语义匹配 | 区分候选多样性与重排收益 |
| CHASE-SQL [10] | Pairwise | 问题、hint、候选 schema 并集、偏好监督 | 位置交换、候选对与 schema-aware 基线 |
| MSc-SQL [11] | 联合候选 critic | SQL 差异与执行结果 | 同候选池的 oracle/SC/critic 分解 |
| GradeSQL [12] | Pointwise ORM | 问题/schema/SQL、执行标签 | 简单、可训练的候选评分基线 |
| OpenSQL [13] | 分步比较 | clause 与 semantic 中间监督 | 错误标签粒度的近邻，不能混成纯采样 |
| Verification [14] | Accept/reject 或评分 | 反译、合成测试输入 | 标注审计、低置信度拒答；需核算成本 |
| MaP-SQL [15] | Listwise | 结构化记忆、多排列 | 免微调替代路线与位置鲁棒性 |

表中方法解决的监督、成本和池分布并不相同。公平比较 selector 时，固定 candidate IDs、SQL 与执行缓存；公平比较课程时还要固定 selector 结构、监督和推理融合规则。

## 7. 当前训练技术：数据、奖励、课程与编排不可混为一谈

### 7.1 合成数据与细粒度监督

[OmniSQL](https://arxiv.org/abs/2503.02240) [16] 强调大规模高质量合成数据；[FINER-SQL](https://arxiv.org/abs/2605.03465v1) [17] 在 RL 中使用 memory 和 atomic reward 提供比 0/1 更细的信号。两者都与错误学习有关，但分别改变数据和奖励，不是对固定候选 pair 重新排序。

[ReViSQL](https://arxiv.org/abs/2603.20004v4) [18] 又把训练标签可靠性放在核心：专家核验数据，并针对单实例结果巧合及忽略 evidence 设计验证/过程信号。它与 [8] 的系统编排路线形成有价值的张力——一个强调可验证流程，一个强调清洁监督和模型训练；不同模型、数据与评测下的结果不能判定某条路线已普遍取代另一条。

本项目最可借鉴的是问题界定：execution-positive 不必然 semantic-positive；AST 重合可作部分信号，但不同合法 SQL 可以实现相同含义。不能把原子结构分数直接当作语义真值，再据此声称错误课程修复了语义。

### 7.2 新补充的 SQL 课程论文

[LeGo-Code](https://arxiv.org/html/2604.18254v1) [19] 直接比较 Text-to-SQL 微调与复杂度课程，并提出按复杂度层级顺序训练和组合 adapters。它报告朴素顺序课程不如预期，但最终方法同时改变 adapter 组织；不能把全部效果归于排序，也不能由它宣布所有课程无效。其一 epoch 顺序设定与基线多 epoch 的文字描述还提醒我们核对曝光和预算，不直接抄论文 headline。

[SQLConductor](https://arxiv.org/abs/2606.23537v1) [20] 用 MCTS 搜索 workflow，结合稳定性监督训练下一步动作策略，再做 curriculum RL。与 LeGo-Code 的生成器适配不同，它学习的是编排策略；与本项目的冻结候选监督 reranker 也不同。它提示离线搜索轨迹可能有可复用价值，但搜索、稳定性估计和 RL rollout 都要计入准备成本。

这两篇是本轮在此前课程综述之外补到的直接 SQL 近邻。它们应加入相关工作，不能再用“没有检索到 SQL 课程”暗示方向空白；但它们也没有直接完成我们的同池、同模型、同预算错误语义增量实验。

### 7.3 训练效率与推理调度

[EllieSQL](https://arxiv.org/abs/2503.22402v2) [21] 按 query 复杂度路由到不同成本的推理管线；[SquRL](https://arxiv.org/abs/2602.15564v1) [22] 学习动态构造 workflow；[20] 进一步关注中间反馈下的逐步动作。它们共同支持按问题分配计算，但研究对象是推理策略，不等于每个 SGD step 应采哪条训练样本。

对当前课题，控制变量应继续收窄：先固定模型和候选，再比较 sample scheduling；不要在同一主实验同时换 generator、加入 memory、修改 reward、扩充 SQL 或学习编排，否则无法判断课程贡献。

## 8. 评价与错误分析：总 EX 只是入口

### 8.1 基准测量的任务不同

[BIRD](https://proceedings.neurips.cc/paper_files/paper/2023/hash/83fc8fab1710363050bbd1d4b8cc0021-Abstract-Datasets_and_Benchmarks.html) [23] 强调数据库内容、外部知识及查询效率；[Spider 2.0](https://proceedings.iclr.cc/paper_files/paper/2025/file/46c10f6c8ea5aa6f267bcdabcb123f97-Paper-Conference.pdf) [24] 把企业 schema、方言、文档与多步工作流纳入任务。因此，SQLite/BIRD train 上的候选选择结果不能直接声称解决企业 Text-to-SQL；不同版本和 setting 的成功率也不能混作同一排行榜。

[Distilled Test Suites](https://aclanthology.org/2020.emnlp-main.29/) [25] 通过多个数据库上的行为测试近似语义正确性；与单库 EX 相比更能揭示偶然相等，但通过有限测试仍不是完整逻辑证明。对 DISTINCT、重复 join、边界和并列极值，可借鉴其反例思想，而不是把现有 Spider 测试库直接套到所有 BIRD 查询。

### 8.2 错误分类和标签审计必须分离

[NL2SQL-BUGs](https://arxiv.org/abs/2503.11984) [26] 提供 9 大类、31 子类的语义错误体系；[Pervasive Annotation Errors](https://arxiv.org/abs/2601.08778) [27] 展示 benchmark 标注问题会改变系统评价。前者帮助描述模型错在哪里，后者提醒负标签可能首先需要审计。BUGs 含正确和错误 SQL，不能把全部 2,018 条都称为错题。

关于标注噪声的百分比必须保留原研究的样本、抽样方式和修订协议：Mini-Dev、公开的 Spider2-Snow 子集或被 verifier 拒绝的定向案例，不是整个 BIRD 的总体错误率。不要挑本模型失败题单方面改 gold 后继续对外宣称官方 EX 上升 [14][27]。

### 8.3 官方脚本与项目本地判分有可核验差异

本次核验的 [BIRD 历史公开 evaluation.py](https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/main/bird/llm/src/evaluation.py) 第 24 行使用 `set(predicted_res) == set(ground_truth_res)`。该实现丢弃重复行次数、忽略行顺序，但保留结果 tuple 内的列顺序。当前项目本地签名保留重复次数并做浮点舍入，因此两者不是同一协议。这个发现不自动改写已有实验结果，也不代表所有官方后续版本都相同。

后续建议双轨报告：固定 evaluator 版本的官方协议 EX，用于外部可比；local multiset diagnostic，用于重复行等语义诊断。另记录 ORDER BY、列对齐、浮点、timeout 和空结果规则。不能把更严格的局部规则直接改名为官方 BIRD EX。

### 8.4 从单分数到模块和错误维度

[SQLMorph](https://amine.io/papers/2026-icde-sqlmorph.pdf) [28] 用受控 mutation 和细粒度结果指标分析过预测/欠预测；[NL2SQLBench](https://arxiv.org/abs/2604.16493) [29] 按 Schema Selection、Candidate Generation、Query Revision 拆解模块效果和成本。二者提示我们把最终错误拆成信息缺失、候选没生成、选择失败和评价不确定，而非仅写“语义理解不足”。

两项边界同样重要：SQLMorph 的新 mutation 若进入训练，就改变 fixed training set；Verification 论文的 selective accuracy 是在选择回答的子集上计算，拒答后精度上升不等于所有题的完整 EX 上升。若未来加 abstention，应同时报告 answer coverage / risk，不能缩小分母掩盖错误。

## 9. 如何改进当前错误感知课程的定义？

Handbook 的“错误位置—错误原因”加上 [26][27] 的语义分类/标注审计，可转成以下记录结构。下面是本项目建议，不是仓库已经实现的标签器：

| 字段 | 例子 | 使用限制 |
|---|---|---|
| `execution_label` | 官方协议正误、本地 multiset 正误 | 两种标签分开保存，不静默覆盖 |
| `semantic_error_types` | 属性缺失、显式条件遗漏、比较符错配、计数对象错误 | AST diff 只生成待复核弱标签 |
| `error_location` | SELECT、WHERE、JOIN、GROUP BY、ORDER BY | 位置不是根因 |
| `suspected_cause` | 值未检索、schema 被删、忽略 hint、组合逻辑错误 | 有输入/轨迹证据才归因，否则 unknown |
| `repairability` | coverage miss、可修复 selection error、SC-correct anchor | 负例校准价值与可修复 EX 空间分开 |
| `annotation_status` | verified、ambiguous、suspected gold issue、unreviewed | 高 loss 不自动表示噪声或高 utility |
| `learning_state(t)` | 当前 margin、loss 变化、遗忘事件 | 随模型变化，不等于静态 SQL 错误标签 |

例如同样是漏 WHERE 条件：schema 中没有必要列、值检索缺失、hint 被忽略、生成时省略过滤，可能产生相同 AST 差异。只有第一种有直接 linker 证据；不能仅凭最后 SQL 反推所有原因。错误类型在单个差异层面应定义清晰，但一条 SQL 可以包含多个差异，不必强迫整题互斥归类。

对小训练池，先把 BUGs 子类映射到 4—6 个有足够支持的粗组，保留细类作分析；记录映射与多标签归一化规则。外部 taxonomy 可以借鉴，但把外部 BUGs 样本直接加入训练、或增加教师纠错解释，都是额外实验变量。

## 10. 与当前结果的交叉解释

### 10.1 为什么不继续用单一 schema 指标作结论？

前述 [3]—[5] 表明裁剪策略依赖模型、规模和输入顺序；本地 [287 题实验](D:/University/Alpha-SQL/LINKER_ABLATION_287_RESULTS.md) 的共同补全也消除了大部分 Jev raw recall 优势。合理结论是 raw recall 不能替代下游验证，不是所有显式 linker 都应删除，也不是新增一个更复杂 linker 必然有收益。

### 10.2 为什么 selector 有局部信号却没有总收益？

[100 题训练报告](D:/University/Alpha-SQL/HARD_CASE_PAIRWISE_100_REPORT.md) 中，hard-focused 模型修复 4 个 hard cases，却让 5 个普通问题回归，总体 75%，SC 76%。与 [10]—[15] 对照，这说明监督选择已有合理任务基础，但当前 tiny hard-only 训练分布、特征表示和安全改选规则仍可能不足；不能把 4/5 修复率当总体改进。

课程研究要问的是如何更快、更稳地学会区分这些情况，不能凭课程声称克服候选 coverage 上限。只有池中同时存在正负候选的题可产生相应 pairwise 排序监督；纯负例题仍可有校准用途，但无法凭重排修复其 coverage。

### 10.3 哪些更新应先于新一轮训练？

1. 把 evaluation 协议、执行标签与语义标签分离；不静默重标历史结果。
2. 在原 train IDs 内核验粗错误组与标签置信度，测试分析不反馈当前 scheduler。
3. 所有组共同采用可形成多次更新的 mini-batch 训练流程。现有 38/82 个 pair 小于默认 batch size 128，仅在全 batch 内换序无法有效检验 order-only。
4. 固定 SC 特征和推理 gate，比较 Uniform、Difficulty-only、Static error-balanced、One-shot error weighting、Dynamic error-aware；涉及 pacing 就加入 random pacing。
5. 用标签打乱、同曝光序列乱序、loss/progress-only 消融，证明错误语义是否提供增量。计入诊断和调度总成本，记录回归与遗忘。

完整训练协议见此前的 [固定训练集课程综述](D:/University/Alpha-SQL/reports/ERROR_AWARE_CURRICULUM_SURVEY_ZH.md)。本轮新增的 [19][20] 和最新 selector [15] 应补入相关工作；它们不改变“先用统一小实验检验机制”的建议。

## 11. 开放问题与建议阅读路线

值得继续研究的具体空隙是：语义错误组相对 loss/margin 的增量；弱标签噪声下的动态采样稳定性；计入诊断后是否真有训练加速；以及何时应保持 SC 而不追逐 minority candidate。已有 [10][13][19][20] 提供相邻机制，但未替本项目完成这些同数据控制实验。

建议按以下顺序精读，阅读后各产出一个可检查的笔记，而非仅摘摘要：

| 顺序 | 论文 | 要回答的问题 / 产物 |
|---|---|---|
| 1 | Handbook Survey [1] + NL2SQL-BUGs [26] | 建立模块图与可复核的错误 taxonomy |
| 2 | CHASE-SQL [10] | 画清 candidate pair、schema 输入、位置交换、推理胜场流程 |
| 3 | MSc-SQL [11] + GradeSQL [12] | 比较联合 critic 与独立 ORM，列清候选池控制条件 |
| 4 | OpenSQL [13] | 区分 clause-level 中间监督与仅有执行标签的监督 |
| 5 | Verification [14] + Annotation Errors [27] | 写清验证信号、错误率分母、标注审计与成本 |
| 6 | LeGo-Code [19] | 核对顺序、曝光次数、adapter 容量和遗忘，不能只看总提升 |
| 7 | DeepEye-SQL [8] | 梳理规则检查与 SC-confidence gate 的能力边界 |
| 8 | MaP-SQL [15] | 理解显式记忆和位置鲁棒性，记录构建/推理成本 |
| 9 | SQLConductor [20] + SquRL [22] | 区分训练采样、workflow 策略和离线搜索成本 |

如果本周只精读四项，优先 BUGs、CHASE-SQL、OpenSQL、LeGo-Code：分别约束错误定义、已有选择器先例、监督粒度与课程比较的公平性。RAT-SQL/RESDSQL/PICARD 用于补基础；不需要因为新论文更复杂就立即实现全部模块。

## 12. 结论

**RQ1：** 当前技术不是仅把问题交给更强 LLM，而是在 grounding、搜索、约束、独立验证、选择以及数据监督之间分配工作；但复杂系统与清洁数据训练是并存路线，不存在由本次阅读即可宣布的统一赢家。

**RQ2：** schema-aware pairwise、联合多候选 critic、ORM 和 clause-level selector 都已有明确先例。我们的研究应以这些为任务基础，把真正的待证增量限定在训练时的错误语义与进度信号，而不是泛称提出 SQL reranker。

**RQ3：** 先保证标签可信、分母明确和训练流程能体现顺序作用，再检验课程。错误位置不等于根因，执行正误不等于语义真值，推理省 token 不等于训练更快；这三条是 Handbook 与近期论文对当前项目最重要的共同约束。

本次使用 deep-research 形成了可追溯的技术地图，并通过反证与近期补检索修正了只看模块分数、只看 hard 修复率和把 fixed-set 当作 fixed-budget 的风险。所有项目建议仍待实验验证；没有修改模型、执行外部仓库脚本、调用付费模型或启动训练。

## 参考文献

[1] Xinyu Liu, Shuyu Shen, et al. “[A Survey of Text-to-SQL in the Era of LLMs: Where are we, and where are we going?](https://arxiv.org/abs/2408.05109v7)” TKDE, 2025；本次阅读 arXiv v7，2026-08-25。

[2] Bailin Wang, Richard Shin, et al. “[RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers](https://aclanthology.org/2020.acl-main.677/).” ACL, 2020.

[3] Haoyang Li, Jing Zhang, et al. “[RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL](https://arxiv.org/abs/2302.05965).” AAAI, 2023.

[4] Karime Maamari, Fadhil Abubaker, et al. “[The Death of Schema Linking? Text-to-SQL in the Age of Well-Reasoned Language Models](https://arxiv.org/abs/2408.07702v2).” Table Representation Learning Workshop at NeurIPS, 2024.

[5] Kyle Luoma, Arun Kumar. “[A Comparative Evaluation of Schema Subsetting for LLM-based NL-to-SQL over Large-Schema Databases](https://vldb.org/2026/program.html).” PVLDB 19(9), 2026. 本次依据官方摘要；[作者项目](https://github.com/KyleLuoma/SKALPEL-subsetting-evaluation)提供论文元数据。

[6] Torsten Scholak, Nathan Schucher, Dzmitry Bahdanau. “[PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models](https://aclanthology.org/2021.emnlp-main.779/).” EMNLP, 2021.

[7] Boyan Li, Jiayi Zhang, et al. “[Alpha-SQL: Zero-Shot Text-to-SQL using Monte Carlo Tree Search](https://arxiv.org/abs/2502.17248v2).” ICML, 2025.

[8] Boyan Li, Chong Chen, et al. “[DeepEye-SQL: A Software-Engineering-Inspired Text-to-SQL Framework](https://arxiv.org/abs/2510.17586v3).” SIGMOD / PACMMOD, 2026；初稿 2025，本次依据 v3。

[9] Yuankai Fan, Zhenying He, et al. “[MetaSQL: A Generate-Then-Rank Framework for Natural Language to SQL Translation](https://arxiv.org/abs/2402.17144).” ICDE, 2024.

[10] Mohammadreza Pourreza, Hailong Li, et al. “[CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL](https://arxiv.org/abs/2410.01943).” ICLR, 2025；初稿 2024。

[11] Satya Krishna Gorti, Ilan Gofman, et al. “[MSc-SQL: Multi-Sample Critiquing Small Language Models For Text-To-SQL Translation](https://aclanthology.org/2025.naacl-long.107/).” NAACL, 2025.

[12] Mattia Tritto, Giuseppe Farano, et al. “[GradeSQL: Test-Time Inference with Outcome Reward Models for Text-to-SQL Generation from Large Language Models](https://arxiv.org/abs/2509.01308v2).” arXiv:2509.01308v2, 2025. 同作者后续稿不作独立重复证据。

[13] Ruilin Hu, Yuyu Luo, et al. “[OpenSQL: Data-Efficient Text-to-SQL for Open-Source LLMs via Synthesized Intermediate Supervision](https://www.vldb.org/pvldb/vol19/p1628-li.pdf).” PVLDB 19(7), 2026.

[14] Tarfah Alrashed, Madhup Sukoon, et al. “[Developing and Benchmarking Verification Algorithms to Improve Text-to-SQL Generation](https://www.vldb.org/pvldb/vol19/p3732-alrashed.pdf).” PVLDB 19(11), 2026.

[15] Yeonseok Jeong, Soyoung Yoon, et al. “[Replacing Training with Memory: Listwise Selection for Text-to-SQL](https://arxiv.org/abs/2609.00834v1).” arXiv:2609.00834v1, 2026-09-01；论文页标注 Findings of EMNLP 2026。

[16] Haoyang Li, Shang Wu, et al. “[OmniSQL: Synthesizing High-quality Text-to-SQL Data at Scale](https://arxiv.org/abs/2503.02240).” PVLDB 18(11), 2025.

[17] Thanh Dat Hoang, Thanh Trung Huynh, et al. “[FINER-SQL: Boosting Small Language Models for Text-to-SQL](https://arxiv.org/abs/2605.03465v1).” arXiv:2605.03465v1, 2026. Handbook 采用扩展题名，本文依 arXiv 题名引用。

[18] Yuxuan Zhu, Tengjun Jin, et al. “[Human-Level Text-to-SQL via Reinforcement Learning on Verified Data, Without Pipeline Engineering](https://arxiv.org/abs/2603.20004v4).” arXiv:2603.20004v4, 2026-08-21. 对应 ReViSQL 研究；标题主张受其具体实验条件约束。

[19] Salmane Chafik, Saad Ezzini, Ismail Berrada. “[LeGo-Code: Can Modular Curriculum Learning Advance Complex Code Generation? Insights from Text-to-SQL](https://arxiv.org/abs/2604.18254v1).” arXiv:2604.18254v1, 2026；全文列 EASE 2026 会议信息。

[20] Yizhang Zhu, Zhangyang Peng, et al. “[SQLConductor: Search-to-Policy Learning for Step-wise Text-to-SQL Orchestration](https://arxiv.org/abs/2606.23537v1).” arXiv:2606.23537v1, 2026.

[21] Yizhang Zhu, Runzhi Jiang, et al. “[EllieSQL: Cost-Efficient Text-to-SQL with Complexity-Aware Routing](https://arxiv.org/abs/2503.22402v2).” COLM, 2025.

[22] Yihan Wang, Peiyu Liu, et al. “[Beyond Static Pipelines: Learning Dynamic Workflows for Text-to-SQL](https://arxiv.org/abs/2602.15564v1).” arXiv:2602.15564v1, 2026. 方法名 SquRL，不与 Think2SQL 混用。

[23] Jinyang Li, Binyuan Hui, et al. “[Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs](https://proceedings.neurips.cc/paper_files/paper/2023/hash/83fc8fab1710363050bbd1d4b8cc0021-Abstract-Datasets_and_Benchmarks.html).” NeurIPS Datasets and Benchmarks, 2023.

[24] Fangyu Lei, Jixuan Chen, et al. “[Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows](https://proceedings.iclr.cc/paper_files/paper/2025/file/46c10f6c8ea5aa6f267bcdabcb123f97-Paper-Conference.pdf).” ICLR, 2025.

[25] Ruiqi Zhong, Tao Yu, Dan Klein. “[Semantic Evaluation for Text-to-SQL with Distilled Test Suites](https://aclanthology.org/2020.emnlp-main.29/).” EMNLP, 2020.

[26] Xinyu Liu, Shuyu Shen, et al. “[NL2SQL-BUGs: A Benchmark for Detecting Semantic Errors in NL2SQL Translation](https://arxiv.org/abs/2503.11984).” KDD, 2025.

[27] Tengjun Jin, Yoojin Choi, et al. “[Pervasive Annotation Errors Break Text-to-SQL Benchmarks and Leaderboards](https://arxiv.org/abs/2601.08778).” PVLDB, 2026.

[28] Mohammadhossein Malekpour, Mohamed Riahi, et al. “[SQLMorph: Query Mutation and Fine-Grained Metrics for Text-to-SQL Evaluation](https://amine.io/papers/2026-icde-sqlmorph.pdf).” ICDE, 2026.

[29] Shizheng Hou, Wenqi Pei, et al. “[NL2SQLBench: A Modular Benchmarking Framework for LLM-Enabled NL2SQL Solutions](https://arxiv.org/abs/2604.16493).” PVLDB 19(5), 2026.
