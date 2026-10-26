# 固定候选池下 SQL 选择器的错误感知课程：从采样收益到 LoRA 净训练效率

更新日期：2026-10-26。本文为文献调研与研究设计，不是新训练结果。没有下载模型、调用付费模型 API、租用 GPU 或启动训练。

## 摘要

本报告研究：固定训练数据、候选 SQL 池、7B 基础模型及损失函数时，能否通过错误感知课程减少 SQL 选择器 LoRA 微调达到共同质量要求的成本。调研围绕 SQL 判别模型、反馈式数据调度、验证器课程以及效率反证四条线，综合 20 项研究及 GradeSQL 官方训练代码。文献已分别覆盖单候选正确性监督、技能分组调度、难度与效用结合、错误反馈和遗忘回放，因此“给 SQL 选择器加课程”不足以构成强创新。较有价值的检验是：SQL 缺陷语义是否在相同训练动态与回放之外提供可利用的信息，并且收益能够抵消标注、探测与调度开销。现有证据支持实现这项研究，但没有证明该具体设置必然提速。报告据此提出 pointwise 共同基线、语义与非语义强对照、质量—回归共同约束和净达标成本评价，并明确哪些阴性结果应推翻原假设。

## 1. 研究问题与固定边界

当前需要回答的不是“能否训练 SQL 选择器”，而是“相同资源配方下，如何安排已有数据更有效”。冻结三个问题：

- RQ1：哪些已有方法能作为 7B SQL 选择器及课程调度的直接基础？
- RQ2：SQL 错误语义相对 loss、学习进度、普通分组与保持性回放，可能提供什么独立增量？
- RQ3：如何区分更少训练步数、更高样本效率和真正更低的达标总成本？

研究对象是文本 SQL selector/verifier，不是 SQL 生成器。正式比较前固定 Question、Evidence、Schema、候选 SQL、执行标签、基础模型版本、LoRA 配置及逐样本损失。各方法只能改变共同池内样本的训练顺序、出现频率及既定回放预算的分配，不生成新 SQL，不修改输入内容，不追加解释监督，不更换损失。

“固定训练集”不意味着必须使用当前小规模 pilot 集；允许先准备足够大的共同数据，再冻结后开展比较。错误类型属于额外训练侧信息，即使没有新增问题或 SQL，也有标注成本，不能称零额外监督。

## 2. 检索方法与证据范围

检索分为 SQL selector 与 LoRA、动态调度与保持性、随机对照与效率审计三个独立视角；第一轮按任务及机制宽检，第二轮沿 GradeSQL、Skill-It、LEAD、DUCL、RoD 等方法及其引用链补检。随后补查错误感知验证器、固定曝光控制及学习率交互。关键词包括 SQL selection/verifier、curriculum LoRA、error-aware curriculum、learning progress、data utility、semantic groups、replay、time-to-target。

本报告精选 20 项研究，覆盖 2021—2026 年，并单独核查 GradeSQL 官方源码固定 commit。采用 ACL Anthology、PMLR、AAAI、ICLR 论文、正式期刊和作者 arXiv；作者主页仅用于确认发表状态。主要核验标题、作者、版本、方法和所引用实验口径，不直接采信二手解读。2026-10-26 的补检没有获得完全对应本设置的新直接证据，但这不是“首次”的证明。

部分网页存在访问限制：HiEval 官方大体积 PDF 未完整取得，因此仅使用官方摘要支持的组成，不引用具体消融；OpenReview 部分入口触发验证，采用可访问的正式论文、作者页面或 arXiv 交叉核对。RoD、EGT 在本文按预印本使用。检索不是穷尽性系统综述，跨任务结果也不等于本设置的独立复现。

## 3. 文献分类：先分清改变了什么

按论文在本研究中承担的主要证据角色分类，而不是假设一篇论文只有一种机制。

| 分支 | 主要问题 | 代表工作 | 可以支持什么，不能支持什么 |
|---|---|---|---|
| SQL 训练对象 | 模型如何判别候选、表达错误 | GradeSQL、CHASE-SQL、MSc-SQL、SAC-SQL、ErrorLLM | 支持训练判别器及错误建模；不证明课程提速 |
| 数据调度信号 | 如何利用表现、效用、组关系和遗忘 | Skill-It、DUCL、LEAD、DoGE、RoD、CURE、FOT | 提供调度机制与强基线；任务、预算和信息条件并不相同 |
| 相邻验证器课程 | 课程是否已用于 judge/reward model | HiEval、RESTL、EGT | 否定“验证器课程本身全新”；不直接回答固定 SQL 池的 LoRA 成本 |
| 效率与反证 | 收益是否来自顺序，是否计入全部成本 | When Do Curricula Work、RHO-LOSS、Beyond Random Sampling、Random Selection、LR Decay | 决定评价和归因方式；不能直接搬用加速倍数 |

空缺不是“没有错误感知课程”，而是本次检索未得到以下组合的直接结论：固定 SQL 判别数据、固定 7B LoRA 配方、相同监督损失，控制训练动态与回放后，错误语义仍降低满足质量及回归约束的净成本。

## 4. SQL 选择器已有成熟训练形式，但不能混淆任务与损失

### 4.1 单候选、成对和联合选择是不同实现，不是不同研究目标

[GradeSQL](https://link.springer.com/article/10.1007/s10844-026-01071-6) [1] 将单候选判为 Yes/No；[CHASE-SQL](https://proceedings.iclr.cc/paper_files/paper/2025/file/974ff7b5bf08dbf9400b5d599a39c77f-Paper-Conference.pdf) [2] 学习成对选择；[MSc-SQL](https://aclanthology.org/2025.naacl-long.107/) [3] 联合评估多个候选及 metadata。三者说明 SQL selection 可以用文本模型专门训练，而不必让选择器重新生成 SQL；也说明 pairwise 不是开展本研究的必要条件。

| 训练粒度 | 输入与输出 | 本研究中的用途 |
|---|---|---|
| Pointwise | 问题、schema、单 SQL → 正确性分数或 Yes/No | 建议作为首个共同基线；正确/错误单一类别的问题仍可用于训练 |
| Pairwise | 同题两个 SQL → 偏好或分数差 | 可作为后续外部验证，不应与主实验同时改动损失 |
| Groupwise | 问题、多个 SQL、metadata → 候选编号/联合判断 | 能比较细微差异，但输入长度和候选位置引入额外控制成本 |

建议先采用 GradeSQL-style pointwise 形式，原因是变量较少、易审计，而不是已知它一定优于 pairwise。若选择评分头加 BCE，同样可以研究课程，但不能把它与自回归 Yes/No 的差异计入课程贡献。

### 4.2 官方 GradeSQL 源码提供了一个重要复现提醒

核查的是官方仓库 commit `9b59d6ca2f5944d22f6d375e097779163c99da21` 的 [finetuning.py](https://github.com/sisinflab/GradeSQL/blob/9b59d6ca2f5944d22f6d375e097779163c99da21/src/finetuning/finetuning.py)。静态代码显示：

- 在 prompt 后拼接 Yes/No；使用 `AutoModelForCausalLM` 和 LoRA。
- tokenization 固定 `max_length=512`、截断及 padding；`labels` 复制 `input_ids`，没有显式把 prompt 和 padding 的标签置为 `-100`。
- 对 tokenized candidate rows 做 10% 随机验证划分，未在该脚本中进行 question-group split。

因此，该源码路径不能描述为已经实现 completion-only Yes/No loss。`attention_mask` 本身也不等于 loss label mask。长输入是否截掉 SQL 或最终标签，需要实测，本文没有统计截断比例。候选行随机划分则存在同题跨内部 train/validation 的风险，不代表论文官方 BIRD dev 结果必然泄漏。

建议采用明确 mask 的共同实现：只在回答标签及预定终止符上计算交叉熵，prompt 与 padding 不计损失；固定答案编码、归一化和推理打分方式。这个版本应称“基于 GradeSQL 思路的标准化实现”，不是逐位复现原脚本。原版复现和规范化版本可以另作审计，但课程各组必须使用同一个版本。

### 4.3 SQL 结构难度不等于 SQL 缺陷语义

[SAC-SQL](https://aclanthology.org/2025.findings-emnlp.190/) [4] 对合成 SQL 数据按结构难度分阶段训练生成器，而 [ErrorLLM](https://arxiv.org/abs/2603.03742) [5] 显式表达 SQL 隐含语义错误，再进行修复。前者的“复杂 JOIN”与后者的“JOIN 错误”不同：正确 SQL 可以结构复杂，错误 SQL 也可以极短。两者支持设计结构及错误标签，但没有给出固定 selector 池内课程加速的答案。

本研究不应同时加入错误解释生成、修复和新候选扩充。否则难以判断收益来自课程，还是来自新的任务及监督。

## 5. 动态调度已超越从易到难：最接近的对照在哪里

### 5.1 技能结构与效用信号已有先例

[Skill-It](https://arxiv.org/abs/2307.14430) [6] 在技能依赖及当前学习状态上分配训练数据，而 [DUCL](https://ojs.aaai.org/index.php/AAAI/article/view/40400) [7] 联合 difficulty 与 utility。二者从不同角度表明：高 loss、静态复杂度或“目前做错”都不是训练效用的充分指标。对 SQL 来说，持续高 loss 可能来自标注歧义或不可学样本，不能无限增加其曝光。

[LEAD](https://www.vldb.org/pvldb/vol19/p426-luo.pdf) [8] 更直接面对动态选样的计算代价：利用训练过程中已有的 loss、近似变化及历史平滑估计实例效用，结合分层选择，避免反复全量额外推理。与 Skill-It 的组关系先验相比，它更适合作为“没有 SQL 缺陷语义、但有训练动态”的近邻基线。其免额外全量 inference 不等于整个调度过程零成本。正式引用为 PVLDB 19(3), 2025，不把会议周期与出版年份混写。

### 5.2 动态曲线是否胜过合适的静态混合，需要单独证明

[DoGE](https://arxiv.org/html/2310.15393v2) [9] 通过泛化估计学习 domain 权重，但其 §4 与 Appendix E 的 stage-wise 实验没有发现分阶段混合优于全局平均混合的清晰平均 perplexity 收益；非常频繁的分阶段切换反而更差。它与 Skill-It 并非简单互相否定：前者涉及代理模型权重迁移及预训练，后者强调技能联系；但共同要求本研究控制“每类最终见了多少次”。

因此，错误感知方法若只胜过 uniform，却没有胜过相同累计类别配额的随机顺序，证据支持的可能是更合适的数据配比，而不是动态课程本身。

### 5.3 遗忘回放也已成为数据调度的一部分

2026-09-30 的预印本 [Replay on Demand](https://arxiv.org/html/2609.40089v1) [10] 按剩余学习潜力和相对预训练状态的遗忘共同分配适应与回放样本，竞争同一训练预算。与 DoGE 的静态混合反证对照，它提示收益可能取决于目标是平均质量还是适应—保持权衡，而不是“动态必优”。RoD 研究 continued pretraining，且有候选评分开销，不能当作 SQL LoRA 净加速的直接证据。

如果本项目方法使用回放，通用动态基线也应有相同回放预算。否则“错误语义有效”可能只是“额外练习旧题有效”。SQL 选择的逐题回归与预训练遗忘不是同一个量，需要自己的指标。

### 5.4 错误组与微调顺序也已有近期应用

[CURE](https://arxiv.org/abs/2601.15408) [11] 按模型表现调整语义类别及数据来源，使用多模态模型微调；[FOT](https://aclanthology.org/2026.findings-acl.1021/) [12] 则将模型相关难度用于 epoch 内顺序调度。相比 Skill-It，前者更接近错误反馈分组，后者更接近细粒度顺序控制。CURE 是 CVPR 2026 工作，其最终配置 warm-up 后一次改权并固定，不能只凭摘要的 dynamically 一词描述成持续高频闭环；完整系统的收益也不能全部归于课程。

| 可迁移思想 | 代表工作 | 必须控制的额外因素 |
|---|---|---|
| 技能组与组间关系 | Skill-It | 分组信息、关系估计及额外试训成本 |
| 难度不等于效用 | DUCL | proxy 定义及其与真实增益的差距 |
| 用训练日志近似选样价值 | LEAD | 历史陈旧度、探索、聚类和调度开销 |
| 合适配比未必需要复杂顺序 | DoGE | 累计类别曝光与调度粒度 |
| 学习与保持竞争共同预算 | RoD | 评分成本、回放来源及共同保持目标 |
| 语义组重加权或 epoch 内排序 | CURE、FOT | 一次改权与持续闭环，顺序与频率 |

这组证据给出的不是现成最优采样器，而是一套不能缺席的对照条件。

## 6. 验证器与 reward model 的课程研究压缩了宽泛创新空间

[HiEval](https://aclanthology.org/2026.findings-acl.332/) [13] 已结合 QA 正确性评分、错误分析、课程和 experience replay；[RESTL](https://ojs.aaai.org/index.php/AAAI/article/view/40324) [14] 对多方面 reward model 本身实施课程，再用其反馈训练形式语言生成器。因此，“不是训练生成器，而是训练判别器”不能独立成为新点。HiEval 的快速评分主要涉及推理效率，不能直接移作训练提速结论；RESTL 也未直接隔离本报告的 SQL 固定池净成本。

与这两项语义或多维评分工作不同，[EGT](https://arxiv.org/html/2602.01884v1) [15] 以输出熵指导多模态 reasoning reward model 的数据清理和难度安排，说明验证器训练还须区分“困难但有用”与“标签含糊”。它同时改变子集并涉及 RL 阶段，不能把整套收益当作固定 CE 损失的顺序消融。

| 相邻工作 | 已覆盖的内容 | 与本课题尚有的距离 |
|---|---|---|
| HiEval，Findings ACL 2026 | judge、错误类别、课程与回放 | 未由本次可得证据确定固定 SQL 池的 LoRA 达标成本 |
| RESTL，AAAI 2026 | 细粒度 reward model 的课程训练 | 多 reward 与生成器 RL 系统，不是只换 selector sampler |
| EGT，2026 预印本 | reward model 的噪声、难度与训练安排 | 清理和课程共同改变，任务与优化阶段不同 |

潜在贡献应是可证伪的增量问题，而不是把这些机制搬到 SQL 后重新命名。

## 7. 提速证据的边界：哪些反证必须进入实验

### 7.1 随机与配平控制可能解释大部分收益

[When Do Curricula Work?](https://arxiv.org/abs/2012.03107) [16] 用同样扩大可用训练集的 random-curriculum 区分 pacing 与难度排序；[Rethinking Data Selection at Scale](https://aclanthology.org/2025.findings-emnlp.146/) [17] 则在大规模 SFT 池中发现，多种选样方法难以稳定超过随机，并强调多样性与长度分布。它们没有证明 SQL 课程无效，但要求 uniform 之外还有同 pacing、问题级配平和长度控制，且不能把单个 seed 的微小领先当作稳定效应。

### 7.2 更少 steps 不是已经证明更少 GPU 小时

[RHO-LOSS](https://proceedings.mlr.press/v162/mindermann22a.html) [18] 优先学习可学、值得学且尚未学会的样本；[Beyond Random Sampling](https://aclanthology.org/2026.eacl-long.271/) [19] 在语言模型预训练中报告达到基线性能所需步数下降。前者的 irreducible-loss 模型与候选评分、后者的预评分和课程构建，都提示效率有准备与运行两部分。

Beyond Random Sampling 的 18–45% 主要是训练步数口径，不能写成已知 LoRA 墙钟加速；其中选择子集和重排对照也说明，数据筛选效应不能全部归于纯顺序。RHO-LOSS 的特定数据集倍数同样不构成本项目的预期结果。两项研究支持测量学习效率，但不替代自己的成本账本。

### 7.3 课程与学习率存在交互，不能独立调一个组的优化器

[How Learning Rate Decay Wastes Your Best Data](https://arxiv.org/abs/2511.18903) [20] 比较恒定与衰减学习率时，发现升序质量课程的收益随标准衰减减弱。与前述 positive curriculum 结果相比，这不是简单反例，而是指出数据价值出现的时间与有效更新幅度有关。论文为 ICLR 2026，研究 1.5B 预训练，不是 7B LoRA；可迁移的是控制要求，而非具体效果。

本项目应把 LR schedule 绑定共同更新或 token 预算，不能每个组按自己更短的停止时间重新缩放；也不能只给错误课程调温和衰减，而给随机组沿用较差设置。若检验交互，应让全部采样器共享同一组预注册 schedule。

## 8. 综合判断：研究什么才有可辨认的贡献

### 8.1 必须区分两层“错误”

第一层是候选 SQL 的静态缺陷，例如投影遗漏、条件遗漏/错误、literal/比较符错误、JOIN 错误、聚合与分组错误、排序与 LIMIT 错误、NULL 或日期逻辑错误。第二层是当前 selector 的动态错误：把错 SQL 接受、把对 SQL 拒绝，以及同题排序失败。

错误 SQL 不一定让当前模型判断困难；模型当前失败也不一定由 SQL 的结构复杂度造成。研究假设应是：缺陷类别能够帮助估计哪些数据现在值得训练，而不是“错误样本越多越好”。正 SQL 不应强贴缺陷标签，可使用 correct/none；若另外标注所需技能，要与负例缺陷标签分开。

### 8.2 可检验的主张

建议表述为：

> 在同一固定训练集、候选池、7B backbone、LoRA 配方和逐样本损失下，SQL 缺陷语义能否改善训练数据分配，使选择器在相同选择质量与回归限制下，以更低净成本达标？该增量能否超过通用训练动态、相同粒度分组和保持性回放？

第一个问句检验应用效果，第二个问句检验机制归因。只证明前者，可以报告有效课程策略；要主张“错误语义是关键”，还必须支持后者。

| 不足以单独成立的贡献 | 更有说服力的证据 |
|---|---|
| 训练一个 7B SQL verifier | 相同训练配方下课程改变学习曲线 |
| 按 JOIN/聚合等类别分组 | 真语义优于同粒度非语义组与静态均衡 |
| 难例多采一些 | 超过 loss、margin、progress 或 utility 的控制 |
| 加 replay 后少退步 | 在共同 replay 预算与同一回归约束下仍有增量 |
| 达标 steps 更少 | 计入标注、probe 和调度后仍降低净成本 |

### 8.3 一个候选调度结构，不是已验证算法

可以让所有动态组共享 instance utility 与 replay，只为实验组增加经过收缩的错误组信号。例如在训练区间内记录标签损失、变化趋势和误判方向，按错误组聚合“尚未学会但可改善”的证据；给小样本组做收缩，保留 uniform floor，限制单题及单组曝光，防止一直追逐噪声。

这里不预先宣布“最高错率组优先”最优。应先检查组级信号是否能在控制 loss、长度、正确性和数据来源后，预测下一段训练的独立收益；若不能，复杂动态方法缺乏必要性。

还需明确：逐样本 loss 不变，不代表有效优化目标不变。非均匀采样改变了训练期的期望梯度权重。本项目允许研究频率，但必须把频率与顺序分开；若要求严格保持均匀经验风险，则需要纯顺序实验，或加入重要性校正控制，不能含混地称“除了顺序什么都不变”。

## 9. 建议实验：统一训练对象，只让 sampler 变化

### 9.1 数据准备与共同基线

1. 优先审计公开 GradeSQL BIRD 训练候选，必要时构建共同训练池；冻结每题候选、generator 来源、去重规则、执行标签和文件哈希。其论文报告 BIRD 候选数据 82,640 行，不是 82,640 个独立问题。
2. 先按问题划分，再展开候选；主要泛化评估尽量数据库隔离。检查相同 SQL、等价模板与同题跨 split。历史反复分析的 100/287 题适合诊断，不适合重新包装成盲测集。
3. 核实 generator/backbone 的训练数据来源。BIRD train 内部留出的题目，若已用于上游 SQL 专用模型训练，不能声称基础模型从未见过；记录污染风险，官方 dev 的最终评估与内部开发分别报告。
4. 使用同一 7B backbone＋LoRA，冻结 tokenizer、模板、label 编码、mask、最大长度、target modules、rank、precision、batch、packing、optimizer、LR schedule、checkpoint 频率和 adapter 初始化。先实测 SQL 与答案是否保留，不默认 512 tokens 足够。
5. 推荐 pointwise completion-only CE：输入 Question、Evidence、Schema、单候选 SQL；输出 Yes/No，推理按预定标签似然排序。标签多 token 时按固定序列规则计算，不混用单 token 概率。所有课程组用完全相同的实现。

执行标签是同库运行下的可观测一致性，不是普遍逻辑等价。对高频难例抽查排序、重复行、列顺序、NULL、数值和空结果；空结果不等于 SQL 无效。固定官方评价脚本和 timeout 规则；不要仅依靠字面 SQL 字符串或 AST 差异判错。

### 9.2 两阶段比较，避免一开始把实验摊得过大

先做三个种子的机制筛查；只有存在稳定信号，再对关键组增加到五个种子并扩大泛化验证。以下为建议协议，不是已运行实验。

| 组 | 唯一允许变化的调度信息 | 回答的问题 |
|---|---|---|
| U | 问题级随机；配平训练规模及必要长度因素 | 最基本共同参照 |
| S | 静态错误类别均衡 | 只靠配比是否足够 |
| D | loss/progress/utility，无 SQL 缺陷语义 | 通用训练动态能做到多少 |
| EA | D＋真实错误组信号 | 语义有没有额外收益 |
| NG | D＋同粒度非语义分组；匹配组大小及长度/难度分布 | 是否只是分组平滑或统计聚合 |
| EA0 | 与 EA 同一代码，语义项系数为零 | 实现应退化为 D，而不是性能新方法 |

第一阶段各组采用相同固定回放规则及预算。第二阶段再比较一个根据遗忘分配回放的通用控制与 EA 对应版本，不能仅给 EA 动态 replay。LEAD-style utility 或 Skill-It-style 分组可以作为代表性近邻；无需第一轮完整复现所有文献方法，但选用的简化版必须说明差异。

必要的附加控制：同 pacing 随机；打乱错误标签但保持组大小；将 EA 实际曝光多重集随机重排。最后一项是事后机制诊断，不是无需先跑 EA 的部署基线。若 EA 仅与 D 极少量采样槽位不同，先审计信号和概率敏感性，不应通过放大预算掩盖失效。

### 9.3 错误标签只给 sampler，不给某一组增加答案提示

允许利用 train 的 gold、经过核验的正例及执行差异辅助标注，但 AST 差异不能直接当错误根因。支持多标签、unknown 和置信度；报告标注覆盖率、抽检一致性及最小组规模。高噪声组可以降权或退出语义实验。

训练侧 monitor 用于调度；dev 用于共同阈值和预定模型选择；独立 test 只用于冻结协议后的确认。不得用 test 的逐题错误或 gold 类型决定下一批训练数据。所有组收到相同输入，EA 的缺陷标签不能拼进 prompt，否则改变了监督条件。

### 9.4 达标指标与成本账本

预先确定共同目标质量 τ、允许回归 δ、监测间隔和稳定窗口；不能事后挑最有利阈值。达标要求同时满足质量与回归限制，并在后续预定窗口保持，而不是某一个 checkpoint 短暂越线。没有达标的运行保留为未达标/截尾，不能只平均成功种子。

- 质量：问题级最终 execution accuracy、candidate coverage 条件下的 selector accuracy、必要的 NLL/校准指标。
- 回归：相对共同冻结参考在“原本正确”的问题上改错的比例；同时报告 fixes 与 regressions，避免总 EX 掩盖退步。参考可选统一基线或 SC，但必须事先固定。
- 效率：updates、非 padding tokens、实际 padded tokens、唯一问题覆盖、GPU 小时、净墙钟及到达阈值的成本。
- 稳定性：不同种子的达标分布、最终预算表现、各错误组及 SQL 长度组曲线。

净成本包括方法特有的错误标注、初始化/预评分、训练、在线 probe、调度、共同评估和 checkpoint。人工时间、API 费用、GPU 时间分别列账；冷启动与可复用标签的摊销口径分别报告。共同候选生成可以单列，但不能只展示 EA 有利的缓存成本。

统计以问题和训练种子为单位；候选行或 pair 不能被当作独立测试题来虚增显著性。开发集阈值用于 time-to-target，独立测试用于确认共同预算或预定停止策略的质量与回归；若事后画 test 学习曲线，也不能再据此调 sampler。

### 9.5 固定池带来的边界与可否定结果

固定候选池时 oracle coverage 不会因课程上升。目标是减少“已有正确 SQL 却选错”，不是让模型在没有正确候选时创造正确答案。

| 结果 | 应得结论 |
|---|---|
| EA 超过 U，但不超过 S | 静态错误配比足以解释收益，动态课程未获支持 |
| EA 超过 U，但不超过 D/NG | 训练动态或分组聚合足以解释，未证明 SQL 语义增量 |
| 减少 steps，但加上 probe 后更慢 | 样本/更新效率改善，不能声称净训练加速 |
| 提升 EX，但超过共同回归限制 | 没有满足原研究目标，不应事后放宽限制 |
| 多种子下净成本更低且质量保持 | 支持本设置的效率收益；再做数据/模型迁移验证 |

## 10. 当前项目证据与下一步优先级

已有项目记录 [P0a 报告](D:/University/Alpha-SQL/reports/SQL_SELECTOR_CURRICULUM_P0A_RESULTS_ZH.md) 是数值 MLP 机制 pilot：训练 129 题、454 个候选；只有 20 题可组成正负 pair，共 82 对；验证 40 题。四组最后都是 27/40，SC 为 28/40；没有观察到修复收益。EA0 与 LP 的逐位等价支持实现控制，但不支持课程有效。EA 与 LP 的采样差异也很小。

这不是 7B 文本选择器实验，不能作为其加速证据，也不能据此断言 7B 一定无效。新主线使用 pointwise 后，全正或全负问题仍可贡献监督，不必继续让“只有 20 个混合标签问题”限定整个研究；但候选展开不增加独立问题数，数据覆盖仍需扩充和审计。

建议优先级：先读 GradeSQL 与源码、Skill-It、LEAD，确定模型和动态控制；再读 DUCL、RoD、HiEval，明确语义与保持性边界；最后用 DoGE、When Do Curricula Work、Random Selection、LR Decay 检查实验归因。不要先做复杂 agent scheduler，再补弱基线。

最先应交付的不是完整新算法，而是三件可审计产物：共同数据 manifest 与标签报告；可稳定训练的统一 7B LoRA 基线；U/S/D/EA 的共享预算、共同 checkpoint 及成本记录。只有机制筛查出现信号，才值得扩大租卡预算。

## 11. 结论：逐项回答研究问题

RQ1：GradeSQL 等工作为文本 SQL 判别器提供直接可行基础，Skill-It、DUCL、LEAD 与 RoD 提供调度及保持性参照；课程不要求 pairwise，也不要求改变损失。官方训练实现须独立审计，不能仅凭论文任务描述推断 loss mask。

RQ2：候选贡献是 SQL 缺陷语义对训练分配的增量信息，而不是 LoRA、错误分组、动态采样或 replay 本身。本次没有检索到完全匹配设置的直接结论；可做实证研究，但不能称已确认首次或已知有效。

RQ3：可信的效率结论需要相同模型配方、强动态与非语义对照、共同质量和回归门槛，并纳入标注、probe 与调度成本。更少步数不足以支持净提速，多种子阴性结果也应保留。

总体上，这是一项工程上可执行、增量效果尚不确定的研究。合理目标不是预设“错误感知一定更快”，而是找到其在哪些错误分布和训练阶段有用，以及何时通用训练动态已经足够。

## 参考文献

[1] M. Tritto, G. Farano, D. Di Palma, et al., “GradeSQL: Outcome reward models for intelligent Text-to-SQL generation from LLMs,” Journal of Intelligent Information Systems, 64:2545–2568, 2026.

[2] M. Pourreza, H. Li, R. Sun, et al., “CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL,” ICLR, 2025.

[3] S. K. Gorti, I. Gofman, Z. Liu, et al., “MSc-SQL: Multi-Sample Critiquing Small Language Models For Text-To-SQL Translation,” NAACL, 2025.

[4] X. Zhu, Q. Li, L. Cui, et al., “Learning SQL Like a Human: Structure-Aware Curriculum Learning for Text-to-SQL Generation,” Findings of EMNLP, 2025.

[5] Z. Hong, H. Chen, Z. Yuan, et al., “ErrorLLM: Modeling SQL Errors for Text-to-SQL Refinement,” KDD, 2026.

[6] M. F. Chen, N. Roberts, K. Bhatia, et al., “Skill-it! A Data-Driven Skills Framework for Understanding and Training Language Models,” NeurIPS, 2023.

[7] Z. Jiang, J. Han, T. Li, et al., “Difficulty Is Not Enough: Curriculum Learning for LLMs Fine-tuning Must Consider Utility,” AAAI, 2026.

[8] X. Lin, Y. Qi, Y. Zhu, et al., “LEAD: Iterative Data Selection for Efficient LLM Instruction Tuning,” PVLDB, 19(3):426–439, 2025.

[9] S. Fan, M. Pagliardini, M. Jaggi, “DoGE: Domain Reweighting with Generalization Estimation,” ICML, 2024.

[10] L. Thede, S. Chen, S. Winzeck, et al., “Replay on Demand: An Emergent Curriculum for Balancing Adaptation and Forgetting in Continued Pretraining,” arXiv:2609.40089, 2026. 预印本。

[11] P. Messina, A. Villa, J. L. Alcázar, et al., “CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation,” CVPR, 2026.

[12] X. Hu, Y. Tang, H. Li, et al., “Fine-Grained Data Ordering Improves Fine-Tuning for Large Language Models,” Findings of ACL, 2026.

[13] Q. Wu, T. Yue, J. Liang, et al., “Curriculum Learning based Hierarchical Scoring and Analysis Framework for Question Answering Task Evaluation,” Findings of ACL, 2026.

[14] Y. Fang, Z. Jin, J. An, et al., “RESTL: Reinforcement Learning Guided by Multi-Aspect Rewards for Signal Temporal Logic Transformation,” AAAI, 2026.

[15] S. Yang, T. Huang, H. Wen, et al., “Entropy-Guided Data-Efficient Training for Multimodal Reasoning Reward Models,” arXiv:2602.01884, 2026. 预印本。

[16] X. Wu, E. Dyer, B. Neyshabur, “When Do Curricula Work?,” ICLR, 2021.

[17] T. Xia, B. Yu, K. Dang, et al., “Rethinking Data Selection at Scale: Random Selection is Almost All You Need,” Findings of EMNLP, 2025.

[18] S. Mindermann, J. M. Brauner, M. T. Razzak, et al., “Prioritized Training on Points that are Learnable, Worth Learning, and not yet Learnt,” ICML, 2022.

[19] Y. Zhang, A. Mohamed, H. Abdine, et al., “Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning,” EACL, 2026.

[20] K. Luo, Z. Sun, H. Wen, et al., “How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining,” ICLR, 2026.
