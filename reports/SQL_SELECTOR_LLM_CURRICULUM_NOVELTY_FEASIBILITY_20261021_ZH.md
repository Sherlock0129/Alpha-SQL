# 文本 SQL 选择器的错误感知课程：创新边界、可行性与决定性实验

检索与评估日期：2026-10-21。本文只进行文献核验、既有实验审查和研究设计；没有租用 GPU、下载模型、调用付费 API 或启动训练。

## 摘要

本报告评估以下主线：固定训练问题、候选 SQL 池、文本选择器与损失函数，通过错误感知数据调度，减少达到共同质量和保持性要求的总训练成本。候选选择、LoRA 验证器、SQL 结构课程、错误类型建模以及错误反馈重加权分别已有明确先例，因此其简单组合不足以构成有力创新。较可辩护的问题是：SQL 错误语义是否提供超越样本 loss、学习进度、分组平滑与保持性回放的额外调度信息，并能覆盖标注、诊断和调度成本。现有数值 MLP pilot 没有支持收益，不能作为正面证据；尚未开展的 7B 文本模型实验也不能从该 pilot 直接推断成功或普遍失败。综合判断为 Accept with Revisions：工程可行性较高，增量效果与论文贡献的风险仍高，应先完成标签审计、共同文本基线与机制对照，再扩大实验。

## 1. 研究问题与边界

- RQ1：训练文本 SQL selector/verifier 与课程学习，分别已经被哪些最近工作覆盖？
- RQ2：错误语义相对训练动态和回放，可能在哪个具体环节产生独立增量？
- RQ3：在可租用 GPU 的条件下，什么实验足以支持“相同质量与回归约束下更快达标”？

训练对象拟为读取 Question、Evidence、Schema 和候选 SQL 的文本打分模型，更新 LLM adapter 和评分头。固定候选池，不训练生成器，不引入推理时 SQL 修复或新增候选；已有 self-consistency 特征若保留，须在全部课程组中一致。

“固定训练集”指正式比较前冻结的共同样本集合，不意味着只能使用目前很小的 pilot 集。错误类型标签属于额外训练侧信息：即使没有增加问题或 SQL，仍有监督和标注成本，不能称为零额外信息或零额外成本。

## 2. 检索方法与证据范围

采用三个独立检索视角：SQL selector 与错误建模；排序、utility 和语义分组课程；随机基线、保持性与成本反证。主线程补检 SQL curriculum、代码排序课程及最新错误 taxonomy 工作，统一综合，不把并列摘要当作结论。关键词包括 SQL selection/curriculum、error-aware curriculum、semantic groups、learning progress、retention replay、time-to-target。

本文精选 24 项核心工作，覆盖 2015—2026 年；优先使用 ACL Anthology、PMLR、AAAI、正式期刊、作者机构记录及作者 arXiv 全文。标题、作者、发表状态与所引方法分别核验。HiEval 官方 PDF 因文件大小无法完整抓取，本文仅采用官方摘要可支持的内容，不引用其表格数字或声称已完整审阅其消融。其他论文也只在可核验范围内使用。

本文不是穷尽检索，未发现完全同构研究不等于证明首次。预印本结果和正式发表结果区分标注，不把整套系统收益、模块耗时或推理加速挪作课程训练加速。

## 3. 相关研究的分类

| 分支 | 主要研究对象 | 对本项目的约束 |
|---|---|---|
| SQL 选择与错误建模 | 候选判别、执行分组、错误定位及修复 | SQL selector、LoRA 和错误 taxonomy 本身不是新点 |
| 数据顺序与反馈调度 | 难度、utility、训练进度、语义组和回放 | 需要超越普通难例采样、静态分组与动态 loss 控制 |
| 保持性与净成本评价 | 随机对照、技能门槛、诊断开销 | 更少更新步数不自动等于更短达标时间 |

## 4. 候选选择和 SQL 错误语义已经有直接先例

### 4.1 训练选择器并不要求训练生成器，但已不是创新

[CHASE-SQL](https://openreview.net/forum?id=CvGqMD5OtX) [1] 采用成对候选判别，[MSc-SQL](https://aclanthology.org/2025.naacl-long.107/) [2] 联合读取多候选及执行 metadata，而 [GradeSQL](https://link.springer.com/article/10.1007/s10844-026-01071-6) [3] 对单候选进行 Yes/No 监督。这些工作在比较粒度上不同，但都说明模型可以专门学习选择，而不必生成 SQL。不能把 CHASE-SQL 的 preference optimization 一概叫作 DPO，也不能把生成一个候选编号或 Yes/No 误认为完整 SQL 生成任务。

| 工作 | 方法特征 | 本项目需新增的证据 |
|---|---|---|
| CHASE-SQL，ICLR 2025 | 问题、hint、schema、两条 SQL；分类式选择 | 固定相同 pair 数据下的数据调度收益 |
| MSc-SQL，NAACL 2025 | 多 SQL、执行结果与 metadata；预测候选 index；QLoRA | 训练效率而非仅推理端最终质量 |
| GradeSQL，JIIS 2026 | 单 SQL 的 Yes/No 自回归监督；LoRA | 错误语义对既有训练动态的额外价值 |
| R³-SQL，Findings ACL 2026 | 排序 reward、执行结果分组与按需重采样 | 关闭重采样后固定池内的课程效应 |

与上述三种判别粒度相比，[R³-SQL](https://aclanthology.org/2026.findings-acl.2146/) [4] 进一步处理执行相同候选的评分一致性，并通过 resampling 补足候选覆盖。它提醒我们：固定池内选择与扩大候选池是两个问题。若作为外部参照，必须把 ranker 与 resampling 分开，不能把 coverage 增益计入本项目的课程收益。

### 4.2 SQL 结构课程不等于 SQL 错误课程

[SAC-SQL](https://aclanthology.org/2025.findings-emnlp.190/) [5] 已将结构难度用于 SQL 生成课程；[ErrorLLM](https://arxiv.org/abs/2603.03742) [6] 则显式建模隐含语义错误，再指导修复。前者的对象主要是生成器及结构复杂度，后者是错误检测与修复，不是固定 selector 训练池中的数据调度。二者共同否定“SQL 结构/错误语义从未进入训练”的宽泛主张，但没有直接回答本报告的增量问题。

2026-10 的 [TEG](https://arxiv.org/abs/2610.05060) [7] 将给定错误诊断映射到应检查的构件及编辑操作，与 ErrorLLM 的错误建模形成邻近路线。这对标签设计有启发，但本项目不应顺势扩大成错误检测、修复、生成和选择的全流程论文。TEG 当前按预印本引用。

必须区分“出现 JOIN”与“JOIN 错误”：正确 SQL 也可以有很多 JOIN，复杂度标签不能直接充当缺陷语义。训练 pair 的 AST 差异也可能只是 alias、等价表达或多解差异，不天然等于已确认错误根因。

## 5. 顺序、utility、语义组与回放也已有覆盖

### 5.1 排序课程有早期和 LLM 版本

[Training Curricula for Open Domain Answer Re-Ranking](https://arxiv.org/abs/2004.14269) [8] 已在神经答案排序中使用 pointwise/pairwise 课程；[CoRNStack](https://arxiv.org/html/2412.01007v3) [9] 则在代码检索训练中通过温度调度渐进采样 hard negatives，并另行训练代码 reranker。因此，不能把“排序模型使用课程”或“代码模型变成 reranker”作为新贡献。CoRNStack 的 negative curriculum 主要作用于检索器对比训练，不能把它的完整 reranker 收益都归于该课程。

[Self-Paced Curriculum Learning](https://ojs.aaai.org/index.php/AAAI/article/view/9608) [10] 已结合先验知识和学习进度；[Automated Curriculum Learning](https://proceedings.mlr.press/v70/graves17a.html) [11] 已通过 learning-progress reward 调整任务分配。与之相比，[Curri-DPO](https://aclanthology.org/2024.findings-emnlp.754/) [12] 将偏好对从易到难组织，[KPO](https://aclanthology.org/2025.findings-acl.250/) [13] 将课程应用于 top-K 排序偏好。它们限制了“首次将进度、偏好排序与课程结合”的表述。

Curri-DPO 的数据对利用与 reference 更新、KPO 的损失形式与课程是不同因素。KPO 的某些更少 steps 结果比较的是损失优化，而不是纯课程消融，不能迁移成我们的预期加速倍数。

### 5.2 最接近的近期工作已经超越固定难度

[DUCL](https://ojs.aaai.org/index.php/AAAI/article/view/40400) [14] 联合 difficulty 和 utility，[FOT](https://aclanthology.org/2026.findings-acl.1021/) [15] 研究细粒度数据顺序，而 [CURE](https://arxiv.org/abs/2601.15408) [16] 根据模型表现调整数据源和语义类别的采样权重。三者的反馈与调度方式不同，但都说明不能把现有课程描述为只看静态难度。CURE 已为 CVPR 2026 oral，不能继续只标注为预印本；其全文最终配置采用 warm-up 后一次改权并固定，不应描述成持续高频闭环。

与 CURE 的语义组重加权相比，[HiEval](https://aclanthology.org/2026.findings-acl.332/) [17] 更接近“训练一个判题器”的任务角色：官方摘要明确包含课程、快速评分、五类错误分析、experience replay 和 contrastive unlikelihood。仅把错误标签与回放接到 SQL verifier 上，是容易被质疑为领域迁移的版本；本项目应提供其未直接回答的固定池增量归因与净训练成本证据。HiEval 的推理速度结论不构成训练速度结论。

### 5.3 更快达标也不能笼统宣称首次

[Beyond Random Sampling](https://aclanthology.org/2026.eacl-long.271/) [18] 已研究预训练早中期收敛与 warm-up 课程；[LogFloor](https://arxiv.org/abs/2608.14936) [19] 更明确地以所有监测技能达到 target floors 的 token 成本为目标，并比较静态混合与阶段顺序。前者不是 SQL selector 微调，后者不是 SQL EX 和逐题回归，但二者已经覆盖“固定数据安排能否更快达到目标”的广泛思想。LogFloor 当前为提交 AAAI 2027 的预印本，不能称已录用。

## 6. 反证：课程的收益与成本都需要强对照

[When Do Curricula Work?](https://arxiv.org/abs/2012.03107) [20] 通过 random-curriculum 区分进入顺序与可用数据范围；[Aioli](https://arxiv.org/abs/2411.05735) [21] 则发现多种既有动态混合策略未稳定胜过简单分层采样，再提出在线混合方法。二者不否定所有 SQL 课程，却要求保留随机、同 pacing 随机及分层采样等强对照，不能只战胜最弱的 uniform sampler。

与上述对照问题相连，[EDCO](https://arxiv.org/html/2601.03725v1) [22] 说明动态诊断本身有成本。其 §5.4 的 83.5% 是熵估计模块开销下降，不是整个训练缩短同样比例；所报固定 500 RL steps 总时间为 33.58 小时，随机采样为 30.67 小时。这不证明它在相同质量目标下更慢，却说明必须分别报告模块成本、固定预算表现和净达标时间。

本项目需要比较的是包含训练侧 probe、调度、验证、checkpoint 以及方法特有准备成本的净时间。共同候选生成成本单列；错误标签准备成本也单列并计入冷启动账本。若可重复使用，应同时报告摊销成本，但不能仅展示摊销后的有利口径。

[Example Forgetting](https://arxiv.org/abs/1812.05159) [23] 已在固定数据训练中追踪逐样本正确到错误的转变；[MIR](https://arxiv.org/abs/1908.04742) [24] 则按预期更新带来的干扰选择回放样本。前者支持在固定训练集内研究回归，后者说明风险感知回放不是新概念，但在线持续学习的结果不能直接移植为固定 SQL 池结论。高 forgetting 也可能与标签噪声有关，所以需要抽检高频回放样本，而非认为它们一定最有学习价值。

## 7. 对当前主线的创新性判断

### 7.1 哪些部分不能作为主创新

7B 模型、LoRA、RankNet、schema-aware 输入、hard negatives、错误 taxonomy、回放以及更快达标的目标都已有先例。把这些组件画在同一架构图里，不足以证明新的方法贡献。

当前更适合定位为“有机会形成新方法的受控实证研究”，而非新的 SQL selection 问题。若最终仅用标准公式给错误组加权、再优于随机一次，贡献更接近应用型 New Setting。只有识别独立信号并提出可重复利用它的机制，才更接近 Novel Method。

### 7.2 可辩护的窄主张

> 在固定训练问题、候选 SQL 池、文本模型与目标函数下，SQL 错误语义能否在相同粒度的训练动态、相同保持性回放和成本预算之外，改善训练数据分配，并降低达到共同质量及回归要求的净成本？

这里的潜在机制是：多个高 loss pair 可能对应不同的可学习性、迁移和干扰模式；跨问题共享的缺陷类型可能帮助稀疏反馈池化。但这也可能仅是统计平滑，或者是 difficulty、SQL 长度与候选来源的代理。只有控制后仍存在收益，才能称为错误语义的增量。

检索到的近邻尚未直接给出这一整组同构对照；这为研究留下可检验空间，而不是首次性证明或正结果承诺。

### 7.3 现有负结果的约束

项目的 [P0a 报告](D:/University/Alpha-SQL/reports/SQL_SELECTOR_CURRICULUM_P0A_RESULTS_ZH.md) 记录：20 个可构造 pair 的训练问题、82 对样本；四组预算末均为 27/40，SC 为 28/40；EA 与 LP 的 1500 个 base draw 仅 4 个槽位不同，η=0 版本与 LP bit-exact。

因此，旧版数值 MLP 加语义项的收益主张应否决，不能把它写成已验证有效。尚未测试的文本 LLM 主线是新的实验条件，不被该小 pilot 普遍证伪，但也不能仅以换成 LLM 为理由预期自动成功。后续若仍使用相同控制公式，须报告其继承关系及此前无增益的事实。

## 8. 决定性实验：先证明增量，再证明净加速

### 8.1 数据与模型先冻结

扩大 BIRD train 范围，增加有正负候选的独立问题；不只收集 oracle-correct/SC-wrong，也保留 SC-correct 和 coverage-miss 问题。正式比较前冻结训练、调度监测、开发验证与最终评估边界，并记录数据库和问题重叠。反复查看过的历史 100 题与 287 题不再当作未触碰的最终 holdout。

使用一个共同的 7B 文本 backbone、adapter 配置、输入序列化、损失、有效 batch 与推理规则。优先建立能学习投影、过滤和值对齐的普通训练基线；不是要求其先达到任意高阈值，而是排除明显训练故障或输入丢失。另一个模型家族或规模用于结论复核，不在每个课程组分别挑模型。

GradeSQL/OmniSQL 等用过 BIRD train 的权重，需要监督数据重叠审计后才可用于内部划分泛化比较。当前候选池若缺失生成器精确版本，也应补新池的 model revision、prompt、采样参数和数据哈希，而不反推历史版本。

### 8.2 主对照

主比较组共享相同 replay 配额、训练预算、监测频率、执行与 SC 特征。另设统一的无 replay 参照即可，不必一开始展开所有因素的全组合。

| 对照 | 排除的替代解释 |
|---|---|
| 随机采样；若有 pacing，再加同 pacing 随机 | 只是逐步开放数据或随机波动 |
| 高 loss / 小 margin 难例采样 | 只是更关注当前难例 |
| 静态错误均衡 | 只是改变最终错误类型频率 |
| loss/progress 动态调度 + 共同回放 | 只是训练动态及保持性 |
| 同粒度非语义分组的动态调度 | 只是分组平滑或统计池化 |
| 完整错误语义调度 + 共同回放 | 待检验主方法 |
| 去语义项及组标签置换 | 语义项实现与标签含义是否真正重要 |
| 保持累计曝光、打乱时间顺序的诊断对照 | 是阶段顺序，还是更多训练某类样本 |

组标签置换应尽量保持组支持量、多标签度及 difficulty/长度/来源分层；随意随机出极不平衡的组不是公平对照。曝光匹配需要完整训练轨迹，仅作为事后机制诊断，其轨迹构造不能偷偷使用测试答案。学习进度与语义组需在相同粒度下比较，避免把单样本噪声和组均值稳定性混成语义优势。

可增加一个 utility 或在线数据混合的近邻适配基线，例如 DUCL/Aioli 风格，但必须说明与原算法的适配差异。主要结论不能仅来自击败 LP 的某个弱实现。

### 8.3 标签质量与机制诊断

训练侧允许用 gold 执行结果构造 correctness，但单个数据库实例的执行相等不是普遍语义等价。审计重复行、列序、排序敏感题、浮点容差、空结果和超时，正式 BIRD EX 与内部签名评价分开报告。

错误类型可以从正负候选、schema 与问题中提取，但要抽样人工审计；支持多标签及 unknown，避免为每条样本强行指定单一根因。保持 alias 和等价改写不改变标签的测试。gold SQL、gold 执行标签和错误类型不作为推理时可用特征泄漏给 selector。

除了最终分数，还检查：语义组是否存在足够跨问题支持；控制初始 loss、margin、长度和来源后，组信息是否仍能预测后续训练收益；调度分布是否确有异质性。预测性分析只提供线索，因果归因仍需要受控训练干预。

### 8.4 达标指标与保持性

预先规定共同质量阈值 τ、固定参考选择器下的回归容忍度 δ、监测间隔及持续达标规则。参考正确题集合必须固定，不能训练中动态改定义。

净达标时间是首次满足质量与回归条件且通过预定稳定性检查的总耗时，包含方法特有准备、诊断、调度、训练与共同验证/checkpoint 成本。有限预算未达标的 run 记为未达标/右删失，不只平均成功的 runs，也不能把训练 loss 达标当作 SQL EX 达标。

报告总 EX、oracle-covered 条件选择准确率、修复数、参考正确题新增错误数及条件回归率；同时报告候选 coverage。相同池的 coverage 本来应保持不变，不能作为 sampler 贡献。SQL 全流程 EX 与 selector 在有正确候选时的能力分别呈现。

至少开展多随机种子、问题级配对分析；有数据库聚类时补数据库层面的不确定性。零个观测回归不等于总体零风险，“差异不显著”也不等于证明非劣。阈值不能在看完各组曲线之后选择最有利的一条。

训练步、有效非 padding tokens、含 padding 计算、峰值显存、GPU-hours 和净时间分别记录。公平归因实验可共享 probe 成本；实际成本比较另提供去除无用 probe 的优化随机基线，避免人为拖慢基线。租用相同 GPU 配置，记录负载并避免单一串行运行顺序混淆首次启动成本。

## 9. 审稿式评估与研究可行性

### 9.1 第一印象与关键风险

一句话故事应是“检验 SQL 缺陷语义能否在保持性约束下提供额外的数据调度价值”，而不是“训练更大的 SQL selector”。这是方法型实证研究候选，当前创新仍待确证。

| 风险 | 严重性 | 处理 |
|---|---|---|
| F1/F3：现有课程与语义分组已覆盖组件，无法隔离独立增量 | MAJOR | 将训练动态、同粒度分组、静态均衡和 replay 作为主对照，而非仅列为 future work |
| F6：把更多 token、更短输入或监测开销差异称作效率 | MINOR，可通过预注册口径防止 | 统一成本日志、阈值、推理规则与未达标处理 |

MAJOR 指需完成实质性实验才能解除的风险，不表示本次已经查明完全重复。旧版 MLP 的已测收益主张不获接受；下面的可行性与评分只针对尚未测试的文本版本窄问题，不复活该旧主张。

### 9.2 生命周期与能力匹配

研究类别是应用型方法研究，含较重数据工程。用户已具备 SQL 候选、执行、选择与课程 pilot 管线，并明确可租用 GPU，工程起点较好。周有效投入、总预算、并行资源、标注支持和投稿期限未知，不能据此承诺几周完成。按该方向的近期工作密度，应尽快用阶段实验决定是否继续，而不是先长期租卡。

| 维度 | 判断 |
|---|---|
| 算力 | 不再是本地 8GB 的硬阻碍；长 schema、pair 训练、多 seed 和 probe 仍影响预算 |
| 数据 | 当前仅 20 个 pairwise 问题，统计支持高风险；需扩大独立问题而非只增加同题 pairs |
| 工程 | 中等风险；已有 pipeline 可复用，但 LLM 训练、长度与吞吐控制尚未实测 |
| 时间 | 暂列 Yellow；必须等吞吐预跑与标签审计后估计 |
| 结论成功率 | 未知；能训练出 selector 不等于课程有效，更不等于语义有增量 |

### 9.3 五维评价

以下为主观选题评分，不是预测实验百分比；高于默认分的项目均依据机制，而非已测收益。

| 维度 | 分数 / 10 | 依据与边界 |
|---|---:|---|
| Higher：最终效果 | 5 | 主目标不是提升性能上限；尚无语义调度泛化收益 |
| Faster：达标效率 | 7 | 有明确可干预的数据分配机制和 time-to-target 判据，但净收益未验证 |
| Stronger：保持性 | 6 | 固定正确题参考集和回放提供可测机制；不是零回归保证 |
| Cheaper：总成本 | 5 | 没增加训练问题不代表免费；标注和 probe 可能抵消训练节省 |
| Broader：推广性 | 5 | 目前限定 SQL 选择；跨模型、候选来源或其他代码任务尚未验证 |

优先强调 Faster 和保持性约束，不把五项都包装成贡献。如果未来效果主要来自更强 backbone、不同数据或后处理，不能将其计入课程评分。

### 9.4 范式与范围

第一性原理：Partial，质疑仅凭难度/进度就足以安排训练，但这并非全领域未经质疑的假设。长期未解大问题：No，当前是明确但相对窄的优化问题。技术周期：Partial，LLM 与执行标注让实验方便，但方法思想早有先例。Hamming 影响：No，现阶段没有足够证据说明会改变领域优先级。因此按增量方法研究表述，不称范式变革。

### 9.5 结论与前三项行动

**Accept with Revisions：值得开展有截止条件的验证，不宜直接按正结果论文全面投入。**

1. 审计 correctness 和错误类型，补足独立训练问题并冻结干净划分；先解决 1616 列序等已知评价问题。
2. 建立所有课程共享的 7B 文本训练基线，实测真实长度下吞吐、显存与 probe 占比；不急于扩大模型。
3. 先比较 loss/progress + replay、同粒度非语义组和真实语义组，再做多 seed 净达标实验。若真实语义未优于这些控制，不能继续主张语义贡献。

若错误语义不增加信息，研究结论应改为“训练动态足够/语义标签在这些条件下冗余”；若仅减少 steps 而增加净时间，只能报告样本效率，不能报告训练加速；若仅靠额外回放有效，则归因于回放。不能通过放宽回归约束或事后挑目标来保存原故事。

## 10. 结论：逐项回答研究问题

RQ1：SQL selector、LoRA、结构课程和错误建模均有直接先例。2026 年 R³-SQL、ErrorLLM、CURE、HiEval 与 LogFloor 明显收紧了可声称的新颖性，只有研究对象换成 SQL 不够。

RQ2：潜在增量是错误语义在相同训练动态、分组粒度和 replay 下是否改变最有价值的训练分配。它必须与普通难度、统计平滑、累计曝光和模型升级区分，目前未获实验证实。

RQ3：可租 GPU 使 7B adapter 实验更现实，但主要工作转向训练数据、标签质量、强对照与成本归因。决定性证据是在冻结池、共同模型与共同标准下，多 seed、更独立评估中仍有计入开销的净加速，且满足事先规定的回归约束。

本报告的核心判断不是“组合已有方法一定有创新”，而是把一个宽泛组合收紧成可被支持、也可被否定的增量问题。

## 参考文献

[1] M. Pourreza, H. Li, R. Sun, et al., “CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL,” ICLR, 2025.

[2] S. K. Gorti, I. Gofman, Z. Liu, et al., “MSc-SQL: Multi-Sample Critiquing Small Language Models For Text-To-SQL Translation,” NAACL, 2025.

[3] M. Tritto, G. Farano, D. Di Palma, et al., “GradeSQL: Outcome reward models for intelligent Text-to-SQL generation from LLMs,” Journal of Intelligent Information Systems, 64:2545–2568, 2026. 本文采用该正式期刊版；同框架其他版本不重复计为独立证据。

[4] H. Han, Y. Jeong, S.-w. Hwang, et al., “R³-SQL: Ranking Reward and Resampling for Text-to-SQL,” Findings of ACL, 2026.

[5] X. Zhu, Q. Li, L. Cui, Y. Du, “Learning SQL Like a Human: Structure-Aware Curriculum Learning for Text-to-SQL Generation,” Findings of EMNLP, 2025.

[6] Z. Hong, H. Chen, Z. Yuan, et al., “ErrorLLM: Modeling SQL Errors for Text-to-SQL Refinement,” KDD, 2026.

[7] S. Lee, W. Song, H. Yu, S. Mo, “Why, Where, How: Taxonomy-guided Error Grounding for Code Repair in NL2SQL,” arXiv:2610.05060, 2026. 预印本。

[8] S. MacAvaney, F. M. Nardini, R. Perego, et al., “Training Curricula for Open Domain Answer Re-Ranking,” SIGIR, 2020.

[9] T. Suresh, R. Gangi Reddy, Y. Xu, et al., “CoRNStack: High-Quality Contrastive Data for Better Code Retrieval and Reranking,” arXiv:2412.01007v3, 2025. 首稿 2024；此处按已核验版本引用，不推定会议状态。

[10] L. Jiang, D. Meng, Q. Zhao, et al., “Self-Paced Curriculum Learning,” AAAI, 2015.

[11] A. Graves, M. G. Bellemare, J. Menick, et al., “Automated Curriculum Learning for Neural Networks,” ICML, 2017.

[12] P. Pattnaik, R. Maheshwary, K. Ogueji, et al., “Enhancing Alignment using Curriculum Learning & Ranked Preferences,” Findings of EMNLP, 2024.

[13] S. Cai, C. Gao, Y. Zhang, et al., “K-order Ranking Preference Optimization for Large Language Models,” Findings of ACL, 2025.

[14] Z. Jiang, J. Han, T. Li, et al., “Difficulty Is Not Enough: Curriculum Learning for LLMs Fine-tuning Must Consider Utility,” AAAI, 2026.

[15] X. Hu, Y. Tang, H. Li, et al., “Fine-Grained Data Ordering Improves Fine-Tuning for Large Language Models,” Findings of ACL, 2026.

[16] P. Messina, A. Villa, J. L. Alcázar, et al., “CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation,” CVPR, 2026. arXiv v2 已标注 oral 与正式页码。

[17] Q. Wu, T. Yue, J. Liang, et al., “Curriculum Learning based Hierarchical Scoring and Analysis Framework for Question Answering Task Evaluation,” Findings of ACL, 2026.

[18] Y. Zhang, A. Mohamed, H. Abdine, et al., “Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning,” EACL, 2026.

[19] S. Choi, J. Sung, M. Umer, et al., “Small Models Scout Bottleneck Order for Large-Model Data Control,” arXiv:2608.14936, 2026. 预印本，标注 Submitted to AAAI 2027。

[20] X. Wu, E. Dyer, B. Neyshabur, “When Do Curricula Work?,” ICLR, 2021.

[21] M. F. Chen, M. Y. Hu, N. Lourie, et al., “Aioli: A Unified Optimization Framework for Language Model Data Mixing,” ICLR, 2025.

[22] J.-C. Pang, L. Sun, C. Zhou, et al., “EDCO: Dynamic Curriculum Orchestration for Domain-specific Large Language Model Fine-tuning,” arXiv:2601.03725, 2026. 预印本。

[23] M. Toneva, A. Sordoni, R. Tachet des Combes, et al., “An Empirical Study of Example Forgetting during Deep Neural Network Learning,” ICLR, 2019.

[24] R. Aljundi, L. Caccia, E. Belilovsky, et al., “Online Continual Learning with Maximally Interfered Retrieval,” NeurIPS, 2019.
