# 回归约束下的错误感知课程：SQL 选择器精确选题分析

日期：2026-10-20。按 deep-research → idea-evaluator 顺序分析。本轮只做检索、方案和评估，不实施训练。本文补充而不替换 [22 篇工作调研](D:/University/Alpha-SQL/reports/FIXED_SET_SQL_CURRICULUM_RESEARCH_UPDATE_ZH.md) 和 [此前选题评估](D:/University/Alpha-SQL/reports/FIXED_SET_SQL_CURRICULUM_IDEA_EVALUATION_ZH.md)。

## 摘要

本文分析一个明确的研究假设：相对于随机、难例采样和静态错误均衡，错误感知课程能否在相同训练池与模型下，以不增加回归为条件，缩短 SQL 选择器达到目标质量的时间。增量检索将已有技能混合、课程负结果与模型更新的 negative flips 联系起来，发现新颖性不在“按错误类别多练”或“关注回归”本身，而在语义错误信号、训练顺序和保持能力之间是否存在可复现的净效率增量。进一步区分相对固定基准的回归事件、不同调度之间的回归差异及真正的总体风险保证，提出质量与回归联合达标的评价协议。当前已有训练结果仅说明修复/回归权衡值得研究，不能确认课程有效。审稿式判定为修改后接受探索：需要先定义回归参照、补足训练动态与保持性对照，并验证真实训练成本是否有改善空间。

## 1. 研究问题：三个目标必须同时成立

用户原始 idea：

> 相比随机采样、难例采样和静态错误均衡，错误感知调度能否在不增加训练数据、不增加回归的前提下，使用课程学习缩短 SQL 选择器达到目标性能的时间？

这是一个可以实验回答的问题，而不是已经完整定义的算法。冻结 question、evidence、schema、candidate SQL、执行标签、pair、模型、特征、优化器和推理融合规则；允许改变训练顺序或同池采样频率，二者分别标记。

- RQ1：错误语义能否提供超出 loss、margin、复杂度与静态组平衡的训练信息？
- RQ2：在同样的回归约束下，动态课程能否比静态或一次性调度更快？
- RQ3：减少更新量能否抵消标签、诊断、采样和评估的额外成本？

不增加原始样本不等于不增加监督。专家错误类别、教师解释、旧模型 logits 均应披露来源和成本；首版只让已有标签派生的错误元信息影响调度，不向 reranker 输入新增解释。

## 2. 调研方法与新增证据范围

本轮沿用已有 22 篇证据池，按技能调度支持、课程/排序反证、保持能力与风险测量三个视角并行补查。两轮关键词包括 `error aware curriculum reranker regression retention`、`negative flips curriculum learning`，以及具体方法 Skill-it!、Aioli、LogFloor、FARE、CURE、Positive-Congruent Training 和 Learn then Test。

下文选取 14 篇一手工作作针对性比较，新增重点是 negative-flip 与统计风险边界；不是声称重新完成整个领域综述。题名、作者与版本经原始论文页面核验；机制和数字查对应正文。检索中发现 TRACT，但其 regression 指数值回归，不是预测退化，故不将它当 negative-flip 近邻。未找到完全相同设定不证明首创。

## 3. 证据分类：加速、终点质量与保持能力不是同一件事

| 文献方向 | 主要问题 | 对本 idea 的直接约束 |
|---|---|---|
| 技能/数据混合 | 哪组数据应多训练 | 已有强近邻，须检验 SQL 错误语义的增量 |
| 课程反例 | 排序或动态调度何时无收益 | 随机、静态和一次性方案不能省略 |
| Positive congruence / forgetting | 学新能力时怎样少破坏旧正确行为 | 回归要有共同参照，保持机制也已有先例 |
| 统计风险控制 | 有限数据可支持多强的可靠性说法 | 未显著退化不等于证明不退化 |

## 4. 文献深度判断：已有机制与尚待验证的交叉点

### 4.1 调度有依据，但“错误组动态配比”不新

[Skill-it!](https://arxiv.org/abs/2307.14430) [1] 与 [Aioli](https://arxiv.org/html/2411.05735v2) [2] 已研究利用分组反馈分配训练；[LogFloor](https://arxiv.org/html/2608.14936v1) [3] 更直接以技能目标的首次达标成本为目标。因而“优先练弱项”“动态改变组比例”“更快达到目标”不能分别作为首次贡献。可能保留的差异是：固定 SQL pair、语义错误组相对标量训练动态的独立信息，以及包含保持约束和诊断开销的实际达标时间。

### 4.2 课程收益有任务条件，不能由生成迁移到判别

[FARE](https://arxiv.org/html/2510.17793v2) [4] 的消融中，课程改善 ProcessBench，却未改善 pairwise 平均分；[When Do Curricula Work?](https://research.google/pubs/when-do-curricula-work/) [5] 则要求分开看难度顺序与逐步开放数据的效应。再与 [CURE](https://arxiv.org/html/2601.15408v1) [6] 最终采用 warmup 后一次配比的配置对照，本 idea 不应先验选择复杂动态版本。上述工作既不能证明 SQL 课程必定失败，也不能提供其净墙钟加速保证。

### 4.3 “不把正确题改错”已经有直接研究

[Positive-Congruent Training](https://arxiv.org/html/2011.09161v3) [7] 直接把旧正确、新错误定义为 negative flips，并针对旧正确行为施加保持偏好；[Learning without Forgetting](https://arxiv.org/html/1606.09282v3) [8] 通过旧模型输出保持能力；[GEM](https://arxiv.org/html/1706.08840v6) [9] 则约束对过去任务记忆的损失变化。三者共同说明不能把回归保持本身当新问题，但作用点不同：蒸馏改变监督/损失，GEM 改变更新规则，都不应偷偷算进“仅调度”的收益。记忆上的局部损失约束也不等于未来 SQL 问题零回归。

### 4.4 风险目标不自动成为风险保证

[Learn then Test](https://arxiv.org/html/2110.01052) [10] 对独立校准数据和多重检验提出明确条件；这与 [7] 降低负翻转的经验目标不同。当前项目可以先做经验性的回归非劣评估，不必扩成完整风险校准课题。若将来声称有限样本保证，则必须明确校准独立性、适应性选择及分布假设，不能仅在反复调参的 validation 上设阈值后写“保证不退化”。

## 5. 这个 idea 的关键修订：给“不增加回归”一个可测定义

### 5.1 回归事件相对谁，调度方法又相对谁？

预先冻结基准选择器 B，例如原 self-consistency。定义一个问题上选择正确为 C(f,q)∈{0,1}，回归事件为 B 正确而新选择器错误。用总问题数作分母：

\[
R_B(f)=\frac1N\sum_q\mathbf1\{C(B,q)=1,\ C(f,q)=0\}.
\]

所有课程/非课程模型都相对同一个 B 计算 R_B，再比较其差异。也可另报在 B-correct 子集内的条件回归率，但分母必须明确且一致。

本报告对用户目标的工作解释是：错误课程相对随机、难例、静态均衡训练的模型，不增加上述共同口径的回归。若用户要求相对 B 本身不增加回归，则因为 R_B(B)=0，实际要求是零负翻转，明显更强，必须另立目标，不能模糊处理。

固定参照的负翻转、相对上一个 checkpoint 的遗忘、组平均准确率下降是不同事件，应分别记录。整体 EX 相同可能同时发生大量修复与回归，因此不能用 EX 非劣替代 R_B 非劣。

### 5.2 不增加回归的三种证据强度

1. 观察性：冻结测试集上回归题数未增加。只对该样本成立。
2. 比较性：给回归差值的配对区间，并用预先规定的非劣边界检验。无显著差异不能证明非劣。
3. 总体风险保证：要求额外假设和独立校准/检验条件；不是首版课题默认具备的能力。

若边界 δ_R=0，就严格检验不增加；若采用 δ_R>0，应写成“退化不超过预设容忍量”，承认它比原始要求弱。阈值不得看 test 后调整。在独立 Bernoulli 事件假设下，即便 100 题观察到零次回归，一侧 95% 精确上界仍约为 2.95%，不是总体风险为零。这是有限样本计算示例；真实 SQL 跨库相关性还需另行处理。

### 5.3 主指标改为质量与回归联合达标时间

对所有调度 m 采用同一目标质量 τ、同一回归预算 r_max、同一检查频率，定义：

\[
T_m^*=\inf\{t:\ A_{\rm val}(f_{m,t})\ge\tau\ \land\ R_{B,\rm val}(f_{m,t})\le r_{\max}\}.
\]

再要求连续预定数量的评估点保持达标，避免一次随机穿越。τ、r_max、持续规则来自独立 pilot 或预先声明的基线规则，不按每组测试成绩修改。历史 oracle 仅约束固定池上限，不能设一个任何方法都不可达到的目标。共同早停/挑 checkpoint 可以使用 validation，但 held-out test 只能在冻结规则后外部审计。

“更快”指新增标签准备、warmup、probe、训练、调度与评估的总耗时降低，不只是 steps 下降；另报预算结束时的质量，以免暂时达标后明显退化。未达标运行单列，不丢弃后只平均成功 seed。

## 6. 最小可行机制与决定性对照

用户给出的是目标而非完整算法。可检验的首版实现是：共同随机 warmup → 在训练诊断面板估计错误组当前排错率与学习进度 → 在基础回放与薄弱组采样间分配下一阶段预算 → 若训练侧保持组恶化，增加其回放。全部信号只来自既有训练数据，任何新标注另披露。

其中，基础回放用于避免完全忘记普通题；保持组使用 B-correct 且确有正负候选对的训练问题。只有正确候选而没有负候选的题不能凭空构造 pair，可通过所有组共同的 pointwise anchor 处理。回放只是缓解机制，不构成测试零回归保证。未独立审计的 AST 错误标签应称弱分组，不能直接称语义根因真值。

必须把“错误分组”与“保持回放”拆开，否则收益可能全部来自已知的 replay。推荐按共同预算运行：

| 训练组 | 要排除的替代解释 |
|---|---|
| Uniform | 基础训练 |
| Hard / loss-margin sampling | 只要多练难例就够了 |
| Static error-balanced | 只是平衡稀疏类别 |
| One-shot error weighting | 无需持续动态诊断 |
| Progress-only / skill controller | 不需要 SQL 错误语义 |
| Dynamic error-aware，无保持回放 | 单独测语义调度 |
| 同样保持回放 + Uniform/Progress | 收益是否仅来自保持机制 |
| Dynamic error-aware + 同样保持回放 | 待验证的组合 |

若要保持首轮规模，可先做四类基础对照和 error-aware 的有/无保持两组；在宣称机制创新前补齐其余强控制。所有组共享 loss、optimizer、训练表示及推理 gate；若给方法新增蒸馏项，也要给对照同样目标，或作为另一轴的消融。

另做真实错误标签与 loss/margin 分层内打乱标签的比较，以及相同曝光多重集的乱序重放。若只有采样频率有效，贡献应称预算分配；若保持回放已能解释全部收益，就不能归功于 SQL 语义课程。事后重放不是在线免费基线，其轨迹构建成本应保留。

## 7. 本地动机与可行性边界

历史 [100 题报告](D:/University/Alpha-SQL/HARD_CASE_PAIRWISE_100_REPORT.md) 显示 4 次修复与 5 次回归，且 hard-focused 仅 9 个训练问题、38 个 pair，更广对照也仅 20 题、82 个 pair。它给出保持能力的具体动机，但没有验证任何本轮课程机制，不能说该机制已获收益或已被数据否定。

专用 [runner](D:/University/Alpha-SQL/alphasql/runner/hard_case_pairwise_experiment.py:600) 默认 batch 128；[训练循环](D:/University/Alpha-SQL/alphasql/algorithm/selection/neural_reranker.py:161) 每个 pair batch 重算所有候选并加全量 BCE。单完整 batch 内重排不能有效检验顺序。应先共同修正 mini-batch/anchor 预算，再独立衡量调度。该工程收益不能记为课程贡献。

既有记录中的 RTX 3070 Ti Laptop GPU 能运行原 MLP，但目前未测基础训练究竟占多少时间。若模型本来极快，在线错误诊断可能更贵。当前池适合管线与机制 pilot，不足以可靠比较多个稀疏错误组或证明严格零回归。正式验证可使用另一个所有方法共享的较大冻结池；增加验证规模不等于只给课程方法增加训练数据，但必须与原小池实验分开报告。不能用重复 pair 数冒充独立题数，也不能人为延长 baseline。

## 8. idea-evaluator 审稿式评估

### 8.1 第一印象

类型：New Setting 下的应用型方法/实证研究。最清楚的定位是“回归约束下的训练效率”，而不是“更高 EX 的新 selector”。一句话故事：相同 SQL 训练池下，以保留原本正确选择为条件，验证错误语义是否改善训练预算分配。

### 8.2 致命问题早审

| 项目 | 严重性 | 检测依据与处置 |
|---|---|---|
| F6：零回归、回归非劣与真实提速未定义 | MAJOR，可修复 | 若无法指定共同 B、联合目标、成本和冻结外部检验，主张不可验证。按第 5–6 节预先定义后再实验 |
| F1：只组合现成弱组采样与 replay | MINOR，针对假设型 pilot；方法论文仍待证明 | 必须说明超越最近邻的具体差异；若无语义增量则降为预算/保持性实证研究，而不是换名声称新算法 |

未发现当前精确假设已被自身受控实验反驳，因其尚未实现隔离对照；不触发“数据已否定核心机制”的直接拒绝。若坚持总体零回归的无条件保证，则现有经验调度不能支持，应拒绝该保证版本，而不是承诺多训即可解决。

五个最接近的方法线索是 Chen/Roberts 等的 Skill-it!（2023）、Chen/Hu 等的 Aioli（2025）、Choi/Sung 等的 LogFloor（2026）、Yan/Xiong 等的 Positive-Congruent Training（2021），以及 Cai/Gao 等的 [KPO](https://arxiv.org/html/2506.00441) [11]。其差异轴分别是 SQL 语义组、净成本/保持联合目标，以及固定 loss/模型的采样干预；这些轴尚待实际贡献验证。另有 Hong/Chen 等的 [ErrorLLM](https://arxiv.org/html/2603.03742v2) [12] 已覆盖 SQL 错误类型，但使用不同数据/表示并修订 SQL，不能把其方法误称为固定池调度。

补查还发现更接近输出任务的 [Measuring and Reducing Model Update Regression in Structured Prediction for NLP](https://papers.nips.cc/paper_files/paper/2022/hash/7af8e3dfefe6e3141144197b8fa44f79-Abstract-Conference.html) [13]：Cai、Mansimov 等已在依存和对话语义解析中研究回归，并提出 BCR 重排。因此不能把结构化输出/重排中的负翻转测量当首次。与之相比，[MIR](https://arxiv.org/abs/1908.04742) [14] 已按预计更新会伤害哪些旧样本来选择回放；“用回归反馈选择下一批样本”也不自动新颖。二者分别是推理重排与持续学习回放，不等于当前固定 SQL pair 的课程净成本问题，但应进入相关工作与保持性控制设计。

### 8.3 生命周期与能力匹配

应用研究，建议按 3–6 个月的竞争窗口规划；这是研究管理判断，不是保证交付时间。已有候选缓存、BIRD 和 MLP 代码使 pilot 可做；有效 pair 支持、标签质量和严格回归证据是主要限制。每周有效投入、截止日期、团队人力和较大模型资源未给定，适配暂为 Yellow，不虚构用户能力评级或工期。

### 8.4 五维评分

以下从 5 分起评；提高分数只针对第 6 节建议的可实现机制，全部未经新数据验证，不把用户的目标当作结果。

| 维度 | 分数 | 依据与最关键验证 |
|---|---:|---|
| Higher | 5/10 | 没有提高最终 EX 的证据，本 idea 本来也不必以此为主；比较固定预算和达标后终点质量 |
| Faster | 7/10 | 机制判断：减少冗余曝光、将预算分给可学习弱组有合理路径；必须用净时间验证，不能承诺倍数 |
| Stronger | 6/10 | 仅对加入共同保持回放的候选版本作机制判断；原始目标陈述本身仍是 5，不能由“不增加回归”四字加分 |
| Cheaper | 5/10 | 随机对照同样不增数据；错误元信息有额外成本，无上调依据 |
| Broader | 5/10 | 只定义 SQL selector，没有跨任务证据 |

优先轴是 Faster，Stronger 是必须通过的约束。当前多项仍在 5，说明需要机制证据，不足以给 Strong Accept。

### 8.5 范式转移探查

第一性原理：No，弱组分配和正确行为保持已有研究。大象问题：Partial，部署稳定性确实重要，但并非无人触及。技术周期：No，所需信号不依赖全新模型能力。Hamming 问题：尚无改变领域优先级的依据。定位为增量研究，不能包装成颠覆训练范式。

### 8.6 可行性风险

计算：原 MLP pilot 低风险，大模型资源未知。数据：高风险，现有独立训练题和错误组太少。工程：中风险，需先统一真实 mini-batch、曝光及共同 anchor。统计：高风险，小样本难支持严格回归非劣。工期：未知，应先做冻结协议与成本测量，不承诺完整论文的确定日期。

### 8.7 判定与前三项行动

**Accept with Revisions：值得作为明确、可证伪的研究假设推进；目前不是已成立的机制或可保证的性能结论。**

1. 先冻结 B、回归分母、质量阈值、回归预算、达标持续规则与全成本计时；严格零回归与允许容忍量分开。
2. 建立所有组共同的训练流程，做采样信号 × 保持回放的机制拆分，保留随机/难例/静态及训练动态强对照。
3. 用配对 seeds 和冻结外部检验决定是否扩大：若 loss-only 或保持回放解释全部收益，停止“错误语义课程”的主张；若只 steps 省而总时间不省，改称更新效率，不声称真实加速。

## 9. 结论：建议保留的精确表述

对 RQ1，语义错误的增量是值得检验的未知量，不是已确认创新。对 RQ2，保持能力使问题更有实际意义，但必须比较共同机制和相同回归预算，不能用更保守的推理 gate 替代课程证据。对 RQ3，净时间与质量/回归联合达标是最有说服力的指标，现有小池尚不能回答。

建议将研究问题表述为：在固定训练数据、模型与候选池下，错误语义能否在 loss/margin 和保持回放之外提供增量，使 SQL 选择器在预先规定的质量与回归约束下更快达标？如要保留严格“不增加回归”，就用 δ_R=0 并接受证据可能不足，不能事后放松后仍用原措辞。

本轮 deep-research 补入 positive congruence 与风险控制边界；idea-evaluator 将它们转化为创新归因、统计证据和实施前置条件。未修改训练代码，未生成新候选，未运行训练或付费模型请求。

## 参考文献

[1] Mayee F. Chen, Nicholas Roberts, et al. “Skill-it! A Data-Driven Skills Framework for Understanding and Training Language Models.” NeurIPS, 2023.

[2] Mayee F. Chen, Michael Y. Hu, et al. “Aioli: A Unified Optimization Framework for Language Model Data Mixing.” ICLR, 2025; arXiv:2411.05735v2.

[3] Seungmin Choi, Jiwon Sung, et al. “Small Models Scout Bottleneck Order for Large-Model Data Control.” arXiv:2608.14936v1, 2026.

[4] Austin Xu, Xuan-Phi Nguyen, et al. “Foundational Automatic Evaluators: Scaling Multi-Task Generative Evaluator Training for Reasoning-Centric Domains.” arXiv:2510.17793v2, 2025.

[5] Xiaoxia Wu, Ethan Dyer, Behnam Neyshabur. “When Do Curricula Work?” ICLR, 2021.

[6] Pablo Messina, Andrés Villa, et al. “CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation.” arXiv:2601.15408v1, 2026.

[7] Sijie Yan, Yuanjun Xiong, et al. “Positive-Congruent Training: Towards Regression-Free Model Updates.” CVPR, 2021.

[8] Zhizhong Li, Derek Hoiem. “Learning without Forgetting.” ECCV, 2016; 方法核对 arXiv:1606.09282v3。

[9] David Lopez-Paz, Marc'Aurelio Ranzato. “Gradient Episodic Memory for Continual Learning.” NeurIPS / NIPS, 2017; arXiv:1706.08840v6.

[10] Anastasios N. Angelopoulos, Stephen Bates, et al. “Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control.” arXiv:2110.01052, 2021 初稿。

[11] Shihao Cai, Chongming Gao, et al. “K-order Ranking Preference Optimization for Large Language Models.” arXiv:2506.00441v1, 2025.

[12] Zijin Hong, Hao Chen, et al. “ErrorLLM: Modeling SQL Errors for Text-to-SQL Refinement.” arXiv:2603.03742v2, 2026.

[13] Deng Cai, Elman Mansimov, et al. “Measuring and Reducing Model Update Regression in Structured Prediction for NLP.” NeurIPS, 2022.

[14] Rahaf Aljundi, Lucas Caccia, et al. “Online Continual Learning with Maximally Interfered Retrieval.” NeurIPS, 2019.
