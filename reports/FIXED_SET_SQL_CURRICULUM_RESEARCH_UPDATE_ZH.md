# 固定训练集下的 SQL 选择器训练：错误信号的增量价值与真实效率

更新日期：2026-10-20。范围：文献深度调研与实验设计；本轮没有训练新模型、补生成候选或调用付费模型 API。

## 摘要

当前问题不是怎样再生成更多 SQL，而是给定相同训练问题、候选、标签和模型，怎样更有效地使用训练预算。本文在已有课程学习调研之上，围绕三个问题做针对性补查：SQL 错误类型是否提供超出 loss/margin 的调度信息，动态反馈是否优于一次性配比，以及减少更新次数能否转化为包含诊断开销的总时间节省。综合课程排序、自适应混合、偏好排序及 SQL 错误建模的 22 篇代表工作，现有证据支持有条件的训练加速，但不支持“错误越多越值得练”或“加入课程必然改善 pairwise 判断”。新补入的 FARE、KPO、Fine-Grained Order Fine-Tuning 与 ErrorLLM 进一步收窄了创新边界。本文提出一个可证伪的研究定位：在固定 SQL pair 池与共同训练流程下，验证语义错误分组相对训练动态的独立增益，并同时审计回归和净训练成本。配套的独立选题评估给出继续探索所需的前提，而非把未实现的机制写成结果。

## 1. 要回答的问题与不改变的条件

研究对象是监督式 SQL candidate selector / pairwise reranker。先冻结上游生成器及其产物，不研究生成器课程，也不把更换 schema linker、增加错误解释、重新标注执行标签或生成新负例混入主实验。

- RQ1：控制当前 loss、pair margin、结构复杂度与组大小之后，projection、filter、join、aggregation 等错误分组是否仍能改善训练调度？
- RQ2：同一训练池中，动态错误反馈是否优于随机、静态均衡、warmup 后一次重加权及无语义的学习进度调度？
- RQ3：在保持最终选择质量、普通样本稳定性与监督预算的条件下，课程能否降低达到预设性能的总训练时间？

“固定训练集”至少固定 question IDs、question/evidence/schema、candidate SQL、执行标签、pair IDs、划分与输入特征。错误标签是调度用的额外元信息，也属于监督：若需要专家或教师生成，必须披露来源及成本，不能称为完全不增加监督。

## 2. 检索方法与核验边界

本轮使用 deep-research 的多视角检索，分别检查支持机制、反证/测量方法、SQL 与排序近邻，再统一综合。先搜索 `error-aware curriculum SQL reranker`、`fixed dataset pairwise curriculum`、`curriculum SQL selection`，再按检出的 FARE、KPO、ErrorLLM、LogFloor 等方法和引用链回查。宽泛 SQL curriculum 搜索混入大量人类教育课程，故第二轮限制到 arXiv、ACL Anthology、OpenReview、PMLR 和作者机构页面。

下文选择 22 篇与三个 RQ 直接相关的代表作，不宣称穷尽。核对题名、作者、版本和摘要；涉及调度、训练对照、版本变化及数字时，另核对正文相关章节。新检索之外的背景来自 [2026-10-18 课程综述](D:/University/Alpha-SQL/reports/ERROR_AWARE_CURRICULUM_SURVEY_ZH.md) 与 [NL2SQL Handbook 学习报告](D:/University/Alpha-SQL/reports/NL2SQL_HANDBOOK_STUDY_ZH.md)。不是把旧综述的全部文献重新宣称为本轮逐页精读。

仅用一手论文支撑技术判断。预印本注明版本，不把投稿信息当录用；ErrorLLM 本轮进一步核对了 2026-06-23 的 v2，避免仅依据三月 v1 解释现方法。FARE 的具体消融依据 arXiv v2。未检索到完全相同设定，只能说明当前检索未发现直接重叠，不能证明首创。

本地代码和结果仅用于判断实验是否可做，不是课程有效性的证据。本轮没有运行新的训练实验。

## 3. 先分清四种不同的干预

| 干预 | 允许改变 | 必须保持或单独披露 | 可回答的问题 |
|---|---|---|---|
| Order-only | 样本顺序 | 最终曝光多重集、batch 规则、优化器相同 | 顺序是否有效 |
| Pacing | 随时间可用的子集 | 同 pacing 的随机对照；各条曝光日志 | 难度排序还是开放规模有效 |
| Fixed-pool allocation | 同池内采样频率 | 相同预算；实际频率及目标权重 | 预算分配是否有效 |
| 监督/模型改变 | 新错误解释、新负例、adapter 或目标 | 作为另一研究轴，不归于纯调度 | 整体系统是否改善 |

后三种常被统称课程，但不能相互替代。固定数据支持集不等于每条曝光相同，也不等于优化目标不变。主实验建议明确采用 fixed-pool allocation，另做 order-only 机制对照，而不是用一个模糊名称同时承诺全部性质。

## 4. 顺序能有帮助，但不能默认优于随机

[When Do Curricula Work?](https://research.google/pubs/when-do-curricula-work/) [1] 指出常规条件下课程收益有限，随机开放训练集也可能同样好；[Beyond Random Sampling](https://aclanthology.org/2026.eacl-long.271/) [2] 则在语言模型预训练中报告早中期收敛收益。两者并不直接冲突：任务、数据规模、有限预算及噪声条件不同。因此应检验特定 SQL selector 的学习曲线，不能把预训练的步数节省照搬到本项目。

与 [2] 的多种数据情景相比，[Fine-Grained Data Ordering](https://aclanthology.org/2026.findings-acl.1021.pdf) [3] 更接近固定池顺序干预。核对其训练设置后，不能仅按摘要将其理解为持续逐 epoch 在线排序：所检查的两 epoch 配方先排序、再随机。它提示随机巩固可能重要，但最终分数改善仍不等于含排序成本的达标时间缩短。主实验既要有强随机基线，也要避免把所有收益解释成“从易到难”。

## 5. 错误、学习进度与技能瓶颈：近邻已很充分

### 5.1 高 loss 不等于高收益

[Automated Curriculum Learning](https://proceedings.mlr.press/v70/graves17a.html) [4] 用学习进度分配任务；[RHO-LOSS](https://arxiv.org/html/2206.07137v3) [5] 区分尚未掌握、值得学习与难以学习的点。两条线共同约束当前设计：不能只追逐高错误率，也不能把低 loss 样本永久删除。前者的进度是观测代理，后者还引入参考模型估计；都不使“每条样本的真实因果训练效用”变得免费可得。

### 5.2 按错误组动态配比并不是新机制

[Skill-it!](https://arxiv.org/abs/2307.14430) [6] 基于技能及其影响关系安排训练；[Aioli](https://arxiv.org/html/2411.05735v2) [7] 在线估计组间混合规律。二者意味着：把 SQL 错误名换成 skill，再按组 loss 调整比例，本身不足以确立新颖性。Aioli 对简单 stratified sampling 的强调尤其要求加入静态错误均衡，而不能只对比全量随机。

[Difficulty Is Not Enough](https://ojs.aaai.org/index.php/AAAI/article/view/40400) [8] 将 difficulty 与 utility 分开；[LogFloor](https://arxiv.org/html/2608.14936v1) [9] 则明确研究达到各技能目标门槛的首次达标成本。与 [6][7] 相比，[9] 进一步提供静态边际混合和逆序路径控制，故“用薄弱技能反馈减少达标训练量”也不是空白。其主要成本是 training tokens，scout 折算不等于全部诊断墙钟；本项目应验证语义错误信号及真实总耗时，而不是重提这一目标。

### 5.3 更好的错误诊断未必需要更频繁地更新

[CURE](https://arxiv.org/html/2601.15408v1) [10] 已按数据源和类别错误率重分配训练；[Confusion-Aware Transfer Teacher](https://arxiv.org/html/2606.17706v1) [11] 则展示更细错误分布评分仍不能保证全数据收益。前者支持 warmup 后一次性配比是强对照，后者要求把评分与 pacing 分开。两者都不支持“动态模块越复杂越好”。

还有一个迁移边界：[11] 的 ConfVar 是对错误类别条件概率的总体方差。若直接用于只有两个输出的二元判断，错误类别只剩一个，在残余概率非零时其条件概率为 1，方差恒为 0。这是依据其公式的推导，不是论文对 SQL 的实验结论。SQL 语义错误组因此不能简单等同于二元输出类别；应单独构造、审计其信息，而非直接搬用多类 confusion 指标。

## 6. 排序与 SQL 的直接近邻限制了哪些创新说法

[Curri-DPO](https://aclanthology.org/2024.findings-emnlp.754/) [12] 已研究排序偏好与课程；[KPO](https://arxiv.org/html/2506.00441) [13] 更直接在 LLM ranking 中按相关候选数量 K 安排课程，并对比随机和降序。因此不能宣称首次对排序模型使用课程，或首次通过课程追求排序训练效率。二者的偏好优化对象与当前数值特征 RankNet 不同，适合作为机制近邻，而不是不改模型就能直接运行的同构 baseline。

[GradeSQL](https://arxiv.org/abs/2509.01308v2) [14] 提供执行监督的候选 outcome verifier；[CHASE-SQL](https://arxiv.org/abs/2410.01943) [15] 提供带 schema 证据的候选对选择。它们说明“训练 SQL selector”和“schema-aware pairwise”均已有先例。研究变量应落在共同模型的训练调度；扩大候选池或改变 verifier 输入所带来的收益应另列。

| SQL 近邻 | 作用对象 | 与纯调度混在一起的变化 | 对本项目的边界 |
|---|---|---|---|
| [LeGo-Code](https://arxiv.org/html/2604.18254v1) [16] | SQL 生成器 | 分层 adapters 与组合 | 朴素复杂度课程有负结果；不能把模块收益归于顺序 |
| [SAC-SQL](https://aclanthology.org/2025.findings-emnlp.190/) [17] | SQL 生成器 | 数据设计/合成及结构课程 | 不是固定候选池的选择器调度 |
| [ErrorLLM](https://arxiv.org/html/2603.03742v2) [18] | 错误检测与 SQL refinement | AST 扰动数据、error tokens 与 LoRA | SQL 错误类别本身不是新贡献 |

与 [16][17] 的生成训练相比，[18] 更直接使用错误类别，但它改变监督和表示，并在推理时修订 SQL。这与只调度既有 pair 的项目仍有差异；差异是具体作用对象和干预方式，而不是它“没有错误感知”。

更需要正视的是 [FARE](https://arxiv.org/html/2510.17793v2) [19]。其通过率课程消融中，pairwise 平均分为有课程 61.14、无课程 61.49；ProcessBench 则为 58.03 与 53.43。与 [12][13] 的正向课程动机相比，这说明训练任务粒度会改变收益。FARE 使用拒绝采样 SFT，不是固定 SQL pool；这些终点分数既不能证明 SQL 课程无效，也不能证明其训练时间更慢，但足以否定“对 evaluator 加课程必然提高 pairwise 质量”的前提。

## 7. 怎样把“更快”变成可验证主张

[Importance Sampling](https://proceedings.mlr.press/v80/katharopoulos18a.html) [20] 区分有信息样本和实际采样收益；[Example Forgetting](https://arxiv.org/abs/1812.05159) [21] 与 [Dataset Cartography](https://aclanthology.org/2020.emnlp-main.746/) [22] 则提示记录正确转错及训练动态。由此得到的设计要求是：不能只报告 hard case 修复和 optimizer steps，还要检查普通题退化、标签可靠性及采样开销。

设相同单步规则下，基础方法需要 N_base 步，课程需要 N_cur 步，平均训练步耗时为 c。一个简化的盈亏条件是：

\[
T_{\mathrm{label}}+T_{\mathrm{probe}}+T_{\mathrm{schedule}}
< (N_{\mathrm{base}}-N_{\mathrm{cur}})c.
\]

实际报告直接测总时间，不依赖单步恒定假设。应分别报告一次性标签/特征准备、每轮诊断、训练、共同评估和缓存复用后成本；共用的候选生成成本可单列，不能把其节省算作本次训练调度收益。MLP 很小时，探测和 Python 调度可能吃掉全部节省，这不是实现细节，而是核心假设的适用条件。

若目标仍是相同的均匀或预设加权风险，非均匀采样需要记录 p_t，并考虑 q_0/p_t 的重要性校正 [20]。未校正就应诚实称为动态重加权。逆权重截断或 batch 内自归一化改变严格无偏性质；“固定集合”不能替代目标定义。

## 8. 本地审计：哪些事实约束下一步

依据 [100 题报告](D:/University/Alpha-SQL/HARD_CASE_PAIRWISE_100_REPORT.md) 和 [机器结果](D:/University/Alpha-SQL/results/hard_case_pairwise_100/report.json)：hard-focused 实际训练 9 题、38 个 pair；all-pairwise 对照是 20 题、82 个 pair。主模型 EX 75%，自一致性 76%；4 次修复被 5 次回归抵消。该实验测试的是 hard-focused 监督与选择策略，不是本轮提出的错误课程，不能用来宣称课程已提升，也不能据此宣布课程已被否定。

[专用 runner](D:/University/Alpha-SQL/alphasql/runner/hard_case_pairwise_experiment.py:600) 默认 batch size 128，大于这两种 pair 数；[共同训练循环](D:/University/Alpha-SQL/alphasql/algorithm/selection/neural_reranker.py:161) 在每个 pair batch 上计算全体候选分数并叠加全量 pointwise BCE。因此只在完整 batch 内重新排列 pair，通常不改变该次梯度；减小 batch 后若仍重复全量 BCE，还会改变更新和成本结构。

应为所有比较组共同建立有效的 mini-batch 流程，固定 pointwise anchor 的计算预算，并区分这个基础工程修正与课程收益。原有 hard-negative/SC loss 权重也要共同冻结或共同取消，避免无记录地重复强化同一信号。

当前记录过 RTX 3070 Ti Laptop GPU，但本轮没有测量可用显存、当前训练吞吐或持续资源。小 MLP 机制 pilot 可依托既有实现；任何大模型微调能力和端到端训练时长仍未确认。旧 100/287 题已反复分析，不再作为新方法完全未见的终局测试。

## 9. 建议冻结的最小方法与对照

首版只使用训练侧的错误组排错率和近期 loss 变化，不使用教师在线解释、不生成新 SQL、不学习复杂技能图。将多标签归属归一化，稀疏组向整体均值收缩；无法可靠归类的 pair 进入 unknown，不因其高 loss 自动判为噪声。部分采样始终来自共同基础分布，维持已有正确模式的回放。

方法组应共享相同模型、初始化、训练标签、batch/anchor 规则、优化器、输入表示、SC 融合与诊断预算：

| 对照 | 排除的替代解释 |
|---|---|
| Uniform；另明确 pair-uniform 或 question-balanced | 基础效果；避免候选多的问题支配训练 |
| Difficulty-only；若有 pacing 加 Random-pacing | 仅难度或开放规模就足够 |
| Loss/margin/progress-only | 不需要错误语义即可得到收益 |
| Static error-balanced | 收益只是类别均衡 |
| Warmup + one-shot error weighting | 无需动态重复诊断 |
| Dynamic error-aware | 待检验的方法 |
| Skill/bottleneck controller 的同预算适配 | 是否只重复已有分组控制思想 |

如果某基线需要额外训练参考模型，单独报告其准备成本，不给它免费的数据或 compute。正式机制检验还要做：在 loss/margin/复杂度分层内打乱错误标签；将动态方法实际消费的曝光多重集乱序重放；固定相同控制器，只移除语义输入。事后重放须计入得到原轨迹的成本，不伪装成在线免费方案。

建议先以共同初始化的五个配对 seed 做低成本 pilot。主指标为预先冻结 validation 目标的稳定首次达标总时间，辅以相同预算 EX、条件选择准确率、fix/regression、SC-correct retention 和每组曝光。未达标 seed 必须报告，不能只平均成功者；pair 不是独立问题样本，准确率区间按 question 重采样。若主张跨数据库泛化，另采用 db-disjoint 划分，不能沿用当前非跨库 split 直接声称。

## 10. 尚未解决的三个问题

**错误标签到底新增了什么？** [6][7][9] 已有技能或组反馈，而 [18] 已有 SQL 错误建模；真正未知的是这些标签在给定相同训练动态后是否提供增量。若打乱标签或纯 loss 调度同样好，不能把收益归于语义理解。

**收益来自顺序还是频率？** [3][13] 强调训练顺序，[9] 又对静态混合与路径顺序做分离。若本项目只有频率有效，应改称错误感知预算分配，而非声称发现不可替代的课程阶段。

**为什么值得为小选择器优化训练时间？** [5][20] 的节省依赖训练与选择成本比例，[19] 提醒 pairwise 未必受益。若现有模型本来训练极快，应把当前阶段定位为机制验证；不能人为加大模型或插入延时制造效率故事。更高成本模型可作为独立、各组共同的后续验证，而非偷换核心条件。

## 11. 结论

对 RQ1，文献支持区分“错了多少”和“错在哪里”，但尚未提供本项目中错误语义超越 loss/margin 的实测依据。可研究的贡献是条件增量，而不是错误感知这一名称。

对 RQ2，动态技能分配、错误组重加权和门槛反馈已有明确先例；必须胜过静态平衡、一次性重加权和训练动态调度，才能主张持续诊断值得其成本。

对 RQ3，减少更新次数不等于真实提速。应同时测总耗时、达标率、同预算质量和回归。当前训练循环和几十个 pair 适合先做 sanity check，不足以直接承诺广泛加速或最终 EX 提升。

综上，值得推进的是一个受约束、可被反证的实验：固定 SQL pair 池，检验语义错误标签是否改善预算分配，并在净成本和回归上通过强对照。是否值得扩大为论文，应由这个实验而非方法命名决定。审稿式评估见 [配套选题评估](D:/University/Alpha-SQL/reports/FIXED_SET_SQL_CURRICULUM_IDEA_EVALUATION_ZH.md)。

## 参考文献

[1] Xiaoxia Wu, Ethan Dyer, Behnam Neyshabur. “When Do Curricula Work?” ICLR, 2021.

[2] Yang Zhang, Amr Mohamed, et al. “Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning.” EACL, 2026.

[3] Xiaomeng Hu, Yixuan Tang, et al. “Fine-Grained Data Ordering Improves Fine-Tuning for Large Language Models.” Findings of ACL, 2026.

[4] Alex Graves, Marc G. Bellemare, et al. “Automated Curriculum Learning for Neural Networks.” ICML, 2017.

[5] Sören Mindermann, Jan Brauner, et al. “Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt.” ICML, 2022.

[6] Mayee F. Chen, Nicholas Roberts, et al. “Skill-it! A Data-Driven Skills Framework for Understanding and Training Language Models.” NeurIPS, 2023.

[7] Mayee F. Chen, Michael Y. Hu, et al. “Aioli: A Unified Optimization Framework for Language Model Data Mixing.” ICLR, 2025; arXiv:2411.05735v2.

[8] Zishang Jiang, Jinyi Han, et al. “Difficulty Is Not Enough: Curriculum Learning for LLMs Fine-tuning Must Consider Utility.” AAAI, 2026.

[9] Seungmin Choi, Jiwon Sung, et al. “Small Models Scout Bottleneck Order for Large-Model Data Control.” arXiv:2608.14936v1, 2026; 投稿信息不等于录用。

[10] Pablo Messina, Andrés Villa, et al. “CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation.” arXiv:2601.15408v1, 2026.

[11] Savini Kommalage, Sanka Mohottala, et al. “Confusion-Aware Transfer Teacher Curriculum Learning Framework: Disentangling Scoring and Pacing Effects.” arXiv:2606.17706v1, 2026.

[12] Pulkit Pattnaik, Rishabh Maheshwary, et al. “Enhancing Alignment using Curriculum Learning & Ranked Preferences.” Findings of EMNLP, 2024.

[13] Shihao Cai, Chongming Gao, et al. “K-order Ranking Preference Optimization for Large Language Models.” arXiv:2506.00441v1, 2025.

[14] Mattia Tritto, Giuseppe Farano, et al. “GradeSQL: Test-Time Inference with Outcome Reward Models for Text-to-SQL Generation from Large Language Models.” arXiv:2509.01308v2, 2025.

[15] Mohammadreza Pourreza, Hailong Li, et al. “CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL.” ICLR, 2025.

[16] Salmane Chafik, Saad Ezzini, Ismail Berrada. “LeGo-Code: Can Modular Curriculum Learning Advance Complex Code Generation? Insights from Text-to-SQL.” arXiv:2604.18254v1, 2026.

[17] Xiaohu Zhu, Qian Li, et al. “Learning SQL Like a Human: Structure-Aware Curriculum Learning for Text-to-SQL Generation.” Findings of EMNLP, 2025.

[18] Zijin Hong, Hao Chen, et al. “ErrorLLM: Modeling SQL Errors for Text-to-SQL Refinement.” arXiv:2603.03742v2, 2026; 作者页面注明 SIGKDD 2026 录用。

[19] Austin Xu, Xuan-Phi Nguyen, et al. “Foundational Automatic Evaluators: Scaling Multi-Task Generative Evaluator Training for Reasoning-Centric Domains.” arXiv:2510.17793v2, 2025.

[20] Angelos Katharopoulos, François Fleuret. “Not All Samples Are Created Equal: Deep Learning with Importance Sampling.” ICML, 2018.

[21] Mariya Toneva, Alessandro Sordoni, et al. “An Empirical Study of Example Forgetting during Deep Neural Network Learning.” ICLR, 2019.

[22] Swabha Swayamdipta, Roy Schwartz, et al. “Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics.” EMNLP, 2020.
