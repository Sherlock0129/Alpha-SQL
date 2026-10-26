# 固定训练集下的错误感知课程学习：从难例排序到可验证的训练效率

调研日期：2026-10-18。面向当前 SQL candidate selector / pairwise reranker 项目。

## 摘要

在不增加训练数据的前提下，改变样本顺序或采样概率，是否能让 SQL 选择器更快学会过滤、投影、聚合等容易混淆的语义？本报告围绕错误信息的增量价值、动态调度机制和训练效率的公平测量，核验了 25 篇相关文献，覆盖经典课程、自适应技能采样、训练动态诊断、SQL 验证器与 2026 年近期工作。已有研究支持课程在部分有限预算、噪声或技能不平衡条件下的价值，但不支持“错误更多就应始终多采样”或“动态一定优于静态”。技能反馈、学习进度和错误分组重加权已有明确先例；当前值得检验的是：SQL 语义错误标签能否在 loss、margin、类别平衡和技能缺口控制之外改善学习，并在计入标注、诊断与调度成本后缩短达标时间。结合本地代码，本报告还发现当前 pair 数小于默认 batch size，单纯更换顺序不足以形成有效课程实验。最后提出分开检验纯顺序与动态重加权的协议，以及以错误标签打乱、相同曝光次数乱序和回归率为核心的证伪实验。本文提供研究依据和实验建议，没有实施新训练，也不宣称课程已提升项目结果。

## 1. 问题边界与研究问题

当前项目已有执行标签和候选 SQL，适合把生成问题暂时固定，专门研究选择器怎样训练。这里的固定 training set 包括 question、evidence、schema、候选 SQL、执行标签、候选 pair 和划分；不在线补生成 SQL，不引入教师改写，不换训练标签，不以扩大数据解释收益。

研究问题限定为：

- **RQ1：错误类型是否有独立信息？** 在控制 loss、pair margin、结构复杂度和组大小后，projection、filter、join、aggregation 等标签是否仍能帮助训练？
- **RQ2：何时应重复训练哪类错误？** 当前错误率、学习进度和遗忘信号应怎样结合；一次性调整是否已足够？
- **RQ3：怎样证明更高效？** 是减少 optimizer steps、减少处理的 pair/token，还是计入全部额外开销后真实 wall-clock 更短？

本文前半部分发展文献证据，后半部分才据此提出项目方案。下面的数学定义和实验建议是本报告的设计，不是已经复现的论文结论。

## 2. 检索与核验方法

采用五个视角：经典难度课程；自适应采样和学习进度；否定结果与评估方法；SQL/偏好学习应用；2025—2026 年近邻。三个视角并行检索，再统一核验、补充相互引用的近邻与反证。

主要检索词包括 `curriculum learning fixed dataset`、`learning progress task sampling`、`error-aware curriculum`、`difficulty utility curriculum`、`When Do Curricula Work`、`SQL curriculum reranker`、`skill bottleneck data mixing`。近邻扩展包括 Skill-it!、Aioli、CURE、DUCL、SAC-SQL、LogFloor 和 random pacing。近期检索延伸至 2026 年 10 月，最后纳入的近期预印本首次提交于 2026-10-09。

只依据作者论文、PMLR、ACL Anthology、NeurIPS、AAAI、作者机构页面和 arXiv；第三方摘要仅用于发现线索。每条参考核验题名、作者、时间及相关摘要；涉及方法、同数据对照、消融或版本差异时另检查正文相关章节。没有声称逐页精读全部附录，也没有复现实验。25 篇中 GradeSQL 与其 2026 年同作者稿件属于同一研究线索，不作为两次独立复证。

纳入标准是至少回答一个 RQ；排除仅面向人类教育的 curriculum、单纯主动标注新数据、纯架构深度调度，以及无法确认的一手引用。没有把所有“使用既有数据”的方法一律视为严格固定样本实验。该检索是问题驱动的叙述性综述，不是具有穷尽数据库召回率的系统综述；未检索到直接同题工作不能证明首创。

本地依据为 [100 题监督实验报告](D:/University/Alpha-SQL/HARD_CASE_PAIRWISE_100_REPORT.md)、[287 题 linker 受控实验](D:/University/Alpha-SQL/LINKER_ABLATION_287_RESULTS.md)、训练代码和已有机器结果。这些历史评估不是新方法的独立测试集。

## 3. 分类：固定集合不等于固定训练过程

以主要控制变量分为四类；一篇论文可以同时涉及多类，讨论时按具体实验归类。

| 类型 | 允许改变什么 | 需要特别控制什么 | 能回答的问题 |
|---|---|---|---|
| Order-only | 同一已冻结多重集的出现顺序 | 每个样本总曝光次数、batch 规则、预算相同 | 顺序是否改变优化路径？ |
| Pacing / admission | 随时间开放不同的可用子集 | 同样扩展速度的随机子集对照 | 收益来自排序还是开放规模？ |
| Adaptive allocation | 固定池内的采样频率、组配比 | 曝光日志、目标权重、评分与监控成本 | 动态预算分配是否更划算？ |
| Data / supervision intervention | 改写输入、扩展 pair、引入教师或额外知识 | 单独拆出 schedule 消融 | 系统总收益不能直接归于课程 |

错误标签还要区分三层：候选 SQL 的语义错误、某个 pair 的差异标签、当前模型把该 pair 排错的事件。SQL 有 GROUP BY 不等于存在 GROUP BY 错误；一个静态“filter error”负例也可能早已被当前模型正确区分。

## 4. 从易到难：有条件的优化工具，不是普遍规律

### 4.1 静态难度与模型能力是两件事

[Curriculum Learning](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf) [1] 把逐渐变化的训练分布与 continuation 联系起来；[Self-Paced Learning](https://papers.nips.cc/paper_files/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html) [2] 则由当前模型迭代决定可接纳样本。因此，自适应能力判断并不是近期才加入课程的概念。两者提供优化动机，但经典实验并不都符合“冻结同一 SQL pair 多重集”的严格设定。

[Competence-based Curriculum Learning](https://aclanthology.org/N19-1119/) [3] 将样本难度和逐渐增长的 competence 分开；[Beyond Random Sampling](https://aclanthology.org/2026.eacl-long.271/) [4] 进一步比较固定集排序、pacing 和 interleaving。在本项目中，应同时冻结难度定义和开放速度；仅说“由简单 SQL 到复杂 SQL”不足以构成可复现算法。

Zhang 等的固定集 S1 与大池 pacing 实验必须分开理解：前者使用同一有限训练集，后者改变预算内的采样。论文报告特定设置下的早中期加速和 warmup 收益，不意味着 SQL selector 会有同等幅度提升，更不能把其汇总的 18%—45% 步数减少设为项目已获收益。[Beyond Random Sampling 正文](https://arxiv.org/html/2506.11300v2) [4]。

### 4.2 随机开放子集是必需的反证基线

[When Do Curricula Work?](https://research.google/pubs/when-do-curricula-work/) [5] 在常规设置中发现课程收益有限，随机顺序但逐渐开放训练集的 random-curriculum 可相当或更好；[Confusion-Aware Transfer Teacher](https://arxiv.org/abs/2606.17706v1) [6] 即便改进了错误分布感知的难度分数，全数据实验也未因此优于标准训练。这与 [4] 的早期加速并不矛盾：任务、预算、噪声和 pacing 不同，最终收敛性能也不是早期学习速度。

因此至少要比较三种情况：全量随机；随机排列后按相同 pacing 开放；真实难度排序后按同样 pacing 开放。只有第三项优于第二项，才能把差异归于排序信号，而非初期减少可用数据。这是对 SQL 实验的设计推论，不是上述分类任务结果对 SQL 的直接否定。

## 5. 动态错误课程：近邻已经覆盖了哪些思想？

### 5.1 从高错误率转向可学习进度

[Automated Curriculum Learning](https://proceedings.mlr.press/v70/graves17a.html) [7] 已用学习进度驱动非平稳 bandit 分配任务；[RHO-LOSS](https://arxiv.org/abs/2206.07137) [8] 则区分尚未掌握、值得学习与难以学习的样本。两条线共同说明：当前 loss 大不等于当前训练收益高，已学会的简单样本与无法解释的高损失样本都可能浪费预算。

二者的机制不同：Graves 等估计训练中的进度，RHO-LOSS 使用 holdout 模型估计可约损失。前者的同批次训练前后 loss 降幅可能乐观，后者有参考模型与数据成本。不能把缓存 loss 的下降直接称为该样本的真实因果 utility，也不能默认高损失、低进度就一定是噪声；它可能缺少前置技能或当前表示能力。

### 5.2 错误分组与在线配比不是空白领域

[Skill-it!](https://arxiv.org/abs/2307.14430) [9] 使用技能及其依赖关系指导在线采样；[Aioli](https://arxiv.org/abs/2411.05735) [10] 在线估计数据组混合规律，同时指出多种复杂 mixing 方法未稳定超过简单 stratified sampling。把 SQL 错误类别改名为“技能”并依据组 loss 采样，很接近这条已有路线；必须与静态组平衡及无语义难度分组比较。

[DUCL](https://ojs.aaai.org/index.php/AAAI/article/view/40400) [11] 已明确提出 difficulty 之外要考虑 utility，并用 DUE 与 Window Ordering 保持多样性；它与 Skill-it! 的差异在于样本分布代理和调度形式，而不是首次认识到“难例不一定有价值”。本项目可检验 SQL 语义标签能否改进组级 utility 估计，但不能仅以 difficulty × utility 的组合公式作为创新。

### 5.3 Error-aware 的直接先例与动态更新的反例

[CURE](https://arxiv.org/html/2601.15408v1) [12] 按数据源和类别错误率调整采样；与 [9][10] 相比，它是更直接的 error-aware 先例。不过其附录最终配置是 3,000 步 uniform warmup 后一次重加权、再固定训练 3,000 步，频繁更新不是最终最佳选择。这支持将“一次诊断后固定”列为强对照，而非预设动态优胜。

[Error-Aware Curriculum Learning for Biomedical Relation Classification](https://arxiv.org/html/2507.14374v1) [13] 使用教师识别错误并提供改写、指导及知识增强；它与 [12] 都有错误感知思想，但前者还改变了输入监督内容，不能作为纯顺序实验的直接证据。迁移到 SQL 时，应只让错误标签控制 scheduler，暂不把额外纠错解释喂给模型。

| 方法线索 | 信号 | 与固定 SQL pool 的关系 | 本项目必须面对的对照 |
|---|---|---|---|
| Graves [7] | 学习进度、bandit | 思想可迁移，原任务不全是固定集 | 无语义的 learning-progress scheduler |
| RHO-LOSS [8] | 可约 holdout loss | 可在固定池选择，额外参考成本 | loss-priority 与 learnability |
| Skill-it! [9] | 技能 loss、技能依赖 | 同池技能分组的强近邻 | Skill-it-style 组调度 |
| Aioli [10] | 在线组混合规律 | 动态配比的强近邻 | 简单 stratified / error-balanced |
| DUCL [11] | difficulty / utility 分布代理 | 需按具体实验核对曝光与预算 | 相同窗口的 difficulty-only |
| CURE [12] | 组错误率 | 不增加原始数据，但有任务重构 | warmup 后一次性重加权 |
| Biomedical error-aware [13] | 教师错误标签与补救 | 输入被增强，不是纯 schedule | 不能把系统总增益当课程增益 |

表中共同结论：应验证错误信号的额外价值，而不是宣称首次动态采样。

## 6. 训练动态、目标漂移与回归风险

[Example Forgetting](https://arxiv.org/abs/1812.05159) [14] 关注样本由正确转错的事件；[Dataset Cartography](https://aclanthology.org/2020.emnlp-main.746/) [15] 使用正确标签置信度及其跨 epoch 波动区分 easy、ambiguous 与 hard 区域。它们提供诊断工具，而非证明所有高波动样本都应优先训练。对 SQL 选择器，应同时记录 pair margin 波动、question winner 的修复和遗忘，避免只看 hard subset 修复率。

[Importance Sampling](https://proceedings.mlr.press/v80/katharopoulos18a.html) [16] 通过逆概率权重保持目标梯度期望；[Group DRO](https://arxiv.org/abs/1911.08731) [17] 则明确改变目标以改善最坏组风险。错误组过采样更接近后者的动态重加权，而不自动等于“更快优化同一均匀目标”。Group DRO 的结果还提示，拟合训练弱组并不保证弱组测试泛化，正则化同样重要。

设固定目标为

\[
R(\theta)=\sum_i q_0(i)\ell_i(\theta).
\]

若按 \(p_t(i)\) 抽样但不校正，梯度期望来自 \(\sum_i p_t(i)\ell_i\)。若使用

\[
\widehat g_i=\frac{q_0(i)}{p_t(i)}\nabla_\theta\ell_i,
\]

在支持集完整、概率计算正确且按该估计量求平均时，可保持原目标梯度的条件期望。该推导不保证 Adam 的完整轨迹等价；截断逆权重或 batch 内自归一化也不再严格无偏。窗口外概率为零时，不能同时宣称对全训练目标无偏。实际效率还要扣除抽样、probing 和概率维护开销 [16]。

本项目建议主实验明示为“固定池下的课程/重加权”；若论文主张严格不改变优化目标，再补完整支持集的 importance-corrected 版本。不要将二者混写。

## 7. 与 SQL 选择和偏好训练的关系

[Curri-DPO](https://aclanthology.org/2024.findings-emnlp.754/) [18] 按同一 prompt 多个响应之间的质量差构造课程；[GradeSQL](https://arxiv.org/html/2509.01308v2) [19] 则用 SQL 执行标签训练 outcome verifier。前者提供 pair 调度思路，后者说明专用候选评分的任务形式；两者合起来仍不能证明错误课程对 SQL verifier 有效。

Curri-DPO 正文包含相同多组 pair 的 pooled-random 对照，也区分固定与迭代 reference，因此不能把其全部收益简单解释成多用了数据。但其最佳方案同时含分阶段训练和 reference 更新。我们的 RankNet selector 不应引入这些变化，才能单独比较 schedule。另一个差异是：SQL 正确/错误二元标签的 pair 标签差都为 1，不能照搬响应质量评分差作为难度，应使用当前 margin 等可观测量 [18]。

GradeSQL 的 balanced 版本通过下采样构建，不能当作固定多重集的课程；同作者 2026 年的 [Test-Time Verification](https://arxiv.org/abs/2606.30851) [20] 属于相关后续稿件，不应当作独立重复验证。与它们相比，[SAC-SQL](https://aclanthology.org/2025.findings-emnlp.190/) [21] 直接研究 SQL 结构难度课程，但完整系统含合成数据，优化对象是生成器而非冻结候选池的选择器。

| 近邻 | 可以借鉴 | 不应照搬的推论 |
|---|---|---|
| Curri-DPO [18] | 同 prompt 的 pair 难度与课程对照 | 最佳结果全来自顺序；二元标签差能表示难度 |
| GradeSQL / 后续稿 [19][20] | 执行监督、候选 verifier、BoN 对照 | 数据平衡收益等于固定样本课程收益 |
| SAC-SQL [21] | AST 结构、复杂度、SQL 任务课程 | SQL 生成器课程已解决 SQL selector 训练效率 |

截至本次关键词与近邻检索，未获得可直接证明“固定 SQL candidate/pair 集上的语义错误课程，计入调度成本后优于强采样基线”的证据。这是尚需验证的具体组合，不是已经确认的首创声明。

## 8. 2026 年近期进展：新的重点是瓶颈顺序，而非泛称动态

[Implicit Curriculum Hypothesis](https://arxiv.org/abs/2604.08510v1) [22] 观察模型技能获得顺序的规律；[RegMix-D](https://arxiv.org/abs/2606.18663v1) [23] 用 proxy 训练轨迹设计动态数据配比。前者主要提供观察证据，后者实施调度干预；不能由“存在自然学习顺序”直接推出某个人工课程必然加速。

[LogFloor](https://arxiv.org/abs/2608.14936v1) [24] 与本项目尤其接近：按监控技能到目标 floor 的缺口分配后续预算，测量所有技能达标的 token 成本，并用静态平均配比与逆序控制检验顺序作用。与 RegMix-D 的固定预算目标不同，它直接研究 first-passage。该文仍为预印本，主要受控技能实验不能证明 SQL 多标签错误组有效，但“关注未解决瓶颈”已不能作为新颖性主张。

[A Step Harder, Not a Step Too Far](https://arxiv.org/abs/2610.13669v1) [25] 在 GRPO 中发现适度增加难度可有益，继续加难却退化；这与 [24] 共同提示课程应匹配当前瓶颈，而非单调追求更难。其在线 rollout 和强化学习信号不同于冻结 SQL pair，不作为当前方案的直接基线，只作为近期边界证据。

近期文献中 [22]—[25] 均按 arXiv 预印本引用，不将投稿状态当作主会录用；[23] 主要 target 实验单种子且目标为单一验证域，不能把其平均结果视为普适结论。[24] 的 source/proxy 成本核算与顺序破坏对照，则是本项目可直接吸收的方法学。

## 9. 综合：可成立的研究定位是什么？

前述证据收敛到三个区别。第一，难度、可学习性和价值不是同一个量；第二，固定数据支持集、相同曝光多重集和相同有效目标不是同一约束；第三，更好错误诊断、早期收敛与更高最终 EX 不能互相替代。

因此建议将研究定位写为：

> 在冻结的 SQL 候选与偏好对训练池上，检验语义错误类别是否能改善有限预算下的学习分配，在控制 loss/margin、组平衡与技能缺口后，仍减少达到目标选择准确率的总训练成本，同时限制普通题回归。

以下内容不能单独作为创新：错误率高的组多采样、easy-to-hard、周期更新、uniform replay、把 difficulty 与 utility 相乘、用执行正误训练 reranker。可能有价值但尚未证实的差异，是多标签 SQL 错误的可靠刻画、其超出常规难度信号的增量、以及净修复与实际成本的联合验证。

错误类型只是分组依据，不是神经网络新的语义输入。当前 48 维统计特征如果无法表示问题—投影对齐或货币类型转换，仅调整样本顺序未必能弥补表示能力；若所有采样方法持平，也应考虑模型/标签限制，而非无限复杂化 scheduler。

## 10. 映射到当前项目：先修实验可识别性，再比较课程

### 10.1 当前数据能回答多大的问题？

[历史报告](D:/University/Alpha-SQL/HARD_CASE_PAIRWISE_100_REPORT.md) 中，269 个可评估问题划分为 129/40/100；hard-focused 实际训练只有 9 个问题、38 个 pair，all-covered 对照也只有 20 个同时含正负候选的问题、82 个 pair。主模型修复 4 个测试 hard case，但引入 5 个回归，overall EX 为 75%，SC 为 76%。这说明训练偏向反多数簇的风险值得研究，不证明课程学习有效。

固定 training set 不应退化成永久只用 9 个 hard questions。应在原 train IDs 内冻结一个所有实验组共用的数据定义：pairwise 部分使用所有已有正负 pair，pointwise/anchor 部分使用预先规定的已有候选；SC-correct 的题也必须保留。不能让 Random 用 9 题、课程用 20 题后声称顺序改进。无正候选的题不能形成正负 pair，也无法被当前 selector 修复 coverage，但其负标签可能用于共同的 pointwise 校准，不能一概称为毫无效用。

### 10.2 当前训练循环的具体障碍

核查 [runner 默认配置](D:/University/Alpha-SQL/alphasql/runner/hard_case_pairwise_experiment.py:600) 和 [训练循环](D:/University/Alpha-SQL/alphasql/algorithm/selection/neural_reranker.py:161)：

1. runner 默认 batch size 为 128，大于 38 和 82。因此每个 epoch 实际只有一个覆盖全部 pair 的 optimizer step。把相同 pair 在该 batch 内重排，在忽略数值求和误差时不改变 rank loss，无法有效检验 order-only。
2. 每一步 `scores = self.model(x)` 计算全部训练候选，pointwise BCE 也覆盖全部候选。若仅改变 rank pair，pointwise 路径依旧是共同全量监督。
3. `build_pair_indices` 已有 hard-negative 和 SC 支持相关权重。课程不能再无记录地叠加相同信号，否则会把旧 loss weighting 与新 sampling 混淆。
4. 当前 early stopping 和最佳 checkpoint 机制会使实际步数不同。固定预算实验应训练至共同预算，并单独记录 best checkpoint 与达标时间。

建议所有组共用真正多次更新的 mini-batch 实验流程，例如在该小池上统一使用较小 pair batch；具体大小由共同 pilot 决定并冻结。可以保留 full pointwise BCE 作为共同 anchor，也可以统一改为预先定义的 mini-batch BCE，但必须所有组一致并明确说明。该流程变更不是课程方法的贡献；旧 75%/76% 不能直接作为新流程的因果对照。

### 10.3 错误标注与监督边界

先以较粗类别起步：projection；filter/literal；join/schema；aggregation/DISTINCT/order/type。若支持样本足够，再细分 GROUP BY、date、ties/NULL、分母等。一条错误 SQL 可以多标签；当前 tiny pool 不宜直接切成十几个低支持组。

训练集标签可用 gold SQL 与执行结果离线辅助，但 AST 差异只能称弱标签，不能自动认作错误根因。执行正确但 AST 不同的候选不能被标成负例；单数据库执行等价也不能证明普适语义等价。低置信标签保留 unknown，不强行归类。人工复核优先覆盖训练各组；已有 287 题分歧案例可用于解释 taxonomy，但不得转成新测试集上的调度反馈或正面独立证据。

所有组共享同一份错误标签和置信度文件；主方法只用标签控制采样，不给 reranker 额外输入 gold、错误解释或正确答案。测试时仍是原有可见输入。统计报告须明确标注成本，不能声称“零额外监督成本”。

## 11. 最小方案、对照与效率协议

### 11.1 分开做两个协议

**协议 A：Order-only。** 预先固定每个 pair 的曝光额度，每次从剩余额度中安排下一批；所有方法在终点消费完全相同的多重集，不重复补 pair。随机打乱、难度顺序、错误交错都使用同一额度和 batch 规则。中途曲线的可用信息不同正是研究变量，但最终曝光次数严格一致。

**协议 B：Fixed-pool adaptive sampling。** 允许在相同训练池内改变曝光频率；固定总 updates、总 pair draw 和模型设置，完整记录每条样本实际次数。不再声称每条样本曝光一致。若需要相同有效目标，另做 importance-corrected 版本。把协议 B 产生的完整样本序列打乱再重放，是事后机制消融，不是可在线提前获得的免费基线，课程生成成本须保留。

固定 steps 不必然固定 token/FLOPs，尤其候选长度和 batch 组成可变。MLP pilot 可以固定 pair draw；未来换文本 encoder 时要采用同样 packing/truncation，并另做相同 token/时间预算比较，不能同时口头承诺多个实际无法严格匹配的预算。

### 11.2 一个低成本、可证伪的 scheduler

先用共同 uniform warmup 获得初始状态。每隔 K 步，在固定训练诊断面板上以 eval 模式计算各组当前 pair 排错率 \(e_g(t)\)、同面板平均 loss 的近期相对下降 \(\Delta_g(t)\)。分组估计先向全局均值收缩，避免只有一两条样本的类别支配分配。\(\Delta_g\) 是观察到的进度代理，不是某组训练的因果收益。

可先试一个只有两个核心信号的优先分数：

\[
z_g(t)=\beta\widetilde e_g(t)+\gamma\max(0,\widetilde\Delta_g(t)).
\]

令 \(a_{ig}\) 是训练 pair 的非负标签归属，\(\sum_g a_{ig}=1\)；unknown 独立成组。定义相对于同一基础分布 \(q_0\) 的倾向：

\[
r_i(t)=\sum_g a_{ig}\exp(z_g(t)),\qquad
p_t(i)=\rho q_0(i)+(1-\rho)
\frac{q_0(i)r_i(t)}{\sum_jq_0(j)r_j(t)}.
\]

\(\rho>0\) 保留基础采样支持；所有系数、K、warmup 和概率上限写入配置。按显式多标签倾向直接归一化，避免“一个 pair 在多个组被重复计数但未计入概率”的问题。固定训练目标的权重另行记录，不把已有 hard-negative 权重再次隐式乘入 scheduler。

\(q_0\) 必须明确定义。若每题等权、题内 pair 等权，则 \(q_0(i)=1/(Q\lvert P_q\rvert)\)，避免拥有很多候选的题主导优化；若为了保持旧实现采用 pair-uniform，也应所有组一致，并把 question-balanced 作为独立消融。两种基线不可混名为 uniform。

首版不同时加入可学习 prerequisite graph、梯度影响、噪声判别器和复杂窗口，以免现有小数据支撑不了参数量。单纯持续高 loss 不自动降权为噪声；只有独立的标签质量标记才能支持该处理。回放由 \(\rho\) 提供，是否需要最后 uniform consolidation 再单独消融。

### 11.3 对照组必须能逐项拆解贡献

| 组别 | 只改变什么 | 排除的替代解释 |
|---|---|---|
| U：Uniform | 共同流程的随机采样 | 基础性能与收敛速度 |
| RP：Random pacing | 随机排列 + 同样开放窗口 | 初期训练集规模效应 |
| D：Difficulty-only | loss/margin 排序 + 同样 pacing | 纯难度即可解释效果 |
| LP：Loss/progress-only | 去掉语义标签，按相同粒度的难度组反馈 | 动态反馈即可解释效果 |
| EB：Static error-balanced | 固定错误组配比 | 类别平衡而非动态课程 |
| E1：One-shot error weighting | warmup 后只更新一次 | 动态频繁更新没有必要 |
| EA：Dynamic error-aware | 当前错误组 + 进度代理 | 待检验主方法 |
| BF：Bottleneck / group-risk | 仅按组缺口或组 loss 分配 | Skill-it/LogFloor/group-DRO 风格即可解释效果 |

资源受限时先运行 U、D、EB、E1、EA 五组进行机制 pilot；若加入窗口，RP 立即成为必需对照。正式提出“超越难度、进度与瓶颈分配”的主张前必须补 LP、BF，不能只挑弱基线。BF 若用 Group DRO，应说明它优化最坏组目标，不伪称与普通 ERM 完全等价。

三个决定性消融：

1. **错误标签打乱**：尽量保持标签频率、多标签数和 difficulty 分层，在层内打乱语义标签，再运行完全相同 scheduler。真实标签若不优于打乱标签，没有错误语义增量证据。
2. **相同曝光序列乱序**：固定 EA 实际消费的多重集，随机打乱后从同一初始化训练。若无差异，收益更可能来自配比而非阶段顺序；这仍可有价值，但贡献应改称 adaptive reweighting。
3. **去掉 progress / 去掉 error**：使用同样 probe 频率和预算，避免 EA 只是多做了诊断。uniform 既报告自然部署成本，也可加 overhead-matched 对照，不人为把所有自然差异藏掉。

### 11.4 数据划分和调参边界

主方案优先使用 train 内固定诊断面板或共享训练轨迹，不在线消费 validation 的 gold 错误分布。若专用 monitor 集驱动 scheduler，必须提前从可训练数据中划出，所有组相同，并另保留用于超参数选择的 validation。在线用 monitor 标签是额外自适应监督，应公开说明；不能再把它当作独立泛化评估。

已有 100 题和 287 题已反复查看，继续用于 debug 可以，但不作为新方法未经接触的终局检验。正式结论需要未反复使用的 held-out 集或预先规定的嵌套验证。现有 split 不是 database-disjoint，只能声称相近分布下的候选选择；跨库主张另需 db-disjoint 设计。候选、SQL、执行标签、pair、error 标签及 split 都保存哈希，所有方法共享。

### 11.5 评价：净收益和总成本共同达标

至少采用五个配对 seed，所有组用相同初始 checkpoint、模型、标准化统计、optimizer、学习率、dropout、batch/pointwise 规则与共同预算。scheduler 的随机数与 dropout 等模型随机数分开管理。测试时固定同一 SC 融合/gate 策略；若各组都调 gate，则须相同搜索空间和 validation 预算，并另报共同 gate 与原始 ranking 的结果。

主效率指标是达到预先冻结 validation 目标、并连续若干评估点保持达标的总耗时。目标由独立 pilot 或预先冻结基线规则确定，不根据每组测试结果反复更改。未达标运行计为未达标/右删失并报告比例，不能删除后只平均成功 seed。

必须同时记录：

- 横轴为 updates、累计 pair/token、总 wall-clock 的 learning curves；同预算下的 EX 和 AULC。
- 达标时的 steps、处理量、GPU-hours，以及最终 EX 是否回落；MLP 下 GPU 时间和系统开销分别报告。
- 覆盖条件下的 selector accuracy、整体 EX、fix/regression、SC-correct retention；coverage 在冻结池下应不变。
- 每类错误的修复、样本数、遗忘和累计曝光；同一问题/多标签误差不当作独立样本。
- 错误标注、proxy、warmup、全量评分、monitor、调度和验证成本。区分一次性准备成本、单次运行成本与多次复用后的摊销成本。

固定 checkpoint 的逐题结果可用 question-level paired bootstrap / McNemar；多 seed 效率差异报告配对差和分布，必要时采用 seed 与 question 分层 bootstrap，不能把同一题多个 pair 当作独立重复。tiny pool 应强调区间和效应量，不以稀疏错误组的一个 p 值作结论。

历史 evaluation 使用的是本地无序多重集执行等价签名，并非完整 BIRD 官方 evaluator。研究论文应分别标注 local EX 与官方口径，报告执行失败、空结果、重复行与排序语义的处理。候选池冻结并不自动解决执行标签的偶然等价问题。

### 11.6 可证伪假设与停止条件

| 假设 | 支持它所需的证据 | 应降低主张的结果 |
|---|---|---|
| H1：错误语义有增量 | EA 优于 loss/progress-only 及标签打乱 | 打乱标签相当或更好 |
| H2：动态比一次诊断有用 | EA 优于 EB、E1，含探测开销 | E1 同样好且更便宜 |
| H3：顺序有独立作用 | EA 优于同曝光多重集的乱序重放 | 只有配比有效 |
| H4：有限预算真实加速 | 达标总耗时减少，最终效果不退化 | 仅 steps 减少而 wall-clock 增加 |
| H5：修复不靠牺牲普通题 | 净 fix 增加、回归/遗忘受控 | hard 修复被普通题回归抵消 |

pilot 若没有足够类别支持，合并组并只验证数据流；若 mini-batch 流程下 EA 仍不超过 EB/E1，不继续添加复杂模块找收益。若只有现有 MLP，可先做机制 sanity check，但不能通过人为放大网络或延时来制造加速空间；正式效率结果需在实际有训练成本、且所有方法共用的模型上复验。更换模型必须是另一阶段，不与 schedule 效果混合。

## 12. 结论与建议阅读顺序

**对 RQ1：** 错误信息可能有用，但语义错误标签优于 loss/margin 和组平衡尚无本项目证据。已有工作覆盖错误分布、技能反馈和效用，创新应落实到标签的增量验证，而非 error-aware 名称。

**对 RQ2：** 学习进度、未解决组缺口和 uniform 保留均有文献基础；频繁更新并不必然更好，必须比较一次重加权、静态平衡与动态版本。先实现低成本、可解释的两个信号，不预先堆叠复杂模块。

**对 RQ3：** 将 order-only、adaptive reweighting 和 importance-corrected sampling 分开；以达标总时间和固定预算性能为共同证据，同时报告失败达标、回归及遗忘。当前单 batch 训练循环不适合直接验证顺序作用，现有数据规模也不足以直接支持广泛效率结论。

建议优先精读：[5] When Do Curricula Work? → [9] Skill-it! → [11] DUCL → [18] Curri-DPO → [24] LogFloor；再读 [16] 重要性采样的预算/目标控制和 [12] CURE 的更新频率消融。[19][20] 作为 SQL verifier 任务基础，[21] 用于区分生成课程与选择课程。项目下一步应产出统一训练流程、冻结错误标签和最小五组 pilot 协议，而不是立即扩数据、重跑 MCTS 或宣布课程收益。

本报告借助 deep-research 的多视角检索与反证核验流程，重点补入了原计划没有充分覆盖的 random pacing、一次重加权、技能瓶颈和曝光匹配对照。它是一份研究工作材料，不是方法效果报告。

## 参考文献

[1] Yoshua Bengio, Jérôme Louradour, et al. “[Curriculum Learning](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf).” ICML, 2009.

[2] M. Pawan Kumar, Benjamin Packer, Daphne Koller. “[Self-Paced Learning for Latent Variable Models](https://papers.nips.cc/paper_files/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html).” NeurIPS / NIPS, 2010.

[3] Emmanouil Antonios Platanios, Otilia Stretcu, et al. “[Competence-based Curriculum Learning for Neural Machine Translation](https://aclanthology.org/N19-1119/).” NAACL-HLT, 2019.

[4] Yang Zhang, Amr Mohamed, et al. “[Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning](https://aclanthology.org/2026.eacl-long.271/).” EACL, 2026. 初稿 arXiv:2506.11300，2025；正文另核查 v2。

[5] Xiaoxia Wu, Ethan Dyer, Behnam Neyshabur. “[When Do Curricula Work?](https://research.google/pubs/when-do-curricula-work/).” ICLR, 2021.

[6] Savini Kommalage, Sanka Mohottala, et al. “[Confusion-Aware Transfer Teacher Curriculum Learning Framework: Disentangling Scoring and Pacing Effects](https://arxiv.org/abs/2606.17706v1).” arXiv:2606.17706v1, 2026；ICML GlobalSouthML Workshop，非主会。

[7] Alex Graves, Marc G. Bellemare, et al. “[Automated Curriculum Learning for Neural Networks](https://proceedings.mlr.press/v70/graves17a.html).” ICML, 2017.

[8] Sören Mindermann, Jan Brauner, et al. “[Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt](https://arxiv.org/abs/2206.07137).” ICML, 2022.

[9] Mayee F. Chen, Nicholas Roberts, et al. “[Skill-it! A Data-Driven Skills Framework for Understanding and Training Language Models](https://arxiv.org/abs/2307.14430).” NeurIPS, 2023.

[10] Mayee F. Chen, Michael Y. Hu, et al. “[Aioli: A Unified Optimization Framework for Language Model Data Mixing](https://arxiv.org/abs/2411.05735).” ICLR, 2025；初稿 2024。

[11] Zishang Jiang, Jinyi Han, et al. “[Difficulty Is Not Enough: Curriculum Learning for LLMs Fine-tuning Must Consider Utility](https://ojs.aaai.org/index.php/AAAI/article/view/40400).” AAAI, 2026.

[12] Pablo Messina, Andrés Villa, et al. “[CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](https://arxiv.org/abs/2601.15408v1).” arXiv:2601.15408v1, 2026.

[13] Sinchani Chakraborty, Sudeshna Sarkar, Pawan Goyal. “[Error-Aware Curriculum Learning for Biomedical Relation Classification](https://arxiv.org/abs/2507.14374v1).” arXiv:2507.14374v1, 2025.

[14] Mariya Toneva, Alessandro Sordoni, et al. “[An Empirical Study of Example Forgetting during Deep Neural Network Learning](https://arxiv.org/abs/1812.05159).” ICLR, 2019；初稿 2018。

[15] Swabha Swayamdipta, Roy Schwartz, et al. “[Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics](https://aclanthology.org/2020.emnlp-main.746/).” EMNLP, 2020.

[16] Angelos Katharopoulos, François Fleuret. “[Not All Samples Are Created Equal: Deep Learning with Importance Sampling](https://proceedings.mlr.press/v80/katharopoulos18a.html).” ICML, 2018.

[17] Shiori Sagawa, Pang Wei Koh, et al. “[Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization](https://arxiv.org/abs/1911.08731).” ICLR, 2020；初稿 2019。

[18] Pulkit Pattnaik, Rishabh Maheshwary, et al. “[Enhancing Alignment using Curriculum Learning & Ranked Preferences](https://aclanthology.org/2024.findings-emnlp.754/).” Findings of EMNLP, 2024. 预印本名含 Curry-DPO，正式论文方法写作 Curri-DPO。

[19] Mattia Tritto, Giuseppe Farano, et al. “[GradeSQL: Test-Time Inference with Outcome Reward Models for Text-to-SQL Generation from Large Language Models](https://arxiv.org/abs/2509.01308v2).” arXiv:2509.01308v2, 2025. 与 v1 题名不同，本文依据 v2。

[20] Mattia Tritto, Giuseppe Farano, et al. “[Test-Time Verification for Text-to-SQL via Outcome Reward Models](https://arxiv.org/abs/2606.30851).” arXiv:2606.30851, 2026；论文页注明 SURGeLLM Workshop at ACL 2026。

[21] Xiaohu Zhu, Qian Li, et al. “[Learning SQL Like a Human: Structure-Aware Curriculum Learning for Text-to-SQL Generation](https://aclanthology.org/2025.findings-emnlp.190/).” Findings of EMNLP, 2025.

[22] Emmy Liu, Kaiser Sun, et al. “[What do Language Models Learn and When? The Implicit Curriculum Hypothesis](https://arxiv.org/abs/2604.08510v1).” arXiv:2604.08510v1, 2026.

[23] Kaiyan Zhao, Zhongtao Miao, et al. “[RegMix-D: Dynamic Data Mixing via Proxy Training Trajectories](https://arxiv.org/abs/2606.18663v1).” arXiv:2606.18663v1, 2026. 论文页标注 work in progress。

[24] Seungmin Choi, Jiwon Sung, et al. “[Small Models Scout Bottleneck Order for Large-Model Data Control](https://arxiv.org/abs/2608.14936v1).” arXiv:2608.14936v1, 2026. 提交于 2026-08-14；投稿信息不等于录用。

[25] Ronghan Li, Fei Chen, Mingyu Lu. “[A Step Harder, Not a Step Too Far: The Benefits and Limits of Curriculum Difficulty Progression in GRPO](https://arxiv.org/abs/2610.13669v1).” arXiv:2610.13669v1, 2026. 提交于 2026-10-09。
