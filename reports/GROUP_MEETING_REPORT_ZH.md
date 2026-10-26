# Jev + Schema-aware SQL 选择实验：组会详细报告

> 2026-10-18 更新：独立运行同一 287 题后，旧 MCTS schema evidence 指标不能代表 LLM linker 本身。原始列 P/R：LLM 94.67%/87.12%，Jev 89.60%/92.32%；共同 PK/FK 补全后的完整召回为 86.41%/85.71%。固定 Qwen/MCTS + SC、仅替换表列 linker 后，EX 为 209/287 对 199/287（−3.48pp，95% CI [−7.32,+0.35]pp，p=0.1102）。旧实验保留为历史记录，最新结论及逐题分析见 [287 题受控替换报告](../LINKER_ABLATION_287_RESULTS.md)。

汇报人：Dongx Xia（50062682）  
日期：2026-10-13

## 1. 研究问题与项目演进

本项目最初关注的是 Text-to-SQL 中的 schema linking：如果模型能更准确地识别问题涉及的表、列和连接路径，是否可以提高最终 SQL 的执行准确率。随着实验推进，问题逐步收敛为两个相互独立的子问题：

1. **候选覆盖**：候选池中是否至少存在一条执行正确的 SQL；
2. **候选选择**：存在正确候选时，selector 能否把它排在第一位。

因此，实验没有把 Jev 当作文本生成模型，也没有替换原有 OpenAI/Qwen 调用，而是增加了一条独立路径：Jev 做结构化 schema/SQL 决策，Python 构建 QueryPlan 和 SQL，SQLite 做语法与执行验证，最后由 Jev 对候选 SQL 重排。与此同时，原 MCTS 候选池被保留，用来单独评估 selector。

当前需要回答的核心问题是：

- Jev 的高 schema-linking 召回是否足以支持替换原有 linker？
- Jev reranker 能否替换 execution self-consistency？
- 如果无监督重排不够可靠，监督式 reranker 应该如何构造训练数据与训练顺序？

## 2. 已完成的系统切片

当前 Jev 垂直切片包含四个阶段：

```text
question + evidence + full schema
              ↓
typed Jev schema linking
              ↓
schema-constrained QueryPlan beam
              ↓
SQLite SQL compilation + execution validation
              ↓
Jev schema-aware reranking
```

Jev linker 对候选表和列提出原子 Noul 问题，并保留概率与置信度；连接路径、聚合、排序、去重和比较符使用 Choice。下游只消费 typed result，不解析自由文本 JSON。SQL 编译器使用 schema graph 约束表、列和 join 的合法性，beam search 使用全量概率而不是单一最高选项。空结果被视为有效执行，仅语法错误、表列不存在和超时等才判为无效。

这条路径的意义不是立即替代开放式生成器，而是把“结构化判断”“SQL 组合能力”和“候选选择”拆开测量。

## 3. 实验路线

### 3.1 12 题探路实验

第一组实验使用 12 个冻结问题，SQL 范围被有意限制在当前 compiler 支持的结构内。MCTS 使用 `qwen3-coder-flash`，2 个 rollout，最大深度 5。

| 方法 | Candidate Coverage | Final EX |
|---|---:|---:|
| MCTS + self-consistency | 12/12 | 12/12 |
| Jev constrained pipeline | 8/12 | 8/12 |
| MCTS pool + Jev reranker | 12/12 | 11/12 |

Jev 自有候选池中，top beam 和 self-consistency 都是 5/12，Jev reranker 达到 8/12，等于该候选池的 oracle。这说明它在自己的小候选池中确实能修复 selection error。但是固定 MCTS 池的 self-consistency 已达到 oracle，Jev 的一次改选反而造成回归。因此，这组实验支持“Jev 有选择信号”，但不支持“直接替换 self-consistency”。

四个 Jev coverage miss 均已召回正确表列，错误来自 DISTINCT 和 literal/比较阈值构造。这是第一次观察到 schema linking 与 SQL coverage 之间并非一一对应。

### 3.2 300 题扩展实验

扩展实验冻结了 300 个 gold SQL 可在本地 SQLite 执行的 BIRD train 问题。Jev 完成了 300/300；低预算 MCTS 完成 287/300 后，Qwen/DashScope 因账户欠费停止，因此严格的三路 300 题对比尚未完成。当前共同集为 287 题。

| 方法 | Candidate Coverage | Final EX |
|---|---:|---:|
| MCTS + self-consistency | 229/287（79.79%） | 217/287（75.61%） |
| Jev constrained pipeline | 84/287（29.27%） | 73/287（25.44%） |
| MCTS pool + Jev reranker | 229/287（79.79%） | 208/287（72.47%） |

在固定 MCTS 候选池上，两种 selector 的配对结果是：204 题都正确，Jev 修复 4 题，Jev 引入 13 题回归，66 题都错误。净变化为 -9/287，即 -3.14 个百分点；paired bootstrap 95% CI 为 [-5.92, -0.35] 个百分点，17 个分歧题上的 exact McNemar/binomial 检验为 `p=0.049`。在当前共同集上，“直接用无监督 Jev 替换 self-consistency 会下降”已经不只是小样本波动。

### 3.3 受约束能力范围分解

原 12 题属于当前 compiler 的目标范围。按同一定义划分 300 题，有 83 题在范围内、217 题在范围外：

| Gold SQL 范围 | 数量 | Jev Coverage | Jev EX |
|---|---:|---:|---:|
| 范围内 | 83 | 64/83（77.11%） | 62/83（74.70%） |
| 范围外 | 217 | 26/217（11.98%） | 16/217（7.37%） |

因此，Jev 总体 EX 只有 26% 并不表示范围内能力退化。相反，范围内 EX 从 12 题实验的 66.7% 上升到 74.7%。总体下降主要是 217 道题包含多投影、多过滤、子查询、GROUP BY/HAVING、ORDER BY/LIMIT、CASE、集合操作和复杂日期/字符串表达式，而当前 QueryPlan 没有完整建模。

### 3.4 Schema linking 诊断

287 题共同集上的平均结果如下：

| 指标 | MCTS 路径 schema evidence | Jev linker |
|---|---:|---:|
| Table precision | 54.24% | 63.35% |
| Table recall | 53.98% | 99.30% |
| Column precision | 47.54% | 60.16% |
| Column recall | 52.86% | 97.35% |

MCTS 的数字受部分路径跳过 schema-selection action 影响，不能视为严格的 linker 对照；但 Jev 的高召回是明确的。更关键的证据是：Jev 的 203 个 coverage miss 中，181 题已经完整召回 gold 表列，只有 22 题存在 schema 不完整。

所以目前不能根据高 linking 准确率直接替换全部原 linker，也没有必要把主要资源继续投入一个新的表列二分类器。更合理的定位是：

- Jev 作为高召回、带置信度的 schema signal；
- 不做硬过滤，避免漏召回时压掉正确 SQL；
- 优先扩展 QueryPlan、literal/operator、聚合层级和 SQL compiler；
- 在同一候选池上训练 benchmark-aligned selector，再讨论替换。

## 4. 无监督 reranker 的错误模式

Jev-on-MCTS 修复了 4 题，代表性信号包括：

- `COUNT(*)` 与 `SUM(qty)` 的计量对象；
- `>=` 与严格 `>`；
- 带 `US$` 和逗号的工资字符串数值转换；
- 日期上界的 benchmark 约定。

但 13 个回归集中在更稳定的模式：

- 5 题加入了自然语言上合理、但 gold 不保留的 DISTINCT；
- 投影从 ID 改成更自然的姓名，或把多列拼成单列；
- 添加额外语义约束；
- 改变 GROUP BY、计数对象或百分比分母；
- 从语义近似的另一张表读取字段。

这说明通用语义判断与 benchmark execution equivalence 并不总是一致。BIRD 的投影、重复行和计数口径需要通过监督数据学习，不能只依赖“哪条 SQL 看起来更自然”。

## 5. 100 题监督式 pairwise reranker

为了绕开完整 MCTS 成本，当前监督实验使用缓存的 `direct_schema_pool_300`。300 个名义问题中有 269 个可评估，共 903 条候选；candidate oracle 覆盖 212 题，满足 `oracle correct && self-consistency wrong` 的 hard case 只有 17 题。冻结划分为 129/40/100，其中训练 hard case 只有 9 题。

模型是两层 MLP（64/32），使用 RankNet pairwise loss 和 0.2 权重的 pointwise loss，5 个随机种子在 GPU 上训练。48 个输入特征中保留 7 个 self-consistency 特征。最终采用 validation 选择的 confidence gate：只有 neural winner 比 self-consistency winner 高至少 0.20 才改选。

| Selector | Test EX | Oracle EX | Gap Closure |
|---|---:|---:|---:|
| Self-consistency | 76% | 81% | 0% |
| Hard-focused pairwise + SC | 75% | 81% | -20% |
| Hard-focused without SC | 76% | 81% | 0% |
| All pairwise-eligible + SC | 76% | 81% | 0% |

主模型在 5 个 hard test cases 上修复 4 个，但在普通题上制造 5 个回归，最终净变化 -1。五个回归全部从较大的 self-consistency 结果簇跳到了 singleton。这个结果证明 hard-case supervision 确实有信号，同时也暴露了数据太小和训练分布过度偏向“反多数簇”的问题。

当前 selector 的第一瓶颈仍然是候选覆盖：100 题 oracle 只有 81%，19 个 coverage miss 无法通过任何重排修复。第二瓶颈才是 selection：5 个可修复 gap 中模型修复了 4 个，但安全边界不够好。

## 6. 五篇论文的主要内容与启发

### 6.1 Bengio et al. (2009), Curriculum Learning

这篇工作把 curriculum learning 形式化为一系列逐步变化的训练分布：开始时更重视容易、干净的样本，随后逐渐增加困难样本的权重，最终回到目标分布。作者把它解释为 continuation method：先优化平滑、容易的目标，再逐步过渡到原目标，从而加速收敛、进入更好的参数区域，并产生类似正则化的效果。

对本项目的直接启发是：pairwise reranker 不应一开始就只看“正确 SQL 与极难 hard negative”。先学习明显的结构差异，再过渡到 DISTINCT、投影、分母和类型转换等微小差异，更符合优化过程。

### 6.2 Zhang et al. (2026), Beyond Random Sampling

该文系统研究 LLM 预训练中的课程学习，训练超过 200 个模型，比较严格排序、pacing 和 interleaved curricula，并考察六类难度指标。主要结论是：课程学习对早中期收敛最稳定，可减少 18% 到 45% 的训练步骤；把课程作为 warmup，随后恢复随机采样，最终性能最多提高 3.5%。Compression Ratio、MTLD 和 Flesch Reading Ease 较有效，而 perplexity 容易把噪声当成难例，后期可能伤害训练。

对本项目的启发有两点。第一，难度指标必须与任务错误类型相关，不能仅用模型 loss 或 SC margin。第二，课程适合作为 warmup，而不是永久严格排序；后期应恢复混合采样，防止模型只学会反多数簇。

### 6.3 Jiang et al. (2026), Difficulty Is Not Enough

该文认为 fine-tuning 不能只按 difficulty 排序，因为容易样本可能已经被模型掌握，困难样本也可能只是噪声。DUCL 同时建模 difficulty 和 utility，用 Wasserstein distance 的比值构造 DUE 分数，再用逐渐扩大的 Window Ordering 在保持难度递增的同时保留 batch diversity。实验中，DUCL 比 PPL 和随机训练更快收敛，CPT 达到相同 loss 所需步骤减少约 25% 到 33%，且额外计算开销约为训练预算的 1%。

对本项目最重要的启发是：hard case 不是天然高价值。训练样本应该同时考虑：它有多难，以及学会它能否真正关闭 selector gap。`oracle correct && SC wrong` 具有高 utility；候选池本身没有正确 SQL 的题，对 reranker 的直接 utility 接近零。Window Ordering 则可避免当前 9 个 hard training questions 造成的分布坍缩。

### 6.4 Pattnaik et al. (2024), Curri-DPO

Curri-DPO 为同一 prompt 采样多个有质量排序的响应，构造多个 preference pairs，并按 chosen/rejected 评分差从大到小安排，即从容易区分到难以区分。每一轮使用前一轮模型作为新的 reference model。结果显示，迭代课程优于把所有 pairs 混在一起训练，Zephyr-7B 在 MT-Bench 达到 7.43；但使用所有可能 pair 继续训练会出现边际收益下降甚至性能回落。

这与当前 SQL selector 非常接近：同一问题有多个候选 SQL，自然形成多个 pair。pair 难度可由正确候选与负候选的结构差、执行结果簇差和模型分数差定义。论文也提醒我们，不应穷举所有 pair；应选择能提供清晰、互补信号的 pair，并逐阶段更新模型。

### 6.5 Tritto et al. (2025), GradeSQL

GradeSQL 是与本项目最直接相关的工作。它生成 N 条 SQL 候选，用与 gold SQL 的执行结果是否一致来打 Yes/No 标签，再以 LoRA 微调 Outcome Reward Model。推理时使用 `P(Yes)` 作为连续分数，对候选做 Best-of-N 排序。

在 BIRD dev、N=32、OmniSQL-7B 设置下，ORM-based BoN 为 68.90%，高于 Majority Voting 的 66.95% 和 execution-based BoN 的 66.04%；Pass@32 为 80.57%，仍有明显 selector gap。论文还发现：

- ORM 在困难题上的相对收益更大；
- SQL-only verifier prompt 优于更冗长的 instruction/data 输入；
- 7B verifier 已足够，继续增大模型收益有限；
- autoregressive Yes/No 训练优于 BCE classification head；
- balanced 小数据可达到与更大不平衡数据接近的效果。

这验证了“训练 SQL selector”本身是合理方向，也说明当前小 MLP 只是阶段性 sanity check。下一阶段先不扩大训练集，也不更换模型规模，而是在完全相同的样本、初始化与训练预算下，研究课程调度能否让 selector 更快、更稳地学会这些执行监督信号。

## 7. 论文与当前实验的统一认识

五篇论文可以归纳成一条连续逻辑：

1. **Bengio**：样本顺序会改变优化路径；
2. **Beyond Random Sampling**：难度指标和调度方式必须匹配，warmup 比永久严格排序更稳；
3. **DUCL**：难度不够，还要考虑样本的训练效用与多样性；
4. **Curri-DPO**：同一输入的多个偏好 pair 应从大间隔到小间隔逐步学习；
5. **GradeSQL**：执行标签可以训练专门的 SQL outcome verifier，优于简单投票。

映射到当前项目，下一步不再把“扩大候选池”作为主要实验变量，而是冻结现有训练集，单独检验训练顺序是否有效：

```text
同一份候选 SQL 与执行标签作为固定 training set
同一模型、初始化、优化器、batch size 与总更新预算
错误类型、当前损失与学习进展共同估计样本价值
课程调度只改变样本出现的时刻与频率
最终比较收敛速度、固定预算性能和错误修复稳定性
```

## 8. 下一阶段方案：固定训练集下的错误感知课程学习

### 8.1 研究问题与公平性约束

核心问题是：**在给定完全相同 training set 的前提下，能否通过错误感知的样本顺序和采样概率，用更少更新步数或更少 GPU 时间达到随机训练的最终性能，并在相同预算下获得更好的泛化？**

主实验冻结以下条件：

- 相同的 train/validation/test 划分及候选 pair；
- 相同的模型结构、参数初始化、优化器、学习率、batch size 与随机种子集合；
- 相同的总 optimizer steps、token 数和样本曝光次数；
- 相同的 validation 频率、early-stopping 规则与最终选择标准；
- 每个方法最终都能访问 training set 中的全部样本，唯一主要变量是样本顺序与采样权重。

因此，若课程方法在较少步骤达到基线分数，提升可以归因于训练效率；若在相同步数获得更高 EX，提升可以归因于更好的优化路径，而不是额外数据或计算。

### 8.2 错误感知的样本表示

每个训练问题或 pair 记录三类量：

1. **错误类型 `E_i`**：projection、filter、join、aggregation/GROUP BY、DISTINCT、ORDER/LIMIT、literal/type/date，以及 benchmark projection/duplicate semantics。一个样本可以有多个标签。
2. **难度 `D_i(t)`**：当前 pairwise loss、正确与错误 SQL 的 score margin、AST 相似度、执行结果簇支持差，以及模型不确定性。它回答“当前模型有多难分辨”。
3. **效用 `U_i(t)`**：该样本近期 loss 下降、对应错误类型在 validation 上仍未解决的频率、梯度与验证目标的一致性，以及该 pair 是否能修复已有 selector error。它回答“现在学习它是否值得”。

同时定义噪声/不稳定性惩罚 `N_i(t)`：持续高损失但多轮无学习进展、执行标签不稳定或候选间存在潜在等价歧义的样本，不应因为“难”而被反复过采样。

候选优先级可写为：

```text
priority_i(t) = window_t(D_i) · exp(β U_i(t) - γ N_i(t))
```

其中 `window_t` 是随训练扩大、按难度开放的窗口；`U_i(t)` 和 `N_i(t)` 每个 epoch 或每 K 步更新一次。全量 probabilities 用于采样，而不是把样本永久分成容易/困难两个桶。

### 8.3 训练调度

课程分为四段，但阶段之间连续过渡：

1. **诊断 warmup**：用随机顺序训练很少的步数，获得每个样本的初始损失、margin 和错误类型统计；
2. **低噪声结构课程**：先开放模型能稳定学习的 projection、filter、aggregation 等差异明显的 pair；
3. **动态错误感知课程**：逐步扩大难度窗口，优先采样“validation 仍常错且近期有学习进展”的错误类型；每个 batch 强制保留跨数据库、跨错误类型和 SC 正确的 anchor 样本；
4. **随机混合巩固**：最后 10%–20% 预算恢复近似均匀采样，减少对课程顺序的过拟合与灾难性遗忘。

每次重评分只使用训练过程可见的信息；不能用 test 标签定义 utility。课程方法也不能额外增加样本曝光次数，否则训练效率比较失去公平性。

### 8.4 对照组与消融

在同一训练集与预算下比较：

- **Random Shuffle**：标准随机训练；
- **Difficulty-only**：仅按初始 loss、margin 或 AST 距离从易到难；
- **Static Error-aware**：训练开始前按错误类型与初始难度固定排序；
- **Self-paced / Competence**：随步数扩大可用样本比例；
- **DUCL-style**：difficulty + utility + window ordering；
- **Dynamic Error-aware**：本文方案，周期性更新错误效用、噪声惩罚和采样概率；
- **Ablations**：去掉错误类型、去掉动态更新、去掉窗口、去掉 anchor replay、去掉最终随机巩固。

主实验先保持当前 reranker 不变，避免“更强模型”与“更好课程”混在一起；只有课程效果成立后，再迁移到更强的 SQL-aware encoder/ORM。

## 9. 可检验假设

- **H1：训练效率。**动态错误感知课程能以更少 optimizer steps、wall-clock time 或 GPU-hours 达到 Random Shuffle 的最终 validation EX，并获得更大的 learning-curve area。
- **H2：效用优于纯难度。**difficulty + error utility 优于 difficulty-only，因为高 loss 既可能代表可学习的 hard case，也可能代表噪声、标签歧义或模型暂时无法利用的样本。
- **H3：动态优于静态。**周期性更新 `U_i(t)` 优于一次性固定排序，因为一个样本的价值会随着模型能力和已解决错误类型变化。
- **H4：窗口与回放提高稳定性。**expanding window、跨错误类型 interleaving 和 anchor replay 能降低 loss spike、普通题 regression 与已修复错误再次出现的比例。
- **H5：收益具有错误类型结构。**课程增益应首先体现在频繁、尚未解决且具有稳定执行标签的错误类型；对潜在 SQL 等价、benchmark projection 或噪声样本的收益较小。

## 10. 评估设计与成功标准

### 10.1 训练效率指标

- 达到 Random Shuffle 最终 validation score 所需的 steps、tokens、wall-clock time 和 GPU-hours；
- 固定预算下的 validation/test EX 与 selector accuracy given coverage；
- learning curve 面积（AULC），同时反映早期与中期学习效率；
- 每 1,000 次更新或每 GPU-hour 的准确率增益；
- 多随机种子的均值、标准差与收敛方差。

### 10.2 错误感知指标

- 各错误类型的错误率下降曲线和首次稳定解决所需步数；
- fix / regression / unresolved selection error；
- 已修复样本的遗忘率与错误类型迁移；
- 课程采样分布、样本曝光次数与有效 batch 多样性；
- 高损失样本中可学习 hard case 与疑似噪声的比例。

### 10.3 公平统计与成功标准

至少运行 5 个相同种子，并使用 paired bootstrap、McNemar test 以及 seed-level mean ± std。训练曲线应同时按 optimizer steps 和真实 GPU 时间绘制，避免课程重评分开销被忽略。

阶段性成功标准满足以下任一项，同时不能显著降低最终 test EX：

1. 用至少 **20% 更少的训练步数或 GPU 时间**达到随机基线的最终 validation 分数；或
2. 在完全相同更新预算下，取得稳定、可重复的 selector accuracy / execution accuracy 提升；并且
3. 至少一个主要错误类型的修复速度显著提高，普通样本 regression 与遗忘率没有明显恶化。

主结果必须明确写成：**固定数据、固定模型、固定预算，只有 curriculum 不同。**

## 11. 当前结论

1. **Jev schema linking 值得保留，但暂不应整体替换原有路径。**它的优势是高召回和结构化置信度；当前端到端瓶颈主要不在表列识别。
2. **Jev constrained pipeline 暂时不是通用 Text-to-SQL 生成器。**范围内表现约 74.7%，范围外仅 7.37%，下一步首先是 compiler/plan 能力扩展。
3. **无监督 Jev reranker 不应直接替换 self-consistency。**287 题共同集下降 3.14 个百分点，主要问题是 benchmark alignment。
4. **监督式 pairwise 方向合理，但当前训练数据远远不够。**9 个 hard training questions 能修复 4/5 hard test cases，也会制造 5 个普通题回归。
5. **下一步的主问题改为固定数据下的训练效率。**在完全相同 training set、模型与预算下，比较随机训练、纯难度课程和动态错误感知课程，检验是否能更快达到目标性能、提高固定预算结果，并降低回归与遗忘。

