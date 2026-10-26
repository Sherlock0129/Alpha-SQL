# SQL 选择器错误感知课程：冻结数据、保持回放与联合达标实验协议

日期：2026-10-20。状态：实验设计，尚未实现新训练器或启动训练。配套参数表为 [protocol YAML](D:/University/Alpha-SQL/config/sql_selector_curriculum_protocol.yaml)，它不是现有 runner 可直接执行的配置。

## 1. 要回答的假设

在固定训练问题、SQL 候选、执行标签、模型和训练目标下，SQL 错误分组是否能在 loss/margin/学习进度和保持性回放之外提供增量，使选择器在相同质量及回归约束下更快达标？

预先分解为四个假设：

| 假设 | 主对比 | 支持条件 | 不支持时的解释 |
|---|---|---|---|
| H1：错误分组有额外信息 | EA+R vs LP+R；EA+R vs 打乱标签的 EA+R | 共同目标下净时间更短且质量/回归通过检验 | 训练动态或分组平滑已足够 |
| H2：反馈更新有额外价值 | EA+R vs OneShot+R | 扣除新增 probe 后仍更快 | 一次性配比更划算 |
| H3：收益不只来自回放 | EA+R vs U+R；以及 EA/U 的有无 R 析因比较 | 加上相同 R 后仍有 EA 增益 | 只支持保持性回放，不支持语义调度 |
| H4：真正加速且没有用回归换速度 | EA+R vs U/H/S/LP 的相同 R 版本 | 总时间、达标率、终点质量与回归共同通过 | 仅 steps/样本效率或有代价的权衡 |

EA=动态错误感知；LP=纯训练动态；R=共同保持回放；U=随机；H=难例；S=静态错误均衡。本协议检验的是方法假设，不预设其有效。

## 2. 当前数据适合什么结论

历史 [100 题实验](D:/University/Alpha-SQL/HARD_CASE_PAIRWISE_100_REPORT.md) 的 269 个可评估问题按 129/40/100 划分；hard-only 实际训练 9 题、38 个 pair，all-pairwise 训练也只有 20 题、82 个 pair。该数据支持流程与小样本机制 pilot，不支持直接宣布稀疏错误组泛化或广泛训练加速。

分两个独立阶段，禁止混报：

| 阶段 | 数据使用 | 可声称内容 |
|---|---|---|
| P0：现有池 pilot | 保持原 train 129 IDs / validation 40 IDs；使用全部 pairwise-eligible train 问题而非 hard-only；历史 100 题不作新 test | 代码正确性、成本分解、方向与方差 |
| P1：确认性实验 | 在开始比较前冻结所有方法共同的 train/dev/audit/test；使用未被反复分析的外部检验集 | 对冻结设置的正式训练效率及回归结论 |

P1 若采用另一个较大训练池，是另一个预先冻结的实验条件，不把其结果与 P0 直接作课程因果比较；任何条件内部，各组训练池完全一致且不在线增加样本。若用户要求训练 IDs 连同历史池都不变，则 P1 继续固定原 129 IDs，只增加独立评估支持，并接受统计能力可能不足。当前没有选定新的 P1 测试 manifest，正式实验保持阻断状态，不能把旧 100/287 题重新称为完全未见测试。

### 2.1 冻结产物

保存文件、版本及 SHA256：question/evidence/schema、数据库文件、candidate IDs/SQL/重复票数、执行结果签名、正误标签、pair IDs、原 linker 输出、特征名与矩阵、错误标签、split 和固定 SC 预测。比较中不重采 SQL、不补 gold candidate、不合成 hard negative、不发新 Jev/LLM 请求。

默认仍采用本地无序 multiset、保留重复行、浮点规范化的既有执行口径；空结果有效，执行错误/超时无效。它不是完整官方 BIRD evaluator。若另报官方 set 口径，作为独立评估轨道冻结，不在训练中混换标签。排序/列顺序处理也写入 manifest。

所有特征只来自问题、实际 schema、原 linker、候选和执行反馈；gold SQL/正误/error labels 不进入模型特征。gold 可用于训练侧标签与错误类别标注，测试 gold 只在最终评估器内可见。候选重复保留其 SC 票数，去重策略和 tie-break 不随组改变。

## 3. 训练器先统一，再比较 sampler

现有 [训练循环](D:/University/Alpha-SQL/alphasql/algorithm/selection/neural_reranker.py:161) 每个 pair batch 都前向全体候选并计算全量 BCE；[专用 runner](D:/University/Alpha-SQL/alphasql/runner/hard_case_pairwise_experiment.py:600) 默认 batch 128。对 82 个 pair 而言，单 batch 内重排通常不能有效检验课程。原训练还包含 hard-negative 权重，并在训练后按各自验证结果挑 fusion/gate。直接复用会混入多种干预。

新实验应新增独立训练入口，不覆盖历史训练器与结果。所有组共同采用：

| 项目 | P0 初始设计值；不是已调优结果 |
|---|---|
| 模型 | 同一 SchemaAwareMLP，hidden 64/32、dropout 0.10；特征名单冻结，保留 SC 特征 |
| 优化器 | AdamW，lr=0.001、weight decay=0.0001、gradient clip=5 |
| 主损失 | RankNet：mean softplus(-(f_pos-f_neg))；另加 0.20×共同 pointwise BCE |
| 原 pair hardness 权重 | 主实验统一设为 1，隔离 sampling；旧权重仅作另一个所有组共同的稳健性条件 |
| Pair mini-batch | 8 个 pair；允许同池有放回采样，记录重复；所有组同 batch size |
| Pointwise anchor | 每步 16 条候选，先均匀抽训练 question、再均匀抽其 candidate；相同 seed 下各组使用完全相同 anchor 序列 |
| BCE 类别权重 | 只由 train labels 计算一次后冻结；不能随当前采样频率变化 |
| 预算 | 每次运行 250 optimizer steps；warmup 前 25 步；每 10 步 probe/checkpoint |
| pilot seeds | 101、202、303；正式比较另用 7、21、42、84、168 |
| 初始状态 | 每个配对 seed 共用初始权重、scaler、优化器状态、dropout seed 和 anchor 序列 |
| 停止 | 不按 validation loss 提前停止；为所有组跑满共同预算并保存曲线，达标时间从曲线按冻结规则读取 |

每步仅前向 batch pair 与 anchor 涉及的唯一候选，再分别计算 pair 和 BCE，禁止仍前向全训练池却宣称省计算。归一化统计只拟合 train。外部计时尽可能隔离其他作业；CPU/GPU 比较作为共同基础工程的 profile，不让某 sampler 换设备。

P1 可通过共同基础 pilot 冻结另一组 batch/总步数，但必须在任何 P1 arm 比较前锁定；不根据哪组占优单独增大预算。建议 P1 起始预算 2000 steps、pair batch 32，仅是待 profile 的配置，尚非正式运行承诺。

### 3.1 Pair 和基础分布

所有训练组使用相同的、同一问题内的正负候选笛卡尔积。设 Q 为有正负 pair 的训练题数，P_q 为其 pair 集，基础抽样概率为 q0(i)=1/(Q×|P_q|)：先题目均匀、题内 pair 均匀，避免候选多的问题支配训练。

没有正确候选的问题不能生成正负 pair；只有正确候选的问题也不凭空造负例。二者保留在所有组共同的 pointwise anchor 中。P0 不删除普通 SC-correct 问题去只训练 hard cases。

主实验明确为固定池自适应重加权，不宣称优化同一个均匀 ERM 的无偏梯度。若要研究同目标 importance-corrected sampling，另设共同目标、显式 q0/p 权重的独立条件，不能混入主表。

## 4. 错误标签：只做可审计的训练调度信息

第一版按已有正候选与负候选的差异构建 pair 标签，不必新增 SQL：

| 组 | 差异范围 |
|---|---|
| Projection / multiplicity | 返回列、返回形状、DISTINCT 和计数对象 |
| Filter / literal | 遗漏或额外过滤、比较符、数值与日期边界 |
| Relation | 表、连接条件、连接方式 |
| Computation / ordering | 聚合、分组、HAVING、排序、LIMIT、类型转换和日期计算 |
| Unknown / mixed | 无法可靠判断或多个交互差异 |

SQLGlot scope-aware AST 差异只生成 weak 标签；不同 AST 可能语义等价，一个负候选也可能有多种错误。人工审计按错误组分层，P0 每组最多抽 10 个 pair、至少覆盖不同 question；报告实际数而非强行凑数。pair 同时属于多个组时其 membership 向量归一化，避免重复计票。分组不可靠时进入 unknown，不把高 loss 自动标成噪声。

正式语义结论要求每个保留组有至少 10 个不同 train question 的支持，这只是防止极端稀疏的设计门槛，不代表统计充分。P0 达不到时按预先顺序合并 projection+filter、relation+computation，再不足则只做数据流/分组 pilot。记录 pair 数和独立题数；不能用重复 pair 替代更多独立例子。只审计 train，禁止看 test 错误来重定义训练分类。

另在训练侧统计同特征向量异标签冲突，以及主要错误组能否被现有表示区分。若正负候选输入完全相同，课程无法凭空补充表示能力；这应记录为当前模型的边界，而不是立即给 EA 单独加语义 encoder。

## 5. 主实验：五种采样策略 × 有无共同保持回放

| ID | 训练信号 | R0：不加专项回放 | R1：相同专项回放 |
|---|---|---|---|
| U | q0 随机 | U0 | U1 |
| H | 当前 pair loss 优先，无进度/语义 | H0 | H1 |
| S | 静态错误组均衡，权重从 train 标签一次计算 | S0 | S1 |
| LP | 当前 loss / margin 与学习进度，无错误标签 | LP0 | LP1 |
| EA | 与 LP 相同的单 pair 动态，再加错误组反馈 | EA0 | EA1 |

10 组构成主析因表。主比较是 EA1 vs U1/H1/S1/LP1，不是 EA1 vs U0。对 U/LP/EA 比较 R1-R0，并报告交互差异，区分课程贡献与 replay 贡献。R1 不是更大 batch：固定总 pair draw，回放槽替代而非追加训练槽。

### 5.1 共同动态观测

每 K=10 steps、eval 模式对 train 的固定 pair 面板计算 raw logit margin、RankNet loss 和排反事件，题目等权聚合。P0 面板可为全体 train pairs；P1 面板只从 train 确定性抽取并冻结，所有组相同。EMA=0.9。进度由同面板相邻轮 loss 的相对下降给出，截到 [0,1]；这是观测代理，不称因果 utility。各方法的 train 观测开销真实计入。

为隔离观测量与调度信号，主机制比较采用 probe-matched：U/H/S/LP/EA 都在同一步评估共同面板，baseline 不使用不需要的字段。另报 production-cost 对照：不需要 probe 的 U/S 直接关闭其额外诊断，其余配置不变；只有 EA1 仍能胜过这些更便宜版本，才称现实净加速。

### 5.2 可直接实现的首版公式

对 pair i，记其 EMA loss 为 L_i，令 h_i=L_i/(1+L_i)，u_i=clip((L_i_prev-L_i)/max(L_i_prev,1e-8),0,1)，定义 d_i=h_i+u_i；首次 probe 的 u_i=0。H 只用 h_i，LP 用 d_i。当前 loss 对二元 RankNet margin 单调，因此不把 loss 和 margin 当两个独立创新信号；margin 同时用于误排与遗忘诊断。

对语义组 g，计算排反率 e_g 与平均进度 u_g。用固定强度 κ=10 个 question-equivalent 的全局均值收缩，避免一个小组决定采样；记 d_g=e_tilde_g+u_tilde_g。EA 的权重为：

\[
w_i^{LP}=\exp(d_i),\qquad
w_i^{EA}=\exp\left(d_i+\eta\sum_g a_{ig}d_g\right),\quad\eta=1.
\]

H 用 exp(h_i)，U 用 1。S 令 c_g=Σ_i q0(i)a_ig，w_i=Σ_g a_ig/c_g，忽略空组，然后归一化。所有权重共同截到 [0.1,10]，基础混合保留 ρ=0.25：

\[
p_{base}(i)=\rho q_0(i)+(1-\rho)\frac{q_0(i)w_i}{\sum_jq_0(j)w_j}.
\]

这些是为了先验证机制的默认值，不是最优或已有论文结论。P0 可发现数值错误，但不能按 test 逐项调参；若需调整，保存变更后重新冻结所有组。

### 5.3 保持回放严格共同

冻结 B=原 self-consistency 选择器及其 train 预测。H_keep 为 B 正确且可形成 pair 的训练题。q_keep 先在这些题间均匀、再在题内 pair 均匀。

R0 的 λ=0；R1 的 λ=0.25，保持分布与主 sampler 混合：

\[
p(i)=(1-\lambda)p_{base}(i)+\lambda q_{keep}(i).
\]

每 batch 8 个 pair 时 R1 固定 2 个 keep 槽和 6 个 base 槽，保持序列在同 seed 下各组一致。若 keep 为空则不偷偷换定义：记录 R 不可检验，禁止宣称保护效果。只有正候选的 B-correct 题由共同 anchor 处理。

主实验不动态调整 λ，不加入 distillation、GEM 梯度投影或教师，以免把不同保持机制混入课程。如果后续研究回归反馈控制 λ，应作为扩展实验，并给 LP 同样的控制器。

## 6. 必需消融与分阶段运行量

| 消融 | 设置 | 要回答的问题 |
|---|---|---|
| OneShot1 | warmup 后冻结 EA 的组项 d_g；单 pair 动态仍与 LP 一样更新 | 语义组持续更新是否必要 |
| ShuffledEA1 | 相同控制器，打乱归一化 membership 向量 | 是否确实需要语义分组 |
| NoGroupEA1 | η=0，必须与 LP1 在相同随机流下等价 | 实现及归因 sanity check |
| ExposureShuffle | 保存 EA1 完整曝光 multiset，乱序后从同初始状态重放 | 频率还是阶段顺序有效 |
| Group-matched controller | 复杂度/初始 loss 分组，组数及支持尽量匹配，使用同一组控制器 | 是否任意分组平滑都有效 |

标签打乱在 warmup loss/margin 与 B-correct 状态的预定分层内进行，交换完整 membership 向量以保留组频次及共现；记录加权组支持变化。小样本分层无法交换时按固定顺序合并 strata，并承认 P0 的标签消融证据不足。P1 预先冻结 3 个 permutation seeds，不能挑最差随机标签结果。整个随机标签对照与 EA 使用相同标签元信息访问范围。

ExposureShuffle 是事后机制分析：训练时不使用 EA 的反馈调整重放，不具有在线可得性；报告生成 EA 轨迹的成本。EA 的真实顺序优于该乱序重放，才支持独立的 order effect；乱序一样好只支持频率/配比解释。

运行顺序：

1. P0a：U1、LP1、EA1、NoGroupEA1，各 1 seed，检查数据/概率/成本，不作性能结论。
2. P0b：主 10 组 + OneShot1 + ShuffledEA1，各 3 pilot seeds，共 36 个训练 run；profile 显示不划算或组支持不足时停止扩大。NoGroup 的等价性通过单元测试，不必重复计为额外性能 arm。
3. P1：在新冻结协议及未反复接触的检验数据上，主 10 组、3 个标签排列版本和 OneShot1，各 5 配对 seed；算量依实际 label arms 显式展开，再加组控制与重放机制消融。不得说是 5-seed ensemble：每 seed 是独立模型，分别报告成本。

主比较若加入 Group-matched 后才完整，可先做探索，但不能先发表“语义增量”结论再补控制。

## 7. 推理规则冻结，防止 gate 冒充课程收益

主轨道：raw neural score 直接在固定有效候选间取 Top-1，同分按冻结 candidate index；无有效候选记错误。只使用候选有效性，不使用 gold。所有组相同，不为每个 checkpoint 单独挑 fusion/gate。

次轨道：所有组共用同一 SC residual/gate，参数由独立 U1 pilot 和 validation 按共同预算选一次后冻结。不得像旧 runner 那样每模型分别挑 residual、confidence、cluster gate 再把差异归于 sampler。

同时报告 raw 和 gated 结果。如果 gated 在初始化就能回退到 B 并满足质量目标，则该目标没有训练增益要求：报告 T=0，但不能用它证明课程加速。另预设一个至少超越 B 一道 validation 题的质量档；若所有方法均未达到，诚实报告不可达。

若只允许在同一执行结果簇内部换 SQL，其执行正确性通常相同，不能作为 EX 选择改善的主要证据；同大小的不同结果簇也不等于同一结果簇。禁止通过这些退化限制制造“无回归”。

## 8. 同质量、同回归的达标指标

对问题 q，C(f,q) 为既有 evaluator 下执行正确指示。所有新模型相对同一 B 计算：

\[
A(f)=N^{-1}\sum_q C(f,q),\qquad
R_B(f)=N^{-1}\sum_q\mathbf1[C(B,q)=1,C(f,q)=0].
\]

另报条件回归率 negative_flips / B_correct_count、positive flips、SC-correct retention、coverage、条件选择准确率。总体 NFR 与条件回归率分母不同，不混名。固定候选池下 coverage 应完全不变，否则数据路径有 bug。

### 8.1 冻结共同门槛的规则

用独立 pilot seeds 的 U1 raw 模型预算末最后 3 个 checkpoint，先每 seed 取中位质量/回归，再跨 seed 取中位数作为 A_ref、R_ref。主门槛 τ=A_ref、r_max=R_ref，按可观察题数网格显式取整并在正式比较前锁定。它是基线质量目标，不是“新模型超过 SC”的自动证明。

同时要求：初始化不已达该联合目标；否则记录任务过易，先在 pilot 阶段重新定义有意义目标，不能到正式结果后再改。若 A_ref 不超过 B，则可做训练机制研究，但不宣传已获可部署选择收益。部署价值档 τ_deploy=max(τ,A(B)+1/N_val)；r_max 不变，单列达标率与不可达结果。

主约束的统计非劣容忍量 δ_A=0、δ_R=0，保持原始“同质量、不增加回归”要求。可以事先另设业务容忍量的辅助轨道，但不能在主轨道失败后放宽还继续称无退化。所有门槛数值、验证题数、来源和哈希必须写入 threshold_manifest；当前尚无这些 pilot 结果，YAML 留空而不是虚构 80%/1% 等目标。

### 8.2 稳定联合达标时间

每 10 updates 评估同一 validation 集，要求连续 3 次满足 A≥τ 且 R_B≤r_max。使用第 3 次确认后的当前 checkpoint，并将确认时间计入，不回溯成第一次穿越的时间。预算末不足 3 次确认的 crossing 不算稳定达标；先保存初始 t=0 结果。

主时间 T_total 为共同特征准备完成后的冷启动训练时钟，包含初始化、warmup、诊断、采样、训练、checkpoint 和 validation；错误标签/特征的一次性生成与审计时间单列后再给冷启动全成本 T_cold。另报缓存已复用的 T_reuse，明确不能拿 EA 的复用成本对比 baseline 的冷启动成本。

T_step、pair draws、唯一题/候选曝光、处理 token（若后续共同换文本模型）、probe 前向量及 GPU 活跃时间为辅指标。GPU 异步计时用同步边界或 CUDA events；各组运行顺序随机、串行占用同设备，记录热机和硬件状态。不要逐步同步/写文件只给某一组增加开销。

## 9. 统计与外部检验

五个正式 seed 使用配对初始状态；sampler、anchor、dropout 和标签排列分别管理 RNG，所有概率和 draw 可重放。5 seeds 是起步重复数，不声称已达到统计功效。pilot 估计时间方差及翻转 discordance，再预先决定是否需要更多 seeds/问题，不用显著性结果决定无限补跑。

- 时间：报告配对差值、比值及区间、每组稳定达标比例；未达标 run 以预算右删失单列。共同 horizon 的 restricted mean time-to-target 或 capped 成本与成功率一起报，不能只平均成功 seed。若 baseline 未达标，不能凭无穷比值写任意倍加速。
- 质量与回归：checkpoint/推理规则冻结后，在未反复使用的 audit/test 上按 question 配对检验。先检验最终 EX 差值下界≥-δ_A、回归差值上界≤δ_R，再判断时间改善；p>0.05 不是非劣证据。
- 依赖结构：同题多个 pair 不作独立样本；主报告保留 seed 配对和 question 配对，辅报按数据库 cluster 的重采样区间。小数据库数导致宽区间时直接披露。
- 多比较：预先指定 EA1 vs LP1 为首要机制比较，EA1 vs U1/H1/S1 为后续比较，使用预定的多重比较控制。报告全部比较，不能只展示有利的一项。
- checkpoint 选择：validation 会因持续监控而适应性使用，不能把它当独立测试；外部 audit/test 不参与采样、调参或早停。正式终点质量对比和各自达标 checkpoint 质量都要报，不能挑不同规则的最优点。

有限测试不能无条件证明未来零回归。若严格 δ_R=0 的区间不支持，即便点估计没有增加，也只报告“未观察到增加但证据不足”。

## 10. 必交付结果、错误分析与判据

### 10.1 主结果表模板

| Method | Replay | Hit seeds | Stable T_total | T_cold | Updates/draws | End EX | NFR vs B | Fix / regression | 外部非劣通过？ |
|---|---|---|---|---|---|---|---|---|---|
| U/H/S/LP/EA | R0/R1 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |

给出质量、NFR 对累计 updates/draws/总时间的三种学习曲线及质量—回归—时间 Pareto 图。记录每个 checkpoint 对每题的固定 B 候选、新候选、正确性和执行签名。额外观察“只是换了 SQL 文本但结果相同”，不计为正确性回归。

逐题错误按 coverage miss、unfixed selection error、fixed、regression、both correct 分解；说明回归来自 singleton、重复行/投影、过滤/边界、join、聚合/类型/排序的哪些弱诊断。人工审计抽样规则在看方法差异前冻结；测试分析只能用于最终解释，不回写方法。每类同时列训练独立题支持和累计曝光，防止只报相对百分比。

### 10.2 什么结果支持完整假设

EA1 在 probe-matched 比较中优于 LP1，并优于标签打乱和组匹配控制；在相同 replay 下优于 U1/H1/S1；对 production-cost U/S 仍有总耗时优势；各自达标与预算末质量/回归通过预设外部非劣，并且非靠明显更低成功率换来的短时间。进一步优于 OneShot1 才声称持续反馈必要。

不必同时胜过所有历史 SQL 框架来验证训练效率，但不能用不同候选池、模型、推理预算的 EX 横比替代上述控制。

### 10.3 停止或收窄

| 观察 | 应采取的结论 |
|---|---|
| 标签打乱或组匹配同样好 | 不支持 SQL 错误语义增量 |
| U1 与 EA1 同样好，U0 较差 | 收益主要来自 replay |
| OneShot 更便宜且同样好 | 优先静态/一次性方案，不堆动态模块 |
| 只省 updates、不省总时间 | 只称更新/样本效率 |
| EX 达标但 NFR 不通过 | 联合目标失败，不用净修复掩盖回归 |
| 小池错误组无法支撑统计 | 只交付 pilot，不宣布正式有效 |
| 有效表示冲突或基础模型不收敛 | 单列表示/模型问题；更换模型需新条件，各方法共同更换 |

## 11. 后续实现接口与测试清单

建议后续新增，不是本轮已存在的命令：`curriculum_dataset.py`（冻结记录/标签），`curriculum_sampler.py`（U/H/S/LP/EA 和 replay），`curriculum_trainer.py`（共同训练/时钟），`curriculum_evaluator.py`（联合达标/翻转/配对统计），独立 runner。不要直接把 protocol YAML 传给历史 hard_case runner。

单元测试至少覆盖：split/哈希一致；无 gold 特征；pair 只来自同题；概率和为 1 且 full support；S 多标签归一化；EA η=0 与 LP 等价；空组/keep 空集明确失败；固定 anchor 重放；train-only 更新；标签排列保留支持；时间确认点不回溯；零次/一次 crossing 不误判；无 hit 不被删除；coverage 不变；无有效候选计错、空结果仍有效。

文献方法学依据：[LogFloor](https://arxiv.org/html/2608.14936v1) 的首次达标与路径/静态对照，以及 [Positive-Congruent Training](https://arxiv.org/html/2011.09161v3) 对平均质量和 negative flips 的区分。本文的 sampler 公式、门槛规则和参数是本项目待验证设计，并非两篇论文已证明适用于 SQL 的结果。

本轮仅新增本协议与参数清单，没有修改运行代码，没有开始 P0/P1，所有性能单元格保留未运行。
