# SQL 选择器课程学习：P0a 实跑与等价性检查

日期：2026-10-20。分支：`codex/alpha-sql-jev`。

## 1. 结论

已完成 U1、LP1、EA1、NoGroupEA1 四组实验，每组 seed=101、250 次更新，使用 RTX 3070 Ti Laptop GPU。

- 实现检查通过：NoGroupEA1（EA 的语义系数 η=0）与 LP1 的完整采样序列、训练损失、probe 概率、验证预测和最终参数逐位一致，参数最大绝对差值为 0。
- 性能没有观察到改善：四组预算末均为 27/40，原自一致性为 28/40；均修复 0 题、新增回归 1 题。
- EA 不是没有生效：与 LP 相比，1500 个 base draw 中有 4 个槽位发生变化，但调度差异很小，没有形成验证收益。
- 本轮仅为 P0a 机制 pilot，不能声称错误语义有增量、训练加速成立，或“不增加回归”成立。

## 2. 冻结数据与评价口径

使用历史 `results/direct_schema_pool_300/train`，不是另一个 MCTS 287 题池。没有运行 MCTS，没有重新生成候选，没有加入 gold SQL 或变异负例。

| 项目 | 数量 |
|---|---:|
| 训练问题 / 原始候选 | 129 / 454 |
| 验证问题 / 原始候选 | 40 / 115 |
| 可组成正负 pair 的训练问题 | 20 |
| 全部训练正负 pair | 82 |
| SC 正确且可组成 pair 的 keep 问题 | 11 |
| SC 错误且有正确候选的训练问题 | 9 |
| 验证集 SC 正确 / candidate coverage | 28 / 31 |
| 数值输入特征 | 48，保留原 SC 特征 |

沿用 `config/hard_case_pairwise_100_split.json` 的 train/validation IDs。读取该 split 文件中的 test ID 清单只用于检查互斥；没有打开 test 候选文件，也没有执行 test gold 或候选。原始 JSON 数据容器会被加载，但只对 train/validation 项访问 gold、构造执行标签。原候选重复次数保留，不去重投票。

这里的正确率采用现有代码的**无序行多重集执行签名**：保留重复行和列顺序，浮点数按原实现保留八位小数。它不是重新运行官方 BIRD evaluator 得到的正式 BIRD EX。空结果只要执行成功，仍属于有效候选。

冻结目录包含 train/validation records、候选文件 SHA256、特征清单、执行签名摘要、代码哈希、错误组 membership、SC 预测和数据哈希；派生诊断已核验数据与训练代码哈希。

## 3. 共同训练设置

模型为原 `SchemaAwareMLP`，隐藏层 64/32、dropout=0.1，scaler 仅 fit train。四组共享初始权重和 dropout 随机流。

每步 8 个 pair：6 个 base + 2 个 keep，不追加额外预算；另有 16 个 pointwise anchor。q0、q_keep 都是先均匀选问题、再均匀选题内 pair。anchor 同样先选问题再选候选，覆盖全部训练问题，不限于 20 个 pairwise 问题。三条采样随机流互相隔离，并在各 arm 使用相同种子。

损失为不带旧 hardness 权重的 RankNet softplus 均值，加 0.2 倍 pointwise BCE；BCE 正类权重由完整训练标签一次确定。AdamW，lr=0.001、weight decay=0.0001、gradient clip=5。

前 25 步统一随机 warmup；t=0 以及每 10 步进行相同的 train probe、验证和 checkpoint，共 26 次。不早停、不按验证结果选 gate、不融合 SC 分数；主推理为可执行候选中的原始神经分数 top-1，同分按候选 index。SC 只作为保留特征和 replay 参考。

每组消耗 2000 个 pair draw、4000 个 anchor draw。训练 forward 每步仅涉及约 29.9 个唯一候选，而非所有 454 个训练候选；全训练池 forward 只用于共同 probe。

LP/EA 沿用设计协议的 EMA=0.9、uniform floor ρ=0.25、组收缩 κ=10、EA η=1、权重范围 [0.1,10]。本次没有搜索这些参数。

## 4. 等价性检查

以下 9 项全部通过：

1. 四组初始参数一致。
2. 四组 warmup draw 一致。
3. 四组 warmup loss 一致。
4. 四组完整 keep 和 anchor 序列一致。
5. LP1 / NoGroupEA1 完整 draw trace 一致。
6. LP1 / NoGroupEA1 完整 loss 和梯度范数 trace 一致。
7. LP1 / NoGroupEA1 所有 probe 的采样概率一致。
8. LP1 / NoGroupEA1 所有 checkpoint 的验证预测及 logits 一致。
9. LP1 / NoGroupEA1 最终所有模型 tensor bit-exact，一切参数的最大绝对差为 0。

此外，单元测试对 30 轮随机 probe 检查 η=0 的概率等价性，并用小型实际训练检查完整轨迹。这是有限输入与有限轨迹的验证，不是对任意输入的数学证明。

## 5. 验证结果

| 方法 | 预算末正确率 | 修复 SC 错误 | 新增 SC 回归 | 回归 / SC 正确问题 |
|---|---:|---:|---:|---:|
| 原 SC | 28/40，70.0% | — | — | — |
| U1 | 27/40，67.5% | 0 | 1 | 1/28，3.57% |
| LP1 | 27/40，67.5% | 0 | 1 | 1/28，3.57% |
| EA1 | 27/40，67.5% | 0 | 1 | 1/28，3.57% |
| NoGroupEA1 | 27/40，67.5% | 0 | 1 | 1/28，3.57% |

四组初始验证均为 27/40；整个训练过程观测到的最高验证正确数均为 28，没有超过 SC。最高值仅作描述，没有用来选择最终 checkpoint。

四组预算末训练集均达到候选 oracle：100/129，训练修复 9 题、回归 0 题。首次在 probe 观察到训练 oracle 的步数为 U1=80、其余三组=70。这不是验证达标，更不是泛化加速证据；LP 与 EA 在这一点上也没有区别。

### 错误明细

四组预算末错误归属完全相同：9 题 coverage miss，3 题未修复 hard case，1 题新增回归。

| 问题 ID | 类型 | 从现存候选可确认的差异 |
|---|---|---|
| 1616 | 新增回归 | 正候选投影为 first name、last name；被选候选交换了两列顺序。两者 schema 覆盖均为 1、执行簇大小均为 1。当前标签判错，但它首先是结果列序与评价口径的审计案例，不能直接当作业务语义完全错误。 |
| 8005 | 未修复 hard | 问题要求全部符合条件大学的数量和其中美国大学的数量；被选候选只计算美国数量，正候选同时输出 total_count、usa_count。 |
| 8896 | 未修复 hard | 被选候选使用 literal `'baking product'`，正候选为数据库中的 `'baking products'`。表、join、聚合结构近似，仅结构统计不足以可靠判断值对齐。 |
| 8902 | 未修复 hard | 被选候选只返回总时长；正候选同时返回 recipe_id 与总时长。涉及投影完整性。 |

9 个 coverage miss 为 2485、4260、4714、5297、5344、7884、8856、8959、972。固定候选池下，这些问题不可能靠更换选择器修复。

当前 MLP 不直接读取 question/SQL token、literal 身份或有序投影语义，主要读取 schema/执行/结构/路径统计。上述案例提示表示能力也可能限制泛化；这只是诊断线索，本轮没有做特征因果消融，不能将失败全部归因于表示或课程。

## 6. EA 的调度作用有多大

错误标签来自已有正候选与负候选的 AST 对比，不使用 gold SQL 文本作分组参考。多标签 membership 归一化。这些标签**未人工审计，而且对 alias、引号和等价表达式敏感**，只能称为弱结构差异组。

| 组 | pair 数 | 不同问题数 | 归一化 question-equivalent |
|---|---:|---:|---:|
| projection_multiplicity | 78 | 19 | 7.005 |
| filter_literal | 56 | 17 | 4.671 |
| relation | 65 | 18 | 4.338 |
| computation_ordering | 48 | 13 | 3.986 |
| unknown_mixed | 0 | 0 | 0 |

pair 数不能相加作为独立样本量。四个活跃组的不同问题数超过 10，但总共仍只有 20 个训练问题；归一化支持量小、组之间高度重叠，unknown 组为空，不满足对完整五组的确认性验证。

EA 与 LP 的最大 pair 概率差约 0.000350，最大 total variation 为 0.001765，1500 个 base 槽位仅 4 个不同。κ=10 的收缩和组重叠使不同组反馈接近。若语义项接近全体共有常数，经过指数权重归一化后就几乎抵消；因此“加了语义项”并不保证产生强调度差异。这里只报告观察，不据此按验证表现增大 η。

## 7. 成本与统计边界

候选读取、SQLite 执行标注和冻结文件准备约 6.05 秒；共同 CUDA backend startup 约 0.265 秒。每组训练/probe/checkpoint/log wall time：U1=1.333 秒、LP1=1.030 秒、EA1=1.019 秒、NoGroupEA1=1.067 秒。

运行很短，顺序执行，首组包含额外首次训练栈开销。这些耗时不能用于声称 EA 比 U 更快；也尚未运行去掉不必要 probe 的 production-cost baseline。全部过程没有模型 API 调用。

quality τ 和 regression r_max 保持 null。没有从本次四组曲线反向挑目标，也没有计算正式 time-to-target。单 seed、40 题验证、20 题 pairwise 训练支持，均不足以推断统计显著性或非劣。

## 8. 代码、测试与复现

新增：

- `alphasql/algorithm/selection/curriculum.py`：冻结 pair、弱分组、LP/EA controller、共享采样流。
- `alphasql/runner/curriculum_pilot.py`：P0a 离线训练、网络阻断、GPU 确定性、哈希冻结和等价性检查。
- `alphasql/runner/curriculum_pilot_analysis.py`：只读冻结训练/验证产物，生成逐题错误和学习曲线明细。
- `tests/test_curriculum_pilot.py`：10 项新测试。

修改 `alphasql/runner/schema_signal_experiment.py`：新增可选 question IDs 白名单，在访问 gold、pickle 和数据库前过滤；为原候选记录执行签名摘要。默认调用方式兼容现有代码。

新测试 10/10 和原 `test_schema_signal.py` 13/13 通过，共 23 项。CUDA 环境未安装 pytest，本轮实际使用 unittest，不将它描述为 pytest 结果。

```powershell
& D:/software/anaconda3/envs/yoloV8/python.exe -m alphasql.runner.cuda_bootstrap unittest discover -s tests -p test_curriculum_pilot.py -v
& D:/software/anaconda3/envs/yoloV8/python.exe -m alphasql.runner.cuda_bootstrap unittest discover -s tests -p test_schema_signal.py -v
```

复现实验时使用新的输出目录，避免覆盖已冻结产物：

```powershell
& D:/software/anaconda3/envs/yoloV8/python.exe -m alphasql.runner.cuda_bootstrap alphasql.runner.curriculum_pilot --output results/sql_selector_curriculum_p0a_seed101_reproduction --seed 101 --device cuda
& D:/software/anaconda3/envs/yoloV8/python.exe -m alphasql.runner.curriculum_pilot_analysis --run-dir results/sql_selector_curriculum_p0a_seed101_reproduction
```

设计 YAML 仍保留原设计文档元数据；新 runner 仅实现并读取 P0a 所需子集，不表示 10 组正式比较、门槛冻结或外部确认实验已经实现。

实际产物目录为 `results/sql_selector_curriculum_p0a_seed101/`，包含冻结 manifest、records、弱标签、run config、四组 checkpoint/曝光/loss/probe/预测/cost、等价性 JSON、配对汇总、diagnostics、160 行逐题 `error_analysis.csv` 和学习曲线。

## 9. 下一步建议

先不直接扩大成 36 runs。优先人工审计有限的训练 pair 弱标签，检查 alias/等价 SQL 的伪差异，以及 1616 这类结果列序口径；按版本冻结修正后的训练标签，不用验证集调 η。利用 train-only 分布诊断确认语义项有足够、合理的异质性，再决定是否进入多 seed P0b。

如果表示能力与候选标签先决定了选择器的上限，单纯换课程很难改善验证表现。未来可以另设固定 question-aware 表示的实验，但必须对所有调度组同时替换并重新冻结，不能把模型升级与课程收益混在一起。
