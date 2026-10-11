# BIRD Train Hard-case Pairwise Reranker：100 题实验报告

## 1. 实验目的

本阶段不修改 SQL 生成器，而是在已有候选池上验证一个更窄的问题：监督式 pairwise reranker 能否从候选 SQL 中选出执行正确的 SQL，并缩小 candidate oracle 与 self-consistency 之间的差距。

训练样本重点筛选为：

```text
oracle correct && self-consistency wrong
```

也就是候选池中至少存在一个执行正确的 SQL，但按最大执行结果簇进行 self-consistency 选择时选错的题。self-consistency 没有被移除，而是通过结果簇大小、簇占比、簇排名、相对最大簇比例、是否属于最大簇以及原有 consistency score 等 7 个特征输入 reranker。

## 2. 数据与划分

本实验使用缓存的 `direct_schema_pool_300`，它是 BIRD train 的 300 题子集，而不是完整的 9,428 题训练集。

| 项目 | 数量 |
|---|---:|
| 名义问题数 | 300 |
| 本地可评估问题数 | 269 |
| 候选 SQL 数 | 903 |
| candidate oracle 可覆盖问题 | 212 |
| hard cases | 17 |
| 训练 / 验证 / 测试问题 | 129 / 40 / 100 |
| 训练 / 验证 / 测试 hard cases | 9 / 3 / 5 |

有 31 题没有进入本地评估，主要来自候选缓存缺失或 gold SQL 无法在当前 SQLite 环境稳定执行。冻结划分按 question ID 进行，并按数据库与 hard-case 状态分层；测试集覆盖 19 个数据库。该划分不是 database-disjoint，因此只能回答“同分布候选选择是否改善”，不能证明跨数据库泛化。

真正可用于 hard-focused pairwise 训练的只有 9 道题、40 个候选和 38 个正负候选对。`all_covered_with_sc` 对照组使用训练集中同时包含正确和错误候选的 20 道题、82 个候选对。这个规模仍然偏小，是解释实验方差和回归现象的关键背景。

## 3. 模型与选择策略

模型是两层 MLP（64/32 hidden units），以 RankNet pairwise loss 为主，并加入权重为 0.2 的 pointwise loss。训练使用 5 个随机种子组成 ensemble，运行在 NVIDIA GeForce RTX 3070 Ti Laptop GPU 上。阈值和最终选择策略只在 40 题 validation set 上确定，测试集没有用于调参。

主模型 `hard_focused_with_sc` 保留 48 个特征，其中包含 7 个 self-consistency 特征。纯神经分数容易覆盖掉强多数簇，因此最终采用 validation 选择出的 confidence gate：只有当 neural candidate 相对 self-consistency winner 的分数优势至少为 0.20 时才改选；否则回退到 self-consistency。

## 4. 100 题主要结果

| Selector | Test EX | Oracle EX | Gap closure | 说明 |
|---|---:|---:|---:|---|
| Self-consistency baseline | 76.0% | 81.0% | 0.0% | 最大执行结果簇 |
| Hard-focused pairwise + SC | 75.0% | 81.0% | -20.0% | 主模型，confidence gate |
| Hard-focused pairwise without SC | 76.0% | 81.0% | 0.0% | SC 特征消融 |
| All pairwise-eligible + SC | 76.0% | 81.0% | 0.0% | cluster-support gate |

主模型相对 baseline 的绝对差为 **-1/100**。逐题 bootstrap 的 95% 置信区间为 **[-7, +5] 个百分点**，区间跨过 0，不能认为总体准确率有可靠提升。

纯 neural ranking 的 EX 只有 72.0%，confidence gate 将其恢复到 75.0%，说明 self-consistency 不仅应作为输入特征，还应作为推理阶段的保守先验。SC 特征消融和更广训练集对照都达到 76.0%，但只是与 baseline 持平，不构成提升证据。

## 5. 选择结果分解

| 结果类别 | 题数 | 含义 |
|---|---:|---|
| baseline 与模型都正确 | 71 | 无需修复且没有回归 |
| candidate coverage miss | 19 | 候选池没有正确 SQL，reranker 无法修复 |
| reranker 修复 | 4 | baseline 错，模型选对 |
| reranker 回归 | 5 | baseline 对，模型改选错误 |
| 仍未修复的 selection error | 1 | 候选池有正确 SQL，但二者都没选到 |

在专门保留的 5 个 hard test cases 上，主模型修复 4 个，选择准确率为 **80%**；self-consistency 在该子集上按定义为 0%。这说明模型确实学到了一部分“少数候选可能比多数结果簇更好”的信号，但 4 次修复被 5 次普通样本回归抵消，因此研究假设只得到局部支持。

## 6. Hard case 与回归逐题分析

### 成功修复的 4 题

| QID | 数据库 | 主要差异 | 判断 |
|---:|---|---|---|
| 2479 | shooting | baseline 使用 `COUNT(DISTINCT officer)`；被选候选使用 `COUNT(*)`，与 gold 的案件计数一致 | 修复投影/计数对象错误 |
| 6969 | ice_hockey_draft | baseline 先用 `WHERE nation='Russia'` 缩小分母；被选候选把 Russia 条件放入分子，分母保持全体球员 | 修复百分比口径错误 |
| 7819 | world | baseline 排除了 `NULL lifeexpectancy`，gold 没有该过滤；模型选择无额外过滤的候选 | 修复不必要过滤 |
| 8121 | university | 模型选择显式 `CAST(... AS REAL)` 的比例计算，避免整数除法或类型差异 | 修复数值语义 |

前两题表明 reranker 不只是匹配 SQL 表面结构，而是在计数对象和比例分母这类执行语义上找到了正确候选。后两题则更接近数据库类型与 gold 执行语义的细节。

### 引入回归的 5 题

| QID | 数据库 | 模型错误 | 根因 |
|---:|---|---|---|
| 3126 | cars | 把 `ORDER BY price ASC LIMIT 1` 改为 `price = MIN(price)` | 在当前数据上 ties/NULL 语义不同；选择了只有一个成员的结果簇 |
| 4646 | disney | 返回 `director.name` 而非 `movie_title`，且直接按带货币符号的文本排序 | 投影和数值类型处理退化；从 4-member 簇跳到 singleton |
| 4725 | disney | 对货币字符串直接降序排序，没有转为数值 | 文本排序造成执行结果错误；从 4-member 簇跳到 singleton |
| 5630 | shipping | `COUNT(DISTINCT city_id)` 与 `GROUP BY city_id` 同时使用，返回多行而非单个城市数 | 聚合层级错误；从 4-member 簇跳到 singleton |
| 8097 | university | 用硬编码 `university_id=123`、`criteria_id=1` 替代语义 join | schema/value 泛化错误；从 2-member 簇跳到 singleton |

五个回归有一个共同模式：模型都从 self-consistency 的较大结果簇跳到了 **group size = 1** 的候选。也就是说，虽然 SC 分数已经进入特征，hard-only 训练仍然过度奖励“反多数簇”行为。validation 选择的 0.20 confidence threshold 对这五个样本仍不够保守。

### 未修复的 1 个 selection error

QID 8944（human_resources）询问 Jose Rodriguez 与 Sandy Adams 谁的职位需要更高教育水平。baseline 和模型都额外投影了 `educationrequired`，而 gold 只返回员工姓名。正确候选存在，但当前特征没有足够的信息区分“排序依据可使用但不应出现在最终投影中”。这说明需要更明确的 question–projection 对齐特征，而不仅是 schema/AST 统计量。

## 7. 其余 25 个模型错误

100 题中模型共错 25 题，其中 19 题是 coverage miss，5 题是回归，1 题是未修复选择错误。错误标签是基于 SQL AST 与 gold 的启发式对比，可以一题多标签，因此计数之和大于错误题数。

| 错误维度 | 总计 | coverage miss 中 | 回归中 | 未修复 selection 中 |
|---|---:|---:|---:|---:|
| Projection | 14 | 11 | 2 | 1 |
| Filter | 12 | 10 | 2 | 0 |
| DISTINCT | 7 | 6 | 1 | 0 |
| Table / join | 4 | 3 | 1 | 0 |
| ORDER BY | 3 | 2 | 1 | 0 |
| LIMIT | 3 | 2 | 1 | 0 |
| GROUP BY | 1 | 1 | 0 | 0 |
| 其他执行/语义不一致 | 3 | 2 | 1 | 0 |

最大的总体瓶颈不是 selector，而是候选覆盖：oracle EX 只有 81%，因此任何只重排当前候选的系统上限都是 81%。19 个 coverage misses 主要集中在 projection（11）、filter（10）和 DISTINCT（6），说明生成阶段经常遗漏返回字段、过滤范围或去重要求。对这些题继续优化 reranker 不会有效，必须增加候选多样性或针对缺失维度进行定向生成。

按数据库看，主模型相对 baseline 的净变化为：`shooting +1`、`ice_hockey_draft +1`、`world +1`，但 `cars -1`、`shipping -1`、`disney -2`；其余数据库持平。Disney 的两个回归都涉及货币字符串的数值转换，显示当前结构特征没有表达 SQLite 类型转换是否正确。

## 8. 结论与下一步

本次实验完成了 hard-case 筛选、监督式 pairwise 训练、SC 特征保留、GPU ensemble 训练以及冻结 100 题评估。结果没有证明总体 EX 提升：主模型为 75%，低于 76% baseline；但它在 hard subset 上修复了 4/5，证明 hard-case supervision 有信号。

下一步最值得做的不是继续扩大同一个小 MLP，而是：

1. 从更大的 BIRD train 候选缓存中把 hard training cases 从 9 道提升到至少 100 道，并保持 test split 完全冻结。
2. 将 reranker 限制为“最大结果簇内部排序”或只在明确检测到 SC 不可靠时越过最大簇；当前五个回归全部来自改选 singleton。
3. 增加 question–SQL 的语义特征，尤其是 projection 对齐、百分比分子/分母、聚合层级、货币/日期数值转换和硬编码 ID 检测。
4. 分开优化 candidate coverage 与 selection：先把 oracle EX 从 81% 提高，再衡量 reranker 的 GapClosure。
5. 后续报告同时给出 overall EX、oracle EX、hard-subset accuracy、回归数和 coverage miss 数，避免只报告 hard-case 修复率。

## 9. 复现实验

冻结划分位于 `config/hard_case_pairwise_100_split.json`。训练命令：

```powershell
D:\software\anaconda3\envs\yoloV8\python.exe -m alphasql.runner.cuda_bootstrap alphasql.runner.hard_case_pairwise_experiment `
  --mode train `
  --candidate-dir results/direct_schema_pool_300/train `
  --data-path data/bird/schema_pool_300/train.json `
  --db-root data/bird/train/train_databases `
  --split-file config/hard_case_pairwise_100_split.json `
  --output-dir results/hard_case_pairwise_100 `
  --device cuda `
  --seeds 7 21 42 84 168 `
  --epochs 250 `
  --patience 40
```

机器可读结果位于：

- `results/hard_case_pairwise_100/report.json`
- `results/hard_case_pairwise_100/per_question.csv`
- `results/hard_case_pairwise_100/errors.csv`
- `results/hard_case_pairwise_100/report.md`

