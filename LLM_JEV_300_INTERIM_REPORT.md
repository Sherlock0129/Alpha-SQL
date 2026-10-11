# LLM/MCTS 与 Jev：300 题扩展实验中间报告

## 当前状态

实验集已经在远端调用前冻结为 **300 个 gold SQL 可在本地 SQLite 执行的 BIRD train 问题**。原 `schema_pool_300` 中有 19 题的 gold SQL 与当前数据库 schema 不匹配，因此从已有 BIRD train 预处理池中按 question ID 升序确定性补入 19 题。替换清单位于 `config/llm_jev_300_split.json`。

Jev constrained pipeline 已完成 300/300。低预算 MCTS 在完成 287/300 个候选池后，Qwen/DashScope 返回 `Arrearage / overdue-payment`，因此剩余 13 题暂时无法生成。本文同时报告：

- Jev 原生 pipeline 的完整 300 题结果；
- 三路方法在共同完成的 287 题上的可比中间结果；
- 13 题补跑后的严格 300 题结果尚未声称完成。

## 实验设置

MCTS 与原 12 题实验保持相同预算：

- 模型：`qwen3-coder-flash`
- rollout：2
- max depth：5
- max output tokens：2048
- temperature：0.8
- 4 个并发 worker

Jev 使用 `jev-latest`、beam width 8，并分别运行 schema linking、受约束 query-plan beam、SQLite 验证和 reranking。固定 MCTS 池上的 Jev reranker 最多读取 8 个候选。

## 主要结果

### 完整 Jev 原生 300 题

| 方法 | Candidate coverage | Final EX | Selector accuracy given coverage |
|---|---:|---:|---:|
| Jev constrained pipeline | 90/300 (30.00%) | 78/300 (26.00%) | 86.67% |

Jev 自有候选池的 top-beam 为 61/300，execution self-consistency 为 45/300，Jev reranker 为 78/300。也就是说，reranker 在该候选池内有效，但 constrained compiler 的 candidate coverage 只有 30%，是主要瓶颈。

### 三路共同完成的 287 题

| 方法 | Candidate coverage | Final EX |
|---|---:|---:|
| MCTS + self-consistency | 229/287 (79.79%) | 217/287 (75.61%) |
| Jev constrained pipeline | 84/287 (29.27%) | 73/287 (25.44%) |
| MCTS 候选池 + Jev reranker | 229/287 (79.79%) | 208/287 (72.47%) |

在固定 MCTS 候选池上，Jev reranker 相比 self-consistency：

- 两者都正确：204
- Jev 修复 self-consistency：4
- Jev 引入回归：13
- 两者都错误：66
- 净变化：-9/287，即 **-3.14 个百分点**

因此，300 题规模上的中间证据比 12 题更明确：当前无监督 Jev reranker 不适合直接替换 execution self-consistency。它能够修复少数语义错误，但回归数量明显更多。

逐题 paired bootstrap 的 Jev-on-MCTS 相对 self-consistency 差值为 **-3.14 个百分点**，95% CI 为 **[-5.92, -0.35] 个百分点**。只看二者结果不同的 17 题，Jev 修复 4 题、破坏 13 题，exact McNemar/binomial 检验 `p = 0.049`。在当前 287 题共同集上，这已经是“直接替换会下降”的证据；但由于 13 个缺失题受欠费与输出截断影响，严格 300 题结论仍应等补跑后确认。

如果把 13 个因欠费缺失的 MCTS 题全部保守计为错误，只能得到下界：MCTS self-consistency 至少 217/300（72.33%），Jev-on-MCTS 至少 208/300（69.33%）。这不是最终 300 题 EX，因为缺失题没有得到运行机会。

## 与原 12 题实验的可比性

原 12 题不是从 BIRD train 随机抽取，而是专门选择的受约束 SQL slice：单个投影、常见聚合、INNER JOIN、最多一个非 join filter，并排除子查询、集合操作、复杂布尔条件、GROUP BY/HAVING、ORDER BY/LIMIT 等结构。因此，直接把完整 300 题总体结果与原 12 题百分比并列，会混淆“模型退化”和“任务超出 compiler 能力”。

按原 12 题实验的能力定义重新划分 300 题：

| Gold SQL 范围 | 题数 | Jev coverage | Jev final EX |
|---|---:|---:|---:|
| 当前 constrained compiler 目标范围内 | 83 | 64/83 (77.11%) | 62/83 (74.70%) |
| 目标范围外 | 217 | 26/217 (11.98%) | 16/217 (7.37%) |
| 全部 | 300 | 90/300 (30.00%) | 78/300 (26.00%) |

因此，Jev 从原 12 题的 66.7% 上升到更大范围内切片的 74.7%，并没有显示范围内能力退化。总体降到 26.0% 的根因是 217 题包含当前 QueryPlan/SQL compiler 没有建模的 SQL 结构。

在 287 题共同集内，范围分解为：

| 子集 | 题数 | MCTS coverage / EX | Jev native coverage / EX | Jev-on-MCTS EX |
|---|---:|---:|---:|---:|
| 范围内 | 77 | 70 / 68 | 59 / 57 | 63 |
| 范围外 | 210 | 159 / 149 | 25 / 16 | 145 |

这也说明 MCTS 的优势主要来自开放式 SQL 生成能力；Jev 的强项是结构化决策和范围内排序，而不是替代通用生成器。

## 选择器误差分解

在 287 题共同集上：

| 系统 | Coverage miss | Selection error | 总错误 |
|---|---:|---:|---:|
| MCTS + self-consistency | 58 | 12 | 70 |
| Jev constrained | 203 | 11 | 214 |
| MCTS pool + Jev reranker | 58 | 21 | 79 |

MCTS 的 oracle 与 self-consistency 差距为 12 题；Jev reranker 只修复其中 4 题，同时破坏了 13 道 self-consistency 已正确的问题。Jev constrained 的 203 个 coverage misses 中，181 题具有完整的 gold table/column recall，因此其主要问题不是 schema linking，而是当前 query-plan/SQL compiler 无法覆盖复杂 SQL 结构、literal 与 benchmark 特定执行语义。

MCTS self-consistency 在已有正确候选的 229 题中选对 217 题，selector accuracy 为 **94.76%**。Jev-on-MCTS 选对 208 题，为 **90.83%**。以 MCTS oracle 与 baseline 的 12 题差距为分母，当前 Jev reranker 的 GapClosure 为 `-9/12 = -75%`，也就是不但没有关闭差距，反而增加了 9 个净错误。

## Schema linking 与候选生成分析

287 题共同集的平均 schema 指标为：

| 指标 | MCTS 路径中的 schema evidence | Jev linker |
|---|---:|---:|
| Table precision | 54.24% | 63.35% |
| Table recall | 53.98% | 99.30% |
| Column precision | 47.54% | 60.16% |
| Column recall | 52.86% | 97.35% |

MCTS 数值偏低的一部分原因是有些保存路径跳过 schema-selection action，因此这里不是纯粹的 linker 对比。尽管如此，Jev 的高 recall 与低 candidate coverage 形成了清楚的诊断：Jev 基本找到了 gold table/column，但没有把这些元素组合成足够丰富的 SQL plan。

Jev 的 203 个 candidate coverage misses 中：

- 181 题 table/column recall 完整，应优先归因于 SQL 语法能力、literal/operator、聚合层级或 beam 组合；
- 22 题存在 schema recall 不完整，才是 linker 本身需要改进的部分。

所以后续资源不应主要投入“再训练一个更强的二分类 schema linker”，而应首先扩展 QueryPlan 与 compiler：多投影、多个过滤条件、子查询、GROUP BY/HAVING、ORDER BY/LIMIT、CASE、集合操作和复杂日期/字符串表达式。

## 分数据库观察

MCTS self-consistency 表现较好的数据库包括 `student_loan`（15/16）、`shipping`（14/15）、`university`（14/15）和 `cars`（18/22）；较弱的包括 `shooting`（8/14）、`law_episode`（9/15）、`beer_factory`（10/16）和 `book_publishing_company`（10/16）。

Jev-on-MCTS 相比 self-consistency 的数据库净变化为：

- 改善：`cars +1`、`human_resources +1`、`shooting +1`；
- 明显下降：`computer_student -3`、`craftbeer -2`；
- 下降 1 题：`beer_factory`、`book_publishing_company`、`car_retails`、`disney`、`law_episode`、`restaurant`、`simpson_episodes`；
- 其余数据库持平。

这不是某一个数据库造成的偶然下降，而是多个 schema 上重复出现的 benchmark alignment 问题。

## Jev-on-MCTS 的修复与回归

成功修复的 question IDs 为：`210, 3099, 8963, 2481`。

- `210`：从 `COUNT(*)` 改为 `SUM(qty)`，修复“订购了多少本书”的计量对象。
- `3099`：从 `>= 400 / >= 30000` 改为 gold 使用的严格 `> 400 / > 30000`。
- `8963`：把带 `US$` 和逗号的工资字符串转换为数值后求最大值。
- `2481`：修复年份上界，将自然语言中的 2014 按 gold 约定映射到 `2013-12-31`。

引入回归的 question IDs 为：`218, 230, 972, 991, 1024, 1327, 1602, 1737, 4346, 4706, 5344, 8859, 8861`。

主要模式包括：

1. **不必要的 DISTINCT**：`218, 972, 991, 1737, 4346`。自然语言上去重看似合理，但 BIRD execution gold 保留重复行。
2. **投影偏离 benchmark**：`230` 把三列姓名拼成一列；`1602` 返回经理姓名而不是 `reportsTo` ID。
3. **错误的额外语义约束**：`1024` 增加 `person.professor = 1`；`1327` 把 episode 字段替换为 `number_in_series`。
4. **聚合与分母错误**：`5344` 改变信用卡类型的分组语义；`8859` 用不同 beer style 数量作为百分比分母；`8861` 按 distinct brewery 而不是 gold 的 beer rows 计数。
5. **等价表面语义但执行结果不同**：`4706` 改从另一个电影表读取 release date。

这些回归说明 Jev 的通用语义判断经常比 benchmark SQL 更“自然”，却不一定与 BIRD 的投影、重复行和计数口径一致。要改进 selector，需要监督式 benchmark-aligned reranker，或至少将 execution-cluster support 作为强约束，而不只是提示信息。

## 效率与中断原因

- Jev native 300 题：平均 1.554 秒/题；4,427,298 input tokens；846,298 output tokens；uncertain rate 88.67%。
- MCTS 已记录：5,835 次成功调用；9,714,613 prompt tokens；1,915,486 completion tokens；共 11,630,099 tokens。
- MCTS 成功调用的聚合 API 时间为 20,050.7 秒；实际首轮墙钟时间约 77 分钟（4 路并发）。
- 用量 JSONL 有 1 条并发退出时留下的损坏尾行，汇总已跳过并显式记录；失败/欠费请求没有 provider usage。
- 中断并非代码异常：DashScope 明确返回账户欠费 `Arrearage / overdue-payment`。

## 完成严格 300 题结果所需步骤

补充 Qwen/DashScope 额度后，重新运行同一 MCTS 命令即可自动跳过 287 个已有文件，只生成缺失的 13 题：

```powershell
$env:ALPHASQL_USAGE_PATH = 'D:\University\Alpha-SQL\results\llm_vs_jev_300\mcts_usage.jsonl'
.venv\Scripts\python.exe -m alphasql.runner.mcts_runner config\llm_jev_300_mcts.yaml
```

当前缺失 question IDs：

```text
987, 1031, 3116, 4415, 4418, 4456, 4501,
4519, 4521, 4653, 4662, 4670, 7884
```

其中 `987, 1031, 3116, 4670, 7884` 在欠费前因 2048-token 输出截断而缺失，其余题在账户欠费后未能生成。补跑后再次执行 Jev-on-MCTS rerank 和 evaluate 即可覆盖完整 300 题。

## 产物

- 冻结 split：`config/llm_jev_300_split.json`
- MCTS 配置：`config/llm_jev_300_mcts.yaml`
- MCTS candidates：`results/llm_vs_jev_300/mcts_candidates/`
- Jev native candidates：`results/llm_vs_jev_300/jev_candidates/`
- Jev-on-MCTS results：`results/llm_vs_jev_300/mcts_jev_rerank/`
- 共同集逐题结果：`results/llm_vs_jev_300/per_question.csv`
- 逐系统错误：`results/llm_vs_jev_300/error_analysis.csv`
- 汇总指标：`results/llm_vs_jev_300/metrics.json`

