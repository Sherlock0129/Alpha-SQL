# 287 题独立 Schema Linking 与替换实验

日期：2026-10-18。分支：`codex/alpha-sql-jev`。

## 已完成：独立 Schema Linking

两侧均已完成 287/287 题，无 linker API、解析或空选择失败。输入使用同一份 Question、Evidence 与 full schema，逐题以 SHA-256 验证。两侧不使用 gold SQL；LLM 保留原 schema_selection 模板，Jev 使用原子 table/column Noul。

| 输出 | Table P | Table R | Table F1 | Column P | Column R | Column F1 | 完整 Schema Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM 原始 | 98.84% | 97.81% | 98.01% | 94.67% | 87.12% | 89.54% | 173/287（60.28%） |
| Jev 原始 | 96.03% | 95.67% | 94.82% | 89.60% | 92.32% | 89.35% | 223/287（77.70%） |
| LLM 共同 PK/FK 补全后 | 98.84% | 97.81% | 98.01% | 86.31% | 95.87% | 90.07% | 248/287（86.41%） |
| Jev 共同 PK/FK 补全后 | 96.03% | 95.67% | 94.82% | 83.80% | 94.82% | 87.40% | 246/287（85.71%） |

所有 P/R/F1 都是逐题 macro average；F1 是逐题 F1 的平均，不能用表中平均 P/R 重新计算。

| 输出 | 平均表数 | 平均列数 |
|---|---:|---:|
| LLM 原始 | 1.94 | 4.07 |
| Jev 原始 | 1.97 | 4.57 |
| LLM 共同补全后 | 1.94 | 4.91 |
| Jev 共同补全后 | 1.97 | 5.06 |

## 如何修正此前判断

旧实验的 LLM 指标来自 MCTS modal schema evidence，部分路径跳过 schema-selection，会造成大量缺失。因此旧的约 50% 指标不能当作原 LLM linker 独立运行的准确率。

新的公平结果不支持“Jev 在 Schema Linking 上全面优于 LLM”：

- LLM 的 table P/R/F1 更高；
- Jev 原始 column recall 高 5.20 个百分点，但 column precision 低 5.07 个百分点，column F1 基本相同；
- Jev 原始完整 schema recall 高 17.42 个百分点；
- 加入相同 PK/FK 补全后，LLM 的 column recall 反而高 1.05 个百分点，完整召回率为 86.41% 对 85.71%；
- 所以下游实际收到的 schema 质量，不能再用原始 column recall 单独代表。

## 键补全为什么会改变排名

| 诊断（gold 列出现次数） | LLM | Jev |
|---|---:|---:|
| 原始遗漏键列 | 133 | 66 |
| 原始遗漏非键列 | 38 | 38 |
| 共同补全找回的 gold 列 | 112 | 32 |
| 补全后仍遗漏的 gold 列 | 59 | 72 |

键列指 PK、FK 或 referenced_by 关系中的列；这里统计跨题的列出现次数，而非去重数据库列数。Jev 原始召回优势有很大部分来自对连接键更主动的选择，LLM 原输出遗漏的连接键大多可被现有规则恢复。相同的非键遗漏总数不代表两侧错在同一批列。

## 下游 EX 实验

两组均完成 287/287 题。每组固定 upfront schema，均使用同一 PK/FK 补全、2 rollouts、depth 5、每次 generation/revision 采样 2 条、同一 seed 规则及 execution self-consistency；完全不使用 Jev selector、SQL intent、Join Choice 或受约束 compiler。

| 方法 | 完成 | Candidate coverage | 最终 EX | 有正确候选时选对 | SQL 可执行率 |
|---|---:|---:|---:|---:|---:|
| 原 LLM linker → Qwen/MCTS → SC | 287/287 | 229/287（79.79%） | 209/287（72.82%） | 209/229（91.27%） | 1089/1115（97.67%） |
| Jev 表列 linker → Qwen/MCTS → SC | 287/287 | 220/287（76.66%） | 199/287（69.34%） | 199/220（90.45%） | 1051/1081（97.22%） |

相对 LLM：Jev 修复 11 题，造成 21 题回归，净减少 10 题，EX 差值 **−3.48 个百分点**。两者同时正确 188 题，同时错误 67 题。

- 问题级 paired bootstrap 95% CI：**[−7.32, +0.35] 个百分点**；
- exact McNemar：**p=0.1102**；
- 该单种子实验没有显示替换收益；差值未达到常用 0.05 显著性标准，因此也不能宣布 Jev 稳定显著更差。

原始完整 schema recall 的 Jev 增益为 +17.42pp，95% CI 为 [+11.50,+23.34]pp、McNemar p≈2.25e−8；共同补全后的差值为 −0.70pp，95% CI 为 [−4.53,+3.48]pp、p=0.864。显著的原始召回增益没有保留到实际下游输入。

这里不是将旧 MCTS 候选池交给 Jev 重排；两组分别根据各自 schema 重新生成候选。LLM 共生成 1115 条去重候选，Jev 组 1081 条，平均每题 3.89 / 3.77 条。相同的是采样与搜索上限，实际候选数会受去重和执行验证影响。

## 错误分解：净下降主要发生在 candidate coverage

| 错误类型 | LLM | Jev |
|---|---:|---:|
| 缺少正确候选，且 gold schema 不完整 | 19 | 24 |
| 缺少正确候选，但 gold schema 已完整 | 39 | 43 |
| 正确候选存在，但 SC 未选中 | 20 | 21 |
| 总错误 | 78 | 88 |

Coverage miss 为 58 → 67，增加 9 题；selection error 为 20 → 21，增加 1 题。最终差距大部分来自候选池变化。上表是按可观测条件分类的诊断，不是人工证明每道错题的因果。

## Schema 补全对 EX 的局部作用

| 共同补全后的情况 | 题数 | LLM EX 正确数 | Jev EX 正确数 |
|---|---:|---:|---:|
| Jev 完整、LLM 不完整 | 16 | 4 | 9 |
| LLM 完整、Jev 不完整 | 18 | 13 | 5 |
| 两侧均完整 | 230 | 180 | 175 |
| 两侧均不完整 | 23 | 12 | 10 |

补齐正确表列具有局部价值：Jev 独有的 16 个完整 schema 带来 5 题净收益；但 LLM 独有的 18 个完整 schema 带来 8 题净优势。两侧 schema 均完整时，仍然有大量生成和选择错误，不能把优化目标压缩成表列召回一个指标。

### 输入相同的对照揭示生成波动

| 两侧实际输入关系 | 题数 | LLM 正确 | Jev 组正确 | Jev 修复 / 回归 |
|---|---:|---:|---:|---:|
| DDL 上下文完全相同 | 97 | 80 | 78 | 1 / 3 |
| 表列集合相同，表顺序不同 | 104 | 78 | 77 | 3 / 4 |
| 表列集合不同 | 86 | 51 | 44 | 7 / 14 |

97 题中连初始 DDL 上下文都完全相同，仍产生 4 个 EX 分歧。这是远端随机生成与后续搜索轨迹波动的证据，不能归因于 linker 召回。104 题集合相同但顺序不同，其差异可能同时受到顺序和采样影响。后续若要更精确估计纯表列集合效应，应统一 DDL 顺序，并进行多种子复验或复用完全相同上下文的生成缓存。

## 32 个分歧题复核

### Jev 修复的 11 题

| IDs | 观察到的机制 |
|---|---|
| 868 | 补回 `course.name`，恢复 Machine Learning Theory 的过滤条件。 |
| 869 | 补回学生 `f_name/l_name`，恢复 Willie Rechert 的实体过滤。 |
| 1274 | 补回 `person.name`，正确候选已存在，但 SC 原先选了没有 Julia Roberts 条件的 SQL。 |
| 1641 | 补回 customers 表和 customerName，输出从客户编号恢复为姓名。执行通过，但其单笔 payment 逻辑与 gold 的 SUM 不完全一致，属于单数据库执行等价而非语义等价证明。 |
| 5656 | 补回司机姓名和城市名，恢复司机、城市、年份的联合过滤。 |
| 8911 | 补回 `recipe.servings`，恢复 servings=7 过滤。 |
| 7897 | Schema 反而不完整，但模型直接使用 `countrycode='BRA'`，在当前数据库执行通过；不能计作 schema 改善带来的修复。 |
| 976、4360、5344、8944 | 表列集合相同，分别涉及额外 professor 条件、DISTINCT、消费总额聚合、姓名投影；不能归因于更高召回。5344 的初始 DDL 也完全相同。 |

### Jev 回归的 21 题

| IDs | 观察到的机制 |
|---|---|
| 230 | 漏掉 `employee.minit`，全名投影缺中间名，且输出了额外 job_lvl。 |
| 1267、1327 | 漏掉 Person 的姓名过滤，模型转而使用错误或无法验证的 person_id 常量。 |
| 1693 | 漏掉餐厅标识与县信息的表，模型把县名误当街道 LIKE 条件。 |
| 3066、3085、3133 | 漏掉 data 表中的车型名/model，无法构造对应的车型过滤。3085 多次输出截断，最终补跑完成但结果仍错。 |
| 3067、3105 | Schema 不完整；3067 正确候选仍存在但 SC 选错，3105 没有正确候选。 |
| 6940 | 漏掉 PlayerInfo，丢失 Pavel Brendl 的实体过滤。 |
| 8116 | 漏掉 university_ranking_year.score，输出学生数而非分数。 |
| 1345 | 额外 person_id 等列伴随计数对象从 award rows 改为 distinct people；这是观察到的关联，单次实验不足以证明额外列是原因。 |
| 4310 | Jev schema 更完整，但 COUNT(*) 被改为 COUNT(DISTINCT person)，仍然回归。 |
| 8008 | 两侧 schema 均完整，模型从 LIMIT 1 改为返回所有最小值并列行。 |
| 991、1666、1745、3126、4695、6969、8963 | 表列集合相同；错误分别涉及 DISTINCT、漏 GROUP BY/SUM、literal 大小写、极值并列、Lion King 字符串、百分比分母、工资字符串数值转换。991/4695/8963 的初始 DDL 完全相同。 |

完整 question、evidence、gold SQL、两侧选中 SQL、coverage、schema 完整性与上下文相同标记均见 `results/linker_ablation_287/paired_cases.csv`。

### 数据库差异

`cars` 下降最明显：18/22 → 12/22，6 个回归，没有修复。`law_episode`、`restaurant`、`ice_hockey_draft`、`university` 各净下降 2 题。`cs_semester` 净增加 2 题，`beer_factory`、`cookbook`、`shipping`、`world` 各增加 1 题。收益和回归的分布与实体过滤、跨表属性定位等模式相关，不能用整体 recall 掩盖数据库差异。

## 延迟和实际用量

| 阶段 | LLM linker 组 | Jev linker 组 |
|---|---:|---:|
| 独立 linking 请求数 | 287 | 287 |
| linking 平均请求耗时 | 3.38 秒 | 1.15 秒 |
| linking input/output tokens | 964,309 / 84,361 | 1,281,716 / 207,401 |
| Qwen 下游实际请求数 | 4,476 | 4,466 |
| Qwen 下游 input/output tokens | 4,130,492 / 1,572,555 | 4,138,380 / 1,640,766 |
| Qwen 下游累计 API 时间 | 17,150.55 秒 | 17,752.68 秒 |
| 成功生成运行平均 wall time | 60.01 秒/题 | 61.08 秒/题 |

Jev linking 延迟更低，但不同 API 的 token 定义和价格不能直接互换；没有根据未配置的价格报告货币成本。生成耗时由 Qwen/MCTS 主导，linker 更快没有转化为明显的端到端时延优势。累计 API 时间不是并发墙钟时间；成功运行平均 wall time 不含先前失败尝试，实际请求/token/API 时间包含失败尝试中已返回的用量。

4 次输出截断发生在 3 个运行：LLM 1327 一次、Jev 1693 一次、Jev 3085 两次。均按相同配置补跑成功；补跑只由异常触发，没有按 SQL 标签重采样。最终失败数为 0。

## 当前结论与交付

本实验不支持直接用 Jev 表列 linker 替换原 LLM linker 以提高最终 EX。Jev 原始列召回的优势主要来自连接键；相同 PK/FK 补全消除了大部分优势。真正影响 EX 的是关键语义列和表是否遗漏，以及下游过滤、聚合、计数、重复行和类型逻辑是否正确。

此前“Jev 明显优于 LLM Schema Linking”的判断应撤回；准确表述是：**Jev 原始输出的完整召回更高，LLM 的 precision 和表识别更好，共同补全后两者完整召回接近；当前单次下游替换没有收益。**

- 实验入口：`alphasql/runner/linker_ablation.py`，支持 prepare/link/generate/evaluate/all 与检查点续跑。
- 完整机器结果：`results/linker_ablation_287/metrics.json`。
- 逐题 schema：`schema_per_question.csv`；逐题 EX：`execution_per_question.csv`；分歧与完整配对：`paired_cases.csv`；错误分类：`error_analysis.csv`。
- 全量输入、模型响应、Jev 概率、候选和用量日志均在该目录。
- 实际路径审计：`audit.json`，574 个运行、2196 条候选路径、2537 个生成/修订节点全部通过固定 schema 和无 Jev selector 检查。
- 新增 9 个测试，加上相关回归共 **31 个测试通过**。

## 评价口径与实现验证

- 复用此前 evaluator 的 SQL usage 提取、SC 和无序多重集结果签名；保留重复行，浮点 8 位；不是 BIRD 官方完整 evaluator。
- 两侧都强制 upfront linking，属于受控替换对照；不能把新 EX 与旧的可跳过 linker 的 75.61% 直接做因果比较。
- 31 个相关测试通过，包括 schema-only Jev、共享输入、PK/FK、无第二次 linker、空结果、重复行、配对统计和失败题分母。
- 全套历史测试另有 6 个错误，涉及 Windows 临时 SQLite 句柄、已移除方法与缺失 torch；未修改无关模块。
- 原始输入、概率、模型输出、完成/失败检查点、实际 token 用量均保存在实验目录；不保存或输出 API key。
