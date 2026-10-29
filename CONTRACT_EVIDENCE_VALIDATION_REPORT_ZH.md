# 候选盲题意约束与定向 JEV 核查：数据库隔离验证集

## 范围与协议

使用 `selection_v2_300` 固定首轮候选池中的验证集：6 个数据库、60 题；训练、验证、测试数据库互不重叠。候选生成、原始 JEV 分数、执行缓存均不变。验证集有 37 个按**同一数据库参考 SQL 执行等价**标记的已覆盖池。该自动标签只用于离线评估，不进入新增 JEV 请求。

比较三种选择：

1. **原 JEV**：有效候选中取 `candidate_raw` 最高者。
2. **结构与执行证据**：复用仅在训练数据库拟合的执行特征排序器，以原 JEV 分数为锚点，修正幅度上限 0.1；不增加 JEV 请求。特征含 SQL AST、原库执行形状和结果簇。这个模型的幅度上限曾在本验证集选择，因此这里是**既有验证集上的诊断对照**，不是独立的新验证成绩。
3. **两阶段 JEV**：先对问题、evidence 和由它们检索的 Schema 发一次**候选盲**请求，选择统计指标、结果重复、输出列数、总体范围和 NULL 约定；候选 SQL、结果、生成路径、原 JEV 分数均不在这一请求中。仅当候选在已确定的约束上存在结构或结果差异时，第二次请求核查每条候选的该项约束。以方法 2 的排序为锚，JEV 定向分最多修正 0.15，改选还须超过 0.08 的预设优势。这些数值在此次在线采集与离线结果评估前固定，没有事后调参。

实现和原始数据见 [实验入口](/D:/University/Alpha-SQL/alphasql/runner/contract_evidence_experiment.py)、[逐题结果](/D:/University/Alpha-SQL/results/new_selector/selection_v2_300/contract_evidence/validation_report.jsonl)及[采集目录](/D:/University/Alpha-SQL/results/new_selector/selection_v2_300/contract_evidence/collection)。第一阶段用 `Choice`，第二阶段用 `Noul`；后者数值是 JEV 判断分，不当作已校准正确概率。没有构造微型区分数据库，也没有使用 SQL/Python 双范式求解。

## 自动标签结果与成本

| 方法 | 全部题参考执行 Top-1 | 已覆盖池 Top-1 | 相对原 JEV 挽救／误伤 | 改选题 | 无有效候选弃判 | 额外 JEV 调用／token |
|---|---:|---:|---:|---:|---:|---:|
| 原 JEV | 33/60 | 33/37 | 0/0 | 0 | 2 | 0／0 |
| 结构与执行证据 | 33/60 | 33/37 | 0/0 | 3 | 2 | 0／0 |
| 两阶段 JEV | 33/60 | 33/37 | 0/0 | 3 | 2 | 110／278,487 |

两阶段的 **60 次题意请求**花费输入 120,152、输出 13,047 token；**50 次定向核查请求**花费输入 139,838、输出 5,450 token。总计输入 259,990、输出 18,497 token。第二阶段相对结构与执行证据方法**没有改变任何一题的最终选择**。三题改选均来自本地排序器：`coinmarketcap/6295` 从一个参考执行不一致的 SQL 改到另一个参考执行不一致的 SQL；`public_review_platform/3854` 与 `world_development_indicators/2128` 则在两个参考执行一致的 SQL 之间切换。

这里的“弃判”仅指池中没有有效 SQL 的两题；当前策略对其余 58 题没有基于置信度的主动弃判。不要把 17 条语义不确定标注误读成实际运行时已经弃判。

## 静态语义复核

对 60 题逐条读取题目、evidence 和三种方法实际选中的 SQL，按“可确认正确／可确认错误／仍不确定／无有效候选”标注。逐题原因保存在[复核清单](/D:/University/Alpha-SQL/results/new_selector/selection_v2_300/contract_evidence/semantic_review_provisional.json)。这是**助手完成的静态语义审查**，不是独立领域专家的盲审，也没有证明跨数据库语义等价；不确定样本不计入条件准确率。

| 方法 | 可确认正确 | 可确认错误 | 可判定部分选对率 | 语义不确定 | 无有效候选 |
|---|---:|---:|---:|---:|---:|
| 原 JEV | 35 | 6 | 35/41（85.4%） | 17 | 2 |
| 结构与执行证据 | 35 | 6 | 35/41（85.4%） | 17 | 2 |
| 两阶段 JEV | 35 | 6 | 35/41（85.4%） | 17 | 2 |

**35/41 不能外推为 60 题语义准确率。**分母只包括静态证据足以判断的题，且复核发生在自动评估之后。三题改选中的两个候选经静态审查均可接受，因此该复核也没有显示两阶段 JEV 的增益。

## 错因审计

- `synthea/1426`：候选盲阶段正确识别需要多列输出，`output_shape=multiple` 的选择概率为 0.97；原 JEV 首选 SQL 却以 `UNION` 把手术和药物并为单列。定向 JEV 仍给首选 SQL 的输出约束 0.77、另一条双列 SQL 0.82，差距不足以克服原始分数。这是**明确结构反证被软分数淹没**的案例。
- `airline/5835`：参考要求包含代码的承运人描述，首选 SQL 输出承运人名称。题目本身只问“哪家航空公司”，静态语义审查认为名称可以回答；参考执行标签的错选不能直接解释为语义错误。
- `airline/5890`：首选 SQL 排除缺失的实际飞行时长，才找最短值；参考 SQL 未排除 NULL，SQLite 将 NULL 排在最前。首选 SQL 的参考执行标签为错，但语义上合理。候选盲阶段把 NULL 约定判为 `unspecified`，所以没有触发 NULL 定向核查。
- `superstore/2386`：首选 SQL 只查 `central_superstore`，遗漏其他分区；自动标正的 `east_superstore` 候选也只查一个分区，恰好返回与参考相同的最大值。当前题意约束没有“跨分区覆盖”一项，因此两阶段方法无法发现这种错误。

## 结论与下一轮实验约束

**此次验证没有观察到选择精度提升。**新增 110 次 JEV 调用未产生相对本地证据排序的改选。下一轮若继续，应先针对结构上可证伪的输出列数、统计对象及分区覆盖建立确定性约束，再让 JEV 处理真正需要语义判断的争议；对 NULL、重复行和输出格式保留不确定或多答案标签。不要用本验证集继续调整修正幅度后再宣称独立提升。

本验证集以及其中的错误案例已在此前多轮研究中被查看，本地排序器上限也曾用它选择。要检验泛化，需要冻结新规则，在另外一批**未用于规则设计和门槛拟合的数据库**重采样固定候选池，并由独立复核者先审定争议题。当前报告只支持“此版本两阶段 JEV 成本较高且未见收益”的结论。

## 复现

```powershell
.\.venv\Scripts\python.exe -m alphasql.runner.contract_evidence_experiment evaluate --input results/new_selector/selection_v2_300/labels/split/validation.jsonl --split-file results/new_selector/selection_v2_300/protocol/splits.json --split validation --execution-dir results/new_selector/selection_v2_300/execution_evidence --collection-dir results/new_selector/selection_v2_300/contract_evidence/collection --model results/new_selector/selection_v2_300/fit/model.json --semantic-review results/new_selector/selection_v2_300/contract_evidence/semantic_review_provisional.json --output results/new_selector/selection_v2_300/contract_evidence/validation_report.json
```

验证：`python -m pytest tests/test_contract_evidence_experiment.py -q -p no:cacheprovider --basetemp results/new_selector/selection_v2_300/contract_evidence/pytest_temp`，3 项通过。测试覆盖候选盲输入、只触发有分歧的约束，以及新增 JEV 请求不含原 JEV 分数和自动正确性标签。
