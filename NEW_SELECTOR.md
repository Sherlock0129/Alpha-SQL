# JEV 双置信度 SQL 生成与选择框架

入口：`python -m alphasql.runner.new_selector`。代码位于 `new_seletor` 分支，复用仓库已有的 Direct Candidate Generator、SQLite 校验器和 TypeSafe Jev SDK 适配器。

## 推理流程

```text
自然语言问题 + evidence + Schema
          │
          ▼
Direct LLM 初轮多提示路径生成或读取冻结候选池
          │
          ▼
SQL 规范化去重 → SQLite 执行校验（空结果仍有效）
          │
          ▼
JEV 同时回答两类正确性问题和四类诊断问题
  pool_covered: 候选集中是否至少有一条完全正确？
  candidate_i: 第 i 条 SQL 是否完全正确？
  gap_schema / projection / filters / logic: 若有错，可能错在哪里？
          │
          ├─ 覆盖或候选置信度低 → 根据诊断决定下一轮动作
          │    schema 问题 → 从全 Schema 重新选表/列 → 重新生成 SQL
          │    投影/过滤/逻辑问题 → 带原 SQL 与所选 Schema 修订
          │    诊断不明显 → 带去重提示重新探索多条生成路径
          └─ 达到阈值或预算耗尽 → 选最高候选正确率
```

这是逐轮闭环：生成器返回 SQL 及其来源路径、所选表；选择器执行校验后把这些信息连同问题和 Schema 送给 JEV；下一轮的 Schema Selection 或 SQL Revision 使用 JEV 诊断和最佳候选。诊断只是模型判断，不是确定的错误标签。执行错误（例如缺表、缺列）可以在无可执行候选时触发重新选 Schema。`--generator direct` 才会实际使用反馈；`--generator pool` 只按顺序揭示已经冻结的 `.pkl` 候选，因此只能评估停止与选择，不能验证反馈生成效果。

当前 Direct 生成器保留 Alpha-SQL 的 Divide-and-Conquer 提示及 `SchemaSelectionAction`、`SQLGenerationAction`、`SQLRevisionAction`，并加入两个独立的单次 LLM 提示路径：`skeleton` 先构造 SQL 子句骨架，优先使用已选 Schema；`query_plan` 从目标结果逆推扫描、连接、过滤与聚合，使用完整 Schema。Schema 诊断触发重新选 Schema 后，在新 Schema 上运行这些路径；SQL 逻辑诊断触发原修订动作和 `contrastive_revision` 语义对照重建。每条 SQL 保存来源路径，方便按路径比较候选覆盖率、质量和调用成本。它没有启动 MCTS 搜索，也没有调用仓库独立的 `deepeye/` 实现；该提示设计不是 CHASE-SQL 或 DeepEye-SQL 的完整复现。

默认每条新增提示路径生成 1 个候选；原 D&C 路径仍由 `--samples` 控制。可用 `--prompt-paths skeleton query_plan --prompt-path-samples 1` 调整，或在 `--prompt-paths` 后直接接下一个选项以关闭新增路径，做同一生成器的消融。新增路径产生的 SQL 由选择器统一执行校验；一次提示路径失败不会丢弃其他路径的候选。

推理前把 `Task.sql` 清空。JEV 只看到问题、证据、Schema、候选 SQL 和执行摘要；标准答案只在离线 `label` 阶段使用。无可执行 SQL 时，候选集覆盖值固定为 0，不能让校准器把它抬高。执行成功但返回空行是有效候选。

`coverage_raw` 和 `candidate_raw` 是 JEV 的原始判断，**并非已经验证的统计概率**。训练后加载 `dual_calibrator.json` 才输出 `coverage_probability` / `candidate_probability`，校准质量以独立验证数据库上的 Brier 分数检验。候选集覆盖概率和单候选正确概率分别拟合，不能把 `max(candidate_probability)` 当作前者。

## 运行

先用已有预处理流程得到包含 `table_schema_dict` 的 `tasks.pkl`，在 `.env.local` 配好 `TYPESAFE_API_KEY` 和直接生成 SQL 所需的模型密钥。以下用仓库现有的 train 候选任务举例；首次试验建议先 `--limit 5`。

```powershell
python -m alphasql.runner.new_selector run `
  --tasks-file-path data/preprocessed/schema_pool_300/train/tasks.pkl `
  --database-root-dir data/bird/train/train_databases `
  --output-dir results/new_selector/train `
  --generator direct --model qwen3-coder-flash `
  --samples 3 --prompt-path-samples 1 `
  --max-rounds 3 --max-candidates 24 --limit 5
```

也可以复用已有的候选 `.pkl`，每轮揭示 `--per-round` 条，以固定候选池测量 JEV 的自适应停止行为：

```powershell
python -m alphasql.runner.new_selector run `
  --tasks-file-path data/preprocessed/schema_pool_300/train/tasks.pkl `
  --database-root-dir data/bird/train/train_databases `
  --output-dir results/new_selector/frozen_train `
  --generator pool --candidate-dir results/direct_schema_pool_300/train `
  --per-round 8 --max-rounds 3 --limit 5
```

每题输出 `question_id.json`，包含逐轮候选数、覆盖估计、候选正确估计、诊断值、反馈动作、每条 SQL 的来源路径和所选表、最终 SQL、是否不确定及执行摘要。Direct 生成的每轮目录还保存候选 `.pkl`、路径异常 `warnings.json` 和下一轮输入的 `feedback.json`。`failures.json` 记录失败题；已有逐题结果默认跳过，可用 `--overwrite` 重跑。增量生成需要模型调用，费用与请求量取决于候选数及轮数。

后续生成器实验可依次加入：多个可区分的 Schema 假设（检验链接错误是否被补救）、问题实体到数据库值的检索（检验 WHERE 字面值错误）、按 JEV 诊断分配路径和候选预算（检验每次额外模型调用的收益）。这些仍是待实现和验证的方向；当前多提示路径尚未在完整 BIRD/Spider 实验中证明优于原 Direct、CHASE-SQL 或 DeepEye-SQL。

## 离线标注、课程训练与校准

只用训练集参考 SQL 执行结果产生标签，按**数据库**划分训练/验证，避免同库泄漏。`label` 产物仅含 JEV 原始值及正确性标签，不含问题文本、候选 SQL 或 gold SQL。

```powershell
python -m alphasql.runner.new_selector label `
  --results-dir results/new_selector/train `
  --data-path data/bird/schema_pool_300/train.json `
  --database-root-dir data/bird/train/train_databases `
  --output-path results/new_selector/train_labels.jsonl

python -m alphasql.runner.new_selector split `
  --labels results/new_selector/train_labels.jsonl `
  --output-dir results/new_selector/split --seed 42

python -m alphasql.runner.new_selector train `
  --train-labels results/new_selector/split/train_labels.jsonl `
  --validation-labels results/new_selector/split/validation_labels.jsonl `
  --output-dir results/new_selector/model
```

训练对两种判断分别拟合 Logistic head。样本按训练题的原始 `coverage_raw` 从高到低排序：初期只见容易题，前 60% epoch 逐渐扩大到全体题；每题对损失贡献相同，避免候选多的题支配训练。验证集选择 Brier 分数最好的 epoch。报告同时保存原始/校准 Brier、候选覆盖率、原始/校准 Top-1 EX。`split_manifest.json` 保存输入哈希与验证数据库。训练时没有新增 SQL 候选，也没有用 gold SQL 作为推理特征。

使用已训练校准器推理时给 `run` 增加：

```powershell
--calibrator results/new_selector/model/dual_calibrator.json
```

当前训练部分针对 **JEV 判断的概率校准和单候选选择**；它不更新 SQL 生成大模型的权重。若要训练生成器本身，需要单独加入 SFT/RL 数据路径、固定候选生成预算和独立评测，不能把这里的校准结果当作生成模型的改进。

## 逐轮训练数据构造（v2）

新运行结果在顶层保存原问题、evidence、完整数据库 Schema、生成时实际使用的 Schema、生成器配置及代码/提示模板哈希。每轮保存截至当时的**全部**候选 SQL、来源路径、所选表列、执行摘要、六类 JEV 原始输出、候选分数、已执行反馈和下一轮反馈。快照独立于最终轮的重评分数；不含标准答案。运行时用 `--generator-version` 为同一批模型与提示配置指定固定标签；同时保存自动计算的 `source_sha256`，后续数据划分拒绝混入不同源码版本。

下面的前两步只读本地运行结果与训练集标准 SQL，**不会调用模型**。参考 SQL 只进入离线标注函数，不写入富标签或动作输入；输出中的执行等价标签仍只是单一数据库上的近似正确性。

```powershell
python -m alphasql.runner.selector_data label `
  --results-dir results/new_selector/train `
  --data-path data/bird/train/train.json `
  --database-root-dir data/bird/train/train_databases `
  --output-path results/new_selector/data/rich.jsonl `
  --calibration-output results/new_selector/data/calibration.jsonl

python -m alphasql.runner.selector_data split `
  --labels results/new_selector/data/rich.jsonl `
  --output-dir results/new_selector/data/split `
  --validation-fraction 0.15 --test-fraction 0.15 --seed 42
```

`rich.jsonl` 每题包含各轮池覆盖 EX 标签、每条候选的 EX 标签、可执行近似错误 SQL 的 AST 编辑距离标签、四类**弱**结构诊断多标签、已发生的单路径反馈收益、以及分开的 `generation_hard` / `selection_hard` / `covered_easy` 分层。近似错误 SQL 必须实际执行成功、结果不等于参考结果，且 AST 编辑距离不超过阈值。`split` 按数据库互斥地产生 train、validation、test，并为现有校准器另导出不含 SQL 文本的 `*_calibration.jsonl`。候选校准课程先看已覆盖池，按正确候选与错误候选的分数差从易到难；覆盖校准课程在早期同时看到正负池，避免全是覆盖正例。

结构诊断不能作为人工真值。先抽样核查；此文件**包含参考 SQL**，只应留在离线标注环境，不能送入推理或动作探测：

```powershell
python -m alphasql.runner.selector_data audit `
  --labels results/new_selector/data/split/train.jsonl `
  --data-path data/bird/train/train.json `
  --output-path results/new_selector/data/diagnostic_audit.jsonl `
  --per-group 20
```

要为反馈动作造标签，先从训练分区的三种难度层抽相同的首轮状态。清单里的问题、Schema 和 SQL 来自原始运行结果，**不含任何标准答案或离线正确性标签**。同一个 `state_id` 的四个动作共享初始候选池哈希、生成器代码哈希和候选/模型调用预算。

```powershell
python -m alphasql.runner.selector_data prepare-actions `
  --labels results/new_selector/data/split/train.jsonl `
  --results-dir results/new_selector/train `
  --output-path results/new_selector/data/action_states.jsonl `
  --per-stratum 50 --max-new-candidates 4 --max-model-calls 3
```

`probe-actions` 会调用配置的 SQL LLM 和 JEV 服务；仅在这些外部调用及其传输的问题、Schema、SQL 已获授权后运行。它分别从同一首轮状态执行 `relink`、`revise`、`diversify`、`adjudicate`。普通 LLM 请求设定每动作调用上限并记录实际 token；JEV 调用另计入上限。每个动作结果独立保存，失败可续跑。`max_tokens` 可用作事后上限校验；API 调用前无法保证实际输入 token 不超过该值。随后离线计算四个动作的候选覆盖增量、最终选择正确性增量和成本；只有四臂齐全且 token 用量完整的状态才有 `best_observed_action`。

```powershell
python -m alphasql.runner.selector_data probe-actions `
  --manifest results/new_selector/data/action_states.jsonl `
  --database-root-dir data/bird/train/train_databases `
  --output-dir results/new_selector/data/action_runs

python -m alphasql.runner.selector_data label-actions `
  --arms results/new_selector/data/action_runs/arms.jsonl `
  --labels results/new_selector/data/split/train.jsonl `
  --data-path data/bird/train/train.json `
  --database-root-dir data/bird/train/train_databases `
  --output-path results/new_selector/data/action_labels.jsonl
```

动作标签是**一次独立探测的观察结果**，受采样随机性影响。后续训练策略前应为关键状态重复探测、报告每动作成功率，并在完全独立的数据库上验证。旧版 `schema_version=1` 结果缺少逐轮六类输出，不能无损转换为这套富数据；需要重新运行或将缺失项明确标为未知。
