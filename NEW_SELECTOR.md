# JEV 双置信度 SQL 生成与选择框架

入口：`python -m alphasql.runner.new_selector`。代码位于 `new_seletor` 分支，复用仓库已有的 Direct Candidate Generator、SQLite 校验器和 TypeSafe Jev SDK 适配器。

## 推理流程

```text
自然语言问题 + evidence + Schema
          │
          ▼
Direct LLM 初轮生成（全 Schema / Schema Selection）或读取冻结候选池
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
          │    诊断不明显 → 带去重提示重新探索两条生成路径
          └─ 达到阈值或预算耗尽 → 选最高候选正确率
```

这是逐轮闭环：生成器返回 SQL 及其来源路径、所选表；选择器执行校验后把这些信息连同问题和 Schema 送给 JEV；下一轮的 Schema Selection 或 SQL Revision 使用 JEV 诊断和最佳候选。诊断只是模型判断，不是确定的错误标签。执行错误（例如缺表、缺列）可以在无可执行候选时触发重新选 Schema。`--generator direct` 才会实际使用反馈；`--generator pool` 只按顺序揭示已经冻结的 `.pkl` 候选，因此只能评估停止与选择，不能验证反馈生成效果。

当前 Direct 生成器复用 Alpha-SQL 的 `SchemaSelectionAction`、`SQLGenerationAction`、`SQLRevisionAction`，初轮会尝试全 Schema 和选后 Schema 两路，后续按反馈选择路径；它没有启动 MCTS 搜索。仓库的 DeepEye 多路径实现位于独立的 `deepeye/`，此入口尚未调用它。完整 Alpha-SQL MCTS、Direct 候选生成、DeepEye 基线和本选择器是不同的运行入口。

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
  --samples 3 --max-rounds 3 --max-candidates 24 --limit 5
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

每题输出 `question_id.json`，包含逐轮候选数、覆盖估计、候选正确估计、诊断值、反馈动作、每条 SQL 的来源路径和所选表、最终 SQL、是否不确定及执行摘要。Direct 生成的每轮目录还保存候选 `.pkl` 和下一轮输入的 `feedback.json`。`failures.json` 记录失败题；已有逐题结果默认跳过，可用 `--overwrite` 重跑。增量生成需要模型调用，费用与请求量取决于候选数及轮数。

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
