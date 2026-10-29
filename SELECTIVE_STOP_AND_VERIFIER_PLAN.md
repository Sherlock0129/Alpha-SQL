# 选择性停止与 SQL 内容验证器：实施记录及后续计划

## 固定数据与目标

- 使用已冻结的 HKUST(GZ) `Qwen3.8-27B` 生成器和 TypeSafe `jev-1.14.0` 轨迹。可标注 281 题按数据库互斥划为训练 192 题/13 库、验证 47 题/3 库、测试 42 题/3 库；19 题参考 SQL 无法在附带 SQLite 执行，保留原始轨迹但不参加监督。
- 参考 SQL 仅在离线标注中使用。单一数据库执行结果相同是代理标签，不是语义正确性证明。
- **预先规定经验错误上限 10%**，在验证库首轮状态上计算 `accepted` 内的 `wrong / accepted`，选择满足上限且接受题数最多的候选门槛。池分数低于 0.8 的题进入追加判别，不计入直接接受；低池分数不等于池未覆盖标签。
- 同时报告单侧 95% Wilson 风险上界作为描述性指标。门槛也是在同一验证集上搜索出来的，逐点 Wilson 区间没有校正门槛搜索，更没有覆盖数据库分布变化；此批数据在逐点上界 ≤10% 的约束下也无法接受任何题。因此下面的 10% 是**验证集经验目标**，不是统计保证。

## 第一步：选择性停止

`alphasql/algorithm/selection/selective_stop.py` 计算各门槛的“接受比例—选错比例”曲线，选择并冻结策略。若验证集相邻分数之间没有样本，门槛取两者中点，避免贴着最低已接受分数而无依据地拒绝区间内的新题。运行时 `alphasql/runner/new_selector.py` 把 `stop`、`adjudicate`、`continue` 和 `budget_exhausted` 记录为显式动作；返回值增加 `accepted_stop` 和 `decision_action`。`adjudicate` 只比较已有 SQL，记录建议和用量，保留原 Top-1 作为带 `uncertain=true` 的尽力结果。它不凭追加判断宣称 SQL 正确，也不会自动重新生成。

| 已冻结策略 | 验证集直接接受 | 验证集错 | 单侧 95% 风险上界 | 既有测试库直接接受 | 既有测试库错 |
|---|---:|---:|---:|---:|---:|
| JEV 候选原始分数 ≥ 0.815，池分数 ≥ 0.8 | 32/47 | 3/32 = 9.4% | 21.3% | 25/42 | 2/25 = 8.0% |
| 内容验证器分数 ≥ 0.5734，池分数 ≥ 0.8 | 33/47 | 3/33 = 9.1% | 20.7% | 25/42 | 3/25 = 12.0% |

当前 JEV 停止策略在既有测试库的直接接受结果与旧 0.8 双阈值相同：25/42 接受、其中错 2，**尚无停止准确率提升证据**；真正变化是低池分数被路由到追加判别并标记不确定，而不是自动重新生成。内容验证器的停止门槛在既有测试库超过经验 10% 上限，暂不用于自动接受。测试库虽与训练/验证数据库互斥，却已在项目历史中被多次分析，不能再充当最终盲测。若要求逐点 Wilson 上界 ≤10%，两者都只能全部拒绝；真正的统计保证还需预先固定门槛选择规程、校正多门槛搜索，并在更多独立数据库上检验。

图与策略文件位于 `results/new_selector/hkust_qwen38_jev114_full300/data/selective_stop_v1/`；严格上界的对照位于同级 `selective_stop_wilson_v1/`。运行新入口时需显式传入 `--stop-policy` 和冻结的 `--generator-version hkustgz-qwen38-selector-v3-direct`。未传策略时保留旧入口行为，便于复现历史实验；新实验必须注明所用策略文件。

## 第二步：读取 SQL 内容的验证器

`content_verifier.py` 用问题、证据、候选 SQL、SQL 涉及的 Schema 列说明、执行状态与结果摘要构造稀疏文本/结构特征，不读取标准 SQL、JEV 分数或离线标签作为输入特征。训练集从每题最终候选池去重取可执行候选；当前有 652 条候选（407 正、245 负），来自 191 题，其中 167 对是同题一正一负。仅有 5 个 AST 近似负例，因此语义近错覆盖不足。

优化目标为按问题均衡的候选二分类损失、同题 `softplus(-(s_correct-s_wrong))` 成对排序损失、L2 正则；AST 近似负例在排序项加权 2 倍。固定超参数 `pair_weight=1`、`regularization=0.1`，只用训练库拟合参数。验证库没有进入梯度训练。该稀疏模型是可检查的首版内容基线，不等于已具备强语义理解的 4B 模型。

| 首轮 Top-1（单库代理标签） | JEV 原始分数 | 内容验证器 |
|---|---:|---:|
| 数据库隔离验证集 | 34/47 | 35/47 |
| 既有测试库（非最终盲测） | 30/42 | 34/42 |

验证器改善候选排序，但高分区仍有误判，因此将“排序能力”和“可信停止”分别验收。模型与训练报告在 `results/new_selector/hkust_qwen38_jev114_full300/data/content_verifier_v1/`。运行入口支持 `--verifier-model` 计算分数；只有加载与停止策略 SHA-256 匹配的模型，才能启用其停止策略。当前建议仅离线分析，不启用该策略自动接受。

## 可复现命令

以下命令在项目根目录执行，不触发外部模型调用：

```powershell
$data = "results/new_selector/hkust_qwen38_jev114_full300/data"
.venv/Scripts/python.exe -m alphasql.runner.selector_verifier train --train "$data/split/train.jsonl" --validation "$data/split/validation.jsonl" --output-dir "$data/content_verifier_v1"
.venv/Scripts/python.exe -m alphasql.runner.selector_verifier fit-policy --validation "$data/split/validation.jsonl" --verifier-model "$data/content_verifier_v1/content_verifier.json" --output-dir "$data/selective_stop_v1" --max-error 0.10
.venv/Scripts/python.exe -m alphasql.runner.selector_verifier evaluate-policy --test "$data/split/test.jsonl" --policy "$data/selective_stop_v1/stop_policy_jev_raw.json" --output-path "$data/selective_stop_v1/test_jev_raw.json"
```

新采集使用 `alphasql.runner.new_selector run` 的现有参数，加上：

```text
--stop-policy results/new_selector/hkust_qwen38_jev114_full300/data/selective_stop_v1/stop_policy_jev_raw.json
--generator-version hkustgz-qwen38-selector-v3-direct
```

## 接下来的执行顺序与验收

1. **数据审核**：人工复核训练库中的高分误判、单库同结果但 SQL 语义不同的样本、近似负例。补充多数据库实例或测试套件执行检查；把弱标签与审定标签分开。目标是增加可执行、结构接近、语义错误的同题负例，而非重复采集容易区分的语法错误。
2. **验证器迭代**：固定生成器与候选池版本，新增负例后训练更强的文本/SQL 编码模型；保留 pointwise + pairwise 两损失，并做去掉执行摘要、去掉 Schema、去掉 pairwise 的消融。当前项目 `.venv` 没有 PyTorch/Transformers，系统 Python 的 PyTorch 与 NumPy 版本有兼容警告；先建立隔离且可复现的训练环境。先看数据库隔离验证库的 Top-1、已覆盖题条件选对率、近错子集与 Brier，再重新拟合停止策略。不得用本次 42 题测试库挑模型、挑门槛或调整特征。
3. **停止与动作评估**：另选此前未看过的数据库作为最终盲测，比较同生成预算下旧 0.8 双阈值、新 JEV 停止策略、内容验证器排序和停止、永远停止、永远追加判别。逐项报告接受率、接受后错误率及上界、全部题最终选对率、JEV/LLM 调用数和新增覆盖。追加判别的建议暂不直接替换 Top-1；先用同起点动作对照确认净收益。
4. **升级门槛**：若目标仍是 10% 错误上限，应扩充数据库隔离验证样本并预先固定多门槛选择的统计检验方法；只有经过校正的风险上界和独立数据库评估都过关，才把内容验证器停止策略作为默认线上动作。Qwen 4B 微调留到数据质量与上述基线明确后再决定。

方法依据：[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) 的拒绝预测风险—覆盖思路；[LEVER](https://proceedings.mlr.press/v202/ni23b.html) 的输入程序与执行结果验证；[GradeSQL](https://arxiv.org/abs/2606.30851) 的候选生成和执行监督；[CHASE-SQL](https://proceedings.iclr.cc/paper_files/paper/2025/hash/974ff7b5bf08dbf9400b5d599a39c77f-Abstract-Conference.html) 的成对选择。这里的稀疏模型和风险门槛是本项目实现，并非这些论文原模型的复现。
