# 独立 Schema Linker 与最终 EX 的受控替换实验

## 实验问题

1. 给定完全相同的 Question、Evidence、Full Schema，Jev 的独立表列判断是否优于原 Qwen Schema Linker？
2. 保持 Qwen SQL 生成、MCTS 预算和 execution self-consistency 一致，仅替换 linker，candidate coverage 和最终 EX 是否提高？

## 冻结与输入

- 题集：原 300 题实验中两路共同完成的 287 题，直接读取既有 `per_question.csv` 的 question IDs；不重新挑题。
- 数据：原 `data/preprocessed/llm_jev_300/train/tasks.pkl`；使用 task 的全量 schema，包含检索后的样例值。
- 共享 schema payload：表名、列名、类型、meaning、description、value_description、PK/FK，每列最多 5 个样例值。
- 两侧逐题共享同一语义 payload，保存 SHA-256；LLM 模板用 JSON 展示 schema，Jev 的 state 使用同一 JSON 对象。
- Gold SQL 仅离线评价使用；调用模型前将 task.sql 清空。

## Linker 对照

| 项目 | LLM | Jev |
|---|---|---|
| 模型 | qwen3-coder-flash | 环境配置的 JEV_MODEL |
| 输入 | 同一 Question/Evidence/Full Schema | 同一 Question/Evidence/Full Schema |
| 决策 | 原 schema_selection 模板，输出表列 JSON | 每张表、每个列独立 Noul |
| 阈值 | 模型直接选择 | 固定原配置 0.5，不在 287 题上调参 |
| 共同后处理 | 原 PK/FK 补全 | 同一 PK/FK 补全 |
| Jev Join Choice / SQL intent | 不使用 | 不使用 |

分别评价 raw 和 completed schema 的 macro P/R/F1、完整召回率、exact match、平均表列数、额外/遗漏表列数。
空选择、未知表列、无效 JSON 与 API 异常均显式失败，不回退全 schema 掩盖错误。

## 下游成对实验

```text
LLM Linker ─→ 相同 PK/FK 补全 ─→ Qwen/MCTS ─→ execution self-consistency
Jev Linker ─→ 相同 PK/FK 补全 ─→ Qwen/MCTS ─→ execution self-consistency
```

所有搜索分支强制使用该题 upfront schema；两组都关闭后续 schema-selection，保留 question rephrase、column values/functions、SQL generation/revision。
因此这是 mandatory-linker MCTS 的受控对照，不与旧的“允许跳过 linker”的 75.61% 直接进行因果比较。

- MCTS rollouts=2，max_depth=5，exploration constant=1.414。
- generation/revision 每次请求采样 2 条，temperature=0.8、top_p=0.8。
- 两组 max_tokens=4096，validation attempts=4。
- 每题两侧使用同一个 Python MCTS 随机种子：42 + question_id；远程生成本身仍有随机性。
- 相同样本/搜索上限；实际调用次数和候选数可能因可执行性重试与去重而不同，需报告用量。
- Jev compiler、聚合/排序/Join 决策、reranker 均不参与下游。
- SQL 生成通过 schema context 约束；继承原生成器行为，不新增 AST 硬过滤。

## 评价

复用原评价脚本的 SQL usage 提取、execution self-consistency 和结果签名。签名是无序多重集，保留重复行、浮点四舍五入到 8 位；不是 BIRD 官方完整 evaluator。

- candidate coverage、最终 EX、可执行率、candidate 数。
- Jev 修复/回归数、问题级 paired bootstrap 95% CI、exact McNemar。
- 错误拆分：schema recall miss / schema 完整但 generation coverage miss / 正确候选存在但 SC 选错。
- 独立 linking 和 downstream generation 的时延与实际 token 用量。
- 单次运行不代表跨种子结论；需多种子复验才能量化生成随机性。
- 额外按实际 DDL 上下文相同 / schema 元素相同但顺序不同 / schema 元素不同分层；前两类的 EX 差异不能直接归因于表列召回提升。

## 实施状态

新增 `alphasql/runner/linker_ablation.py`，全部工作在 `codex/alpha-sql-jev`。
两组独立 linking 与下游生成均完成 287/287；31 个相关单元测试通过。
最终 LLM EX 为 209/287，Jev EX 为 199/287，修复 11、回归 21；完整分析见 `LINKER_ABLATION_287_RESULTS.md`。
全量结果及检查点位于 `results/linker_ablation_287/`，最终无未解决失败。

全测试发现的既有问题：Windows 临时 SQLite 文件句柄未关闭、旧测试调用已移除的生成方法、当前环境缺少 torch；未修改这些无关模块。
