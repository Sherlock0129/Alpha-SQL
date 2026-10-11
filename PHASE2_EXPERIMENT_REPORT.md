# 第二阶段实验：Alpha-SQL MCTS 与 Jev Schema-aware Selection

## 实验目的

本阶段不再使用第一阶段缓存的 direct candidates，而是在冻结的新集合上实际运行受预算的 Alpha-SQL MCTS，并把“候选生成能力”和“候选选择能力”分开评估：

1. Alpha-SQL MCTS 候选池 + execution self-consistency；
2. Jev schema linking + constrained query-plan beam + SQLite validation + Jev reranking；
3. 固定 MCTS 候选池，仅把 self-consistency 替换为 Jev schema-aware reranker。

测试集在 API 调用前冻结，包括 12 个问题、5 个第一阶段未使用的数据库。所有 gold SQL 均先通过 SQLite 执行检查。SQL slice 包含一到三表 INNER JOIN、单个过滤条件、DISTINCT 和常见聚合；不包含子查询、多过滤条件、GROUP BY、ORDER BY 和 LIMIT。MCTS 使用 `qwen3-coder-flash`、2 个 rollout、最大深度 5，因此这是完整但受预算的 MCTS，而不是论文中的大预算设置。

## 主要结果

| 指标 | MCTS + self-consistency | Jev constrained pipeline |
|---|---:|---:|
| Candidate coverage | 12/12 (100.0%) | 8/12 (66.7%) |
| Candidate executable rate | 100.0% | 100.0% |
| Final execution accuracy | 12/12 (100.0%) | 8/12 (66.7%) |
| Average candidate count | 4.92 | 7.33 |

Jev 自有候选池上的 top-beam 和 self-consistency 都只达到 5/12；Jev reranker 达到 8/12，与该候选池 oracle 相同。因此，reranker 在存在选择空间时修复了 3 个 selection error，并关闭了这个小样本中的全部 selector gap。

固定 MCTS 候选池上的结果为：

| Selector | EX |
|---|---:|
| Execution self-consistency | 12/12 (100.0%) |
| Jev schema-aware reranker | 11/12 (91.7%) |
| Candidate oracle | 12/12 (100.0%) |

这组数据中 MCTS self-consistency 已达到 oracle，因而不存在可被 reranker 提升的 selector gap。Jev 改选一题后反而下降 8.3 个百分点。该结果不能证明 reranker 无效，但能证明当前无监督 Jev 打分不应直接替换 self-consistency；更安全的策略应是学习式融合或只在 self-consistency 低置信度时触发 reranking。

## 误差分析

Jev constrained pipeline 的四个 coverage miss 都具有完整的 gold table/column recall，瓶颈发生在 literal 和 plan beam，而不是 schema 语义召回：

- `1602`：Jev 把 DISTINCT 判为需要，删除了 gold 结果中的重复 manager IDs。
- `1658`：问题中的 `45300` 与 SQLite 中浮点文本 `45300.0` 无法通过当前 exact-value 检索匹配，beam 没有生成过滤条件。
- `3074`：`40000` 是比较阈值而不是数据库中的现存值；当前 literal 候选只来自数据库值，无法生成 `price > 40000`。
- `8864`：同样需要从问题直接保留阈值 `5`，不能依赖该值恰好存在于目标列。

固定 MCTS 池唯一错误是 `1602`。BIRD gold 返回 `employees.reportsTo` 编号，而问题 “To whom ... inform” 在自然语言上也可合理解释为经理姓名。Jev 将返回姓名的 SQL 排在返回编号的 gold-equivalent SQL 之前。这说明仅凭通用语义一致性会偏离 benchmark 的投影约定，reranker 仍需要监督数据、字段描述或输出类型约束。

## 效率

- MCTS：记录到 260 次成功 API 调用，451,330 prompt tokens、72,394 completion tokens，共 523,724 tokens；聚合 API 时间 1,661.1 秒。失败的连接尝试没有 provider usage 记录，货币成本因本地价格表不支持该模型而未估计。
- Jev 原生 pipeline：平均 1.664 秒/题，共 180,929 input tokens 和 35,033 output tokens。
- Jev 对固定 MCTS 池 rerank：平均 0.696 秒/题，共 40,715 input tokens 和 4,779 output tokens。

MCTS 首轮出现 TLS/远端断连，3 个缺失问题随后以单进程重试；成功文件采用断点续跑，没有重复生成其余 9 题。

## 结论与下一步

本阶段最重要的结论是：当前 12 题不足以验证 reranker 对 MCTS selector 的正向提升，因为 baseline 已达到 candidate oracle。下一阶段应直接构建 hard candidate pool：在更大的 BIRD train 集上保留满足 `oracle correct && self-consistency wrong` 的问题，用 train 训练 pairwise reranker、validation 调权重和阈值、完全冻结 test。模型输入应加入全量 schema 描述、候选投影类型和执行结果簇大小，并把 self-consistency 分数作为特征而不是丢弃。

Jev constrained generator 的后续修复应单独进行，不能回写本次主结果：数值 literal 需要规范化 `45300`/`45300.0`，比较阈值必须直接从问题抽取，DISTINCT 应保留 yes/no 两个高概率 beam。修复后应在新的未见集合上复验。

完整机器可读结果位于 `results/llm_vs_jev_phase2/metrics.json`，逐题结果位于 `results/llm_vs_jev_phase2/per_question.csv`，原始候选与 rerank 输出保存在同目录下。
