# DeepEye-SQL 接入记录

完成日期：2026-10-19。代码位于当前 `codex/alpha-sql-jev` 分支。

## 实现内容

- `external/DeepEye-SQL/`：官方源码、文档、模型模板与 MIT License；`UPSTREAM_SOURCE.json` 记录下载归档和上游版本。
- `.venv-deepeye/`：Python 3.12 独立环境，使用远程 LLM/embedding 的 BIRD 运行依赖。
- `requirements-deepeye.txt` / `requirements-deepeye.lock.txt`：依赖范围和本次实际安装版本。
- `deepeye/runner.py`：prepare、doctor、run、单阶段恢复、inspect、export、evaluate。
- `.env.example`：独立 DeepEye 模型/embedding 配置；也支持复用当前 DashScope 设置。
- `tests/test_deepeye_integration.py`：15 项离线接入测试。
- `DEEPEYE_SQL_SETUP.md`：中文运行、复现和数据/评估口径说明。
- `README.md`：新增 DeepEye-SQL baseline 入口。

上游算法直接运行：值检索、动态 few-shot 检索、direct/reversed/value schema linking、DC/skeleton/ICL SQL 生成、八类 checker 修复、基于执行结果与 LLM pairwise 比较的选择器。

上游两个文件包含本地适配：`app/config/config.py` 支持环境变量占位符、BIRD train、question IDs，处理配置复制到自身的情况；`app/dataset/dataset.py` 支持 UTF-8、冻结问题子集和缺省标签字段。

## 验证结果

| 检查 | 结果 |
|---|---|
| 独立环境导入检查 | 通过 |
| `pip check` | No broken requirements found |
| 接入测试 | 15/15 通过 |
| 上游 schema profile 测试 | 2/2 通过 |
| mock LLM/embedding + 真实 SQLite + 值索引的完整流程 | 通过 |
| 动态 few-shot 的索引、masking、preliminary SQL 与检索流程 | mock 模型条件下通过 |
| snapshot 恢复 | 重跑阶段不新增模型调用 |
| token 累计与 pairwise selector | mock API 条件下通过 |
| 真实 BIRD dev 问题预处理 | question ID 0，成功加载 schema 并保存快照 |
| 真实模型端到端 EX / 论文分数复现 | 尚未运行 |

当前准备好的运行目录：`results/deepeye/bird-dev-smoke-qwen35/`，包含 1 题的不可混用输入 manifest、安全配置、dataset snapshot 与预处理日志。

另已准备 `results/deepeye/bird-dev-full-qwen35/` 的 1,534 题冻结 manifest 与配置，并通过配置加载和 dry-run；该全量运行尚未发起模型请求。

测试期间禁止远程网络，LLM 和 embedding 被 mock；查询、检索索引、上游算法调度、checkpoint 和导出评估实际执行。mock 测试中的 EX 不能作为模型性能。

## 运行命令

在项目根目录执行：

```powershell
.\.venv-deepeye\Scripts\python.exe -m deepeye.runner run --run-name bird-dev-smoke-qwen35
.\.venv-deepeye\Scripts\python.exe -m deepeye.runner evaluate --run-name bird-dev-smoke-qwen35
```

正式运行会访问已配置模型和 embedding 服务。输出中的 `candidate_sqls`、`candidate_sqls_with_votes`、`selected_sql` 和 schema evidence 可用于下一阶段 selector / 课程学习数据构建。

## 与论文设置的差别

默认本机模型为 `qwen3.5-35B-a3b`；smoke 使用减小预算并关闭动态 few-shot。full profile 使用官方采样预算和动态 few-shot，但仍需要配置目标模型与官方推荐的数据版本。当前本地训练标签复用现有 BIRD train，未替换为官方新 runbook 推荐的 filtered 标签。

评估同时报告行集合与保留重复行的多重集合口径，空结果计为可执行。当前本地评估不等价于完整官方 BIRD evaluator。Spider2 云数据库凭据与本地 sentence-transformer 模型不在本次运行环境中。
