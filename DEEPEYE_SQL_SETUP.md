# DeepEye-SQL 本地接入与复现

本项目已引入 [DeepEye-SQL 官方实现](https://github.com/HKUSTDial/DeepEye-SQL)，源码位于 `external/DeepEye-SQL`。当前接入面向 BIRD / SQLite，运行入口为 `python -m deepeye.runner`，支持 PowerShell，无需 Bash 或 uv。

源码版本记录在 `external/DeepEye-SQL/UPSTREAM_SOURCE.json`，官方 MIT LICENSE 原样保留。官方 `master` 下载归档 SHA256 为 `c8f78edce13fb69e13369961fe634a8d17e504a78f1c59da93f77188b44e7d17`，下载时 GitHub API 返回 HEAD `252ab45a7adf9f88fb93d1ff27ca019d20f0449a`。本地改动仅涉及配置和 BIRD 数据加载，算法使用上游实现。

## 环境与配置

已创建独立 Python 3.12 环境 `.venv-deepeye`，安装 BIRD 远程 LLM/embedding 所需依赖，完整安装版本保存在 `requirements-deepeye.lock.txt`。重建环境时使用本机 Python 3.12：

```powershell
py -3.12 -m venv .venv-deepeye
.\.venv-deepeye\Scripts\python.exe -m pip install -r requirements-deepeye.lock.txt
.\.venv-deepeye\Scripts\python.exe -m deepeye.runner doctor
```

`.env.local` 或 shell 可配置：

```dotenv
DEEPEYE_MODEL=qwen3.5-35B-a3b
DEEPEYE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DEEPEYE_API_KEY=
DEEPEYE_EMBEDDING_MODEL=text-embedding-v4
DEEPEYE_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DEEPEYE_EMBEDDING_API_KEY=
```

shell 优先于 `.env.local`，后者优先于 `.env`。独立配置缺省时复用现有 `OPENAI_*` / `DASHSCOPE_*` 和 `EMBEDDING_*`；LLM 模型优先使用 `DEEPEYE_MODEL`，然后 `LLM_MODEL`、`PREPROCESS_MODEL`，默认 `qwen3.5-35B-a3b`。所有阶段共用该模型。新建配置中的模型名称是本机设置，不能据此声称复现了论文 Qwen3.6-27B 的分数。

配置文件使用 `${DEEPEYE_API_KEY}` 等占位符，密钥只在内存解析，日志会遮盖已加载密钥。`doctor` 仅报告依赖、文件和密钥是否配置，不访问远程 API。

当前安装的是 CPU 版 PyTorch；远程 embedding 与 LLM 由服务端执行，本地相似度检索使用 CPU。需要本地 embedding 模型时，可另行安装官方 `pyproject.toml` 的 `sentence-transformers` 等依赖，并调整配置。本次未配置 Spider2 的 Snowflake/BigQuery 服务；官方源码仍保留其入口。

## 运行流程

```text
preprocess
  → index-values
  → retrieve-values
  → index-few-shot / prepare-few-shot（启用时）
  → link：direct + reversed + value linking
  → generate：divide-and-conquer + skeleton + ICL
  → revise：syntax / join / order / time / select / max-min / null / result checkers
  → select：执行结果簇 + pairwise LLM 比较
  → predictions.json + candidates.jsonl
```

`smoke` 使用 linking 各 1 个样本、DC/skeleton 各 1 个样本、checker/selector 各 1 个样本，默认关闭 few-shot 与 ICL。`full` 使用官方模板的 12/12 linking、12/12/12 generation、16 checker/selection 预算并开启动态 few-shot。完整 profile 的模型、并行度、batch size 和超时仍使用本机配置。若在 smoke 中指定 `--few-shot`，检索索引最多使用 16 条训练样本、每题检索 3 个示例。

### 已准备的 1 题 BIRD dev smoke

当前 `bird-dev-smoke-qwen35` 已冻结 1 题，并已完成真实数据库预处理：

```powershell
.\.venv-deepeye\Scripts\python.exe -m deepeye.runner inspect --run-name bird-dev-smoke-qwen35
.\.venv-deepeye\Scripts\python.exe -m deepeye.runner run --run-name bird-dev-smoke-qwen35 --dry-run
.\.venv-deepeye\Scripts\python.exe -m deepeye.runner run --run-name bird-dev-smoke-qwen35
.\.venv-deepeye\Scripts\python.exe -m deepeye.runner evaluate --run-name bird-dev-smoke-qwen35
```

`run` 会调用真实模型及 embedding 服务。预处理和成功的模型阶段均有快照，可直接重跑以继续未完成阶段。

### 自选问题与训练候选池

```powershell
.\.venv-deepeye\Scripts\python.exe -m deepeye.runner prepare --run-name train-pilot --split train --question-ids 1742,7854 --max-samples 2 --no-few-shot
.\.venv-deepeye\Scripts\python.exe -m deepeye.runner run --run-name train-pilot
```

为防止训练题自己的 gold SQL 被动态 few-shot 检索出来，train 目标必须使用 `--no-few-shot`。这是一种候选池构建设置，和官方 dev few-shot 实验的设定不同。

可通过 `--ids-file config/llm_jev_300_split.json` 读取 `evaluation_ids`，或提供纯 question ID 列表。注意 `smoke` 默认只取前 3 题；需要全部 IDs 时使用 `--profile full --no-few-shot`（train）或明确传入 `--max-samples`。

### BIRD dev 完整流程

```powershell
.\.venv-deepeye\Scripts\python.exe -m deepeye.runner prepare --run-name bird-dev-full-qwen35 --profile full --split dev
.\.venv-deepeye\Scripts\python.exe -m deepeye.runner run --run-name bird-dev-full-qwen35 --dry-run
.\.venv-deepeye\Scripts\python.exe -m deepeye.runner run --run-name bird-dev-full-qwen35
.\.venv-deepeye\Scripts\python.exe -m deepeye.runner evaluate --run-name bird-dev-full-qwen35
```

full profile 首次运行会为现有 BIRD train 建 few-shot 索引。当前复用的是本地已有训练标签；官方新 runbook 推荐 `birdsql/bird23-train-filtered`。数据版本与模型不同的实验应独立报告，不能直接作为论文分数复现。

### 单阶段执行

```powershell
.\.venv-deepeye\Scripts\python.exe -m deepeye.runner stage --run-name bird-dev-smoke-qwen35 --stage link
.\.venv-deepeye\Scripts\python.exe -m deepeye.runner stage --run-name bird-dev-smoke-qwen35 --stage generate
.\.venv-deepeye\Scripts\python.exe -m deepeye.runner export --run-name bird-dev-smoke-qwen35
```

阶段输入依赖前面的快照。上游阶段报告的未完成问题会转为非零退出，并记录日志，防止把部分失败误报为成功。更改输入、预算、模型或端点需创建新 run name；同名实验可恢复已成功的 checkpoints。

## 产物与现有项目的连接

所有运行结果在 `results/deepeye/<run-name>/`：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 冻结 question IDs、配置与输入哈希、模型/端点信息 |
| `config.toml` | 不含明文密钥的运行配置 |
| `*.snapshot`、`*.snapshot.data/` | 各阶段官方结构化快照；具体目录由快照 manifest 指定 |
| `logs/*.log`、`stages.jsonl` | 阶段输出、退出码和墙钟时间 |
| `predictions.json` | question ID → 最终 SQL |
| `candidates.jsonl` | question ID、db_id、candidate_sqls、selected_sql、schema_linking 和累计 token/时间 |
| `metrics.json` | Candidate coverage、最终 EX、SQL 可执行率及逐题结果 |

`candidates.jsonl` 使用当前项目的候选集合字段命名，可以用于后续 selector 和错误感知课程学习实验。`candidate_sqls` 来自 revision 后的去重 SQL；`candidate_sqls_with_votes` 保留原始采样频次。阶段原始输出也保存在官方快照中。比较 self-consistency 时应读取未去重的原始列表，避免丢失票数。

本地评估同时提供 set EX（与上游 evaluation.py 的行集合比较口径一致）和 multiset EX（保留重复行）。空结果属于可执行 SQL，Gold 执行失败会单独报告。两种口径都不是完整官方 BIRD evaluator，正式 benchmark 结果需使用官方评估。

## 验证范围

```powershell
.\.venv-deepeye\Scripts\python.exe -m unittest discover -s tests -p test_deepeye_integration.py -v
```

21 项接入测试全部通过，另有 7 项分析测试（主环境运行）、4 项恢复测试和 2 项上游测试，共 34 项。离线测试覆盖：密钥占位符解析、缺少配置报错、冻结子集、隐式 train 题号、输入/模型变更检测、train few-shot 泄漏防护、空结果、普通与动态 few-shot 全流程、断点恢复、候选导出、set/multiset 评估差异、多次采样 token 累计、API 用量记录、SC 票数、限流时不重采成功样本、计费错误立即停止、可恢复检查点隔离，以及包含真实执行结果的 pairwise selection。

测试中 LLM 与 embedding 均 mock，网络连接被禁止；数据库、检索索引和官方 pipeline 实际执行。其正确率仅用于软件验证，不能视为 DeepEye-SQL 的真实任务性能。300 题真实运行另见 `DEEPEYE_SQL_300_EXPERIMENT.md` 和相应运行目录。

## 冻结 300 题实验

API 实测模型 ID 为小写 `qwen3.5-35b-a3b`；大写 `B` 的旧 ID 在当前端点返回 404。新实验已在 `.env.local` 中使用小写模型名。Qwen3.5 非流式请求显式设置 `enable_thinking=false`。

```powershell
.\.venv-deepeye\Scripts\python.exe -m deepeye.runner prepare --run-name bird-train-300-qwen35-p32 --profile full --split train --ids-file config/llm_jev_300_split.json --max-samples 300 --workers 32 --no-few-shot
.\.venv-deepeye\Scripts\python.exe -u -m deepeye.runner run --run-name bird-train-300-qwen35-p32
.\.venv-deepeye\Scripts\python.exe -m deepeye.runner evaluate --run-name bird-train-300-qwen35-p32
.\.venv\Scripts\python.exe -m deepeye.analyze --run-name bird-train-300-qwen35-p32
```

最后一步为纯离线分析，使用主环境中已安装的 SQLGlot。输出 `REPORT.md`、`analysis.json`、`error_analysis.jsonl`；只有题号完整匹配 manifest 才生成最终报告。`api_usage.jsonl` 逐次记录成功 Chat API 的 usage，不存提示词或密钥。Embedding 用量未纳入，因此不能把它当成全部计费用量。
