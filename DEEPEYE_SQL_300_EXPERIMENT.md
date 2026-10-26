# DeepEye-SQL 300 题扩展实验协议

港科广 Qwen3.6 混合模型续跑已完成 300/300 题。结果：候选覆盖 225/300、最终 multiset EX 216/300、同池 SC EX 214/300；详细统计与逐题错误见 `results/deepeye/bird-train-300-hkust-qwen36-resume/REPORT.md` 和 `error_analysis.jsonl`。生成阶段 7200 条输出逐题去重后只有 576 条，属于本次实验的主要瓶颈。下文最初的欠费状态只描述旧 Qwen3.5 运行。

实验目录：`results/deepeye/bird-train-300-qwen35-p32/`。

状态：2026-10-19 真实推理被服务端 `400 Arrearage` 账户欠费错误阻断，所有调用已停止。索引与检索完成，保留 286 条可靠 linking，14 题待补跑；生成/选择尚未完成，**没有最终 EX**。详情见运行目录 `STATUS.md`，未完成时不会生成最终 `REPORT.md`。

## 冻结设置

- 题目：`config/llm_jev_300_split.json` 的 300 个 evaluation IDs，覆盖 19 个 BIRD train 数据库。
- 已逐题核对先前任务缓存与原始 BIRD train：question、db_id、gold SQL 全部一致。原始 train 没有显式 question_id 时使用过滤前的零基行号。
- 实现：本地接入的官方 DeepEye-SQL 源码；源归档哈希见运行 manifest。不是论文原模型原样复现。
- 所有生成/判别 LLM：`qwen3.5-35b-a3b`，关闭 thinking。向量模型：`text-embedding-v4`。
- full profile：direct/reversed linking 各 12；DC/skeleton 生成各 12；checker/selection sampling budget 16；32 workers。
- train 目标题禁止动态 few-shot，ICL budget 为 0。Gold SQL 只用于离线诊断/执行评估，不进入生成提示词。
- 不改动此前 MCTS/Jev 结果，不在这 300 题上训练选择器或调阈值。

## Pipeline

数据库预处理 → 值索引 → 值检索 → 多路 schema linking → 多路 SQL 生成 → SQL revision → 执行反馈与候选选择 → 本地 SQLite 评测。

成功阶段可从 checkpoint 恢复；同名实验禁止更改题集、模型和配置哈希。任何阶段失败不得记为“实验完成”。

最初的 8-worker 运行保留在 `bird-train-300-qwen35/`，在 linking 初期停止；为提高并行度创建新 manifest，而非修改已冻结配置。19 个索引、300 题检索结果和已落盘的 linking 检查点复用；具体见新目录的 `RESUME_PROVENANCE.json`。已有成功 Chat usage 也保留。中断时尚未返回或落盘的调用可能重跑，其实际计费可能无法完全观测。

32-worker 阶段触发过 TPS/TPM 限流。已将上游 `LLM.ask(n)` 的重试粒度修复为单次 HTTP 请求，防止失败时重复整批成功样本；20 项离线接入测试（包含 429 回归测试）通过后从检查点恢复。算法、模型、输入和目标采样预算不变。限流说明见 [阿里云官方错误码](https://help.aliyun.com/zh/model-studio/error-code)。

## 必报结果

1. 覆盖数与完成数（固定 300 分母；无效金标另报）。
2. 候选覆盖率、最终 EX、同池 SC EX、可执行率、平均去重候选数。
3. Multiset 为主，set 为辅；均忽略行顺序，不冒充官方 BIRD evaluator。
4. 选择器修复/退化数、candidate coverage 到最终 EX 的 gap、配对 bootstrap 区间。
5. SQLGlot scope-aware 表列 P/R、完整 schema recall、parser 可评估题数与额外表列数量。
6. 按数据库分组；无 difficulty 标签则不杜撰难度分组。
7. 逐题错误：候选缺失、选择错误、全候选无效；schema omission 与结构差异仅作诊断，不冒充人工语义根因。
8. 分阶段墙钟时间、成功 Chat API 请求数与 input/output tokens。Embedding 用量不在当前 telemetry 内；不猜测费用。
9. SC vote share 不是校准后的正确概率。未经独立校准，不宣称已测得模型置信度校准表现。

## 对照解释

本次首要受控对照是 **同一 DeepEye 候选池 + SC** 与 **DeepEye 完整选择器**。SC 保留重复 SQL 的票数，按执行结果簇计票，空结果有效；同票按最早出现打破，不使用 gold。

旧 MCTS/Jev 的共同题目对照仅是描述性结果：旧 MCTS 使用不同 Qwen 模型和低预算，不可把差异归因于框架；缺失的旧 MCTS 题目不补成错误。后续严格比较必须统一模型、候选/API 预算、输入和评估规则。

## 产物

- `manifest.json` / `config.toml`：冻结配置与输入哈希，无明文密钥。
- `logs/` / `stages.jsonl` / `api_usage.jsonl`：运行审计。
- `*.snapshot*`：官方阶段输出与中间候选。
- `candidates.jsonl` / `predictions.json`：兼容当前项目的候选与预测。
- `metrics.json`：逐题执行评估及候选审计。
- `analysis.json` / `error_analysis.jsonl` / `REPORT.md`：离线统计与中文分析报告，仅完整跑完后生成。

## 港科广 Qwen3.6 续跑（2026-10-23）

根据[官方 V2 API 说明](https://docs.hpc.hkust-gz.edu.cn/docs/aigc/api-usage/)和[模型列表](https://docs.hpc.hkust-gz.edu.cn/docs/aigc/api-model-list/)，新运行使用 `https://aigc-api.hkust-gz.edu.cn/api/v2` 与大小写准确的 `Qwen3.6-35B-A3B`。模型标价为输入 0.90 元/百万 tokens、输出 5.40 元/百万 tokens；此前阿里云 22 元实付不适合作为新平台账单预测。

已用 `deepeye.resume` 离线创建独立目录 `results/deepeye/bird-train-300-hkust-qwen36-resume/`，冻结原 300 题和 full profile，使用 8 workers。复制了 19 个值索引、300 题预处理与值检索快照，以及经过校验的 286 条 Qwen3.5 Schema Linking 检查点；剩余 14 题由 Qwen3.6 补跑。原目录保持不变，新目录的 `RESUME_PROVENANCE.json` 记录题号和模型归属。旧 token 记录留在源目录，新目录只记录新调用。

这是**混合模型续跑**：Schema Linking 是 Qwen3.5 的 286 题加 Qwen3.6 的 14 题；SQL 生成、修订与选择将由 Qwen3.6 完成。最终分析按题标记 linking 模型，不能作为纯 Qwen3.6 效果或两模型受控比较。

在校园网或学校 VPN 可访问 API 时，使用交互式启动器输入学校密钥，先做一次低 token 探测，成功后才进入完整运行与本地评测：

```powershell
.\.venv-deepeye\Scripts\python.exe -u -m deepeye.hkust_run --execute
```

启动器从 `AIGC_API_KEY` 环境变量或隐藏输入读取密钥，只传给当前进程及子进程，不写入 manifest、日志或源文件；学校端点不会自动继承旧阿里云密钥。任何探测或阶段失败都会停止后续步骤。评测完成后才生成 `metrics.json`、`analysis.json`、`error_analysis.jsonl`、`REPORT.md`。
