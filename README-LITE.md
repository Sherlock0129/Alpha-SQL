# 百炼 API 五题闭环

基于 HKUSTDial/Alpha-SQL commit `216e61fc86467591fa1dc5b7031ca536577b4d21`。
主模型和关键词提取均使用 `qwen3-coder-flash`，embedding 使用 `text-embedding-v4`。
这属于更换模型后的轻量方法复现，不等同于论文 Qwen2.5-Coder 的成绩复现。

## 环境与密钥

已验证 Python 3.12。使用工程内 `.venv`；首次安装：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lite.lock.txt
```

项目配置在 `.env.local`，示例见 `.env.example`。将 `DASHSCOPE_API_KEY` 填入本地配置，
程序会加载为自身进程的环境变量，并映射到主模型和 embedding 客户端。
也可以在启动程序的终端设置同名环境变量；终端值优先。
不会修改全局 shell 配置，不读取上游附带的 `.env`。
真实密钥文件由 Git 忽略；建议权限保持为 600。
也可运行 `.venv/bin/python script/set_api_key.py`，在本地隐藏输入密钥。

## 检查与运行

在工程根目录依次执行：

```sh
.venv/bin/python script/smoke5.py check
.venv/bin/python script/smoke5.py probe
.venv/bin/python script/smoke5.py run
```

- `check`：不调用 API，检查五题、数据库和标准答案执行。
- `probe`：调用两次主模型和一次 embedding，验证权限、采样及输出格式。
- `run`：值检索预处理 → 每题 MCTS → 执行一致性选择 → 结果比对。

结果在 `results/qwen3-coder-flash/smoke5/<运行时间>/`：
`pred_sqls.json`、`evaluation.json`、逐题日志和 `usage.jsonl`。
日志不保存密钥；调用统计记录主模型 tokens，不包含 embedding 费用及失败重试的精确账单。
费用以百炼控制台为准。接口使用非流式返回，每次请求只采样一个答案。

## 预算与实验范围

固定五题 ID：721、742、1382、1437、1461，随机种子 42。
来自 BIRD `dev_20240627`，仅保留 superhero、student_club 两个数据库，
包含 3 道 simple、1 道 moderate、1 道 challenging。这是工程冒烟测试，不是论文 SDS。
来源和原始 JSON 哈希见 `data/bird/smoke5_manifest.json`。
数据源：https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip ，BIRD 数据许可 CC BY-SA 4.0。

配置：2 rollouts、普通动作采样 1、SQL 生成与修正各采样 2、并发 1。
主模型逻辑请求上限：预处理进程 120，每题进程 120；SDK 最多额外重试一次。
预处理和每题分别有 900 秒进程超时，单请求 60 秒。
这不是货币消费上限；即便五题也可能产生大量 tokens。
`max_depth` 是上游保留字段，尚未实施深度截断；本方案依靠请求预算和超时约束运行。

保留上游动作、提示词和基于执行结果的一致性奖励。参考答案仅用于离线评估，
进入 MCTS 前会从 Task 中移除。评估使用结果集合相等的 BIRD-style EX；
尚未接入完整官方评估器，不用于声明正式榜单成绩。失败题按未正确计入五题分母。
SQL 选择计时重复次数从 20 改成 1，可能影响同票时的选择。
embedding 模型已改变，0.6 阈值仅沿用初始值，后续应单独评估。

预处理缓存仅供本次固定配置使用。更换模型、数据或检索参数时应使用新的缓存目录。
当前 API Key 尚未配置，真实 API 与端到端五题结果待密钥提供后验证。

## 阶段一：Schema-aware 轻量重排序实验

该实验保持原 Schema Linker、MCTS 和候选池不变。它从每条候选路径读取原有
`selected_schema_dict`（硬选择而非置信度），构造 SQL 结构、Schema 覆盖与外键连接、
执行结果簇以及可选 MCTS 元数据，训练无额外机器学习依赖的 Logistic Regression。
gold SQL 只用于生成执行等价标签，不进入特征。

推荐使用互不重叠的训练候选和评估候选：

```sh
python -m alphasql.runner.schema_signal_experiment \
  --train-results-dir results/train_candidates \
  --train-data-path data/bird/train/train.json \
  --eval-results-dir results/dev_candidates \
  --eval-data-path data/bird/dev/dev.json \
  --db-root-dir data/bird/train/train_databases \
  --eval-db-root-dir data/bird/dev/dev_databases \
  --output-dir results/schema_signal
```

如果暂时只有一套候选，可省略 `--eval-*` 参数，程序默认按数据库划分 80/20；
数据库不足两个时才回退为按问题划分。同一问题的候选不会跨集合。正式结论仍应采用
独立数据集或按数据库划分的交叉验证。

输出包括：

- `report.json`：execution-only、SQL+execution、+schema、+MCTS 四组消融的
  Selection EX、Oracle EX、GapClosure，以及标准化特征系数；
- `schema_aware_logistic.npz`：默认 Schema-aware 线性模型；
- `predictions.jsonl`：评估集中每条候选的分数和标签。

判断 Schema 信号是否有效，应主要比较 `schema_aware` 与 `sql_execution` 的
`model_ex` 和 `gap_closure`。`schema_aware_mcts` 单列报告，因为一致性奖励可能与
执行结果簇重复，不能用它替代 Schema 信号的消融结论。

### BIRD 数据与候选池

官方 Train/Dev 数据安装在 `data/bird/train` 和 `data/bird/dev`，安装脚本会防止
Zip Slip、展开嵌套的数据库压缩包、核对题目和 SQLite 数据库数量并记录 SHA-256：

```sh
python script/install_bird.py train data/bird/_downloads/train.zip
python script/install_bird.py dev data/bird/_downloads/dev.zip
```

阶段一默认从体积最小的 12 个 train 数据库和 6 个 dev 数据库中，按数据库
（以及可用时的难度）均衡抽取 100/50 题。该边界避免首次实验被数百万唯一值的
极端大库拖慢；正式扩大实验时可增加两个 `--*-db-count` 参数。以下命令可重新生成清单：

```sh
python script/prepare_schema_pool.py --train-size 100 --dev-size 50 --seed 42
```

配置好项目本地 `.env.local` 中的 `DASHSCOPE_API_KEY` 后，构建候选池：

```sh
python script/build_schema_pool.py all --split both
```

预处理结果位于 `data/preprocessed/schema_pool/{train,dev}`，每题 8 次 MCTS rollout
产生的候选路径位于 `results/schema_pool/{train,dev}`。运行器会跳过已有的 `.pkl`，
因此中断后可执行同一命令续跑；进入 MCTS 前会移除 gold SQL，避免标签泄漏。

阶段一建议先运行数据库均衡的 pilot（train 24 题/12 库，dev 18 题/6 库，
每题 2 次 rollout），先验证 Schema 信号再承担完整搜索成本：

```sh
python script/prepare_schema_pool.py --train-size 24 --dev-size 18 --seed 42 \
  --train-db-count 12 --dev-db-count 6 --output-root data/bird/schema_pool_pilot
python script/build_schema_pool.py generate --profile pilot --split both
```

pilot 候选保存到 `results/schema_pool_pilot/{train,dev}`，详细调用日志分别写入
对应目录的 `generation.log`。请求上限按题重置，避免复用 worker 时前几题耗尽整个进程额度。

需要形成非零 selector gap 时，使用 dense profile。该配置执行 4 次 rollout、
真正限制搜索深度，并保留 generation/revision action 已经采样出的全部 SQL：

```sh
python script/build_schema_pool.py generate --profile dense --split both
```

dense 候选写入 `results/schema_pool_dense/{train,dev}`。它仍使用原 Schema Linker；
同一问题采用 linker 输出众数作为稳定的 question-level schema，避免把“路径没有运行
Schema Selection”误编码成“linker 不支持该 SQL”。

运行 pilot 的 Logistic Regression 消融：

```sh
python -m alphasql.runner.schema_signal_experiment \
  --train-results-dir results/schema_pool_pilot/train \
  --train-data-path data/bird/schema_pool_pilot/train.json \
  --eval-results-dir results/schema_pool_pilot/dev \
  --eval-data-path data/bird/schema_pool_pilot/dev.json \
  --db-root-dir data/bird/train/train_databases \
  --eval-db-root-dir data/bird/dev/dev_databases \
  --output-dir results/schema_signal_pilot
```

GPU pairwise MLP 使用 RankNet loss，并对可执行、Schema 覆盖高的错误 SQL 增加权重。
hard negatives 只增强训练问题，内部验证集和 dev 保持真实候选：

```sh
python -m alphasql.runner.neural_schema_reranker \
  --train-results-dir results/schema_pool_dense/train \
  --train-data-path data/bird/schema_pool_pilot/train.json \
  --eval-results-dir results/schema_pool_dense/dev \
  --eval-data-path data/bird/schema_pool_pilot/dev.json \
  --db-root-dir data/bird/train/train_databases \
  --eval-db-root-dir data/bird/dev/dev_databases \
  --output-dir results/neural_schema_reranker_dense \
  --device cuda --epochs 300 --hard-negatives-per-question 4
```

正式结论应至少报告 3 个固定随机种子的均值和标准差；不得按 dev 选择最好 seed。

候选生成完成后运行轻量重排序实验：

```sh
python -m alphasql.runner.schema_signal_experiment \
  --train-results-dir results/schema_pool/train \
  --train-data-path data/bird/schema_pool/train.json \
  --eval-results-dir results/schema_pool/dev \
  --eval-data-path data/bird/schema_pool/dev.json \
  --db-root-dir data/bird/train/train_databases \
  --eval-db-root-dir data/bird/dev/dev_databases \
  --output-dir results/schema_signal
```
