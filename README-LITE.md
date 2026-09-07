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
