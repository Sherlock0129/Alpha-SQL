# Kev-4B 集群试跑

本目录用于 Qwen3.5-4B 底座的开源 Jev-style 模型 [Kev-4B](https://github.com/jaredpalmer/kev)，与 `training/qwen25_coder_lora` 的文本生成微调不同。

## 当前状态

已准备 Slurm `debug` 模板和合成样本；尚未准备正式标注集、集群 Kev/Python 环境和模型权重，也尚未提交 GPU 作业。合成数据只检查流程，不能衡量模型质量。

## 集群准备

1. 将本目录上传至 `/hpc2hdd/home/dxia440/jhupload/kev_sql`。在 PyCharm 中只同步本目录。
2. 在集群允许的软件安装位置克隆官方 Kev 仓库，使用 Python 3.12/3.13 和 `uv sync --extra serve` 创建环境。`pip install kev` 会安装同名的另一个包，不要使用。
3. 按集群规定预先缓存 `Qwen/Qwen3.5-4B-Base` 和 `jaredpalmer/kev-4b` 权重；首次下载不要放进最长 30 分钟的 `debug` 作业。
4. 确认下面的 `KEV_REPO` 和 `KEV_PYTHON` 路径确实存在，再提交：

```bash
cd /hpc2hdd/home/dxia440/jhupload/kev_sql
export KEV_REPO=/hpc2hdd/home/dxia440/src/kev
export KEV_PYTHON="$KEV_REPO/.venv/bin/python"
export TRAIN_JSONL="$PWD/smoke_train.jsonl"
export OUTPUT_DIR="$PWD/runs/smoke"
sbatch --export=ALL run_debug.slurm
```

训练日志在提交目录的 `kev-sql-smoke-<jobid>.log`，输出权重在 `runs/smoke`。数据每行是含 `state`、`questions` 的 JSON；每个问题含 `type`、`instructions` 和 `label`。`choice` 标签为选项名，`noul` 为布尔值，`score` 为从 0 开始的等级索引。正式数据应留出独立验证集，并将本项目已有的 300 道评估题排除在训练和调参之外。

后续用官方 `python -m kev.benchmark --run <checkpoint> --data <heldout.jsonl> --out <report-dir>` 比较原始与微调模型；持续服务应在分配了 GPU 的开发容器中运行，不能在登录节点启动。Alpha-SQL 的 `JevClient` 还需增加本地 `base_url` 设置。

参考：[Kev 训练格式](https://github.com/jaredpalmer/kev#by-hand)、[HPC 二期队列](https://docs.hpc.hkust-gz.edu.cn/docs/hpc12/slurm/queue/)。
