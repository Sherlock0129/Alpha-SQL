# Kev-4B 集群试跑

本目录用于 Qwen3.5-4B 底座的开源 Jev-style 模型 [Kev-4B](https://github.com/jaredpalmer/kev)，与 `training/qwen25_coder_lora` 的文本生成微调不同。

## 当前状态

已上传 Slurm `debug` 模板和合成样本。集群上已克隆 Kev、安装 Python 3.13 和 PyTorch 2.8.0+cu128，并从官方 GitHub Release 下载、校验和解压 Kev-4B 检查点。Qwen 基座已下载到 8.8 GB 的本地目录，两个权重分片及索引文件存在。2026-10-29 的试跑作业 `13385088` 已完成，Slurm 状态 `COMPLETED`、退出码 `0:0`，运行 2 分 1 秒，生成了 LoRA 适配器和训练指标。合成数据只检查流程，不能衡量模型质量。

## 集群准备

1. 将本目录上传至 `/hpc2hdd/home/dxia440/jhupload/kev_sql`。在 PyCharm 中只同步本目录。
2. Kev 仓库位于 `/hpc2hdd/home/dxia440/src/kev`，使用 Python 3.13 和 `uv sync --extra serve` 创建 `.venv`。`pip install kev` 会安装同名的另一个包，不要使用。
3. Kev-4B 检查点位于 `/hpc2hdd/home/dxia440/models/kev-4b`，来自[作者发布的归档](https://github.com/jaredpalmer/kev/releases/tag/kev-1.0)，SHA-256 已核对。Qwen 基座从[官方 ModelScope 镜像](https://modelscope.cn/models/Qwen/Qwen3.5-4B-Base)下载到 `/hpc2hdd/home/dxia440/models/Qwen/Qwen3.5-4B-Base`，提交前须检查完整性。首次下载不要放进最长 30 分钟的 `debug` 作业。
4. Kev 的检查点元数据使用 `Qwen/Qwen3.5-4B-Base` 这个基座名，因此在 Kev 仓库中建立同名本地软链接。确认虚拟环境和权重完整后再提交：

```bash
cd /hpc2hdd/home/dxia440/jhupload/kev_sql
export KEV_REPO=/hpc2hdd/home/dxia440/src/kev
export KEV_PYTHON="$KEV_REPO/.venv/bin/python"
mkdir -p "$KEV_REPO/Qwen"
test -e "$KEV_REPO/Qwen/Qwen3.5-4B-Base" || ln -s "$HOME/models/Qwen/Qwen3.5-4B-Base" "$KEV_REPO/Qwen/Qwen3.5-4B-Base"
export KEV_INIT_FROM="$HOME/models/kev-4b"
export TRAIN_JSONL="$PWD/smoke_train.jsonl"
export OUTPUT_DIR="$PWD/runs/smoke"
test -x "$KEV_PYTHON"
test -f "$KEV_REPO/Qwen/Qwen3.5-4B-Base/model.safetensors.index.json"
test -f "$KEV_INIT_FROM/head.pt"
sbatch --export=ALL run_debug.slurm
```

本次训练日志在 `/hpc2hdd/home/dxia440/jhupload/kev_sql/kev-sql-smoke-13385088.log`，输出权重和 `training_metrics.json` 在 `/hpc2hdd/home/dxia440/jhupload/kev_sql/runs/smoke`。PyCharm 的 SSH 终端可用 `tail -n 50 kev-sql-smoke-13385088.log` 查看日志，用 `sacct -j 13385088 --format=JobID,State,ExitCode,Elapsed` 查看 Slurm 记录。数据每行是含 `state`、`questions` 的 JSON；每个问题含 `type`、`instructions` 和 `label`。`choice` 标签为选项名，`noul` 为布尔值，`score` 为从 0 开始的等级索引。正式数据应留出独立验证集，并将本项目已有的 300 道评估题排除在训练和调参之外。

后续用官方 `python -m kev.benchmark --run <checkpoint> --data <heldout.jsonl> --out <report-dir>` 比较原始与微调模型；持续服务应在分配了 GPU 的开发容器中运行，不能在登录节点启动。Alpha-SQL 的 `JevClient` 还需增加本地 `base_url` 设置。

参考：[Kev 训练格式](https://github.com/jaredpalmer/kev#by-hand)、[HPC 二期队列](https://docs.hpc.hkust-gz.edu.cn/docs/hpc12/slurm/queue/)。
