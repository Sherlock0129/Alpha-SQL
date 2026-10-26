# Qwen2.5-Coder-7B-Instruct 单卡 LoRA 模板

训练脚本和合成样本已上传到 HPC 二期的 `/hpc2hdd/home/dxia440/jhupload/qwen25_coder_lora`。尚未在集群安装依赖、下载基座模型或提交作业。

`smoke_train.jsonl` 和 `smoke_eval.jsonl` 是用于检查训练流程的合成小样本，不用于评价模型质量。

## 数据格式

UTF-8 JSONL，每行一条对话，最后一条消息必须是目标回答：

```json
{"messages":[{"role":"system","content":"你是一个代码助手。"},{"role":"user","content":"写一个 Python 排序函数。"},{"role":"assistant","content":"def sort_items(xs):\n    return sorted(xs)"}]}
```

训练时只对最后一条 assistant 回答计算损失。请用互不重复的样本分别制作训练集和验证集；正式任务还需检查样本质量及授权。

## 集群准备

1. 在登录节点检查集群可用的 Python、CUDA 和 PyTorch 环境，并确认 A40 计算节点的驱动兼容性。创建独立 Python 环境，安装与集群 PyTorch 兼容的 `transformers`、`datasets`、`accelerate`、`peft`、`trl`；记录实际版本。安装软件和下载权重时遵守集群登录节点的资源使用规则。
3. 确保基座模型目录已在集群可读取；如需从 Hugging Face 下载，先确认集群网络可用以及模型存储路径。`MODEL_PATH` 可设为本地模型目录。
4. 首次用 `debug` 分区和少量样本做不超过 30 分钟的代码调试。`run.slurm` 默认只跑 10 个优化步骤，并在训练后计算一次验证损失。完成后检查日志、损失与输出的 LoRA adapter；合成样本不能说明模型质量。
5. 正式训练时，将 `run.slurm` 的分区改为 `i64m1tga40u`（A40）或 `i64m1tga800u`（A800），根据实验估算设置时长、步骤数和成本，再提交。

首次试跑示例（需在集群 shell 中执行；先完成依赖和模型准备）：

```bash
cd /hpc2hdd/home/dxia440/jhupload/qwen25_coder_lora
export TRAIN_JSONL="$PWD/smoke_train.jsonl"
export EVAL_JSONL="$PWD/smoke_eval.jsonl"
export OUTPUT_DIR=/hpc2hdd/home/dxia440/jhupload/qwen25_coder_lora/smoke_adapter
export MODEL_PATH=/hpc2hdd/home/dxia440/models/Qwen2.5-Coder-7B-Instruct
export PYTHON_BIN=/hpc2hdd/home/dxia440/venvs/lora/bin/python
sbatch --export=ALL run.slurm
```

使用 `squeue -u "$USER"` 查看排队或运行状态。正式训练前先验证数据格式、模型路径、环境依赖和 GPU 显存；不得在登录节点直接运行训练。

参考：[港科广 Slurm 队列](https://docs.hpc.hkust-gz.edu.cn/docs/hpc12/slurm/queue/)、[作业提交](https://docs.hpc.hkust-gz.edu.cn/docs/hpc12/slurm/job2/)、[TRL SFT 数据格式](https://huggingface.co/docs/trl/sft_trainer)。
