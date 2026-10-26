# LLM 微调与 SQL 选择器课程学习：算力门槛、已有进展与证据边界

检索日期：2026-10-21。本文是文献与可行性说明，没有下载模型、安装训练依赖、训练新模型或调用付费模型 API。

## 摘要

本报告区分三个容易混淆的问题：训练对象是否为 LLM、更新哪些参数，以及如何安排训练样本。课程学习属于第三个维度，并不要求使用大模型或生成式损失。LoRA、QLoRA 降低了适配预训练模型的显存门槛，但基础模型、激活与前后向计算仍有成本。相关研究已覆盖偏好对课程、排序课程、样本 utility、错误反馈重加权和技能达标成本，SQL 领域也已有专门微调的选择器。因此，本项目不能仅以“加入课程”或“训练 SQL reranker”为贡献。现有 MLP 适合验证调度实现；若主张 LLM 微调效率，则需要实际更新语言模型参数，并在相同质量、回归与总成本约束下验证。当前 20 个 pairwise 训练问题、82 个 pair 的有效支持仍是主要证据限制。

## 1. 研究问题

- RQ1：LLM 微调是否必须依赖多卡或巨量算力？哪些成本能由参数高效微调减少，哪些不能？
- RQ2：课程学习是否要求微调 LLM？当前数值 MLP、文本选择器和 LLM adapter 分别能支持什么结论？
- RQ3：与固定训练集下的 SQL 选择器最相关的课程和微调研究，已经验证到什么程度？

## 2. 检索方法与范围

采用四个视角：参数高效微调及其内存边界、课程与排序训练主流方法、随机基线及保持性/成本反证、SQL selector/verifier 应用。三个文献视角独立检索后合并，主文由同一视角统一综合。优先使用作者论文、ACL Anthology、正式会议论文集和期刊全文，核验标题、作者、版本与引用内容。检索涵盖经典工作至 2026 年公开文献，精选 18 篇核心论文；这不是穷尽整个领域或证明不存在其他方法的系统综述。

检索不将作者的整套系统收益自动归于课程，不将训练环境当作最低硬件要求，也不将训练 tokens、优化步数、单步时间和总 wall-clock 混称为“训练效率”。论文作者的报告不等于本项目复现。LoRA-FA 引用核验到 2026 年 v3，GradeSQL 优先采用 2026 年正式期刊版。

## 3. 三个独立的设计维度

| 维度 | 选择示例 | 对当前项目的含义 |
|---|---|---|
| 输入表示与模型 | 数值 MLP、文本 encoder、decoder LLM | 决定模型能否直接读取问题、schema、SQL 的语义 |
| 参数更新范围 | 全参数、LoRA/QLoRA、冻结 backbone 后只训练 head | 决定适配能力及一部分内存/计算成本 |
| 数据调度 | 随机、静态难度、utility、学习进度、错误组反馈 | 决定训练时选什么、何时选、分配多少曝光 |

课程可以用于任意有参数学习过程的模型，更新范围也不限定损失必须是 SFT。相同课程可与分类、RankNet、DPO 或生成式监督搭配，但作用与成本需要分别验证。完全不更新模型、只改变 API 推理提示或 few-shot 顺序，与本项目的参数训练课程不是同一个研究问题。

## 4. 算力门槛：更新参数少，不等于整个训练免费

[LoRA](https://arxiv.org/abs/2106.09685) [1] 冻结预训练权重、训练低秩更新矩阵；[QLoRA](https://arxiv.org/abs/2305.14314) [2] 进一步将冻结基础模型量化为 4-bit，并训练 adapter。二者针对的是不同内存组成，不能将“LoRA 参数少”直接理解为基础模型也无需加载。

训练峰值显存可粗略分解为：基础权重、可训练参数及梯度/优化器状态、激活、算子临时空间。冻结 backbone 后仅训练末端 head，允许在适当表示设计下预计算并缓存文本特征，不必为 encoder 构建训练反向图；而将 LoRA 插入多层时，仍要执行相关层前后向并保存或重算激活。[LoRA-FA v3](https://arxiv.org/abs/2308.03303v3) [3] 对激活内存和低秩更新结构的处理，也说明可训练参数量不是完整资源指标。

### 4.1 权重占用的算术下界

| 基础参数规模 | 全部按 16-bit 存储 | 理想全部按 4-bit 存储 |
|---|---:|---:|
| 1B | 2 GB | 0.5 GB |
| 3B | 6 GB | 1.5 GB |
| 7B | 14 GB | 3.5 GB |

这里采用十进制 GB，仅计算 N×位宽/8。它不是训练显存预测：未包含量化元数据、未量化模块、adapter、梯度、优化器状态、激活和临时空间。长 schema、长候选和 pairwise/listwise 文本输入会进一步影响峰值。Hugging Face 的[官方量化文档](https://huggingface.co/docs/transformers/quantization/bitsandbytes)也明确区分量化基础模型与额外参数训练。

本次只读检查确认本机 GPU 为 RTX 3070 Ti Laptop、8192 MiB 总显存。可以将约 1B 级模型的短上下文 adapter 实验作为待测配置，之后再评估更大模型；这属于工程起点判断，不是已经证明 1B 或 3B 在本机可训练。必须实际测量最大输入长度、batch=1 时峰值显存和吞吐，并确认 Windows、PyTorch、量化库兼容性。不能据 7B 的理想 3.5 GB 权重大小保证 8 GB 显存能完成训练。

全参数微调通常比 adapter 微调资源要求高，但不能不指定参数规模、精度、序列长度、batch、优化器和分片策略，就说所有 LLM 微调都要多卡。反过来，QLoRA 的低显存也不自动保证更高吞吐或更短总时间。

## 5. 课程不依赖 LLM，但结论必须匹配训练对象

[Bengio 等的 Curriculum Learning](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf) [4] 已包含简单分类网络等实验；[Competence-based Curriculum Learning](https://aclanthology.org/N19-1119/) [5] 则在神经机器翻译中安排训练可采样集合，均早于当前指令 LLM 微调范式。这些工作说明课程是优化过程中的数据安排，不是一种 LLM 专属结构。后者的 competence 在所述实验中是训练步数函数，不能误写为实时错误反馈。

[When Do Curricula Work?](https://arxiv.org/abs/2012.03107) [6] 通过同样逐步开放数据、但进入顺序随机的 random-curriculum，对比“顺序”与“开放范围”的作用；其常规任务上收益有限、有限预算或噪声条件下更有效的发现，限制了从经典正结果直接推断普适收益的做法。它不证明 SQL 或 LLM 的课程失败，而是要求配置强随机与相同 pacing 对照。

当前模型是 48–64–32–1 数值 MLP，所有 5,441 个参数从头训练。它可以检查 sampler、replay、损失和 η=0 等价性，但不能证明语言模型适配更快。若训练一个文本 encoder 的分类/排序模型，可以支持文本 SQL selector 的课程结论；若冻结文本 backbone、仅训练打分 head，结论应限定为该 head 的训练，不能称为完成 LLM backbone 微调。若目标明确为“LLM 微调效率”，则应至少在实际更新语言模型 adapter 或其他 backbone 参数的设置中验证。

## 6. LLM 课程已从固定难度扩展到 utility、反馈与达标控制

### 6.1 偏好和排序任务已有直接先例

[Curri-DPO](https://aclanthology.org/2024.findings-emnlp.754/) [7] 根据同一 prompt 的多个回答质量差构造阶段课程；[KPO](https://arxiv.org/abs/2506.00441) [8] 则面向 top-K 排序偏好，并按 K 安排课程。前者主要研究对齐，后者的任务形式更贴近候选排序，但两者都不是当前数值特征 RankNet。Curri-DPO 还包含多偏好对和 reference 更新，需结合其 pooled-random、固定 reference 等对照理解，不能把完整收益全部归于顺序。

KPO 使用 LoRA 训练文本 LLM，并提供同一排序数据的随机、升序、降序对照，因此“给排序模型加从易到难课程”已有先例。其独立单步 runtime 对比是在比较损失实现，不能用来证明课程缩短达标时间。两篇均不直接验证 SQL 错误组的增量价值。

### 6.2 难度之外的训练价值与采样反馈

[Aioli](https://arxiv.org/abs/2411.05735) [9] 分析在线数据组混合，并指出多种既有方法未稳定超过简单分层采样；[Difficulty Is Not Enough](https://ojs.aaai.org/index.php/AAAI/article/view/40400) [10] 进一步在 LLM 微调中联合考虑 difficulty 与 utility。二者共同限制“高 loss/高错误率就应优先训练”的假设，但 Aioli 的混合规律估计与 DUCL 的分布代理 utility 不同，不能把它们都描述为直接观测每条样本的因果学习收益。

DUCL 已报告保持性任务和课程构造成本，不能说它忽略这些问题。不过其所报 Qwen2.5-7B-Instruct 数学微调中，IFEval 从 base 的 73.4 到 DUCL 的 57.7，仍有明显下降，尽管优于随机训练的 54.6。因此，部分减轻遗忘不等于不增加回归，更不等于保持每一道原本正确问题。

### 6.3 顺序干预和持续反馈不应混写

[Fine-Grained Data Ordering](https://aclanthology.org/2026.findings-acl.1021/) [11] 在 CPT 与 SFT 中研究数据顺序；其明确两 epoch 配方是首轮排序、第二轮随机。[CURE](https://arxiv.org/html/2601.15408v1) [12] 则依据数据源/类别错误率重加权，但最终所述配置同样采用 3000 步 uniform warmup 后一次更新并固定 3000 步，而非持续频繁重估。这两篇分别说明有益顺序未必需要持续反馈，错误感知也未必以高频更新最佳。

FOT 的最终任务分数改善不能自动转化成净达标时间缩短。CURE 是医学 VLM 多任务适配，还改变任务表述和监督组织，其完整系统改善也不能直接作为固定 SQL pair 池课程的收益。两篇都有保持性相关分析，但都不提供本项目所要求的逐问题零新增回归保证。

### 6.4 净时间和“达到共同目标”已经成为更明确的研究对象

[EDCO](https://arxiv.org/html/2601.03725v1) [13] 用更新的推理熵选择训练数据，可用于 SFT 和 RLFT；[LogFloor](https://arxiv.org/abs/2608.14936) [14] 则直接以所有监测技能达到门槛所需训练 tokens 为目标，并研究小模型轨迹向大模型迁移。相比只报训练终点分数，两者分别触及动态诊断开销和目标达成成本，但其成本单位和任务约束不同。

EDCO §5.4 报告的 83.5% 是熵估计由 2.24 秒/样本降至 0.37 秒/样本，不是完整训练加速。其报告的 500 步 RL 总时间为 33.58 小时，随机采样为 30.67 小时，约多 9.5%。这个结果不代表它在相同质量目标下必然更慢，却足以说明“模块加速”与“固定步数总训练加速”不能混用。

LogFloor 是 2026-08 预印本，页面标记为投 AAAI 2027，不能称为已经被 AAAI 2027 接收。它在指定技能切片、明确 monitor floors 的实验中报告 token 节省及部分 held-out audit；跨架构迁移不保证。其 loss-floor/训练-token 结果不等于 SQL EX、逐题回归约束或计入所有 probe 的 GPU wall-clock 已获得同样结论。因此，“以更少训练量达到共同门槛”也不能再作为无人研究过的宽泛目标。

## 7. SQL 选择器微调：已有文本监督，任务形式并不唯一

与上述通用课程相比，[CHASE-SQL](https://proceedings.iclr.cc/paper_files/paper/2025/file/974ff7b5bf08dbf9400b5d599a39c77f-Paper-Conference.pdf) [15]、[MSc-SQL](https://aclanthology.org/2025.naacl-long.107/) [16] 和 [GradeSQL](https://link.springer.com/article/10.1007/s10844-026-01071-6) [17] 直接训练文本 SQL selector/verifier，但优化形式分别偏向 pairwise、listwise 和 pointwise。它们证明候选选择不必训练 SQL 生成器，却不能证明本项目的课程一定有效。

| 工作 | 选择监督 | 参数适配与资源证据 | 本项目不能直接继承的结论 |
|---|---|---|---|
| CHASE-SQL，ICLR 2025 | 问题、hint、schema、两条 SQL，监督二选一 | selector SFT，包含 LoRA 配方；云端环境不是本地显存下界 | pairwise selector 已有先例，且不是因为标题有 preference 就等于 DPO |
| MSc-SQL，NAACL 2025 | 同时读取多个候选及执行 metadata，预测正确候选编号 | 7B/8B/9B 系列，QLoRA；论文报告单张 48 GB GPU | 单卡可行不代表本机 8 GB 能照搬，也不代表已验证课程 |
| GradeSQL，JIIS 2026 正式期刊版 | 问题、schema、单 SQL，生成 Yes/No，以 P(Yes) 排序 | LoRA 自回归分类监督，模型包含 1B–8B | 不是 MLP，也不是只修改生成器投票规则；未证明错误课程与零回归 |

特别需要澄清：LLM 的 next-token 训练目标可以用于生成很短的分类标签或候选编号，不一定要求生成完整 SQL。也可以另设实数打分 head 并用 RankNet 训练，但这属于本项目的架构选择，不是上述论文都用了同一 loss。

GradeSQL 已于 2026-07-07 正式发表，不能因 arXiv 旧稿残留 under review 而说目前仍在审。MSc-SQL 的单卡 QLoRA 是反驳“LLM selector 微调必然需要多卡”的直接领域证据；它不构成任何更低显存配置的实测证明。

## 8. 对当前项目的综合判断

第一，当前不做大模型全参微调是可理解的阶段安排，但“微调必需巨量算力”不是充分理由。可以研究较小文本模型、冻结表示加 head，或小型 LLM 的 adapter，分别承担不同层次的结论。需要在所有课程组共享同一模型，而不是只升级 EA。

第二，当前瓶颈不仅是硬件。129 个训练问题中仅 20 个可形成正负 pair、共 82 对；重复采样不增加独立问题支持。不能断言所有 LLM 都必须有某个固定样本量才能微调：[LIMA](https://arxiv.org/abs/2305.11206) [18] 展示过少量高质量适配数据的价值，但其指令对齐条件不能证明本项目 20 个选择问题足以识别课程效应。

第三，研究贡献必须比“给 LLM 加课程”更具体。本轮核验的作品尚未提供“固定 SQL 候选池和共同训练数据、SQL 错误语义相对 loss/progress 的独立增量、保持性回放共同控制、明确回归约束、包含诊断成本的净达标时间”这一完整同构证据。这是限定检索范围的观察，不是首次性证明。DUCL 和 FOT 已涉及保持性；LogFloor 已直接研究门槛成本；不能通过忽略近邻来制造缺口。

第四，下一步应先修复弱结构标签的可信度和模型输入语义不足。否则 MLP 无收益可能来自不可区分表示，换 LLM 后的收益也可能来自模型升级而非课程。可以用文本选择器建立共同基线，再在同一冻结池里比较随机、同 pacing 随机、静态错误均衡、LP、EA 和 η=0。若主要主张升级为 LLM 微调效率，至少加入实际更新 adapter 的共同模型实验。

## 9. 效率验证所需的共同口径

质量固定目标、回归上限和连续达标次数应事先冻结。计算准备/标注、训练诊断、调度、训练更新和共同验证的总时间，另列可复用缓存与冷启动成本。共同候选生成成本单列，不把没有重新生成候选当作 sampler 的加速。

对文本模型，还应报告实际非 padding tokens、最大输入长度、截断规则、有效 batch、峰值显存、checkpointing 与适配参数。相同 pair 数并不保证相同 token 成本，动态重采样也可能改变训练目标权重。所有组应共享推理规则，且不在每组单独选择 gate 后把结果归于课程。

在“不增加回归”目标下，不仅看总体 EX，还应报告原 SC 正确的新错误数量、条件回归率、修复数和不同问题类型。统计不显著不等于已证明非劣，单 seed 成功也不是稳健的训练加速结论。

## 10. 结论：逐项回答

RQ1：不是所有 LLM 微调都需要巨大算力。全参数训练通常昂贵，LoRA/QLoRA 能降低门槛，领域内已有单卡 48 GB 的直接例子；本机 8 GB 的具体配置仍待实测。

RQ2：课程学习不要求微调 LLM。它可以作用于 MLP、文本 encoder、head 或 adapter，但关于 LLM 微调效率的主张需要相应参数更新实验。当前 MLP 只能承担机制 pilot。

RQ3：课程已覆盖排序、utility、错误反馈和目标达标成本，SQL selector 微调也已有不同监督范式。值得检验的是这些因素在固定 SQL 候选池下能否提供独立、可解释、计入成本且满足回归约束的增量，而不是简单证明“课程可以接到 LLM 上”。

## 参考文献

[1] E. J. Hu, Y. Shen, P. Wallis, et al., “LoRA: Low-Rank Adaptation of Large Language Models,” arXiv:2106.09685, 2021.

[2] T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer, “QLoRA: Efficient Finetuning of Quantized LLMs,” NeurIPS, 2023.

[3] L. Zhang, L. Zhang, S. Shi, et al., “LoRA-FA: Efficient and Effective Low Rank Representation Fine-tuning,” arXiv:2308.03303v3, 2026; original submission 2023.

[4] Y. Bengio, J. Louradour, R. Collobert, J. Weston, “Curriculum Learning,” ICML, 2009.

[5] E. A. Platanios, O. Stretcu, G. Neubig, et al., “Competence-based Curriculum Learning for Neural Machine Translation,” NAACL-HLT, 2019.

[6] X. Wu, E. Dyer, B. Neyshabur, “When Do Curricula Work?,” ICLR, 2021.

[7] P. Pattnaik, R. Maheshwary, K. Ogueji, et al., “Enhancing Alignment using Curriculum Learning & Ranked Preferences,” Findings of EMNLP, 2024.

[8] S. Cai, C. Gao, Y. Zhang, et al., “K-order Ranking Preference Optimization for Large Language Models,” arXiv:2506.00441, 2025.

[9] M. F. Chen, M. Y. Hu, N. Lourie, et al., “Aioli: A Unified Optimization Framework for Language Model Data Mixing,” ICLR, 2025.

[10] Z. Jiang, J. Han, T. Li, et al., “Difficulty Is Not Enough: Curriculum Learning for LLMs Fine-tuning Must Consider Utility,” AAAI, 2026.

[11] X. Hu, Y. Tang, H. Li, et al., “Fine-Grained Data Ordering Improves Fine-Tuning for Large Language Models,” Findings of ACL, 2026.

[12] P. Messina, A. Villa, J. L. Alcázar, et al., “CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation,” arXiv:2601.15408v1, 2026.

[13] J.-C. Pang et al., “EDCO: Dynamic Curriculum Orchestration for Domain-specific Large Language Model Fine-tuning,” arXiv:2601.03725v1, 2026.

[14] S. Choi, J. Sung, M. Umer, et al., “Small Models Scout Bottleneck Order for Large-Model Data Control,” arXiv:2608.14936v1, 2026.

[15] M. Pourreza, H. Li, R. Sun, et al., “CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL,” ICLR, 2025.

[16] S. K. Gorti, I. Gofman, Z. Liu, et al., “MSc-SQL: Multi-Sample Critiquing Small Language Models For Text-To-SQL Translation,” NAACL, 2025.

[17] M. Tritto, G. Farano, D. Di Palma, et al., “GradeSQL: Outcome reward models for intelligent Text-to-SQL generation from LLMs,” Journal of Intelligent Information Systems, 64:2545–2568, 2026.

[18] C. Zhou, P. Liu, P. Xu, et al., “LIMA: Less Is More for Alignment,” NeurIPS, 2023.
