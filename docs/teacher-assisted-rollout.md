# Teacher 卡先辅助 Student 生成，再切回评分

日期：2026-09-25。状态：本机常驻互斥已实现并完成 0.6B←8B 实卡测量；源码按当前主路版本冻结。8B←32B 的 H800 尚未远端验证。

veRL 分支：`experiment/teacher-assisted-rollout`，从当前项目的 `main` 创建，基础为已实现的 `sd-early`。本机 worktree 为 `fopd/.worktrees/teacher-assisted-rollout`。本文随该 veRL 工作区保存；外层 fopd 的启动器和结果文档仍沿用现有位置。

## 目标与基本决定

每步开始时，原 Teacher 卡临时运行一个额外 Student vLLM 副本，与原 Student 副本共同生成本步轨迹。完成一定数量的 Student 轨迹后，额外副本停止接单，将未完成请求分散到原 Student 副本，释放其执行资源，原 Teacher 恢复评分。全部 Student 完成后，仍按 `sd-early` 在原 Actor 卡组上执行流式 F/B，与剩余 Teacher 评分重叠。

目标是缩短完整训练步，而不只是提前结束 Student。先验证短时间借用 Teacher 卡的收益，不默认等待半批轨迹结束。切换门槛可配置，数值待组件测量后选定；当前没有已经验证的最优比例或加速承诺。

Actor 的资源组、梯度累计、优化器和单步更新语义保持 `sd-early`。Teacher 卡上的额外 Student 只做推理，不创建 Actor、梯度或 optimizer，不引入跨训练组梯度合并。

## 资源布局与配方

| 场景 | 固定 Student / Actor 卡 | 临时辅助生成、随后评分的卡 | 借用期间 Student 副本 |
|---|---|---|---|
| 本机，6×3090，0.6B←8B | 4 卡，2 个 Student TP2 副本，4 卡 Actor | 2 卡，辅助 Student TP2 / Teacher TP2 | 3 个 TP2 副本 |
| 目标，8×H800，8B←32B | 6 卡，3 个 Student TP2 副本，6 卡 Actor | 2 卡，辅助 Student TP2 / Teacher TP2 | 4 个 TP2 副本 |

新增的是同版本 Student TP2 副本，不把模型改成 TP6/TP8。总物理卡数不变，卡数命名不重复计算共享卡上的两个角色。

本机沿用 `scripts/fair_compare/6gpu-0.6b-from-8b.sh`，通过 `FOPD_VERL_DIR` 选择本 worktree。batch 48、prompt 1024 / response 2048、Student SD k=3、Teacher follow 关闭、early-actor 和现有 OPD fast path / 发布配置与对照一致。默认 5 step，前 2 warmup，后 3 取平均。模型、draft、精度、offload 和并发配置在对照与候选间一致，额外副本使用匹配的 draft。

H800 目标沿用现有 8 卡 4K `sd-early` 配方：batch 96、prompt 1024 / response 4096、8B←32B、Student SD k=3。没有用户后续明确指令，不在 SuperPOD 提交或启动任务。本机结果不替代 H800 容量及性能结论。

## 每步调度

### 1. 准备与借用

- 在本步第一条请求发出前，原 Student 和辅助 Student 均持有本步同一版已发布参数；Teacher 模型保持固定。
- Teacher 处于可安全切换的暂停状态，辅助 Student 已可执行。两套引擎进程保留，每步不重新启动服务或从磁盘重载模型。
- 当前 logical batch 的请求在全部 Student 副本间分配，不能因增加副本而增加 batch。
- 完整 Student 轨迹立即发布 Student-ready 状态；Teacher 尚未就绪时，评分请求在现有调度层等待。等待不能占住整个调度器、阻止其他生成或阻止切换。
- Teacher 请求只有在 Teacher 已恢复且可服务后才进入其引擎，不能依赖向 sleeping engine 发请求后自动唤醒。

### 2. 触发切换

- 首版以本步已完成 Student 轨迹数作为可配置触发门槛，可用比例换算并向上取整。统计包含全部 Student 副本，只计完整逻辑轨迹，不计被中断的部分输出。
- 增加可配置的最大借用时间，避免长答案导致 Teacher 长时间无法开始；计时以本步首次 Student 请求提交为起点。条数达到门槛或达到时间上限即切换，所有 Student 已完成时也必须立即切换。
- 门槛不能依赖 Teacher-ready 数，因为借用期间 Teacher 没有评分。每步只执行一次辅助生成到 Teacher 的角色切换。
- 初始门槛根据额外副本收益、已有轨迹完成分布和实测切换耗时选取，不把 50% 写成未经验证的默认策略。不预先构建自适应调度框架。
- 待评分 token 量比条数更接近 Teacher 工作量，可用于分析门槛选择；自动按 token 负载决定切换属于后续决策，不作为本轮必须实现的第二套算法。

### 3. 迁出辅助 Student

1. 先从路由中撤下辅助副本，禁止新请求进入，并处理撤下时仍在提交的请求。
2. 在已接受 token 边界中断未完成请求，保存已有 token、对应 rollout logprob、停止状态和剩余 response 配额。
3. 清除这些请求指向辅助副本的 sticky 路由，分散到仍在运行的原 Student 副本。首版复用现有负载均衡，同时检查目标并发和 KV 空间，避免集中迁给同一个副本；不要求精确预测剩余长度。
4. 使用完整 prompt 加已接受前缀重新 prefill，继续剩余生成。首版不实现直接 KV block 搬运；迁移成本包含目标重算及其对原请求的干扰。
5. 确认辅助副本不再执行生成且部分输出已交接后，释放其必要显存并恢复 Teacher。无须等待被迁请求在目标副本上全部完成。

迁移请求最终只产生一条完整轨迹，Teacher 只接收这条最终完整轨迹。已自然结束的请求应直接完成，不得因切换竞争而重复续写或重复评分。当前迁移分支的 partial-rollout 拼接和撤销路由机制可以复用，但不能整体引入其早训组和梯度交换实现。现有续写循环有固定 1 秒重试等待，本方案应改为随路由就绪恢复，不能无条件照搬该等待。

### 4. Teacher 评分与原有 sd-early

- Teacher 恢复后处理之前等待的完整轨迹，并继续接收后来完成的轨迹，沿用现有完整 response 评分接口和 vLLM batching。
- 原 Student 全部结束后才允许 Actor 开始 F/B；辅助副本的迁移中间状态不能被当成整批 Student 已完成。
- 原 Actor 卡组 sleep Student vLLM，再流式消费已评分轨迹。保留 `sd-early` 的归一化、一次梯度裁剪、optimizer / scheduler 更新和发布顺序。
- Teacher 必须完成本步所有评分，Actor 必须消费完整本步 batch，才能结束本步更新流程。

### 5. 下一步准备

- Teacher 所有本步请求结束且结果已交付后，可以在剩余 Actor 训练期间准备下一次角色切换，前提是不会干扰当前训练和发布。
- 参数发布必须覆盖原 Student 和辅助 Student。辅助副本不与 Actor 共卡，需要接入已有跨资源池权重传输能力，不能仅沿用原同卡发布并漏掉这个副本。
- 辅助 Student target 参数每步更新；draft 的固定参数和必要状态按当前 SD 发布实现保留或恢复，不能被深度 sleep 后的随机内存替代。
- 下一步请求只能在本步更新及全部 Student 副本发布完成后发出，不跨步预生成、不使用旧策略续写。
- Teacher sleep、辅助 Student wake、额外副本发布等成本必须计入连续训练周期，不移到计时区间外。若跨 Actor 阶段隐藏部分开销，以实际完整墙钟体现收益，不能重复扣减。

## 两套 vLLM 的驻留和切换

两套引擎使用同一 Teacher 物理卡组上的独立进程及各自 TP 通信组。先沿用现有 Ray 资源池创建方式安排共享角色，不增加实际 GPU 资源申请，也不要求 MPS 或同时执行两个模型。

首版使用项目安装版本实际支持的 sleep/wake 能力。vLLM level-1 会把权重备份到 CPU 并丢弃 KV，wake 恢复权重；level-2 丢弃权重，必须明确恢复来源。sleep 不是保存全部运行现场，迁移中的 KV 和 SD 请求状态不能依赖 sleep 自动保留。具体语义参见 [vLLM Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)，实现前仍以当前安装版本为准。

Teacher 不走 `STANDALONE`。`TeacherModelManager._initialize_llm_servers` 对每个副本调用 `init_colocated`，`RolloutReplica.rollout_mode` 因此是 `COLOCATED`。`vLLMHttpServer.sleep` 在 `node_rank==0` 且 `free_cache_engine` 为真时，对 `COLOCATED` 执行 `engine.sleep(level=1)`（权重备份到 CPU，丢弃 KV）。`wake_up` 对 `COLOCATED` 调用 `engine.wake_up(tags=["kv_cache","weights"])` 并 `reset_prefix_cache`，不检查 `free_cache_engine`。`enable_sleep_mode` 来自 `RolloutConfig` 默认值 `True`（teacher `distillation.yaml` 未覆盖），并传入 vLLM 启动参数；为 `False` 时引擎不会按 sleep mode 分配，后续 sleep/wake 不能假定生效。`STANDALONE` 的 sleep/wake 确实直接返回，但那不是当前 Teacher 路径。`distillation.yaml` 里 teacher `free_cache_engine: true`。角色切换仍要显式调用这对 sleep/wake，并在 Teacher 恢复前挡住评分请求。

容量检查需覆盖：

- 两套引擎初始化、profile / CUDA Graph capture 期间的峰值。必要时顺序初始化并让已初始化引擎先 sleep，不能只验证稳态轮换。
- Teacher 权重备份所需主机内存、sleep 后仍驻留的上下文和缓冲，以及 wake 时的临时峰值。
- 辅助 Student 的 target、draft、KV 和执行缓冲，以及 Teacher 恢复后完整评分的容量。
- 第二步及以后、Actor optimizer 状态已经分配、额外副本重新发布时的容量和角色状态。

H800 上 32B Teacher 与 8B Student 的 BF16 TP2 权重粗估合计约 40 GB/卡，不含 draft、KV 和执行缓冲。两套权重常驻、仅轮换 KV / 执行资源可能减少搬运，但当前没有容量保证。它是后续在容量和切换耗时明确后可选的优化，不作为首版前提；也不能把最新上游的仅释放 KV 接口视为项目当前版本已经可用。

## 训练语义和适用范围

- 本步所有生成和续写使用同一 Student 策略版本。Teacher 模型、蒸馏目标、batch、response 上限、SD 参数、精度和优化器与对照一致。
- 每条轨迹恰好训练一次，每个有效 response token 使用正确的 rollout / Teacher logprob；迁移不重复前缀、不多给生成预算、不把中断视为样本丢弃。
- 不要求迁移前后的随机生成与未迁移逐 token 相同，但续写的采样参数和策略版本必须一致；在线生成长度变化必须在性能解释中保留。
- 首版限定当前单节点、单轮文本 OPD、一个 Teacher TP2 副本、同构 Student TP2 副本、同步 `sd-early`。Teacher follow 关闭。
- 不与 colocated Actor overlap、早训迁移或固定分离 Actor 模式同时启用；不新增训练组，不训练未完成前缀，不改变完整 batch 更新边界。
- 功能默认关闭，关闭时直接恢复原 `sd-early` 路径，不启动辅助副本，也不保留额外切换和参数发布成本。
- 初始化或切换失败时应清理本次任务资源并报告失败，不静默丢弃样本、用旧参数运行或继续声称完成整步。

## 现有实现的接入位置

以下路径均相对本 veRL worktree，沿用已有接口做必要修改，不提前搭建通用弹性调度框架：

| 位置 | 需要解决的问题 |
|---|---|
| `verl/trainer/ppo/v1/trainer_base.py` | Teacher 共享卡组、辅助推理副本初始化和资源映射 |
| `verl/trainer/ppo/v1/trainer_sync.py` | 本步借用、切换、Student barrier、原 sd-early 及下一步准备 |
| `verl/trainer/ppo/v1/agent_loop_tq.py`、`replay_buffer.py` | Student-ready 与 Teacher-ready 独立推进，评分等待不阻塞切换 |
| `verl/experimental/teacher_loop/teacher_model.py`、`teacher_manager.py` | Teacher 请求准入、暂停与恢复 |
| `verl/workers/rollout/llm_server.py`、`vllm_rollout/vllm_async_server.py` | 副本路由、部分输出续写、停止接单和实际 sleep/wake |
| `verl/checkpoint_engine/` 及当前发布路径 | 辅助 Student 参数更新、SD draft 保留和同步完成屏障 |

配置沿用 `trainer.v1.sync` 层级和现有启动器覆盖方式，仅增加启用开关、完成门槛、最大借用时间等本方案必要参数。具体字段名随实现确定；本文没有宣布新的可用命令行开关。

## 验证与性能判断

先回答三个组件问题：多一个 TP2 Student 副本实际减少多少生成时间；两引擎切换和迁移续写需要多久；Teacher 延后启动是否推迟最终评分和更新。组件验证沿用现有输入和计时方式，不从单个组件结果推算已经实现的整步收益。

H800 现有 4K `sd-early` 记录为 Student 12.27 秒、Teacher 尾部 8.33 秒、整步 25.09 秒，采用历史 step 2–5 口径。这些数值只说明机会与瓶颈，正式候选比较仍需相同节点、配置和统计口径。Teacher 在第一条完整 Student 结束后就能评分，不是一直等到全部 Student 完成；借用期间因此可能损失原本可重叠的评分工作。

原三个 TP2 副本承接 B96 时约每副本 32 条，新增副本后约 24 条。即使没有请求排队，副本增多也未必线性缩短 SD 单轮延迟或最长轨迹时间。理想线性吞吐下，6 卡临时变 8 卡，借用 t 秒相当于原六卡约 t/3 秒的额外工作；这只是容量估算，未包含长尾、切换、重算和评分延迟。

实现验证应覆盖迁移边界的 token / logprob 拼接、停止条件、长度预算、版本一致性、自然完成与中断竞争、门槛/时间上限、空迁移、全部 Student 已结束、异常清理以及关闭功能的原路径。使用现有 CPU 调度测试和必要的实卡验证；不要求为文档编写测试。

完整实卡验证至少走完既定五步，覆盖后续步重新切换和更新后的辅助副本。检查每步 batch 和有效 token 训练量完整、loss / grad norm 有限、一次更新、下一步使用新版本，以及完整训练正常退出。

性能以同配方 `sd-early` 对照的完整整步墙钟为主，复用已有运行日志和结果文档，补充定位问题必需的切换/迁移/额外发布耗时即可。关注 Student 完成、Teacher 最终评分、Actor 更新完成和下一步可生成的实际时间，不能把重叠区间简单相加。保留全部计时步，不剔除较慢结果。

本机结果写入外层 `docs/results/6gpu-0.6b-from-8b.md`，遵循已有版式，H800 结果单独记录。生成 token 量和轨迹不同、运行波动与较小收益必须如实说明；一次短跑或 Student 提前结束不能单独证明稳定加速。不为本方案生成哈希、校验和、provenance、证据包或审计台账。

## 第二阶段：常驻显存，只互斥请求

对齐后的 level-1 轮换整步 41.323 秒，对照基线 37.251 秒，慢 10.9%。后 3 步 Student 32.615 秒、Teacher 尾 3.644 秒、Actor 在 Student 后 4.454 秒、主发布 1.097 秒。准备 3.078 秒里 Teacher sleep 0.718 秒、aux wake 0.152 秒、aux 发布约 2.2 秒。token 55869 对基线 55861。每步 16 条迁移，版本 0 到 4，前缀非零。多出来的时间主要是每步开头的第二次发布，不是生成本身。

vLLM 0.24 的 `request_memory` 按整卡容量乘 `gpu_memory_utilization` 计算，不按当时剩余空闲。Teacher 保持 0.4。辅助副本单独用 0.2，NCCL bucket 256MB；引擎同时持有 send/recv，峰值是两倍 bucket。主路 naive 仍用配置里的 2048MB IPC bucket，不改 Student / Teacher 原配方。初始化时只在 aux profile 和 CUDA graph 期间让 Teacher sleep 一次，之后两套引擎都保持醒着。每步只关一边的 `set_serving`、打开另一边；迁移后 aux 引擎处于 pause，下次借用前 `resume_generation`。功能关闭或借用时间为 0 时不创建辅助副本，也不走 fused 发布。

新权重在 optimizer 之后才存在，来不及和本步 Teacher 评分重叠。步末 `naive_aux` 用同一次 FSDP `get_per_tensor_param` 写入 aux NCCL bucket，再交给原来的 naive IPC。broadcast 在独立 NCCL 组上启动，下一次 flush 复用这块 buffer 之前才等待完成。实测这次重叠没有把 `update_weights` 从大约 2.0 秒降下来：aux 的 cupy 拷贝加 NCCL 仍叠在 naive IPC 的墙上。SD draft 不在这次权重里，aux 不再每步 sleep，draft 留在显存里。

同配方实测见外层 `docs/results/6gpu-0.6b-from-8b.md` 的最终代码汇总。aux 进程只能看见 Teacher 卡，不能打开 Student 卡上的 IPC buffer，所以没有做成同一次 IPC。

## 实际实现状态

当前主路源码已冻结，供只读 review。本机 0.6B←8B 已跑完：功能开且借用 0 的对照两组（另有一次 wake OOM 后重跑成功），以及同一冻结二进制上的 2、8、12、16 秒窗口。4 秒只存在于更早二进制，不在这张最终表里，也不从 12 秒外推。没有稳定的整步加速。不再改发布架构。

## 实际配置

- 入口：外层 `scripts/profile/teacher_assisted_6gpu.sh`，`FOPD_VERL_DIR` 指向本 worktree。`CUDA_VISIBLE_DEVICES=2,3,4,5,6,7`。不用 MPS。`param_offload=False`，`optimizer_offload=False`。两侧 `max_num_seqs=32`。Student SD k=3，`TEACHER_FOLLOW=False`，`EARLY_ACTOR_LITE=True`，`OPD_PUBLICATION_GC_FREEZE_STEP=2`。5 step，前 2 warmup，后 3 取平均。
- `teacher_assisted_rollout=True` 且 `teacher_assisted_max_borrow_s>0` 才建辅助副本并走 fused 发布。借用时间为 0，或功能关闭，跳过辅助初始化和额外发布。
- Teacher `gpu_memory_utilization` 保持 0.4。辅助副本 0.2，NCCL bucket 256MB，`multi_sender=False`（只有 actor rank 0 发送）。主路 naive IPC bucket 仍是 2048MB。初始化只在 aux profile / CUDA graph 期间让 Teacher sleep 一次，之后两套常驻，每步只切换 `set_serving`；迁移后 aux `pause_generation`，下次借用前 `resume_generation`。
- 步末一次 FSDP gather：aux 走 NCCL，主 Student 走原来的 naive IPC。IPC 拷贝留在 gather 所在 stream；fused 握手只同步当前 stream。NCCL flush / finish 仍做设备同步。

## 限制

本机 3090 上 Teacher 与 0.6B 辅助 Student 能同时放下，不能推出 8B←32B 在 H800 上也能同时放下。该目标配方没有在远端跑过。aux 看不见 Student 卡，不能共用一块 IPC buffer。校正后的 fused 发布大约 1.62–1.81 秒，借用 0 大约 0.97–0.99 秒，多出来的约 0.65 秒没有被稳定更短的生成关键路径盖住。窗口之间不是单调的，未测窗口不能由邻窗代替。不生成哈希、校验和、provenance、证据包或审计台账。

## 验证命令

CPU，在本 worktree：

```bash
/localdata/hlife/miniconda3/envs/fopd/bin/python -m pytest tests/trainer/ppo/v1/test_teacher_assisted_on_cpu.py -q
```

相关 CPU 集合（v1 on_cpu trainer、vLLM 权重更新、engine worker LoRA 同步）在冻结前为 121 项通过。

实卡，在外层 fopd，先确认 GPU 2–7 空闲，每次换一个 `FOPD_RUN_DIR`：

```bash
export TEACHER_ASSISTED_BORROW_S=8
export FOPD_RUN_DIR=/csproject/fyp26_bl1/fopd/runs/6gpu-0.6b-from-8b/teacher-assisted-resident-8s-<时间>
bash scripts/profile/teacher_assisted_6gpu.sh
python scripts/profile/sd_early_borrow_window.py "$FOPD_RUN_DIR" --warmup 2
```

借用 0 把 `TEACHER_ASSISTED_BORROW_S` 设为 0。数字与有效/失效组写在外层 `docs/results/6gpu-0.6b-from-8b.md` 最后一节。

## 四卡 TP1

本机四卡公平比较是 3 个 Student/Actor TP1 加 1 个 Teacher TP1，不是把 3+1 改成 2+2，也不是六卡。Student TP 与 Teacher TP 必须相同，Teacher 池必须正好是这一份副本。TP1 和原来的 TP2 都能过校验。

Teacher `gpu_memory_utilization=0.85` 时记录占用约 21.4GiB。辅助副本仍是 0.2。vLLM 按整卡比例预留，0.85+0.2 超过一张 24GiB 3090，所以四卡不常驻。`resident_engines` 在两者之和大于 0.95 时强制 level-1 sleep/wake，即使把 `teacher_assisted_resident` 写成 true 也不会两套同时醒着。六卡 Teacher 0.4 加辅助 0.2 仍走原来的常驻互斥。四卡启动脚本显式写 `teacher_assisted_resident=False`。

轮换：借用开始时 Teacher `sleep`（COLOCATED level-1），再 wake 辅助 Student，用已有 NCCL `update_weights` 把当前已发布版本送进辅助副本，然后才接请求。切换时迁移、辅助 `sleep`、Teacher `wake_up`。步末主 Student 仍走原来的 naive 发布，不在辅助睡着时做 fused 发布。Teacher 显存比例、batch、SD draft、offload、`max_num_seqs` 都不为了常驻改掉。辅助副本复制整份 Student rollout 配置，draft 块留在里面。

完成率是本步完整 Student 轨迹数除以 48，`ceil` 后 5%–25% 为 3、5、8、10、12。时间上限只做兜底（120 秒），`max_borrow_s=0` 仍是关闭功能。触发原因记在日志里；`switch_reason_code` 1 是 `complete_count`，2 是时间上限，3 是全部 Student 已完成。poll 约 0.05 秒，超过门槛的条数在 `complete_overshoot`。

四卡复跑，GPU 固定为同一次实验选中的四张空闲卡（本次 2、3 已被占用，套用 4、5、6、7）：

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TEACHER_ASSISTED_RATIO=0.10
export TEACHER_ASSISTED_BORROW_S=120
export FOPD_RUN_DIR=/csproject/fyp26_bl1/fopd/runs/4gpu-0.6b-from-8b/teacher-assisted-ratio-0.10-<时间>
bash scripts/profile/teacher_assisted_4gpu.sh
python scripts/profile/sd_early_borrow_window.py "$FOPD_RUN_DIR" --warmup 2
```

结果写在外层 `docs/results/4gpu-0.6b-from-8b.md`，与六卡和 H800 分开。本机两套能否放下仍然不说明 8B←32B 的 H800 可以，那边没有跑过。
