# 四卡 bubble 诊断

唯一运行：`runs/4gpu-0.6b-from-8b/gpu-bubble-profile-20260926_185229`。`train_exit=0`，5 step 自然结束，每步 48 条。这是诊断，不是胜负对照。trace 只在该目录的 `window_trace/`。分析脚本是 `scripts/analyze_gpu_bubble.py`，NVML 采样是 `scripts/gpu_busy_sampler.py`。

这次运行的代码在每个 Actor chunk 上做了 CUDA event synchronize。step 3/5 的 Actor 墙钟因此每个 chunk 大约多串行不超过 0.1 秒（`fb_gpu_span` 与 `fb_launch` 之差）。下面的比较用 `fb_gpu_span` 对 `fb_wall`，以及 trace 里的空隙。

副本 r0/r1/r2 对应 GPU 4/5/6，Teacher 在 GPU 7。`student_replica_rank` 在 step 1–5 都是 0/1/2 各 16 条。

后 3 步（秒）：

| step | 整步 | Student | Teacher 尾 | Actor | 发布 |
|---:|---:|---:|---:|---:|---:|
| 3 | 22.86 | 17.35 | 2.52 | 3.84 | 1.335 |
| 4 | 26.52 | 17.38 | 3.96 | 6.69 | 1.360 |
| 5 | 24.18 | 17.97 | 3.53 | 4.59 | 1.336 |

step 4 的 Actor 含约 2.17 秒 trace 导出。

三副本按条数静态 16/16/16。先排空的卡在屏障前空闲，合计约 4.9–7.1 GPU·秒。按 token 完全均衡的理论上限约 1.6–2.3 秒/步，但提交时不知道长度。第一个副本排空时，慢副本还剩 6–8 条，其中 5–7 条会顶到 2048，并在大约 3–4 秒后一起结束。迁走这些请求最多快约 0.3–0.5 秒，还要付 abort 和重新 prefill。约 2 秒的理论值不可实现，排空迁移不值得做。

Teacher 尾 2.5–4.0 秒是这些长样本的打分，GPU 7 占用 0.95–0.98，队列和 prefill 等待为 0。

发布约 1.33 秒。`pre` 0.23–0.28 秒没有 GPU kernel；`wake_weights` 0.16 秒本 rank 没有 kernel；`gather_ipc_load` 0.85–0.87 秒。step 4 rank0 trace 里这是同一份权重 all-gather 两次：`SHARDED state_dict()` unshard（NCCL 357 毫秒），然后 `full_tensor()` 再 all-gather（NCCL 407 毫秒）。接收端 load 0.013–0.015 秒，wait 约 0.42 秒是在等生产端。消费端没有可与生产端 gather 重叠的工作。

n=3 chunk 的 GPU span 约 0.79 秒，拟合为 0.56 秒固定开销加每千 token 0.036 秒。固定部分主要是每个 chunk 的全量 fp32 grad reduce-scatter 和两次参数 all-gather。GPU 空隙约 49 毫秒。只合并尾部 chunk 在其中一种到达模式下会变慢，不作为首选。

下一步只做单次 gather，不把 wake 重叠或 `no_sync` 混进同一次对照。正式对照不用这条诊断分支，因为这里每个 Actor chunk 都有 `ev1.synchronize()`。
