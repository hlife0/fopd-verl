# Student decode graph

四卡配方把 Student 的 `cudagraph_mode` 设成 `PIECEWISE`。vLLM 0.24 在这个模式下给 target 和 draft 的第一步（prefill）抓 piecewise 图；draft 的后续 decode 不支持 piecewise，代码把 decode 图改成 `NONE`。EAGLE3 k=3 时，每个 verify 步后面有 2 次 draft decode，这两次是 eager。

`20260926-sd-early-followoff-052727` 的 `train.log` 有 `Capturing CUDA graphs (PIECEWISE)` 和 `Capturing prefill CUDA graphs (PIECEWISE)`，没有 `Capturing decode CUDA graphs`。异步调度对 EAGLE 是打开的（组件日志：`Asynchronous scheduling is enabled`）。固定 k=3 不是 dynamic speculative decoding，所以 `FULL_AND_PIECEWISE` 不会被 vLLM 改回 `PIECEWISE`。

Student `calculate_log_probs=True`，每个生成 token 要一个被选 logprob。这和 Teacher k1 的输出路径是同一类对象构造。整步约 5.5 万 response token 的这类构造在 Teacher 探针里是零点几秒，盖不住 Student 的 19–21 秒。这条不改 logprob 返回。

同期对照样本里，后 3 步各有 13、15、12 条 response 顶到 2048。整批最后 2 秒仍有 13、17、12 条结束。时间轴终点的 active=1 只是最后一瞬间，不是尾段只剩 1–2 条。样本 index 每 16 条切开后长答集中在一组，这还没有对上 Ray 副本编号，不把它写成某个副本。

## GPU 1 组件

一个 Student 副本的形状：Qwen3-0.6B，EAGLE3 k=3，draft greedy，batch 16，`logprobs=1`，temperature 0，seed 1，新 token 128。不是训练的 2048，也不当整步性能。GPU 1，跑完 4 MiB。capture 尺寸 4、8、16。

| 模式 | 第 1 次 | 第 2 次 | decode 图 |
|---|---:|---:|---|
| PIECEWISE | 1.083 秒，1891 tok/s | 1.007 秒，2035 tok/s | 没有 |
| FULL_AND_PIECEWISE | 0.857 秒，2390 tok/s | 0.824 秒，2486 tok/s | `Capturing decode CUDA graphs (FULL)` |

两次 16 条、每条 128 token 的 id 完全相同。采样、rejection method、k 都没改。只多覆盖 draft decode 的 FULL 图。

这是高并发窗口，不是训练的 2048。profile 和正式计时分开。

## 四卡单次

`runs/4gpu-0.6b-from-8b/student-decode-graph-20260926_055951`。`train_exit=0`，`teacher_follow=False`。Hydra 打印 `cudagraph_mode=FULL_AND_PIECEWISE`。三个 Student 服务有 `Capturing decode CUDA graphs (FULL)`（4/4），prefill 同时有 PIECEWISE 和 FULL。另一进程 `2018527` 是 enforce eager，关掉了 CUDA graph，decode 图不是它抓的。没有写入 `052727`。k 仍是 3。

对照是已有的 `20260926-sd-early-followoff-052727`，没有重跑。口径：整步 / Student / Teacher 尾 / Actor-after-T / 发布 / response token。Actor-after-T 是样本 `actor_done_ts` 减最晚 `teacher_done_ts`。token 是 48 条 `response_len` 之和。

| | 整步 | Student | Teacher尾 | Actor-after-T | 发布 | resp |
|---|---:|---:|---:|---:|---:|---:|
| 本组 3–5 | 25.816 | 16.465 | 5.432 | 1.201 | 2.647 | 55547 |
| 052727 3–5 | 28.908 | 19.709 | 5.454 | 0.997 | 2.689 | 54902.67 |
| 本组减 052727 | -3.092 | -3.244 | -0.022 | +0.204 | -0.042 | +644 |

| step | 整步 | Student | Teacher尾 | Actor-after-T | 发布 | resp | 顶到 2048 | loss | grad |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 38.104 | 17.739 | 1.569 | 2.901 | 2.828 | 50326 | 7 | 0.292 | 12.691 |
| 2 | 22.144 | 16.020 | 1.634 | 1.635 | 2.783 | 46719 | 5 | 0.296 | 11.718 |
| 3 | 23.668 | 15.634 | 4.290 | 1.041 | 2.633 | 49644 | 11 | 0.144 | 12.480 |
| 4 | 25.253 | 15.820 | 5.653 | 1.061 | 2.646 | 54250 | 12 | 0.136 | 11.580 |
| 5 | 28.525 | 17.941 | 6.353 | 1.499 | 2.662 | 62747 | 17 | 0.090 | 10.474 |

每步 48 条，`aborted_ratio=0`，loss 和 grad 有限。五步 Student 都低于 052727 的 21.218 / 19.121 / 19.723 / 19.617 / 19.787。step 5 整步是 28.525，高于 052727 的 28.042；这一步 17 条顶到 2048（052727 是 12 条），response 62747 对 55006，Teacher 尾 6.353 对 4.431。后 3 步 response/Student 秒大约是 3175、3429、3497，052727 是 2673、2905、2780。采样轨迹不同，loss 不能当逐 token 核对。这一组和 `052727` 不是同一时刻的交错对，单独不能当稳定结论。

## 机制

`baseline/sd-early` 的公平脚本把 Student 设成 `PIECEWISE`。vLLM 0.24 的 `FULL_AND_PIECEWISE` 是 `(FULL, PIECEWISE)`：decode 走 FULL，prefill/mixed 走 PIECEWISE。`PIECEWISE` 的 `decode_mode()` 仍是 `PIECEWISE`。EAGLE speculator 只在 `decode_mode()==FULL` 时把 draft decode manager 设成 `FULL_DECODE_ONLY`，否则设成 `NONE`。k=3 时 `num_speculative_steps>1`，所以会去抓 decode 图；`PIECEWISE` 下这张图是空的，每个 verify 之后的 2 次 draft decode 走 eager。`FULL_AND_PIECEWISE` 补的是这段 draft decode，prefill 的 piecewise 图还在。

固定 `num_speculative_tokens=3` 没有 `num_speculative_tokens_per_batch_size`，不是 dynamic speculative decoding，vLLM 不会把 FULL 改回 PIECEWISE。`eagle3` 属于 async scheduling 默认可开的方法。Student `calculate_log_probs=True` 仍在公平脚本里，两个启动脚本都不改它，也不改 `rejection_sample_method=standard`、`draft_sample_method=greedy`。`cudagraph_mode` 不是 `SamplingParams` 的字段。GPU 1 上 temperature 0 的 16×128 token id 相同，只说明这条贪心路径的输出没变；四卡训练温度不是 0，loss 不能当逐 token 核对。

覆盖范围按 vLLM 的候选规则：decode 图只收 `decode_query_len <= 尺寸 <= max_num_seqs`。本配方 `max_num_seqs=32`，capture 列表是 4、8、16、32、64、96、128，所以 decode FULL 是 4 档，prefill 仍是 7 档。正式日志里 decode 进度条是 4/4，prefill 是 7/7。图捕获发生在第一步计时之前，不进 `timing_s/step`。GPU 1 的 128-token profile 只在组件日志里，没有和这五次正式计时混在一起。

Hydra 覆盖的键是 `actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode`。Teacher 的 `distillation.yaml` 仍是 `enforce_eager: true`、`engine_kwargs: {}`。Teacher 进程日志里出现的 `FULL_AND_PIECEWISE` 来自 `vllm_async_server` 在没有 compilation_config 时的 `setdefault`，紧接着被 `--enforce_eager` 关掉，没有 decode 图。Actor 的 offload、loss、学习率没有改。相对 `391ac71f` 没有改 verl 运行时代码，只有本分支的启动脚本、说明和 CPU 测试。

## 交错复验

同一工作区、同一脚本，只换上面这一个键。顺序是 PIECEWISE、FULL、PIECEWISE、FULL。都是 `train_exit=0`，`teacher_follow=False`，每步 48 条，aborted 0，loss 和 grad 有限。没有重跑 `052727`，没有 profiler。

| 运行 | 模式 | 整步 | Student | Teacher尾 | Actor-after-T | 发布 | resp |
|---|---|---:|---:|---:|---:|---:|---:|
| piecewise-060904 | PIECEWISE | 28.750 | 19.146 | 5.798 | 1.127 | 2.616 | 55873.67 |
| graph-061430 | FULL | 25.384 | 16.719 | 4.736 | 1.200 | 2.660 | 54747.33 |
| 061430 减 060904 |  | -3.366 | -2.427 | -1.062 | +0.073 | +0.044 | -1126 |
| piecewise-061942 | PIECEWISE | 28.491 | 19.111 | 5.318 | 1.374 | 2.626 | 56581.67 |
| graph-062506 | FULL | 25.715 | 17.488 | 4.412 | 1.103 | 2.644 | 55543.33 |
| 062506 减 061942 |  | -2.776 | -1.623 | -0.906 | -0.271 | +0.018 | -1038 |

两对里，FULL 的五步整步和五步 Student 都低于同对 PIECEWISE。后 3 步 Student：060904 是 19.175 / 19.354 / 18.909，061430 是 16.618 / 16.165 / 17.374；061942 是 18.929 / 19.046 / 19.359，062506 是 17.948 / 17.289 / 17.226。三组 FULL 的后 3 步整步均值（25.816、25.384、25.715）都低于两组同期 PIECEWISE（28.750、28.491）和 `052727`（28.908）。Student 均值 16.465、16.719、17.488 都低于 19.146、19.111、19.709。

后 3 步 response/Student 秒：060904 约 2873、3008、2873；061430 约 3355、3264、3207；061942 约 2977、2859、3045；062506 约 2964、3258、3316。061942 的 step 3 和 062506 的 step 3 速率接近，Student 少的约 1 秒对着更少的 response（56359 对 53193），这一步的时间差主要是长度。其余 profile 步 FULL 的速率更高；061430 的 step 3 response 还更多（55761 对 55094），Student 仍是 16.618 对 19.175。

Teacher 尾在 FULL 运行上也更短，但 Teacher 两边都是 enforce eager，没有 decode 图。这段不记成 Teacher 优化。发布仍在 2.6 秒附近。`055951` 的 step 5 整步仍可以高于 `052727` 的 step 5，长答顶到 2048 时单步整步会重叠。

上面四组和 `052727`、follow 的发布都在 2.6–2.8 秒，而且从 step 1 起就是这条线。这不是 FULL 和 PIECEWISE 之间的差：两对发布差是 +0.044 和 +0.018 秒。

## 发布翻倍

`035937` 后 3 步发布 1.314 秒，逐步 2.075、1.703、1.259、1.320、1.364。`publication-state-20260926_043847` 把 `OPD_PUBLICATION_GC_FREEZE_STEP=2` 只放进 Hydra `runtime_env`，worker 上 `publication_opt=True`，`freeze_called=True` 只在 `global_steps=2`。那次发布逐步 2.049、1.640、1.257、1.273、1.286，后 3 步 1.272 秒。warning 没进 `train.log`，不能用来判断 freeze 有没有跑。

sd-early 的开关在 `update_weights`：变量非空时，`aggressive_empty_cache` 从最多 3 次改成 1 次，并跳过权重同步后的第二次清理；变量等于当前 step 时再 `gc.freeze()`。公平脚本不写这个变量。`035937` 的 Hydra `runtime_env` 里没有它，但发布曲线和确认开过开关的那次一样，对应启动 shell 继承进 worker。`052727`、follow 和上面四组的 ray init 都没有这个键；本分支脚本还 `unset` 了 shell 变量，又没有把它写进 `runtime_env`。发布因此停在大约 2.6 秒，没有 step 2 之后的下落。

其余公共项与 `035937` 的 train.log 一致：offload false、checkpoint backend naive、bucket 2048、`free_cache_engine` true、Teacher enforce eager、EAGLE3 k=3、`calculate_log_probs=True`。没有为了比较把这些改弱。缺的是这一项已经在 sd-early 里的发布开关。交错四组彼此仍然同配置，Student 的差距不是靠拿掉这个开关造出来的；它们不能代替带这个开关的 `035937` 当强基线。

两个启动脚本现在都保持 shell unset，并加上同一条 `runtime_env` 覆盖。图模式仍是唯一差别。用这一对再验一次，不重跑已有目录。

复跑（新目录，GPU 4–7，不要把 profiler 开在计时上）：

```bash
bash /csproject/fyp26_bl1/fopd/.worktrees/student-decode-graph/scripts/student_decode_piecewise_4gpu.sh
bash /csproject/fyp26_bl1/fopd/.worktrees/student-decode-graph/scripts/student_decode_graph_4gpu.sh
```

PIECEWISE 的 `train.log` 应是 Student `cudagraph_mode=PIECEWISE`，有 prefill PIECEWISE，没有 `Capturing decode CUDA graphs`。FULL 应是 `FULL_AND_PIECEWISE`，并且有 `Capturing decode CUDA graphs (FULL)`。两边的 ray init `env_vars` 都要有 `OPD_PUBLICATION_GC_FREEZE_STEP` 为 `2`。有效时发布在 step 2 之后落到 1.3 秒附近，而不是停在 2.6 秒。没有 `OPD publication GC frozen` 这句 warning 不能当成没冻结。Teacher 进程的 enforce eager 警告两边都会出现。
