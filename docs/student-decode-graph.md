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

Hydra 覆盖的键是 `actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode`。Teacher 的 `distillation.yaml` 仍是 `enforce_eager: true`、`engine_kwargs: {}`。Teacher 进程日志里出现的 `FULL_AND_PIECEWISE` 来自 `vllm_async_server` 在没有 compilation_config 时的 `setdefault`，紧接着被 `--enforce_eager` 关掉，没有 decode 图。Actor 的 offload、loss、学习率没有改。`baseline/sd-early` 本身停在 `391ac71f`。本分支在 `update_weights` 上加了一行两边共用的 `PUBLICATION_STATE` stdout，用来看 worker 里的 freeze 环境；发布冻结条件对 `OPD_PUBLICATION_GC_FREEZE_STEP=2` 与 sd-early 相同。

## 旧共同缺GC

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

两对后 3 步均值：P 臂 28.6205 / 19.1285 / 5.558 / 1.2505 / 2.621 / 56227.67，F 臂 25.5495 / 17.1035 / 4.574 / 1.1515 / 2.652 / 55145.33。F 相对 P：整步 −3.071 秒（10.73%），Student −2.025 秒（10.59%），response −1082.34（−1.92%）。发布差约 +0.03 秒。这两对发布都在 2.6 秒，是共同漏了 `OPD_PUBLICATION_GC_FREEZE_STEP`。这组数字单独保留，不和下面强基线的均值混在一起。

## 验收

捕获图不等于每步都 replay。vLLM 在 draft 的第 2、3 步把 batch 标成每个请求 1 token，再 `dispatch`。命中 FULL 描述符时 `run_fullgraph` 调用 `graph.replay()`；对不上则返回 `NONE`，draft decode 走 eager。本配方 `max_num_seqs=32`，decode FULL 只捕获 4、8、16、32。token 数 1–32 会 pad 到下一档再 replay，不是掉回 eager。单副本不能超过 32 条，所以 draft decode 不会因尺寸超出已捕获档而回退。prefill 和长短不一的 mixed batch 仍走 piecewise。四卡 `train.log` 没有逐步 replay 次数（`cudagraph_metrics` 没开），不把捕获行写成每步计数。

GPU 1 第二次：capture 在 `LLM()` 初始化里，计时在那之后。batch 16 正好是捕获档。PIECEWISE 1.007 秒，FULL_AND_PIECEWISE 0.824 秒，16×128 token id 相同。这是 replay 窗口，不是 capture 开销，也不是四卡整步。

四组 Hydra 的 Student/Teacher 显存 0.4/0.85、两边 `max_num_seqs=32`、`max_num_batched_tokens=8192` 相同。`train.log` 没有 preemption 字样，也没有 GPU block 数。KV 容量的实际块数缺失，不补。接受长度、acceptance rate 也没有记录。`actor/distillation/ppo_kl` 有：四组五步都在 0.000388–0.001250，loss 有限，在 0.088–0.300。没有爆炸。采样仍是 `rejection_sample_method=standard`、k=3、`draft_sample_method=greedy`。Teacher 四组都是 `enforce_eager: true`，decode FULL 图不在 Teacher 上。

## 强baseline正确GC

已完成两对、四组新强基线配置。启动内容对应 `7141d545`：两边脚本都把 `runtime_env.env_vars.OPD_PUBLICATION_GC_FREEZE_STEP='2'` 放进 Hydra，shell 变量 unset。`a9bb6c9c` 曾从脚本拿掉这一行，用来对齐上面缺 GC 的四组；这四组没有改用那一版脚本。GPU 4–7，配方其余不变，只差 Student 图模式。`train_exit=0`，`teacher_follow=False`。四组 ray init 都有该键。日志里没有 `OPD publication GC frozen` warning；缺 warning 是日志传播，发布从 step 3 起约 1.3 秒，说明冻结已经生效。

`PUBLICATION_STATE` 这行 stdout 是在 P1 结束之后、F1 启动之前写进工作区的。所以第一对的观测不一样：P1 的 `train.log` 没有这行；F1 有，`env='2'`、`publication_opt=True`，`freeze_called=True` 只在 `global_steps=2`。第二对两边都保留同一行，运行时代码与 `2b655a22` 相同。

| 运行 | 模式 | 整步 | Student | Teacher尾 | Actor-after-T | 发布 | resp |
|---|---|---:|---:|---:|---:|---:|---:|
| P1 piecewise-154529 | PIECEWISE | 26.068 | 19.160 | 4.542 | 1.003 | 1.300 | 52980.67 |
| F1 graph-155943 | FULL | 23.958 | 17.304 | 4.046 | 1.237 | 1.300 | 55804.33 |
| F1 减 P1 |  | -2.110 | -1.856 | -0.496 | +0.234 | 0.000 | +2823.67 |
| P2 piecewise-gc-160530 | PIECEWISE | 25.971 | 18.571 | 5.030 | 1.018 | 1.290 | 52929.67 |
| F2 graph-gc-161532 | FULL | 22.699 | 16.712 | 3.424 | 1.195 | 1.304 | 52961.00 |
| F2 减 P2 |  | -3.272 | -1.859 | -1.606 | +0.177 | +0.014 | +31.33 |

P1 逐步：39.788 / 19.926 / 2.923 / 1.494 / 2.060 / 48193（9）；25.378 / 19.196 / 3.403 / 1.009 / 1.687 / 45534（8）；26.624 / 19.249 / 4.949 / 1.070 / 1.293 / 54250（11）；25.873 / 19.449 / 4.199 / 0.894 / 1.270 / 49497（11）；25.707 / 18.781 / 4.478 / 1.046 / 1.338 / 55195（10）。没有 `Capturing decode CUDA graphs`。

F1 逐步：35.768 / 15.756 / 3.008 / 1.901 / 1.998 / 46927（7）；22.476 / 16.649 / 2.496 / 1.559 / 1.704 / 48779（9）；24.765 / 17.323 / 4.643 / 1.439 / 1.297 / 57754（10）；21.613 / 16.797 / 2.043 / 1.384 / 1.313 / 51697（9）；25.497 / 17.792 / 5.453 / 0.889 / 1.289 / 57962（14）。有 `Capturing decode CUDA graphs (FULL)`。

P2 逐步：39.093 / 20.188 / 2.791 / 1.708 / 1.963 / 52472（7）；26.198 / 18.504 / 5.144 / 0.861 / 1.620 / 49939（11）；25.935 / 18.589 / 4.954 / 1.060 / 1.272 / 50801（11）；24.630 / 18.429 / 3.780 / 1.033 / 1.326 / 50657（8）；27.348 / 18.696 / 6.355 / 0.962 / 1.271 / 57331（13）。没有 decode 图。`PUBLICATION_STATE`：`env='2'`、`publication_opt=True`，`freeze_called=True` 只在 step 2。

F2 逐步：36.994 / 16.561 / 3.711 / 1.277 / 1.938 / 50532（10）；20.797 / 15.223 / 2.215 / 1.649 / 1.647 / 46837（7）；22.333 / 15.816 / 4.270 / 0.879 / 1.300 / 51882（10）；21.576 / 16.371 / 2.201 / 1.647 / 1.296 / 49309（9）；24.188 / 17.949 / 3.801 / 1.059 / 1.317 / 57692（12）。有 decode FULL。worker 行与 P2 相同：`env='2'`、`publication_opt=True`，`freeze_called=True` 只在 step 2。

两对里，五步整步和五步 Student 都是 FULL 更短。P 臂后 3 步均值 26.020 / 18.866 / 4.786 / 1.011 / 1.295 / 52955.17。F 臂 23.329 / 17.008 / 3.735 / 1.216 / 1.302 / 54382.67。F 相对 P：整步 −2.691 秒（10.34%），Student −1.858 秒（9.85%），response +1427.50（+2.70%），发布 +0.007 秒。第二对 response 几乎相同（+31.33），Student 仍少 1.859 秒。Teacher 尾更短，Teacher 仍是 enforce eager。发布后 3 步四组都在 1.290–1.304，与 `035937` 的 1.314、`publication-state-20260926_043847` 的 1.272 同一水平。每步 48 条，aborted 0。P1 loss 0.112–0.298、`ppo_kl` 0.000596–0.000899；F1 loss 0.110–0.279、`ppo_kl` 0.000553–0.001236；P2 loss 0.109–0.282、`ppo_kl` 0.000546–0.001227；F2 loss 0.144–0.197、`ppo_kl` 0.000588–0.001232。没有显存报错，日志无 preemption。已完成两对、四组新强基线配置。

不进入这四组均值：`student-decode-piecewise-gc-20260926_155411` 是第一对之间多出来的 PIECEWISE。`153646`、`155110`、`155254`、`155956`、`161100` 是中止或初始化失败，不是结果。

复跑（新目录，GPU 4–7，不要开 profiler，不要复用上述目录）：

```bash
bash /csproject/fyp26_bl1/fopd/.worktrees/student-decode-graph/scripts/student_decode_piecewise_4gpu.sh
bash /csproject/fyp26_bl1/fopd/.worktrees/student-decode-graph/scripts/student_decode_graph_4gpu.sh
```

两个脚本都 unset shell 变量，并传入 `+ray_kwargs.ray_init.runtime_env.env_vars.OPD_PUBLICATION_GC_FREEZE_STEP='2'`。ray init 应有该键。`PUBLICATION_STATE` 里 `env='2'`、`freeze_called=True` 只在 `global_steps=2`。缺 warning 不能当成没冻结。PIECEWISE 没有 `Capturing decode CUDA graphs`。FULL 有 `Capturing decode CUDA graphs (FULL)`。Teacher 的 enforce eager 警告两边都有。
