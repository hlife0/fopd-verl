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

每步 48 条，`aborted_ratio=0`，loss 和 grad 有限。五步 Student 都低于 052727 的 21.218 / 19.121 / 19.723 / 19.617 / 19.787。step 5 整步是 28.525，高于 052727 的 28.042；这一步 17 条顶到 2048（052727 是 12 条），response 62747 对 55006，Teacher 尾 6.353 对 4.431。后 3 步 response/Student 秒大约是 3175、3429、3497，052727 是 2673、2905、2780。采样轨迹不同，loss 不能当逐 token 核对。

单次运行。不写成稳定加速。接下来用同一工作区交错两对：`scripts/student_decode_piecewise_4gpu.sh` 与 `scripts/student_decode_graph_4gpu.sh`，顺序是 PIECEWISE、FULL、PIECEWISE、FULL。两边只差 `cudagraph_mode`。不重跑 `052727`，也不把 profiler 开在计时上。
