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

这是高并发窗口。还没有四卡整步数字。正式计时用 `scripts/student_decode_graph_4gpu.sh`，对照仍是 `20260926-sd-early-followoff-052727`，不要重跑那份。profile 和正式计时分开。
