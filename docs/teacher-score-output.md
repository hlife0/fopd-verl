# Teacher 评分输出路径

四卡配方 `distillation_loss_mode=k1`，`topk=64` 不参与这条损失。`use_topk` 为假，Teacher 请求的是 `prompt_logprobs=0`：每个位置只要被选 token 的一个 logprob。`detokenize=False` 已经写在 `_get_teacher_sampling_params` 里。一次性评分不设置 `skip_reading_prefix_cache`，也不要再包一层。

vLLM 0.24 在 `num_logprobs=0` 时不做 `torch.topk`。logsumexp 只写出被选 token，不把整张 vocab 搬到 CPU。另外还有一次 vocab rank 扫描；k1 损失不用这个 rank。

返回张量形状是 `(seq_len, 1)`，float32。`prompt_logprobs[0]` 是空。抽出的第 `i` 行是 token `i+1` 的 logprob，最后一行是 dummy 0。序列末尾的 EOS 落在 dummy 的前一行。数值与源 logprob 的 float32 强制转换一致。

## 2026-09-26 测量

位置数是 **58224**（四万八乘一千二百一十三），不是 5824。它是一次整步的 prompt+response，对应已公布基线 step 5 的 48 条、合计 58245。response token 本身是 54213。0.370 秒已经覆盖这 58224 个位置；按长度线性放大到约 5.8 万不会再乘十。CPU 跑了两遍，取较慢的一遍。GPU 只占用物理 GPU 1，用 Qwen3-8B 的 hidden 4096、vocab 151936，不加载权重；结束后该卡 4 MiB。没有改评分实现，没有提交 GPU 4–7。

| 段 | 整步 58224 个位置 |
|---|---:|
| vLLM 逐位置 dict（detokenize 关闭） | 0.370 秒 |
| 同一数据走 flat 列表 | 0.057 秒 |
| verl extract + `torch.tensor` | 0.038 秒 |
| pickle 现有 list | 0.012 秒，1.05 MB |
| pickle 紧凑 tensor | 0.005 秒，0.49 MB |
| LM-head gemm（57344 token） | 0.938 秒 |
| 被选 token 的 logsumexp | 0.047 秒 |
| rank 扫描 | 0.020 秒 |

把 dict 换成 flat 列表，整步大约少 0.31 秒。现有 extract 如果直接去迭代 flat 容器，会把 dict 重建回来，extract 从 0.038 秒变成 0.078 秒，所以不能只打开 `flat_logprobs`。队列没有在这次探针里分开计时。LM-head 的 0.94 秒是整步 token 的矩阵乘，不是 Teacher 尾，也不含 attention。

整步输出路径上限约 0.42 秒（0.370 + 0.038 + 0.012）。换成 flat 列表大约少 0.31 秒。这 0.31 秒是把 58224 个位置全部算进关键路径的上限。

Student 还在生成时已经评完的序列，dict 构造叠在 Student 里，不延长 Teacher 尾。关键路径上只剩最后一条 Student 结束时仍未 `teacher_done` 的序列。同期对照 `20260926-sd-early-followoff-052727` 的后 3 步，这些尾序列的 prompt+response 是 23602、32119、19343。按 0.370 秒 / 58224 个位置线性缩放，尾上的 dict 时间约 0.150、0.204、0.123 秒；flat 能省下的约 0.127、0.173、0.104 秒。两种口径都低于 0.5 秒整步。不改返回格式，不跑四卡。
