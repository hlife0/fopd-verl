# Teacher follow：增量计算 / 前缀复用

只改 Teacher 对已经生成的 prefix 怎么评分。不缩短生成，不改训练算法。四卡配方与已公布的 sd-early 基线相同，唯一开关是 `teacher_follow=True`。不要读 `20260926-strong-sd-early-baseline-035937` 目录里现在的 hydra；那次计时表仍然是对照（后 3 步：整步 26.195、Student 19.372、Teacher 尾 4.315、Actor-after-Teacher 1.128、发布 1.314）。

## vLLM 0.24 接口

本地 `vllm 0.24.0`。`SamplingParams.__post_init__` 在 `prompt_logprobs` 不为空且调用方没写 `skip_reading_prefix_cache` 时，把它设成 `True`。注释写明原因：读了 prefix cache 之后，prompt logprob 行数会少于 prompt token 数。`KVCacheManager.get_computed_blocks` 在这个标志为真时直接返回 0 个命中，不查 cache。

标志为假时才会查 cache。命中的 token 不再计算；prompt logprob 只覆盖新算的那段。全命中时最后一个 token 仍要重算，才能拿到 logits。块大小默认 16，和 follow 的 `FOLLOW_KV_BLOCK_SIZE` 一致。

现有 follow 只在 follow 请求里把 `skip_reading_prefix_cache=False` 传进 `SamplingParams`。payload 仍是当前 prefix（中途按 16 对齐，Student 结束后是完整 prefix）。cache 命中时拼接后缀 logprob；没命中就整段重算，结果仍是同一条 response 的 teacher logprob，不是语义错误。

## GPU 1 探针（Qwen3-0.6B，eager，prefix caching 开）

同一条 96 token 序列，先一次性评分（`skip=True`，`cached=0`，95 行 logprob），`reset_prefix_cache` 之后按 32/64/96 增长：

| 请求 | prompt | cached | logprob 行数 | 非空行 |
|---|---:|---:|---:|---:|
| hop32 | 32 | 0 | 32 | 31 |
| hop64 | 64 | 32 | 32 | 31 |
| hop96 | 96 | 64 | 32 | 31 |
| 随后默认 skip | 96 | 0 | 96 | 95 |

命中请求返回的第 i 行（非空）对应绝对 token `cached+i`，不是列表下标 i。这些位置上、被评分 token 的 logprob 相对一次性评分：hop64 中位绝对差 0.00011、最大 0.025；hop96 中位 0.00017、最大 0.0079。复用同一 `request_id` 的命中和差值相同。短请求 hop32 相对长序列的同位置：中位 0.0087、最大 0.088（位置 11），这是两次前向的数值差，不是 cache 把 token 对错。

每个命中块的列表第 0 行是空的，所以 `cached` 那个 token 不在本次 prompt logprob 里。follow 用上一次请求的 decode top-k 补这一个边界。

结论：`prompt_logprobs` 可以按需复用 KV。默认标志会主动跳过 cache 并重算整段 logprob；显式 `skip_reading_prefix_cache=False` 时，命中的 prefix 不再出 logprob，新算的是后缀，且与一次性评分对齐。接口没有否定这条路线。四卡上要看的是并发时最后一次长请求是否仍然 `cached≈0`。

## 四卡一组

`scripts/teacher_follow_incremental_4gpu.sh`。`FOPD_SAMPLE_TRACE_DIR` 和 `FOPD_FOLLOW_CACHE_LOG` 通过 Hydra `runtime_env` 传给 worker。不要扫 chunk / block 参数。

读 `follow_cache.jsonl`：每条有 `payload_len`、`new_tokens`、`num_cached`、`prefill_s`、`student_done`、`hole_fills`。每个 `request_id` 的最后一条是收尾请求。

- 收尾请求 `num_cached/payload_len` 接近 1，且 `sum(new_tokens)/sum(payload_len)` 明显小于 1：复用在这批评分里发生了。
- 中途命中、收尾 `num_cached≈0`：最后的完整 prefix 没吃到 cache（挤掉或被清掉），长序列仍集中重算。
- 全部 `num_cached≈0`：这组训练里标志没有让 worker 去读 cache。
- `cached=0` 只说明没有复用 KV，不说明 logprob 语义错了。
