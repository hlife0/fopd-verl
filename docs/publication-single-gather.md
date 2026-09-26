# 单次 gather 对照

候选是 `experiment/publication-single-gather`（`961bca2c`），从强基线 `63b06066` 只改 FSDP1 非 LoRA 的 `get_per_tensor_param`：一次 `FULL_STATE_DICT`（`offload_to_cpu=False`，`rank0_only=False`），不再在 `SHARDED state_dict()` 之后对每个参数 `full_tensor()`。没有 wake 重叠，没有 `no_sync`。基线是 `experiment/student-decode-graph` 的同一提交。两边都没有诊断分支里每个 Actor chunk 的 `ev1.synchronize()`。

四次都是新目录，GPU 4–7，一次只跑一个作业。5 step，前 2 步 warmup，后 3 步平均。`FULL_AND_PIECEWISE`，Ray `runtime_env` 里 `OPD_PUBLICATION_GC_FREEZE_STEP=2`。`train_exit=0`，无 OOM。每步 48 条，`response/aborted_ratio=0`，distillation loss 和 grad norm 有限。`freeze_called=True` 只在 step 2。日志有 `Capturing decode CUDA graphs (FULL)`。没有 `FOPD_WINDOW_TRACE`。发布数是 `update_weights` 墙钟，这次没有再加分段 trace。

顺序 A/B/B/A。后 3 步均值（秒）：整步 / Student / Teacher 尾 / Actor-after-Teacher / 发布 / response tokens。

| 运行 | 整步 | Student | Teacher 尾 | Actor-after-T | 发布 | resp |
|---|---:|---:|---:|---:|---:|---:|
| A1 `publication-ab-A1-20260926_191721` | 24.269 | 16.733 | 4.855 | 1.319 | 1.308 | 55230.67 |
| B1 `publication-ab-B1-20260926_192313` | 23.606 | 16.806 | 4.690 | 1.144 | 0.917 | 54224.33 |
| B2 `publication-ab-B2-20260926_192842` | 23.105 | 17.099 | 3.798 | 1.222 | 0.926 | 53698.00 |
| A2 `publication-ab-A2-20260926_193344` | 23.013 | 16.200 | 4.479 | 0.978 | 1.297 | 52567.00 |

两对发布：A 1.308 / 1.297，B 0.917 / 0.926。每个 profile step 的 B 都低于每个 A（A 最小 1.281，B 最大 0.952）。后 3 步发布均值 A 1.303，B 0.922，少 0.381 秒。

整步均值 A 23.641，B 23.356，少 0.286 秒。这不是稳定的整步加速：同一臂的 Student 就能差约 0.5 秒，A1 到 A2 的整步差 1.26 秒。response token 两臂总均值接近（A 约 53900，B 约 53960），但单次可以差几千 token。全量 state dict 和原来的 shard 再 `full_tensor` 不是逐位相同的采样路径，后续 step 的生成长度会变，Teacher 尾跟着变。发布墙钟才是这次对照里分开的量。不能用以前 FULL 臂的 23.329 秒代替这四次。

诊断里约 2 秒的副本均衡上限仍然不可实现，不记成这次的加速。
