# Publication state diagnostic

One `PUBLICATION_STATE` line is printed from each rollout worker on every `update_weights` call. It records only `global_steps` (value and type), `repr` of `OPD_PUBLICATION_GC_FREEZE_STEP`, the effective mode, whether that variable is non-empty (`publication_opt`), `gc.isenabled()`, and whether `gc.freeze()` returned on this call.

The 4-GPU launcher unsets the shell variable and passes `2` only through Hydra `runtime_env`.

Read the lines, not the source:

- `env=''` means this worker did not receive the variable.
- `mode` other than `'naive'`, `global_steps` of `None` or a list, or steps that skip `2`, means the freeze condition was not met. `freeze_called=True` only after `gc.freeze()` returns, and only when the value equals `2`.
- `env='2'`, `mode='naive'`, a step whose `global_steps=2`, and `freeze_called=True` means the worker ran the existing switch. A missing older `GC frozen` warning in that case is log propagation.

Compare step time and `timing_s/update_weights` with `runs/4gpu-0.6b-from-8b/20260926-strong-sd-early-baseline-035937` (step 26.195 s, publish 1.314 s). The switch, if it runs, sits inside publish.

## 2026-09-26 结论

判别是日志传播，不是 worker 没收到变量，也不是 naive / step 条件没走到。

运行 `runs/4gpu-0.6b-from-8b/publication-state-20260926_043847`，`train_exit=0`。启动 shell 没有 `OPD_PUBLICATION_GC_FREEZE_STEP`；ray init 的 `runtime_env.env_vars` 里是 `'2'`。三个 rank 的 worker stdout 共 18 行：`global_steps` 为 int 的 0、1、2、3、4、5，没有 `None`、没有列表、没有跨过 2。`env='2'`，`mode='naive'`，`publication_opt=True`，`gc_enabled=True`。`freeze_called=True` 只出现在 `global_steps=2`，并且是在 `gc.freeze()` 返回之后。`global_steps=0` 是训练步之前的一次发布，条件是等于 2，所以不冻结。

同一调用里的 `logger.warning("OPD publication GC frozen ...")` 没有出现在该次 `train.log`、Ray session `session_2026-09-26_04-38-53_676822_1737272` 的 worker stdout/stderr，或该 session 的其它日志里。`PUBLICATION_STATE` 的 `print` 在 `train.log` 和 worker stdout 里都有。因此早先基线里缺少这句 warning，不能用来判断 freeze 有没有执行。

后 3 步平均：整步 26.360（基线 26.195，+0.165），Student 18.822（19.372），Teacher 尾 5.075（4.315），Actor-after-Teacher 1.128（1.128），发布 1.272（1.314，−0.042），response tokens 53933.67（52587）。单次运行，发布和整步都没有改善。不改冻结逻辑，不再跑一组。

`20260926-strong-sd-early-baseline-035937` 的 hydra yaml 曾被一次误启动覆盖成这次诊断的配置。那次基线的计时仍以当时的 train.log 和样本为准，不要读该目录里现在的 hydra。
