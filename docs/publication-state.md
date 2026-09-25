# Publication state diagnostic

One `PUBLICATION_STATE` line is printed from each rollout worker on every `update_weights` call. It records only `global_steps` (value and type), `repr` of `OPD_PUBLICATION_GC_FREEZE_STEP`, the effective mode, whether that variable is non-empty (`publication_opt`), `gc.isenabled()`, and whether `gc.freeze()` returned on this call.

The 4-GPU launcher unsets the shell variable and passes `2` only through Hydra `runtime_env`.

Read the lines, not the source:

- `env=''` means this worker did not receive the variable.
- `mode` other than `'naive'`, `global_steps` of `None` or a list, or steps that skip `2`, means the freeze condition was not met. `freeze_called=True` only after `gc.freeze()` returns, and only when the value equals `2`.
- `env='2'`, `mode='naive'`, a step whose `global_steps=2`, and `freeze_called=True` means the worker ran the existing switch. A missing older `GC frozen` warning in that case is log propagation.

Compare step time and `timing_s/update_weights` with `runs/4gpu-0.6b-from-8b/20260926-strong-sd-early-baseline-035937` (step 26.195 s, publish 1.314 s). The switch, if it runs, sits inside publish.
