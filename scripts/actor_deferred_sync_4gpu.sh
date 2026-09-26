#!/usr/bin/env bash
# One 5-step 4-GPU run of the single-gather candidate (publication_single_gather_4gpu.sh), with
# ACTOR_DEFER_GRAD_SYNC=1 turning on trainer.v1.sync.actor_defer_grad_sync: streamed Actor chunks
# skip the gradient reduce-scatter until the chunk that closes the optimizer step.
# ACTOR_DEFER_GRAD_SYNC=0 (default) is the same code with the switch off, i.e. the A arm.
# Everything else matches the single-gather launcher. Do not reuse an existing run directory.
set -euo pipefail
cd /csproject/fyp26_bl1/fopd
unset OPD_PUBLICATION_GC_FREEZE_STEP
case "${ACTOR_DEFER_GRAD_SYNC:-0}" in
    1|true|True) defer=True ;;
    0|false|False) defer=False ;;
    *) echo "ACTOR_DEFER_GRAD_SYNC must be 0 or 1" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export ACTOR_ROLLOUT_OVERLAP=0
export FOPD_VERL_DIR=/csproject/fyp26_bl1/fopd/.worktrees/actor-deferred-sync
export FOPD_RUN_DIR="${FOPD_RUN_DIR:-/csproject/fyp26_bl1/fopd/runs/4gpu-0.6b-from-8b/actor-deferred-sync-${defer}-$(date +%Y%m%d_%H%M%S)}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2.1}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda-13.2.1/bin/ptxas}"
export HF_HOME="${HF_HOME:-/homes/hlife/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/cachedir/cache/hlife/triton}"
export TMPDIR="${TMPDIR:-/tmp}"
export OPD_ENABLE_SD=1
export TEACHER_FOLLOW=False
export EARLY_ACTOR_LITE=True
mkdir -p "$FOPD_RUN_DIR"
exec bash scripts/fair_compare/4gpu-0.6b-from-8b.sh \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    distillation.teacher_models.teacher_model.inference.max_num_seqs=32 \
    actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_AND_PIECEWISE \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPD_PUBLICATION_GC_FREEZE_STEP='2'" \
    "trainer.v1.sync.actor_defer_grad_sync=${defer}" \
    "$@"
