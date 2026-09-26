#!/usr/bin/env bash
# One 5-step 4-GPU diagnostic run on the strong-baseline Student graph recipe
# (FULL_AND_PIECEWISE, publication GC freeze at step 2). Not a timing comparison.
# Step 4: rank-0 torch CUDA trace of one 3-sample Actor chunk and of publication.
# Steps 3 and 5: only the PUBLICATION_* / ACTOR_* stdout timers and NVML busy samples.
set -euo pipefail
cd /csproject/fyp26_bl1/fopd
unset OPD_PUBLICATION_GC_FREEZE_STEP
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export ACTOR_ROLLOUT_OVERLAP=0
export FOPD_VERL_DIR=/csproject/fyp26_bl1/fopd/.worktrees/gpu-bubble-profile
export FOPD_RUN_DIR="${FOPD_RUN_DIR:-/csproject/fyp26_bl1/fopd/runs/4gpu-0.6b-from-8b/gpu-bubble-profile-$(date +%Y%m%d_%H%M%S)}"
export FOPD_WINDOW_TRACE_STEP="${FOPD_WINDOW_TRACE_STEP:-4}"
export FOPD_WINDOW_TRACE_DIR="${FOPD_WINDOW_TRACE_DIR:-$FOPD_RUN_DIR/window_trace}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2.1}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda-13.2.1/bin/ptxas}"
export HF_HOME="${HF_HOME:-/homes/hlife/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/cachedir/cache/hlife/triton}"
export TMPDIR="${TMPDIR:-/tmp}"
export OPD_ENABLE_SD=1
export TEACHER_FOLLOW=False
export EARLY_ACTOR_LITE=True
mkdir -p "$FOPD_RUN_DIR" "$FOPD_WINDOW_TRACE_DIR"
exec bash scripts/fair_compare/4gpu-0.6b-from-8b.sh \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    distillation.teacher_models.teacher_model.inference.max_num_seqs=32 \
    actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_AND_PIECEWISE \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPD_PUBLICATION_GC_FREEZE_STEP='2'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.FOPD_WINDOW_TRACE_STEP='${FOPD_WINDOW_TRACE_STEP}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.FOPD_WINDOW_TRACE_DIR='${FOPD_WINDOW_TRACE_DIR}'"
