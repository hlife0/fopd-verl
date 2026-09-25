#!/usr/bin/env bash
# One 5-step 4-GPU diagnostic. Same recipe as 20260926-strong-sd-early-baseline-035937,
# except the shell does not export OPD_PUBLICATION_GC_FREEZE_STEP. That value is
# passed only through Hydra runtime_env so worker receipt is not mixed with inheritance.
set -euo pipefail
cd /csproject/fyp26_bl1/fopd
unset OPD_PUBLICATION_GC_FREEZE_STEP
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export ACTOR_ROLLOUT_OVERLAP=0
export FOPD_VERL_DIR=/csproject/fyp26_bl1/fopd/.worktrees/publication-state
export FOPD_RUN_DIR="${FOPD_RUN_DIR:-/csproject/fyp26_bl1/fopd/runs/4gpu-0.6b-from-8b/publication-state-$(date +%Y%m%d_%H%M%S)}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2.1}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda-13.2.1/bin/ptxas}"
export HF_HOME="${HF_HOME:-/homes/hlife/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/cachedir/cache/hlife/triton}"
export TMPDIR="${TMPDIR:-/tmp}"
export OPD_ENABLE_SD=1
export TEACHER_FOLLOW=False
export EARLY_ACTOR_LITE=True
exec bash scripts/fair_compare/4gpu-0.6b-from-8b.sh \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    distillation.teacher_models.teacher_model.inference.max_num_seqs=32 \
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPD_PUBLICATION_GC_FREEZE_STEP='2'"
