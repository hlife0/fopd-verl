#!/usr/bin/env bash
# One 5-step 4-GPU run. Same sd-early recipe, one config change:
# Student vLLM cudagraph_mode FULL_AND_PIECEWISE instead of PIECEWISE.
# EAGLE3 k stays 3. teacher_follow stays false. Do not reuse an existing run directory.
# The shell freeze variable is unset and is not passed in runtime_env.
# That matches the measured pairs. Do not add it on only one side.
set -euo pipefail
cd /csproject/fyp26_bl1/fopd
unset OPD_PUBLICATION_GC_FREEZE_STEP
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export ACTOR_ROLLOUT_OVERLAP=0
export FOPD_VERL_DIR=/csproject/fyp26_bl1/fopd/.worktrees/student-decode-graph
export FOPD_RUN_DIR="${FOPD_RUN_DIR:-/csproject/fyp26_bl1/fopd/runs/4gpu-0.6b-from-8b/student-decode-graph-$(date +%Y%m%d_%H%M%S)}"
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
    actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_AND_PIECEWISE
