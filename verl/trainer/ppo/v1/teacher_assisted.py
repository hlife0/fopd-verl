# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Teacher-GPU borrow for one extra Student replica.

The auxiliary replica is created only when the feature is on and the borrow
limit is positive. While it is on, Teacher and the auxiliary Student stay
resident; a step only closes one engine's request admission and opens the other.
"""

import math
from typing import Optional


def _cfg(config, path, default=None):
    value = config
    for key in path.split("."):
        if value is None:
            return default
        try:
            value = value.get(key, default) if hasattr(value, "get") else getattr(value, key)
        except Exception:
            return default
        if value is default and not hasattr(config, "get"):
            return default
    return value


def _finite(name: str, value, *, allow_none: bool = False, low: float | None = None, high: float | None = None):
    if value is None and allow_none:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if low is not None and number < low:
        raise ValueError(f"{name} must be >= {low}, got {number}")
    if high is not None and number > high:
        raise ValueError(f"{name} must be <= {high}, got {number}")
    return number


def validate_teacher_assisted_scope(config) -> None:
    """Reject an enabled borrow that would leave the sd-early Actor path.

    Inactive configs are not checked, so a zero borrow stays on the original path.
    """
    mode = str(_cfg(config, "trainer.v1.trainer_mode", "sync"))
    if mode != "sync":
        raise ValueError(f"teacher-assisted rollout requires trainer_mode=sync, got {mode}")
    sync = config.trainer.v1.sync
    for flag in ("separate_rollout", "actor_rollout_overlap", "actor_rollout_migrate"):
        if bool(sync.get(flag, False)):
            raise ValueError(f"teacher-assisted rollout cannot run with sync.{flag}=true")
    if not bool(sync.get("early_actor_lite", False)):
        raise ValueError("teacher-assisted rollout requires early_actor_lite")
    if not bool(sync.get("opd_no_task_reward_fast_path", False)):
        raise ValueError("teacher-assisted rollout requires the OPD fast path so Actor streams on teacher-ready samples")
    rollout = config.actor_rollout_ref.rollout
    actor = config.actor_rollout_ref.actor
    if int(rollout.n) != 1:
        raise ValueError("teacher-assisted rollout requires rollout.n=1")
    student_tp = int(rollout.tensor_model_parallel_size)
    if student_tp not in (1, 2):
        raise ValueError(f"teacher-assisted rollout supports Student TP 1 or 2, got {student_tp}")
    if int(getattr(rollout, "pipeline_model_parallel_size", 1) or 1) != 1:
        raise ValueError("teacher-assisted rollout does not support pipeline parallel")
    if int(getattr(rollout, "data_parallel_size", 1) or 1) != 1:
        raise ValueError("teacher-assisted rollout requires data_parallel_size=1 inside a replica")
    sp = int(getattr(actor, "ulysses_sequence_parallel_size", 1) or 1)
    fsdp_sp = int(getattr(getattr(actor, "fsdp_config", None), "ulysses_sequence_parallel_size", 1) or 1)
    if sp != 1 or fsdp_sp != 1:
        raise ValueError("teacher-assisted rollout does not support sequence parallel")
    if int(actor.get("ppo_epochs", 1) if hasattr(actor, "get") else getattr(actor, "ppo_epochs", 1)) != 1:
        raise ValueError("teacher-assisted rollout requires ppo_epochs=1")
    train_bsz = int(config.data.train_batch_size)
    mini = int(actor.get("ppo_mini_batch_size", train_bsz) if hasattr(actor, "get") else train_bsz)
    if mini != train_bsz:
        raise ValueError("teacher-assisted rollout requires ppo_mini_batch_size == train_batch_size")
    strategy = str(actor.get("strategy", "fsdp") if hasattr(actor, "get") else "fsdp")
    if strategy not in ("fsdp", "fsdp2"):
        raise ValueError(f"teacher-assisted rollout requires FSDP Actor strategy, got {strategy}")
    if bool(config.get("critic", {}).get("enable", False) if hasattr(config, "get") else False):
        # need_critic is the real switch; a missing critic block is off.
        pass
    critic = getattr(config, "critic", None)
    if critic is not None and bool(critic.get("enable", False)):
        raise ValueError("teacher-assisted rollout does not support a critic")
    teacher_models = config.distillation.teacher_models
    if len(teacher_models) != 1:
        raise ValueError("teacher-assisted rollout requires exactly one Teacher model")
    teacher = next(iter(teacher_models.values()))
    inference = teacher.inference
    teacher_tp = int(inference.tensor_model_parallel_size)
    teacher_pp = int(getattr(inference, "pipeline_model_parallel_size", 1) or 1)
    if teacher_pp != 1:
        raise ValueError("teacher-assisted rollout does not support Teacher pipeline parallel")
    if teacher_tp != student_tp:
        raise ValueError(
            "the auxiliary Student replica must use the same TP as the one Teacher replica, "
            f"got student TP {student_tp} and teacher TP {teacher_tp}"
        )
    pool = int(config.distillation.n_gpus_per_node) * int(config.distillation.nnodes)
    if pool != teacher_tp:
        raise ValueError(
            f"teacher-assisted rollout requires the Teacher pool to be one replica ({teacher_tp} GPUs), got {pool}"
        )
    borrow_s = _finite("teacher_assisted_max_borrow_s", sync.get("teacher_assisted_max_borrow_s", 0), low=0.0)
    if borrow_s <= 0:
        raise ValueError("teacher-assisted rollout was validated while the borrow limit is not positive")
    _finite(
        "teacher_assisted_complete_ratio",
        sync.get("teacher_assisted_complete_ratio", None),
        allow_none=True,
        low=0.0,
        high=1.0,
    )


def teacher_assisted_active(config) -> bool:
    """True only when the switch is on, the borrow limit is positive, and the scope matches sd-early."""
    sync = config.trainer.v1.sync
    enabled = bool(sync.get("teacher_assisted_rollout", False))
    raw_borrow = sync.get("teacher_assisted_max_borrow_s", 0)
    if not enabled:
        return False
    borrow_s = _finite("teacher_assisted_max_borrow_s", 0 if raw_borrow is None else raw_borrow, low=0.0)
    if borrow_s <= 0:
        return False
    _finite(
        "teacher_assisted_complete_ratio",
        sync.get("teacher_assisted_complete_ratio", None),
        allow_none=True,
        low=0.0,
        high=1.0,
    )
    validate_teacher_assisted_scope(config)
    return True


def auxiliary_rollout_config(rollout_config, gpu_memory_utilization: float, bucket_megabytes: int):
    """Copy the Student rollout config and override only the auxiliary engine.

    The caller's config keeps the main naive bucket and the original memory util.
    """
    from omegaconf import OmegaConf, open_dict

    util = _finite("teacher_assisted_aux_gpu_memory_utilization", gpu_memory_utilization, low=0.05, high=0.9)
    if util <= 0:
        raise ValueError("teacher_assisted_aux_gpu_memory_utilization must be positive")
    bucket = int(bucket_megabytes)
    if bucket < 64:
        raise ValueError(f"teacher_assisted_aux_bucket_megabytes must be >= 64, got {bucket}")
    copied = OmegaConf.create(OmegaConf.to_container(rollout_config, resolve=True))
    with open_dict(copied):
        copied.gpu_memory_utilization = util
        copied.enable_sleep_mode = True
        copied.free_cache_engine = True
        copied.checkpoint_engine.backend = "nccl"
        copied.checkpoint_engine.update_weights_bucket_megabytes = bucket
        if "engine_kwargs" not in copied.checkpoint_engine or copied.checkpoint_engine.engine_kwargs is None:
            copied.checkpoint_engine.engine_kwargs = {}
        nccl_kwargs = dict(copied.checkpoint_engine.engine_kwargs.get("nccl") or {})
        nccl_kwargs["multi_sender"] = False
        copied.checkpoint_engine.engine_kwargs["nccl"] = nccl_kwargs
    return copied


_COLOCATE_UTIL_LIMIT = 0.95


def teacher_gpu_memory_utilization(config) -> float:
    teacher = next(iter(config.distillation.teacher_models.values()))
    inference = teacher.inference
    raw = inference.get("gpu_memory_utilization", 0.5) if hasattr(inference, "get") else 0.5
    return float(0.5 if raw is None else raw)


def resident_engines(config) -> bool:
    """Keep both engines awake only when their requested fractions fit on one card.

    vLLM reserves ``gpu_memory_utilization`` times the whole card, not the free
    remainder. Teacher 0.85 plus auxiliary 0.2 does not fit a 24 GiB card.
    An explicit true cannot override that. Six-card Teacher 0.4 plus 0.2 still fits.
    """
    sync = config.trainer.v1.sync
    aux = float(sync.get("teacher_assisted_aux_gpu_memory_utilization", 0.2) or 0.2)
    fits = teacher_gpu_memory_utilization(config) + aux <= _COLOCATE_UTIL_LIMIT
    explicit = sync.get("teacher_assisted_resident", None)
    if explicit is False:
        return False
    if not fits:
        return False
    if explicit is None:
        return True
    return bool(explicit)


def teacher_assisted_settings(config) -> tuple[Optional[float], float]:
    sync = config.trainer.v1.sync
    ratio = sync.get("teacher_assisted_complete_ratio", None)
    if ratio is not None:
        ratio = float(ratio)
    borrow_s = float(sync.get("teacher_assisted_max_borrow_s", 0) or 0)
    return ratio, borrow_s


def completion_threshold(ratio: Optional[float], total_trajectories: int) -> Optional[int]:
    """Ceil of ratio * batch. None means the count trigger is unused."""
    if ratio is None:
        return None
    if total_trajectories < 0:
        raise ValueError(f"total_trajectories must be >= 0, got {total_trajectories}")
    if not math.isfinite(ratio) or ratio < 0 or ratio > 1:
        raise ValueError(f"teacher_assisted_complete_ratio must be in [0, 1], got {ratio}")
    return math.ceil(ratio * total_trajectories)


def should_switch(
    n_complete: int,
    threshold: Optional[int],
    elapsed_s: float,
    max_borrow_s: float,
    all_done: bool,
) -> Optional[str]:
    """Return the switch reason, or None to keep borrowing.

    Count, time, and "every Student trajectory is done" each switch once.
    The caller must not call this after a switch has already happened.
    """
    if all_done:
        return "all_students_done"
    if threshold is not None and n_complete >= threshold:
        return "complete_count"
    if max_borrow_s > 0 and elapsed_s >= max_borrow_s:
        return "max_borrow_time"
    return None


def assign_migrations(n_requests: int, loads: dict[str, int]) -> list[str]:
    """Spread interrupted requests onto the least-loaded original replicas."""
    if n_requests < 0:
        raise ValueError(f"n_requests must be >= 0, got {n_requests}")
    if n_requests == 0:
        return []
    if not loads:
        raise RuntimeError("teacher-assisted migration has no original Student replica")
    current = dict(loads)
    assigned = []
    for _ in range(n_requests):
        server_id = min(current, key=lambda sid: (current[sid], sid))
        assigned.append(server_id)
        current[server_id] += 1
    return assigned


def migration_fits(n_requests: int, loads: dict[str, int], max_num_seqs: int) -> bool:
    """True when an even spread stays within each replica's concurrency cap."""
    if max_num_seqs <= 0:
        raise ValueError(f"max_num_seqs must be positive, got {max_num_seqs}")
    assigned = assign_migrations(n_requests, loads)
    if not assigned:
        return True
    projected = dict(loads)
    for server_id in assigned:
        projected[server_id] += 1
    return max(projected.values()) <= max_num_seqs

