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
import json
import logging
import math
from copy import deepcopy
import os
import socket
import time

import transfer_queue as tq
import ray
from omegaconf import open_dict
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _first_output(output):
    if isinstance(output, (list, tuple)):
        for item in output:
            if item is not None:
                return item
        return None
    return output


def _chunk_metric(output, name: str) -> float | None:
    output = _first_output(output)
    if output is None:
        return None
    try:
        value = output["metrics"][name]
    except Exception:
        return None
    if isinstance(value, (list, tuple)):
        value = value[-1]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _chunk_fb_s(output) -> float | None:
    return _chunk_metric(output, "chunk_fb_s")


@register_trainer("sync")
class PPOTrainerSync(PPOTrainer):
    """Synchronous PPO trainer
    1. Trainer and rollout are colocated
    2. Partial rollout is disabled
    """

    def _setup(self):
        if self.config.trainer.v1.sync.get("actor_rollout_migrate", False):
            from verl.utils.torch_dtypes import PrecisionType
            import torch
            precision = self.config.actor_rollout_ref.actor.fsdp_config.get("mixed_precision") or {}
            if PrecisionType.to_dtype(precision.get("param_dtype", "bf16")) == torch.float16:
                raise ValueError("actor_rollout_migrate supports BF16/FP32, not FP16 GradScaler training")
            with open_dict(self.config.actor_rollout_ref.actor.fsdp_config):
                self.config.actor_rollout_ref.actor.fsdp_config.use_orig_params = True
        super()._setup()

    def get_llm_client(self):
        if self.config.trainer.v1.sync.get("actor_rollout_migrate", False):
            from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
            return self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient)
        return super().get_llm_client()

    def on_init_end(self):
        self._configure_opd_no_task_reward_fast_path()
        self._configure_early_actor_lite()
        self._configure_actor_rollout_overlap()
        self._configure_actor_rollout_migrate()
        # update weights after loading checkpoint
        self.checkpoint_manager.update_weights(self.global_steps)

    def step(self, metrics: dict, timing_raw: dict):
        batch = super().step(metrics, timing_raw)
        if self.opd_no_task_reward_fast_path:
            metrics.update(
                {
                    "opd_no_task_reward_fast_path/enabled": 1,
                    "opd_no_task_reward_fast_path/old_log_prob_alias": 1,
                    "opd_no_task_reward_fast_path/task_advantage_materialized": 0,
                    "opd_no_task_reward_fast_path/task_ppo_loss_executed": 0,
                }
            )
        if self.early_actor_lite:
            metrics["early_actor_lite/enabled"] = 1
            metrics["early_actor_lite/stream_fb"] = int(getattr(self, "early_actor_stream_fb", False))
        if getattr(self, "actor_rollout_overlap", False):
            metrics["actor_rollout_overlap/enabled"] = 1
        return batch

    def _configure_opd_no_task_reward_fast_path(self) -> None:
        sync_config = self.config.trainer.v1.sync
        self.opd_no_task_reward_fast_path = bool(sync_config.get("opd_no_task_reward_fast_path", False))
        if not self.opd_no_task_reward_fast_path:
            return
        rollout_correction = self.config.algorithm.get("rollout_correction", None)
        if not rollout_correction or not rollout_correction.get("bypass_mode", False):
            raise ValueError("the OPD no-task-reward fast path requires bypass_mode=true")
        if int(self.config.actor_rollout_ref.rollout.n) != 1:
            raise ValueError("the OPD no-task-reward fast path requires rollout.n=1")
        if not self.config.actor_rollout_ref.rollout.calculate_log_probs:
            raise ValueError("the OPD no-task-reward fast path requires rollout log probabilities")
        if str(self.config.algorithm.adv_estimator) != "grpo":
            raise ValueError("the OPD no-task-reward fast path requires GRPO")
        if self.config.algorithm.use_kl_in_reward:
            raise ValueError("the OPD no-task-reward fast path requires use_kl_in_reward=false")
        if not self.use_teacher_policy:
            raise ValueError("the OPD no-task-reward fast path requires teacher scoring")
        if self.distillation_config is None:
            raise ValueError("the OPD no-task-reward fast path requires distillation")
        if self.distillation_config.distillation_loss.use_task_rewards:
            raise ValueError("the OPD no-task-reward fast path requires use_task_rewards=false")
        if not self.distillation_config.distillation_loss.use_policy_gradient:
            raise ValueError("the OPD no-task-reward fast path requires use_policy_gradient=true")
        if self.distillation_config.distillation_loss.loss_mode != "k1":
            raise ValueError("the OPD no-task-reward fast path requires K1 distillation")
        if self.use_critic or self.use_reference_policy:
            raise ValueError("the OPD no-task-reward fast path requires critic and reference updates to be disabled")
        filter_groups = self.config.algorithm.get("filter_groups", None)
        if filter_groups is not None and filter_groups.get("enable", False):
            raise ValueError("the OPD no-task-reward fast path requires group filtering to be disabled")
        if self.reward_loop_manager.reward_loop_worker_handles is None:
            raise ValueError("the OPD no-task-reward fast path requires agent-loop reward scoring")

    def _configure_early_actor_lite(self) -> None:
        sync_config = self.config.trainer.v1.sync
        self.early_actor_lite = bool(sync_config.get("early_actor_lite", False))
        if not self.early_actor_lite:
            self.early_actor_stream_fb = False
            return
        filter_groups = self.config.algorithm.get("filter_groups", None)
        if filter_groups is not None and filter_groups.get("enable", False):
            raise ValueError("early_actor_lite requires group filtering to be disabled")
        actor_cfg = self.config.actor_rollout_ref.get("actor", {})
        data_cfg = self.config.get("data", {})
        ppo_epochs = int(actor_cfg.get("ppo_epochs", 1)) if actor_cfg else 1
        ppo_mini = int(actor_cfg.get("ppo_mini_batch_size", 0)) if actor_cfg else 0
        train_bsz = int(data_cfg.get("train_batch_size", 0)) if data_cfg else 0
        strategy = str(actor_cfg.get("strategy", "fsdp")) if actor_cfg else "fsdp"
        self.early_actor_stream_fb = bool(
            self.early_actor_lite
            and getattr(self, "opd_no_task_reward_fast_path", False)
            and ppo_epochs == 1
            and ppo_mini > 0
            and ppo_mini == train_bsz
            and strategy in ("fsdp", "fsdp2")
            and not getattr(self, "use_critic", False)
            and not getattr(self, "use_reference_policy", False)
        )
        if self.early_actor_stream_fb:
            logger.info(
                "early_actor_lite: sleep Student vLLM after the Student barrier, then stream "
                "Actor F/B on teacher-ready samples while leftover Teacher scoring continues"
            )
        else:
            logger.info(
                "early_actor_lite: sleep Student vLLM after the Student barrier and overlap "
                "old_log_prob with leftover Teacher scoring"
            )

    def _configure_actor_rollout_overlap(self) -> None:
        sync_config = self.config.trainer.v1.sync
        self.actor_rollout_overlap = bool(sync_config.get("actor_rollout_overlap", False))
        if not self.actor_rollout_overlap:
            return
        if not self.early_actor_stream_fb:
            raise ValueError(
                "actor_rollout_overlap requires early_actor_lite with the OPD fast path, "
                "FSDP/FSDP2, ppo_epochs=1 and ppo_mini_batch_size=train_batch_size"
            )
        if self.config.actor_rollout_ref.actor.loss_agg_mode != "token-mean":
            raise ValueError("actor_rollout_overlap requires loss_agg_mode=token-mean")
        if self.parameter_sync_step != 1:
            raise ValueError("actor_rollout_overlap requires parameter_sync_step=1")
        if self.config.trainer.critic_warmup > 0:
            raise ValueError("actor_rollout_overlap requires critic_warmup=0")
        if int(self.config.data.max_response_length) <= 0:
            raise ValueError("actor_rollout_overlap requires a positive max_response_length")
        if int(sync_config.get("actor_rollout_overlap_chunk_size", 0)) < 0:
            raise ValueError("actor_rollout_overlap_chunk_size must be nonnegative")
        logger.info("actor_rollout_overlap: accumulate scored trajectories before the Student barrier")

    def _configure_actor_rollout_migrate(self):
        cfg = self.config.trainer.v1.sync
        self.actor_rollout_migrate = bool(cfg.get("actor_rollout_migrate", False))
        if not self.actor_rollout_migrate:
            return
        if self.actor_rollout_overlap or not self.early_actor_stream_fb:
            raise ValueError("actor_rollout_migrate requires early_actor_lite OPD streaming and disables route 1")
        actor = self.config.actor_rollout_ref.actor
        rollout = self.config.actor_rollout_ref.rollout
        if actor.loss_agg_mode != "token-mean" or self.parameter_sync_step != 1:
            raise ValueError("actor_rollout_migrate requires token-mean and parameter_sync_step=1")
        if actor.fsdp_config.ulysses_sequence_parallel_size != 1 or actor.fsdp_config.forward_only:
            raise ValueError("actor_rollout_migrate requires FSDP training without sequence parallelism")
        if self.distillation_config.teacher_follow:
            raise ValueError("actor_rollout_migrate currently requires teacher_follow=false")
        if rollout.name != "vllm" or rollout.data_parallel_size != 1 or rollout.pipeline_model_parallel_size != 1:
            raise ValueError("actor_rollout_migrate requires vLLM TP replicas without PP/engine DP")
        if self.config.trainer.nnodes != 1 or len(self.llm_server_manager.rollout_replicas) < 2:
            raise ValueError("actor_rollout_migrate requires one node and at least two rollout replicas")
        self._migration_fraction = float(cfg.actor_rollout_migrate_after_fraction)
        if not 0 < self._migration_fraction < 1:
            raise ValueError("actor_rollout_migrate_after_fraction must lie strictly between 0 and 1")
        self._migration_dp = int(rollout.tensor_model_parallel_size)
        self._migration_chunk = int(cfg.actor_rollout_migrate_chunk_size) or self._migration_dp
        if self._migration_chunk < self._migration_dp or self._migration_chunk % self._migration_dp:
            raise ValueError("actor_rollout_migrate_chunk_size must be divisible by early FSDP DP size")
        from verl.single_controller.ray.base import RayClassWithInitArgs, RayWorkerGroup, split_resource_pool
        from verl.trainer.ppo.utils import Role
        from verl.workers.engine_workers import ActorRolloutRefWorker
        role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        pool = self.resource_pool_manager.get_resource_pool(role)
        early_pool = split_resource_pool(pool, [pool.world_size - self._migration_dp, self._migration_dp])[-1]
        early_config = deepcopy(self.config.actor_rollout_ref)
        with open_dict(early_config.actor.fsdp_config):
            early_config.actor.fsdp_config.param_offload = True
            early_config.actor.fsdp_config.optimizer_offload = False
        self.migration_actor_wg = RayWorkerGroup(
            resource_pool=early_pool,
            ray_cls_with_init=RayClassWithInitArgs(
                cls=ray.remote(ActorRolloutRefWorker), config=early_config, role="actor",
                distillation_config=self.config.distillation, migration_fb_only=True,
            ),
            name_prefix="migration_actor", device_name=self.config.trainer.device,
        )
        self.migration_actor_wg.init_model()
        self._migration_units = self.actor_rollout_wg.migration_units()[0]
        if self.migration_actor_wg.migration_units()[0] != self._migration_units:
            raise ValueError("Main and early FSDP wrapping must use identical parameter units")
        self._init_gradient_exchange()

    def _init_gradient_exchange(self):
        # Separate Ray groups already own process groups. A stateless NCCL
        # communicator lets the early full gradient land in the main shards
        # without a host copy through the driver.
        main = self.actor_rollout_wg.world_size
        early = self.migration_actor_wg.world_size
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        # Early ranks occupy the last early GPUs. Rank 0 of that group shares
        # its GPU with this main rank.
        src = main - early
        addr = "127.0.0.1"
        main_refs = self.actor_rollout_wg.init_gradient_exchange(addr, port, src, False)
        early_refs = self.migration_actor_wg.init_gradient_exchange(addr, port, src, True)
        ray.get([*main_refs, *early_refs])

    def _merge_migration_gradients(self):
        units = tuple(self._migration_units)
        early_refs = self.migration_actor_wg.send_migration_gradients(units)
        main_refs = self.actor_rollout_wg.recv_migration_gradients(units)
        ray.get([*early_refs, *main_refs])

    def _sync_migration_weights(self):
        # Called only after the migrated replica has slept. The other Student
        # cards are still decoding; this copy is not on the step-start path.
        units = tuple(self._migration_units)
        main_refs = self.actor_rollout_wg.send_migration_parameters(units)
        early_refs = self.migration_actor_wg.recv_migration_parameters(units)
        ray.get([*main_refs, *early_refs])

    def _migrate_and_train(self, metrics, timing_raw, sample_batch_size):
        from verl.trainer.ppo.padding_utils import upsample_batch_to_divisible_size
        from verl.utils.ray_utils import auto_await
        prompts = self._actor_overlap_prompt_uids
        if len(prompts) != sample_batch_size:
            raise RuntimeError("Migration logical batch differs from submitted prompts")
        manager = self.llm_server_manager
        replica = manager.rollout_replicas[-1]
        address = manager.server_addresses[-1]
        main_dp = self._actor_dp_size()
        threshold = math.ceil(sample_batch_size * self._migration_fraction)
        consumed = set()
        padded_keys = []
        events = []
        migrated = early_started = main_started = asleep = finalized = False
        early_tokens = 0.0
        early_state = None
        old_poll = self.replay_buffer.poll_interval
        self.replay_buffer.poll_interval = min(old_poll, 0.05)
        denominator = sample_batch_size * int(self.config.data.max_response_length)
        gen_start = time.time()
        self._trace_actor_start_ts = None
        try:
            self.on_sample_begin()
            while len(consumed) < sample_batch_size:
                students, ready = self.replay_buffer.peek_actor_overlap_batch("train", prompts, self.global_steps)
                all_students = len(students) == sample_batch_size
                if not migrated and not all_students and len(students) >= threshold and ready:
                    migrate_start = time.time()
                    ray.get(manager.global_load_balancer.remove_servers.remote([address]))
                    migrated = True
                    reports = ray.get([server.migrate_requests.remote() for server in replica.servers])
                    metrics["actor_rollout_migrate/aborted_requests"] = sum(r["aborted_count"] for r in reports)
                    auto_await(replica.sleep)()
                    timing_raw["migration"] = time.time() - migrate_start
                    self._migration_start_ts = migrate_start
                    self._migration_done_ts = time.time()
                    weight_sync_start = time.time()
                    self._sync_migration_weights()
                    timing_raw["migration_weight_sync"] = time.time() - weight_sync_start
                if all_students and not asleep:
                    self._trace_student_barrier_ts = time.time()
                    self._trace_sleep_start_ts = time.time()
                    # The evacuated replica is already asleep; sleep only remaining replicas.
                    replicas = manager.rollout_replicas[:-1] if migrated else manager.rollout_replicas
                    for item in replicas:
                        auto_await(item.sleep)()
                    self._trace_sleep_end_ts = time.time()
                    asleep = True
                    timing_raw["gen"] = time.time() - gen_start
                    if early_started and early_state is None:
                        early_state = self.migration_actor_wg.migration_loss_state()[0]
                        early_tokens = early_state["tokens"]
                    if not main_started:
                        main_started = True
                        self.actor_rollout_wg.begin_actor_accumulate(loss_normalization_tokens=denominator)
                    if self._trace_actor_start_ts is None:
                        self._trace_actor_start_ts = time.time()
                available = [key for key in ready if key not in consumed]
                remaining = sample_batch_size - len(consumed)
                if asleep:
                    take_n = min(len(available), main_dp)
                    if len(available) < remaining:
                        take_n -= take_n % main_dp
                    group, dp, phase = self.actor_rollout_wg, main_dp, "main"
                elif migrated:
                    # Keep one full main DP chunk so every main rank has valid
                    # gradient storage before importing the early contribution.
                    take_n = min(len(available), self._migration_chunk, remaining - main_dp)
                    take_n = max(0, take_n - take_n % self._migration_dp)
                    group, dp, phase = self.migration_actor_wg, self._migration_dp, "early"
                else:
                    take_n = 0
                if not take_n:
                    time.sleep(self.replay_buffer.poll_interval)
                    continue
                take = available[:take_n]
                chunk = students.select_keys(take)
                expected_version = self.global_steps - 1
                if any(
                    tag.get("min_global_steps") != expected_version
                    or tag.get("max_global_steps") != expected_version
                    for tag in chunk.tags
                ):
                    raise RuntimeError(
                        f"Migration trajectories must use frozen rollout weight version {expected_version}"
                    )
                chunk.extra_info.update(self._actor_update_extra_info())
                chunk = upsample_batch_to_divisible_size(chunk, dp, self.tokenizer.eos_token_id)
                padded_keys.extend(key for key, tag in zip(chunk.keys, chunk.tags, strict=True) if tag.get("is_padding"))
                if phase == "early" and not early_started:
                    self._trace_actor_start_ts = time.time()
                    early_started = True
                    group.begin_actor_accumulate(loss_normalization_tokens=denominator)
                dispatch_ts = time.time()
                result = group.accumulate_actor(chunk)
                events.append({"n": take_n, "phase": phase, "dispatch_ts": dispatch_ts,
                               "return_ts": time.time(), "chunk_fb_s": _chunk_fb_s(result)})
                consumed.update(take)
                if phase == "main" and early_started:
                    merge_start = time.time()
                    self._merge_migration_gradients()
                    self.actor_rollout_wg.add_migration_loss_state(early_state)
                    self.migration_actor_wg.abort_actor_accumulate()
                    early_started = False
                    timing_raw["migration_gradient_merge"] = time.time() - merge_start
            batch = self.replay_buffer.materialize_actor_overlap_batch("train", prompts, self.global_steps)
            self._trace_actor_finish_start_ts = time.time()
            result = _first_output(self.actor_rollout_wg.finish_actor_accumulate())
            self._trace_actor_finish_end_ts = self._trace_actor_done_ts = time.time()
            finalized = True
        finally:
            self.replay_buffer.poll_interval = old_poll
            try:
                if early_started:
                    self.migration_actor_wg.abort_actor_accumulate()
                if main_started and not finalized:
                    self.actor_rollout_wg.abort_actor_accumulate()
            finally:
                if padded_keys:
                    tq.kv_clear(partition_id="train", keys=padded_keys)
                if migrated and not finalized:
                    auto_await(replica.wake_up)()
                    ray.get([server.resume_generation.remote() for server in replica.servers])
                    ray.get(manager.global_load_balancer.add_servers.remote({address: manager.server_handles[-1]}))
            if self._trace_actor_start_ts is not None:
                timing_raw["update_actor"] = time.time() - self._trace_actor_start_ts
        self._trace_actor_chunk_events = events
        self._apply_actor_update_metrics(result, metrics)
        batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        self._record_gen_split_timing(batch, timing_raw)
        last_student = max(float(tag["student_gen_done_ts"]) for tag in batch.tags)
        metrics["actor_rollout_migrate/enabled"] = 1
        metrics["actor_rollout_migrate/early_samples"] = sum(e["n"] for e in events if e["phase"] == "early")
        metrics["actor_rollout_migrate/early_samples_completed_before_student"] = sum(
            e["n"] for e in events if e["phase"] == "early" and e["return_ts"] < last_student
        )
        metrics["actor_rollout_migrate/migrated_trajectories"] = sum(bool(tag.get("migration_count")) for tag in batch.tags)
        metrics["actor_rollout_migrate/migrated_prefix_tokens"] = sum(tag.get("migrated_prefix_tokens", 0) for tag in batch.tags)
        metrics["actor_rollout_migrate/early_loss_tokens"] = early_tokens
        self._dump_actor_timeline(batch)
        self._migration_replica_removed = migrated
        return batch

    def prepare_step(self) -> dict:
        if not (getattr(self, "actor_rollout_overlap", False) or getattr(self, "actor_rollout_migrate", False)):
            return super().prepare_step()
        batch = self._next_train_batch()
        # Capture the actual submitted IDs, rather than selecting any ready groups
        # left in the queue (which can belong to another logical batch).
        self._actor_overlap_prompt_uids = list(batch["uid"])
        self._submit_batch_to_rollout(batch)
        return {}

    def _step_once(self, metrics: dict, timing_raw: dict, sample_batch_size: int):
        if getattr(self, "actor_rollout_migrate", False):
            return self._migrate_and_train(metrics, timing_raw, sample_batch_size)
        if getattr(self, "actor_rollout_overlap", False):
            return self._stream_actor_during_rollout(metrics, timing_raw, sample_batch_size)
        if not self.early_actor_lite:
            return super()._step_once(metrics, timing_raw, sample_batch_size)

        sessions_per_prompt = int(self.config.actor_rollout_ref.rollout.n)
        leftover_poll = self.replay_buffer.poll_interval
        self.replay_buffer.poll_interval = min(leftover_poll, 0.05)
        if getattr(self, "_trace_step_start_ts", None) is None:
            self._trace_step_start_ts = time.time()
        try:
            with marked_timer("gen", timing_raw, color="red"):
                self.on_sample_begin()
                self.replay_buffer.wait_until_students_done(
                    partition_id="train",
                    batch_size=sample_batch_size,
                    sessions_per_prompt=sessions_per_prompt,
                )
                self._trace_student_barrier_ts = time.time()
                student_batch: KVBatchMeta = self.replay_buffer.peek_student_ready(
                    partition_id="train",
                    batch_size=sample_batch_size,
                    sessions_per_prompt=sessions_per_prompt,
                )
                student_batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                # Free Student vLLM on the colocated GPUs; Teacher pool keeps scoring.
                self._trace_sleep_start_ts = time.time()
                self.on_sample_end()
                self._trace_sleep_end_ts = time.time()

            if self.reward_loop_manager.reward_loop_worker_handles is None:
                with marked_timer("reward", timing_raw, color="yellow"):
                    student_batch = self._compute_reward_colocate(student_batch, metrics=metrics)

            if getattr(self, "early_actor_stream_fb", False):
                return self._stream_actor_with_leftover_teacher(
                    student_batch, metrics, timing_raw, sample_batch_size
                )

            student_batch = self._balance_batch(student_batch, metrics=metrics)

            if not self.opd_no_task_reward_fast_path:
                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    student_batch = self._compute_old_log_prob(student_batch, metrics=metrics)

            with marked_timer("gen", timing_raw, color="red"):
                batch, off_policy_metrics = self.replay_buffer.sample(
                    global_steps=self.global_steps,
                    partition_id="train",
                    batch_size=sample_batch_size,
                )
                metrics.update(off_policy_metrics)
                batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        finally:
            self.replay_buffer.poll_interval = leftover_poll

        self._record_gen_split_timing(batch, timing_raw)

        if self.reward_loop_manager.reward_loop_worker_handles is None and "reward" not in timing_raw:
            with marked_timer("reward", timing_raw, color="yellow"):
                batch = self._compute_reward_colocate(batch, metrics=metrics)

        batch = self._balance_batch(batch, metrics=metrics)

        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                batch = self._compute_ref_log_prob(batch, metrics=metrics)

        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                batch = self._compute_values(batch, metrics=metrics)

        if not self.opd_no_task_reward_fast_path:
            with marked_timer("adv", timing_raw, color="brown"):
                batch = self._compute_advantage(batch, metrics=metrics)

        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                batch = self._update_critic(batch, metrics=metrics)

        if self.config.trainer.critic_warmup <= self.global_steps:
            self._trace_actor_start_ts = time.time()
            with marked_timer("update_actor", timing_raw, color="red"):
                batch = self._update_actor(batch, metrics=metrics)
            self._trace_actor_done_ts = time.time()

        return batch

    def _stream_actor_during_rollout(
        self, metrics: dict, timing_raw: dict, sample_batch_size: int
    ) -> KVBatchMeta:
        prompt_uids = self._actor_overlap_prompt_uids
        if len(prompt_uids) != sample_batch_size:
            raise RuntimeError("actor_rollout_overlap submitted prompt count differs from the logical batch")
        dp_size = self._actor_dp_size()
        chunk_cap = int(self.config.trainer.v1.sync.get("actor_rollout_overlap_chunk_size", 0)) or dp_size
        if chunk_cap < dp_size or chunk_cap % dp_size:
            raise ValueError("actor_rollout_overlap_chunk_size must be a positive multiple of Actor DP size")
        denominator = sample_batch_size * int(self.config.data.max_response_length)
        consumed: set[str] = set()
        chunks: list[dict] = []
        padding_keys: list[str] = []
        started = finalized = students_asleep = False
        old_poll = self.replay_buffer.poll_interval
        self.replay_buffer.poll_interval = min(old_poll, 0.05)
        gen_start = time.time()
        self._trace_step_start_ts = getattr(self, "_trace_step_start_ts", None) or gen_start
        self._trace_actor_start_ts = None
        self._trace_actor_done_ts = None
        try:
            self.on_sample_begin()
            while len(consumed) < sample_batch_size:
                students, teacher_ready = self.replay_buffer.peek_actor_overlap_batch(
                    "train", prompt_uids, self.global_steps
                )
                if len(students.keys) == sample_batch_size and not students_asleep:
                    self._trace_student_barrier_ts = time.time()
                    self._trace_sleep_start_ts = time.time()
                    self.on_sample_end()
                    self._trace_sleep_end_ts = time.time()
                    timing_raw["gen"] = timing_raw.get("gen", 0.0) + time.time() - gen_start
                    students_asleep = True
                ready = [key for key in teacher_ready if key not in consumed]
                remaining = sample_batch_size - len(consumed)
                take_n = min(len(ready), chunk_cap)
                # Only the final chunk may need synthetic zero-loss padding.
                if len(ready) < remaining:
                    take_n -= take_n % dp_size
                if take_n == 0:
                    time.sleep(self.replay_buffer.poll_interval)
                    continue
                take = ready[:take_n]
                chunk = students.select_keys(take)
                chunk.extra_info.update(self._actor_update_extra_info())
                chunk = self._balance_batch(
                    chunk, metrics=metrics, logging_prefix="early_actor_chunk", align_to_mini_batch=False
                )
                padding_keys.extend(
                    key for key, tag in zip(chunk.keys, chunk.tags, strict=True) if tag.get("is_padding", False)
                )
                if not started:
                    self._trace_actor_start_ts = time.time()
                    started = True  # Abort also cleans up a partially successful collective begin.
                    self.actor_rollout_wg.begin_actor_accumulate(loss_normalization_tokens=denominator)
                    self._trace_fsdp_load_end_ts = time.time()
                dispatch_ts = time.time()
                output = self.actor_rollout_wg.accumulate_actor(chunk)
                return_ts = time.time()
                consumed.update(take)
                chunks.append(
                    {
                        "n": take_n,
                        "teacher_ready_when_dispatched": len(ready),
                        "leftover_when_dispatched": remaining - len(ready),
                        "is_last": len(consumed) == sample_batch_size,
                        "dispatch_ts": dispatch_ts,
                        "return_ts": return_ts,
                        "chunk_fb_s": _chunk_fb_s(output),
                        "students_done_when_dispatched": len(students.keys),
                        "chunk_fb_start_ts": _chunk_metric(output, "chunk_fb_start_ts"),
                        "chunk_fb_end_ts": _chunk_metric(output, "chunk_fb_end_ts"),
                    }
                )

            # Every consumed group is terminal and immutable. Validate and consume
            # this exact set before the only optimizer update, never generic sample().
            batch = self.replay_buffer.materialize_actor_overlap_batch("train", prompt_uids, self.global_steps)
            if not students_asleep:
                raise RuntimeError("actor_rollout_overlap completed training before the Student barrier")
            self._trace_actor_finish_start_ts = time.time()
            output = _first_output(self.actor_rollout_wg.finish_actor_accumulate())
            self._trace_actor_finish_end_ts = time.time()
            finalized = True
            self._trace_actor_done_ts = time.time()
        finally:
            self.replay_buffer.poll_interval = old_poll
            try:
                if started and not finalized:
                    self.actor_rollout_wg.abort_actor_accumulate()
            finally:
                # Padding has its own UID and will not appear in the final real
                # batch returned to the trainer's normal trajectory cleanup.
                if padding_keys:
                    tq.kv_clear(partition_id="train", keys=padding_keys)
            if self._trace_actor_start_ts is not None:
                timing_raw["update_actor"] = timing_raw.get("update_actor", 0.0) + (
                    time.time() - self._trace_actor_start_ts
                )

        self._trace_actor_chunk_events = chunks
        batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        if output is not None:
            self._apply_actor_update_metrics(output, metrics)
        self._record_gen_split_timing(batch, timing_raw)
        last_student = max(float(tag["student_gen_done_ts"]) for tag in batch.tags)
        metrics["actor_rollout_overlap/chunks"] = len(chunks)
        metrics["actor_rollout_overlap/samples"] = len(consumed)
        metrics["actor_rollout_overlap/chunks_completed_before_student"] = sum(
            event["return_ts"] < last_student for event in chunks
        )
        metrics["actor_rollout_overlap/samples_completed_before_student"] = sum(
            event["n"] for event in chunks if event["return_ts"] < last_student
        )
        metrics["actor_rollout_overlap/actor_tail_s"] = max(0.0, self._trace_actor_done_ts - last_student)
        self._dump_actor_timeline(batch)
        return batch

    def _stream_actor_with_leftover_teacher(
        self,
        student_batch: KVBatchMeta,
        metrics: dict,
        timing_raw: dict,
        sample_batch_size: int,
    ) -> KVBatchMeta:
        traj_keys = [
            key
            for key, tag in zip(student_batch.keys, student_batch.tags, strict=True)
            if not tag.get("is_padding", False)
        ]
        full_tokens = sum(
            int(tag.get("response_len") or 0)
            for tag in student_batch.tags
            if not tag.get("is_padding", False)
        )
        dp_size = self._actor_dp_size()
        extra_info = self._actor_update_extra_info()
        extra_info["batch_num_tokens_override"] = int(full_tokens)
        consumed: set[str] = set()
        chunk_sizes: list[int] = []
        chunk_events: list[dict] = []
        finalized = False
        output = None

        self._trace_actor_start_ts = time.time()
        with marked_timer("update_actor", timing_raw, color="red"):
            self.actor_rollout_wg.begin_actor_accumulate()
            self._trace_fsdp_load_end_ts = time.time()
            try:
                while len(consumed) < len(traj_keys):
                    ready = [
                        key
                        for key in self.replay_buffer.peek_teacher_ready_keys(
                            student_batch.partition_id, traj_keys
                        )
                        if key not in consumed
                    ]
                    leftover = len(traj_keys) - len(consumed) - len(ready)
                    if leftover > 0:
                        take_n = len(ready) - (len(ready) % dp_size)
                    else:
                        take_n = len(ready)
                    if take_n == 0:
                        time.sleep(self.replay_buffer.poll_interval)
                        continue
                    take = ready[:take_n]
                    is_last = leftover == 0 and len(consumed) + take_n == len(traj_keys)
                    chunk = self.replay_buffer.peek_trajectories(student_batch.partition_id, take)
                    if chunk.extra_info is None:
                        chunk.extra_info = {}
                    chunk.extra_info.update(student_batch.extra_info)
                    chunk.extra_info.update(extra_info)
                    chunk = self._balance_batch(
                        chunk,
                        metrics=metrics,
                        logging_prefix="early_actor_chunk",
                        align_to_mini_batch=False,
                    )
                    dispatch_ts = time.time()
                    output = self.actor_rollout_wg.accumulate_actor(chunk)
                    return_ts = time.time()
                    consumed.update(take)
                    chunk_sizes.append(take_n)
                    chunk_events.append(
                        {
                            "n": take_n,
                            "teacher_ready_when_dispatched": len(ready),
                            "leftover_when_dispatched": leftover,
                            "is_last": is_last,
                            "dispatch_ts": dispatch_ts,
                            "return_ts": return_ts,
                            "chunk_fb_s": _chunk_fb_s(output),
                        }
                    )
                    if is_last:
                        finish_ts = time.time()
                        output = _first_output(self.actor_rollout_wg.finish_actor_accumulate())
                        self._trace_actor_finish_end_ts = time.time()
                        self._trace_actor_finish_start_ts = finish_ts
                        finalized = True
                        break
                if not finalized:
                    raise RuntimeError("early_actor_lite streaming F/B ended without a finalize chunk")
            finally:
                if not finalized:
                    self.actor_rollout_wg.abort_actor_accumulate()
        self._trace_actor_done_ts = time.time()
        self._trace_actor_chunk_events = chunk_events

        if output is not None:
            self._apply_actor_update_metrics(output, metrics)
        metrics["early_actor_lite/chunks"] = len(chunk_sizes)
        if chunk_sizes:
            metrics["early_actor_lite/first_chunk"] = chunk_sizes[0]

        with marked_timer("gen", timing_raw, color="red"):
            batch, off_policy_metrics = self.replay_buffer.sample(
                global_steps=self.global_steps,
                partition_id="train",
                batch_size=sample_batch_size,
            )
            metrics.update(off_policy_metrics)
            batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        self._record_gen_split_timing(batch, timing_raw)

        last_teacher = None
        for tag in batch.tags:
            if tag.get("is_padding", False) or tag.get("teacher_done_ts") is None:
                continue
            done = float(tag["teacher_done_ts"])
            last_teacher = done if last_teacher is None else max(last_teacher, done)
        if last_teacher is not None and self._trace_actor_start_ts is not None:
            metrics["early_actor_lite/overlap_s"] = max(0.0, last_teacher - self._trace_actor_start_ts)
            metrics["early_actor_lite/teachers_after_actor_start"] = sum(
                1
                for tag in batch.tags
                if not tag.get("is_padding", False)
                and tag.get("teacher_done_ts") is not None
                and float(tag["teacher_done_ts"]) > self._trace_actor_start_ts
            )
        self._dump_actor_timeline(batch)
        return batch

    def _dump_actor_timeline(self, batch) -> None:
        out_dir = os.environ.get("FOPD_SAMPLE_TRACE_DIR")
        if not out_dir:
            return
        t0 = getattr(self, "_trace_step_start_ts", None)
        if t0 is None:
            return

        def rel(ts):
            if ts is None:
                return None
            return round(float(ts) - float(t0), 4)

        last_student = None
        last_teacher = None
        for tag in batch.tags:
            if tag.get("is_padding", False):
                continue
            if tag.get("student_gen_done_ts") is not None:
                done = float(tag["student_gen_done_ts"])
                last_student = done if last_student is None else max(last_student, done)
            elif tag.get("student_last_token_ts") is not None:
                done = float(tag["student_last_token_ts"])
                last_student = done if last_student is None else max(last_student, done)
            if tag.get("teacher_done_ts") is not None:
                done = float(tag["teacher_done_ts"])
                last_teacher = done if last_teacher is None else max(last_teacher, done)
        chunks = []
        for event in getattr(self, "_trace_actor_chunk_events", []):
            chunks.append(
                {
                    **event,
                    "dispatch_s": rel(event.get("dispatch_ts")),
                    "return_s": rel(event.get("return_ts")),
                    "chunk_fb_start_s": rel(event.get("chunk_fb_start_ts")),
                    "chunk_fb_end_s": rel(event.get("chunk_fb_end_ts")),
                }
            )
        payload = {
            "step": self.global_steps,
            "t0": "step_start",
            "step_start_s": 0.0,
            "student_barrier_s": rel(getattr(self, "_trace_student_barrier_ts", None)),
            "last_student_done_s": rel(last_student),
            "sleep_start_s": rel(getattr(self, "_trace_sleep_start_ts", None)),
            "sleep_end_s": rel(getattr(self, "_trace_sleep_end_ts", None)),
            "update_actor_start_s": rel(self._trace_actor_start_ts),
            "fsdp_load_end_s": rel(getattr(self, "_trace_fsdp_load_end_ts", None)),
            "chunks": chunks,
            "optimizer_start_s": rel(getattr(self, "_trace_actor_finish_start_ts", None)),
            "optimizer_end_s": rel(getattr(self, "_trace_actor_finish_end_ts", None)),
            "update_actor_end_s": rel(self._trace_actor_done_ts),
            "last_teacher_done_s": rel(last_teacher),
        }
        if chunks:
            payload["first_chunk_dispatch_s"] = chunks[0]["dispatch_s"]
            payload["first_chunk_return_s"] = chunks[0]["return_s"]
            payload["last_chunk_dispatch_s"] = chunks[-1]["dispatch_s"]
            payload["last_chunk_return_s"] = chunks[-1]["return_s"]
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"step_{self.global_steps}_actor_timeline.json"), "w") as fh:
            json.dump(payload, fh, indent=2)

    def on_step_end(self):
        self._trace_weights_start_ts = time.time()
        with marked_timer("update_weights", self.timing_raw, color="red"):
            # wake up all replicas to update weights
            self.checkpoint_manager.update_weights(self.global_steps)
        if getattr(self, "_migration_replica_removed", False):
            manager = self.llm_server_manager
            ray.get([server.resume_generation.remote() for server in manager.rollout_replicas[-1].servers])
            ray.get(manager.global_load_balancer.add_servers.remote({manager.server_addresses[-1]: manager.server_handles[-1]}))
            self._migration_replica_removed = False
        self._trace_weights_done_ts = time.time()

    def on_sample_end(self):
        # sleep all replicas to discard weights and kv cache
        self.checkpoint_manager.sleep_replicas()
        if self.curr_step_profile:
            self._stop_rollout_profiling()
