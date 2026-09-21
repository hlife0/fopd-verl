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
import os
import time

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


def _chunk_fb_s(output) -> float | None:
    output = _first_output(output)
    if output is None:
        return None
    try:
        value = output["metrics"]["chunk_fb_s"]
    except Exception:
        return None
    if isinstance(value, (list, tuple)):
        value = value[-1]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@register_trainer("sync")
class PPOTrainerSync(PPOTrainer):
    """Synchronous PPO trainer
    1. Trainer and rollout are colocated
    2. Partial rollout is disabled
    """

    def on_init_end(self):
        self._configure_opd_no_task_reward_fast_path()
        self._configure_early_actor_lite()
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

    def _step_once(self, metrics: dict, timing_raw: dict, sample_batch_size: int):
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
        self._trace_weights_done_ts = time.time()

    def on_sample_end(self):
        # sleep all replicas to discard weights and kv cache
        self.checkpoint_manager.sleep_replicas()
        if self.curr_step_profile:
            self._stop_rollout_profiling()
