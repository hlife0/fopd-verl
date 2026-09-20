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
import logging
import os
import time

from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("sync")
class PPOTrainerSync(PPOTrainer):
    """Synchronous PPO trainer
    1. Trainer and rollout are colocated
    2. Partial rollout is disabled
    """

    def on_init_end(self):
        self._configure_opd_no_task_reward_fast_path()
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
