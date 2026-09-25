# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync


def _trainer(actor_gpus=1, rollout_gpus=2, teacher_gpus=1, tp=1):
    trainer = PPOTrainerSync.__new__(PPOTrainerSync)
    trainer.config = OmegaConf.create({
        "data": {"train_batch_size": 48, "max_response_length": 2048},
        "actor_rollout_ref": {
            "model": {"lora_rank": 0},
            "actor": {
                "strategy": "fsdp2", "loss_agg_mode": "token-mean",
                "fsdp_config": {"param_offload": False, "optimizer_offload": False},
            },
            "rollout": {
                "name": "vllm", "nnodes": 1, "n_gpus_per_node": rollout_gpus,
                "tensor_model_parallel_size": tp, "data_parallel_size": 1, "pipeline_model_parallel_size": 1,
                "free_cache_engine": False, "enable_sleep_mode": False, "temperature": 1.0,
                "checkpoint_engine": {"backend": "nccl"},
            },
        },
        "trainer": {"nnodes": 1, "n_gpus_per_node": actor_gpus, "critic_warmup": 0, "v1": {
            "trainer_mode": "sync", "sync": {
                "separate_rollout": True, "actor_rollout_overlap": True,
                "early_actor_lite": True, "opd_no_task_reward_fast_path": True,
                "actor_rollout_overlap_chunk_size": 0,
            },
            "sampler": {"max_off_policy_threshold": 1, "max_off_policy_strategy": "drop", "sampler_kwargs": {}},
        }},
        "algorithm": {"filter_groups": {"enable": False}},
        "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
        "distillation": {"enabled": True, "nnodes": 1, "n_gpus_per_node": teacher_gpus},
    })
    trainer.trainer_mode = "sync"
    trainer.use_reference_policy = trainer.use_critic = False
    trainer.actor_rollout_overlap = trainer.early_actor_stream_fb = True
    trainer.parameter_sync_step = 1
    trainer.global_steps = 1
    trainer.curr_step_profile = False
    trainer.timing_raw = {}
    trainer._configure_separate_rollout()
    return trainer


@pytest.mark.parametrize("layout", [(1, 2, 1, 1), (2, 4, 2, 2)])
def test_fixed_roles_allocate_actor_only_and_standalone_rollout(layout):
    trainer = _trainer(*layout)
    with (
        patch("verl.trainer.ppo.v1.trainer_base.ray.remote", side_effect=lambda cls: cls),
        patch("verl.trainer.ppo.v1.trainer_base.need_critic", return_value=False),
        patch("verl.trainer.ppo.v1.trainer_base.need_reference_policy", return_value=False),
    ):
        trainer._init_resource_pool_mgr()
    assert list(trainer.role_worker_mapping) == [Role.Actor]
    assert trainer.mapping[Role.Actor] == "global_pool"
    assert trainer.resource_pool_manager.resource_pool_spec == {
        "global_pool": [layout[0]], "teacher_pool": [layout[2]],
    }
    trainer.resource_pool_manager.get_n_gpus = lambda: layout[0] + layout[2]
    assert trainer._get_n_gpus_for_throughput() == sum(layout[:3])

    trainer.actor_rollout_wg = object()
    replicas = [object(), object()]
    server = Mock()
    server.get_replicas.return_value = replicas
    with (
        patch("verl.trainer.ppo.v1.trainer_base.LLMServerManager.create", return_value=server) as create,
        patch("verl.trainer.ppo.v1.trainer_base.omega_conf_to_dataclass", side_effect=lambda cfg: cfg),
        patch("verl.trainer.ppo.v1.trainer_base.CheckpointEngineManager") as checkpoint,
    ):
        trainer._init_rollout_and_checkpoint_manager(object())
    create.assert_called_once_with(config=trainer.config)
    assert checkpoint.call_args.kwargs["actor_wg"] is trainer.actor_rollout_wg
    assert checkpoint.call_args.kwargs["replicas"] is replicas
    assert checkpoint.call_args.kwargs["config"].backend == "nccl"
    checkpoint.return_value.sleep_replicas.assert_not_called()


def test_original_sync_keeps_hybrid_server_and_naive_transport():
    trainer = _trainer()
    trainer.separate_rollout = False
    trainer.actor_rollout_wg = object()
    resource_pool = object()
    with (
        patch("verl.trainer.ppo.v1.trainer_base.LLMServerManager.create") as create,
        patch("verl.trainer.ppo.v1.trainer_base.omega_conf_to_dataclass", side_effect=lambda cfg: cfg),
        patch("verl.trainer.ppo.v1.trainer_base.CheckpointEngineManager") as checkpoint,
    ):
        trainer._init_rollout_and_checkpoint_manager(resource_pool)
    create.assert_called_once_with(
        config=trainer.config, worker_group=trainer.actor_rollout_wg, rollout_resource_pool=resource_pool
    )
    assert checkpoint.call_args.kwargs["config"].backend == "naive"


@pytest.mark.parametrize("field,value,match", [
    ("trainer.v1.trainer_mode", "separate_async", "trainer_mode=sync"),
    ("trainer.v1.sync.actor_rollout_overlap_chunk_size", 3, "chunk_size=0"),
    ("trainer.v1.sync.actor_rollout_overlap", False, "requires actor_rollout_overlap"),
    ("actor_rollout_ref.rollout.n_gpus_per_node", 0, "positive dedicated"),
    ("actor_rollout_ref.rollout.tensor_model_parallel_size", 3, "divisible"),
    ("actor_rollout_ref.rollout.checkpoint_engine.backend", "naive", "backend=nccl"),
    ("actor_rollout_ref.rollout.free_cache_engine", True, "free_cache_engine=false"),
    ("actor_rollout_ref.rollout.enable_sleep_mode", True, "enable_sleep_mode=false"),
    ("actor_rollout_ref.actor.fsdp_config.param_offload", True, "offload disabled"),
    ("actor_rollout_ref.actor.fsdp_config.optimizer_offload", True, "offload disabled"),
])
def test_separate_rejects_role_switching_or_invalid_allocation(field, value, match):
    trainer = _trainer()
    OmegaConf.update(trainer.config, field, value)
    with pytest.raises(ValueError, match=match):
        trainer._configure_separate_rollout()


def _wire_pipeline(trainer, wrong_version=False, ready_counts=(1, 48)):
    events, submitted, trained = [], [], []
    published = [0]
    steps = {}
    trainer._actor_dp_size = lambda: trainer.config.trainer.n_gpus_per_node
    trainer._actor_update_extra_info = lambda: {"opd_no_task_reward_fast_path": True}
    trainer._apply_actor_update_metrics = lambda *args: None
    trainer._record_gen_split_timing = lambda *args: None
    trainer._dump_actor_timeline = lambda *args: None
    def balance(batch, **kwargs):
        assert kwargs["align_to_mini_batch"] is False
        pad = (-len(batch)) % trainer._actor_dp_size()
        return KVBatchMeta(
            partition_id=batch.partition_id,
            keys=batch.keys + [f"padding_{len(trained)}_{i}" for i in range(pad)],
            tags=batch.tags + [{"is_padding": True} for _ in range(pad)],
            extra_info=batch.extra_info,
        )

    trainer._balance_batch = balance

    def submit(batch):
        assert published[0] == trainer.global_steps - 1
        submitted.extend(batch["uid"])
        steps[trainer.global_steps] = {"uids": list(batch["uid"]), "trained": [], "last_student": None, "calls": 0}
        events.append((trainer.global_steps, "submit"))

    trainer._next_train_batch = lambda: {"uid": [f"s{trainer.global_steps}p{i}" for i in range(48)]}
    trainer._submit_batch_to_rollout = submit

    class Replay:
        poll_interval = 2.0

        def peek_actor_overlap_batch(self, partition_id, prompt_uids, global_steps):
            state = steps[global_steps]
            assert prompt_uids == state["uids"]
            # Final rollout cannot complete until the early F/B has returned.
            count = ready_counts[min(state["calls"], len(ready_counts) - 1)]
            if count == 48 and state["last_student"] is None:
                state["last_student"] = time.time()
                events.append((global_steps, "last_student"))
            version = published[0] - int(wrong_version)
            keys = [f"{uid}_0_0" for uid in prompt_uids[:count]]
            tags = [{
                "student_gen_done_ts": state["last_student"] or time.time(), "teacher_done_ts": time.time(),
                "min_global_steps": version, "max_global_steps": version,
            } for _ in keys]
            return KVBatchMeta(partition_id="train", keys=keys, tags=tags), keys

        def materialize_actor_overlap_batch(self, partition_id, prompt_uids, global_steps):
            assert len(steps[global_steps]["trained"]) == 48
            return self.peek_actor_overlap_batch(partition_id, prompt_uids, global_steps)[0]

    class Actor:
        def begin_actor_accumulate(self, loss_normalization_tokens):
            assert loss_normalization_tokens == 48 * 2048
            events.append((trainer.global_steps, "begin"))

        def accumulate_actor(self, chunk):
            state = steps[trainer.global_steps]
            real = [key for key, tag in zip(chunk.keys, chunk.tags, strict=True) if not tag.get("is_padding")]
            assert len(chunk) % trainer._actor_dp_size() == 0
            assert not set(real).intersection(trained)
            expected = ready_counts[min(state["calls"], len(ready_counts) - 1)] - len(state["trained"])
            assert len(real) == expected
            trained.extend(real)
            state["trained"].extend(real)
            state["calls"] += 1
            events.append((trainer.global_steps, "fb"))
            return {"metrics": {"chunk_fb_s": 0.001}}

        def finish_actor_accumulate(self):
            assert len(steps[trainer.global_steps]["trained"]) == 48
            events.append((trainer.global_steps, "update"))
            return {"metrics": {}}

        def abort_actor_accumulate(self):
            events.append((trainer.global_steps, "abort"))

    def publish(version):
        assert events[-1] == (version, "update")
        assert len(submitted) == 48 * version
        events.append((version, "publish_done"))
        published[0] = version
        return {}

    trainer.replay_buffer = Replay()
    trainer.actor_rollout_wg = Actor()
    trainer.checkpoint_manager = SimpleNamespace(update_weights=publish, sleep_replicas=Mock())
    return events, submitted, trained


def test_two_sync_batches_train_early_once_and_publish_before_next_submit():
    trainer = _trainer()
    assert type(trainer._build_replay_buffer()) is ReplayBuffer
    events, submitted, trained = _wire_pipeline(trainer)
    for step in (1, 2):
        trainer.global_steps = step
        trainer.on_train_begin()  # No async warmup submission.
        assert len(submitted) == (step - 1) * 48
        trainer.prepare_step()
        metrics = {}
        trainer._step_once(metrics, {}, 48)
        trainer.on_step_end()
        assert events.count((step, "fb")) == 2
        assert metrics["actor_rollout_overlap/samples"] == 48
        assert metrics["actor_rollout_overlap/samples_completed_before_student"] == 1
        assert events.count((step, "begin")) == events.count((step, "update")) == 1
        assert events.index((step, "fb")) < events.index((step, "last_student"))
    assert len(submitted) == len(set(trained)) == 96
    assert events.index((1, "publish_done")) < events.index((2, "submit"))
    trainer.checkpoint_manager.sleep_replicas.assert_not_called()
    assert not hasattr(trainer, "_trace_sleep_start_ts")


def test_wrong_rollout_weight_version_is_rejected_before_actor_training():
    trainer = _trainer()
    events, _, trained = _wire_pipeline(trainer, wrong_version=True)
    trainer.prepare_step()
    with pytest.raises(RuntimeError, match="published Actor version"):
        trainer._step_once({}, {}, 48)
    assert trained == [] and (1, "begin") not in events


@pytest.mark.parametrize("ready_counts", [(3, 8, 48), (48,)])
def test_separate_drains_ready_samples_and_cleans_intermediate_padding(ready_counts):
    trainer = _trainer(actor_gpus=2)
    events, _, trained = _wire_pipeline(trainer, ready_counts=ready_counts)
    trainer.prepare_step()
    with patch("verl.trainer.ppo.v1.trainer_sync.tq.kv_clear") as clear:
        trainer._step_once({}, {}, 48)
    assert len(trained) == len(set(trained)) == 48
    assert events.count((1, "fb")) == len(ready_counts)
    assert events.count((1, "update")) == 1
    if ready_counts == (3, 8, 48):
        clear.assert_called_once_with(partition_id="train", keys=["padding_0_0", "padding_3_0"])
    else:
        clear.assert_not_called()
