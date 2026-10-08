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

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.single_controller.ray.base import RayResourcePool
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.rollout.replica import RolloutMode


@pytest.fixture
def config():
    config_dir = Path(__file__).resolve().parents[4] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(
            config_name="ppo_trainer",
            overrides=[
                "trainer.v1.sync.teacher_shared_actor=True",
                "trainer.n_gpus_per_node=2",
                "trainer.nnodes=1",
                "distillation.enabled=True",
                "distillation.n_gpus_per_node=1",
                "distillation.nnodes=1",
                "distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-4B",
                "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1",
                "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
                "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
                "+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.multi_sender=False",
                "algorithm.adv_estimator=grpo",
                "critic.enable=False",
            ],
        )


def make_trainer(config):
    trainer = PPOTrainerSync.__new__(PPOTrainerSync)
    trainer.config = config
    trainer.use_critic = trainer.use_reference_policy = False
    trainer.teacher_shared_actor = True
    trainer.teacher_shared_actor_no_wait_sleep = config.trainer.v1.sync.teacher_shared_actor_no_wait_sleep
    trainer.early_actor_lite = trainer.overlap_actor = False
    trainer.curr_step_profile = False
    trainer.global_steps = 1
    trainer.timing_raw = {}
    return trainer


@pytest.mark.parametrize("student,teacher,tp", [(2, 1, 1), (3, 1, 1), (6, 2, 2)])
def test_actor_pool_includes_teacher_without_double_reserving_gpus(config, student, teacher, tp):
    config.trainer.n_gpus_per_node = student
    config.distillation.n_gpus_per_node = teacher
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = tp
    trainer = make_trainer(config)
    trainer._init_resource_pool_mgr()
    assert trainer.resource_pool_manager.resource_pool_spec == {"global_pool": [student + teacher]}
    assert trainer.mapping[Role.ActorRollout] == trainer.mapping[Role.TeacherModel] == "global_pool"
    assert trainer.resource_pool_manager.get_n_gpus() == student + teacher
    pool = RayResourcePool(process_on_nodes=[student + teacher])
    pool.pgs = []  # Already allocated; no Ray runtime needed for sub-pool slicing.
    trainer.resource_pool_manager.resource_pool_dict["global_pool"] = pool
    trainer._split_teacher_shared_pool()
    teacher_pool = trainer.resource_pool_manager.get_resource_pool(Role.TeacherModel)
    assert teacher_pool.start_bundle_index == 0
    assert teacher_pool.world_size == teacher
    assert trainer.student_resource_pool.start_bundle_index == teacher
    assert trainer.student_resource_pool.world_size == student


def test_disabled_keeps_separate_teacher_pool(config):
    config.trainer.v1.sync.teacher_shared_actor = False
    trainer = make_trainer(config)
    trainer._init_resource_pool_mgr()
    assert trainer.resource_pool_manager.resource_pool_spec == {"global_pool": [2], "teacher_pool": [1]}


def test_no_wait_sleep_requires_shared_actor(config):
    config.trainer.v1.sync.teacher_shared_actor = False
    config.trainer.v1.sync.teacher_shared_actor_no_wait_sleep = True
    with pytest.raises(ValueError, match="requires teacher_shared_actor=True"):
        make_trainer(config)._init_resource_pool_mgr()


def test_no_wait_sleep_config_enabled(config):
    config.trainer.v1.sync.teacher_shared_actor_no_wait_sleep = True
    trainer = make_trainer(config)
    trainer._init_resource_pool_mgr()
    assert trainer.teacher_shared_actor_no_wait_sleep
    assert trainer.resource_pool_manager.resource_pool_spec == {"global_pool": [3]}


def test_multiple_batches_per_publication_rejected(config):
    OmegaConf.update(config, "trainer.v1.sync.parameter_sync_step", 2, force_add=True)
    with pytest.raises(ValueError, match="parameter_sync_step=1"):
        make_trainer(config)._init_resource_pool_mgr()


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("trainer.v1.sync.early_actor_lite", True, "waits for all Teacher scores"),
        ("trainer.v1.sync.overlap_actor", True, "waits for all Teacher scores"),
        ("actor_rollout_ref.actor.fsdp_config.param_offload", False, "offload"),
        ("actor_rollout_ref.actor.fsdp_config.optimizer_offload", False, "offload"),
        ("actor_rollout_ref.rollout.free_cache_engine", False, "Student sleep"),
        ("distillation.teacher_models.teacher_model.inference.free_cache_engine", False, "Teacher sleep"),
        ("actor_rollout_ref.rollout.checkpoint_engine.backend", "naive", "NCCL"),
        ("actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.multi_sender", True, "duplicate GPU"),
        ("trainer.nnodes", 2, "single node"),
        ("actor_rollout_ref.rollout.tensor_model_parallel_size", 3, "divisible"),
    ],
)
def test_invalid_layout_rejected_before_allocation(config, key, value, match):
    OmegaConf.update(config, key, value)
    with pytest.raises(ValueError, match=match):
        make_trainer(config)._init_resource_pool_mgr()


def test_all_scores_arrive_before_teacher_sleeps_and_training_starts(config):
    trainer = make_trainer(config)
    events = []
    batch = SimpleNamespace(extra_info={})

    def sample(**kwargs):
        events.append("all_scores")
        return batch, {}

    trainer.replay_buffer = SimpleNamespace(sample=sample)
    trainer.on_sample_begin = lambda: None
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: events.append("student_sleep"))
    trainer.teacher_model_manager = SimpleNamespace(sleep=lambda: events.append("teacher_sleep"))
    trainer._record_gen_split_timing = lambda *args: None
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=[])

    class ReachedTraining(Exception):
        pass

    def balance(*args, **kwargs):
        events.append("training")
        raise ReachedTraining

    trainer._balance_batch = balance
    with pytest.raises(ReachedTraining):
        trainer._step_once({}, {}, 48)
    assert events == ["all_scores", "student_sleep", "teacher_sleep", "training"]


@pytest.mark.parametrize("blocked", [("student",), ("teacher",), ("student", "teacher")])
@pytest.mark.parametrize("oom", [False, True])
def test_no_wait_dispatches_actor_while_sleep_is_pending(config, blocked, oom):
    config.trainer.v1.sync.teacher_shared_actor_no_wait_sleep = True
    trainer = make_trainer(config)
    trainer.opd_no_task_reward_fast_path = True
    trainer.curr_step_profile = True
    trainer._stop_rollout_profiling = Mock()
    batch = SimpleNamespace(extra_info={})
    scores_ready = Event()
    started = {name: Event() for name in ("student", "teacher")}
    release = Event()

    def sample(**kwargs):
        scores_ready.set()
        return batch, {}

    def sleep(name):
        assert scores_ready.is_set()
        started[name].set()
        if name in blocked:
            assert release.wait(10), "test did not release sleep"

    trainer.replay_buffer = SimpleNamespace(sample=sample)
    trainer.on_sample_begin = lambda: None
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: sleep("student"))
    trainer.teacher_model_manager = SimpleNamespace(sleep=lambda: sleep("teacher"))
    trainer._record_gen_split_timing = lambda *args: None
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=[])
    trainer._balance_batch = lambda batch, **kwargs: batch
    trainer._actor_update_extra_info = lambda: {}
    trainer._apply_actor_update_metrics = lambda *args: None
    failure = torch.OutOfMemoryError("simulated Actor OOM")

    def update_actor(actual_batch):
        assert actual_batch is batch
        assert all(event.wait(5) for event in started.values())
        assert not release.is_set()
        for index, name in enumerate(("student", "teacher")):
            if name in blocked:
                assert not trainer._pending_vllm_sleep[index].done()
        trainer._stop_rollout_profiling.assert_not_called()
        if oom:
            raise failure

    trainer.actor_rollout_wg = SimpleNamespace(update_actor=Mock(side_effect=update_actor))
    with ThreadPoolExecutor(max_workers=1) as runner:
        step = runner.submit(trainer._step_once, {}, {}, 48)
        try:
            if oom:
                with pytest.raises(torch.OutOfMemoryError) as exc:
                    step.result(timeout=5)
                assert exc.value is failure
            else:
                assert step.result(timeout=5) is batch
            trainer.actor_rollout_wg.update_actor.assert_called_once_with(batch)
        finally:
            release.set()
    for future in trainer._pending_vllm_sleep:
        future.result(timeout=5)


@pytest.mark.parametrize("slow_side", ["student", "teacher"])
def test_no_wait_finishes_both_sleeps_before_waking_inference(config, slow_side):
    config.trainer.v1.sync.teacher_shared_actor_no_wait_sleep = True
    trainer = make_trainer(config)
    trainer.curr_step_profile = True
    started = {name: Event() for name in ("student", "teacher")}
    release = Event()
    wake = Event()
    events = []

    def sleep(name):
        started[name].set()
        if name == slow_side:
            assert release.wait(10), "test did not release sleep"
        events.append(name + "_slept")

    def wake_student(tags):
        assert "student_slept" in events and "teacher_slept" in events
        wake.set()
        events.append(tuple(tags))

    trainer.checkpoint_manager = SimpleNamespace(
        sleep_replicas=lambda: sleep("student"),
        wake_up_replicas=wake_student,
        update_weights=lambda step: events.append("publish"),
    )
    trainer.teacher_model_manager = SimpleNamespace(
        sleep=lambda: sleep("teacher"), wake_up=lambda: events.append("teacher_wake")
    )
    trainer.actor_rollout_wg = SimpleNamespace(to=lambda device: events.append(device))
    trainer._stop_rollout_profiling = lambda: events.append("stop_profile")
    trainer.on_sample_end()
    # fit() advances this flag before publication; cleanup must use the saved flag.
    trainer.curr_step_profile = False
    with ThreadPoolExecutor(max_workers=1) as runner:
        publication = runner.submit(trainer._publish_actor_weights)
        try:
            assert all(event.wait(5) for event in started.values())
            assert not wake.wait(0.1)
        finally:
            release.set()
        publication.result(timeout=5)
    assert set(events[:2]) == {"student_slept", "teacher_slept"}
    assert events[2:] == ["stop_profile", ("weights",), "publish", "cpu", ("kv_cache",), "teacher_wake"]
    assert trainer._pending_vllm_sleep == ()


@pytest.mark.parametrize("failed_side", ["student", "teacher"])
def test_no_wait_sleep_failure_propagates_before_publication(config, failed_side):
    config.trainer.v1.sync.teacher_shared_actor_no_wait_sleep = True
    trainer = make_trainer(config)

    def sleep(name):
        if name == failed_side:
            raise RuntimeError("sleep failed")

    trainer.checkpoint_manager = SimpleNamespace(
        sleep_replicas=lambda: sleep("student"), wake_up_replicas=Mock(), update_weights=Mock()
    )
    trainer.teacher_model_manager = SimpleNamespace(sleep=lambda: sleep("teacher"), wake_up=Mock())
    trainer.on_sample_end()
    try:
        with pytest.raises(RuntimeError, match="sleep failed"):
            trainer._publish_actor_weights()
        trainer.checkpoint_manager.wake_up_replicas.assert_not_called()
        trainer.checkpoint_manager.update_weights.assert_not_called()
        trainer.teacher_model_manager.wake_up.assert_not_called()
    finally:
        for future in trainer._pending_vllm_sleep:
            try:
                future.result(timeout=5)
            except RuntimeError:
                pass


@pytest.mark.parametrize("fail_publication", [False, True])
def test_teacher_wakes_only_after_all_actor_ranks_finish_publication(config, fail_publication):
    trainer = make_trainer(config)
    events = []

    def publish(step):
        assert step == 1
        events.append("publish")
        if fail_publication:
            raise RuntimeError("publication failed")
        return {}

    trainer.checkpoint_manager = SimpleNamespace(
        update_weights=publish, wake_up_replicas=lambda tags: events.append(tuple(tags))
    )
    trainer.actor_rollout_wg = SimpleNamespace(to=lambda device: events.append(device))
    trainer.teacher_model_manager = SimpleNamespace(wake_up=lambda: events.append("teacher_wake"))
    if fail_publication:
        with pytest.raises(RuntimeError, match="publication failed"):
            trainer._publish_actor_weights()
        assert events == [("weights",), "publish"]
    else:
        trainer._publish_actor_weights()
        assert events == [("weights",), "publish", "cpu", ("kv_cache",), "teacher_wake"]


def test_rollout_reuses_only_student_subpool(config, monkeypatch):
    import verl.workers.rollout.llm_server as server_module

    pool = SimpleNamespace(world_size=2)
    pieces = [object(), object()]
    placed = []

    class Replica:
        def __init__(self, replica_rank, **kwargs):
            self.replica_rank = replica_rank
            self._server_handle = object()
            self._server_address = str(replica_rank)

        async def init_colocated(self, subpool):
            placed.append((self.replica_rank, subpool))

    def split(actual_pool, size):
        assert actual_pool is pool and size == 1
        return pieces

    manager = LLMServerManager.__new__(LLMServerManager)
    manager.rollout_replica_class = Replica
    manager.rollout_config = config.actor_rollout_ref.rollout
    manager.model_config = config.actor_rollout_ref.model
    manager.worker_group = None
    manager.rollout_resource_pool = pool
    manager.start_rank = 0
    monkeypatch.setattr(server_module, "split_resource_pool", split)
    asyncio.run(manager._initialize_llm_servers())
    assert placed == [(0, pieces[0]), (1, pieces[1])]


def test_colocated_vllm_wakes_only_requested_memory():
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    server = SimpleNamespace(
        node_rank=0,
        rollout_mode=RolloutMode.COLOCATED,
        engine=SimpleNamespace(wake_up=AsyncMock(), reset_prefix_cache=AsyncMock()),
        _get_wake_up_tags=lambda: ["weights", "kv_cache"],
    )
    asyncio.run(vLLMHttpServer.wake_up(server, tags=["weights"]))
    server.engine.wake_up.assert_awaited_once_with(tags=["weights"])
