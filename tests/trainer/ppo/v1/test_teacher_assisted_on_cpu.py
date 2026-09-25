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

import asyncio

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
from verl.trainer.ppo.v1.teacher_assisted import (
    assign_migrations,
    auxiliary_rollout_config,
    completion_threshold,
    migration_fits,
    should_switch,
    resident_engines,
    teacher_assisted_active,
)
from verl.workers.rollout.llm_server import AssistedRolloutLLMServerClient, FullyAsyncLLMServerClient, LLMServerClient
from verl.workers.rollout.replica import TokenOutput


def _config(enabled, borrow_s, ratio=None, **overrides):
    cfg = {
        "trainer": {
            "v1": {
                "trainer_mode": "sync",
                "sync": {
                    "teacher_assisted_rollout": enabled,
                    "teacher_assisted_max_borrow_s": borrow_s,
                    "teacher_assisted_complete_ratio": ratio,
                    "early_actor_lite": True,
                    "opd_no_task_reward_fast_path": True,
                },
            }
        },
        "data": {"train_batch_size": 48},
        "actor_rollout_ref": {
            "actor": {
                "strategy": "fsdp",
                "ppo_epochs": 1,
                "ppo_mini_batch_size": 48,
                "ulysses_sequence_parallel_size": 1,
                "fsdp_config": {"ulysses_sequence_parallel_size": 1},
            },
            "rollout": {
                "n": 1,
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 1,
                "data_parallel_size": 1,
            },
        },
        "critic": {"enable": False},
        "distillation": {
            "n_gpus_per_node": 2,
            "nnodes": 1,
            "teacher_models": {
                "teacher_model": {"inference": {"tensor_model_parallel_size": 2, "pipeline_model_parallel_size": 1}}
            },
        },
    }
    cfg = OmegaConf.merge(OmegaConf.create(cfg), OmegaConf.create(overrides))
    return cfg


def test_auxiliary_config_does_not_change_the_main_rollout():
    rollout = OmegaConf.create(
        {
            "gpu_memory_utilization": 0.4,
            "enable_sleep_mode": True,
            "free_cache_engine": True,
            "checkpoint_engine": {"backend": "naive", "update_weights_bucket_megabytes": 2048},
        }
    )
    copied = auxiliary_rollout_config(rollout, 0.2, 256)
    assert float(rollout.gpu_memory_utilization) == 0.4
    assert int(rollout.checkpoint_engine.update_weights_bucket_megabytes) == 2048
    assert str(rollout.checkpoint_engine.backend) == "naive"
    assert float(copied.gpu_memory_utilization) == 0.2
    assert int(copied.checkpoint_engine.update_weights_bucket_megabytes) == 256
    assert str(copied.checkpoint_engine.backend) == "nccl"
    assert copied.checkpoint_engine.engine_kwargs.nccl.multi_sender is False
    assert "engine_kwargs" not in rollout.checkpoint_engine or rollout.checkpoint_engine.get("engine_kwargs") in (
        None,
        {},
    )


def test_feature_off_and_zero_borrow_do_not_activate():
    assert teacher_assisted_active(_config(False, 30, n=2)) is False
    assert teacher_assisted_active(_config(True, 0)) is False
    assert teacher_assisted_active(_config(True, 4)) is True


def test_active_borrow_rejects_a_different_actor_path():
    with pytest.raises(ValueError, match="rollout.n=1"):
        teacher_assisted_active(_config(True, 4, **{"actor_rollout_ref": {"rollout": {"n": 2}}}))
    with pytest.raises(ValueError, match="early_actor_lite"):
        teacher_assisted_active(_config(True, 4, **{"trainer": {"v1": {"sync": {"early_actor_lite": False}}}}))
    with pytest.raises(ValueError, match="finite"):
        teacher_assisted_active(_config(True, float("nan")))


def test_completion_threshold_ceils_ratio():
    assert completion_threshold(None, 48) is None
    assert completion_threshold(0.5, 48) == 24
    assert completion_threshold(0.1, 48) == 5
    assert completion_threshold(0, 48) == 0
    assert [completion_threshold(r, 48) for r in (0.05, 0.10, 0.15, 0.20, 0.25)] == [3, 5, 8, 10, 12]


def test_tp1_teacher_pool_activates_and_mismatched_tp_does_not():
    tp1 = {
        "actor_rollout_ref": {"rollout": {"tensor_model_parallel_size": 1}},
        "distillation": {
            "n_gpus_per_node": 1,
            "teacher_models": {
                "teacher_model": {
                    "inference": {
                        "tensor_model_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                        "gpu_memory_utilization": 0.85,
                    }
                }
            },
        },
    }
    assert teacher_assisted_active(_config(True, 120, ratio=0.05, **tp1)) is True
    with pytest.raises(ValueError, match="same TP"):
        teacher_assisted_active(
            _config(
                True,
                120,
                **{
                    "distillation": {
                        "n_gpus_per_node": 1,
                        "teacher_models": {
                            "teacher_model": {"inference": {"tensor_model_parallel_size": 1}}
                        },
                    }
                },
            )
        )


def test_resident_follows_card_capacity_not_an_unsafe_override():
    six = _config(True, 4)
    six.distillation.teacher_models.teacher_model.inference.gpu_memory_utilization = 0.4
    assert resident_engines(six) is True
    four = _config(
        True,
        120,
        **{
            "actor_rollout_ref": {"rollout": {"tensor_model_parallel_size": 1}},
            "distillation": {
                "n_gpus_per_node": 1,
                "teacher_models": {
                    "teacher_model": {
                        "inference": {"tensor_model_parallel_size": 1, "gpu_memory_utilization": 0.85}
                    }
                },
            },
        },
    )
    assert resident_engines(four) is False
    four.trainer.v1.sync.teacher_assisted_resident = True
    assert resident_engines(four) is False
    six.trainer.v1.sync.teacher_assisted_resident = False
    assert resident_engines(six) is False


def test_auxiliary_config_keeps_the_draft_block():
    rollout = OmegaConf.create(
        {
            "gpu_memory_utilization": 0.4,
            "enable_sleep_mode": True,
            "free_cache_engine": True,
            "checkpoint_engine": {"backend": "naive", "update_weights_bucket_megabytes": 2048},
            "speculative": {"method": "eagle3", "num_speculative_tokens": 3},
        }
    )
    copied = auxiliary_rollout_config(rollout, 0.2, 256)
    assert copied.speculative.method == "eagle3"
    assert int(copied.speculative.num_speculative_tokens) == 3
    assert rollout.speculative.method == "eagle3"


def test_switch_on_count_time_or_all_done_only_once_each_check():
    assert should_switch(4, 5, 1.0, 10.0, False) is None
    assert should_switch(5, 5, 1.0, 10.0, False) == "complete_count"
    assert should_switch(1, None, 10.0, 10.0, False) == "max_borrow_time"
    assert should_switch(48, 40, 1.0, 10.0, True) == "all_students_done"


def test_migration_spreads_across_original_replicas():
    assigned = assign_migrations(5, {"a": 3, "b": 1, "c": 1})
    assert assigned.count("a") == 1
    assert assigned.count("b") == 2
    assert assigned.count("c") == 2
    assert migration_fits(2, {"a": 1, "b": 1}, max_num_seqs=2) is True
    assert migration_fits(2, {"a": 2, "b": 2}, max_num_seqs=2) is False
    assert assign_migrations(0, {"a": 1}) == []


def test_assisted_client_does_not_use_the_fixed_retry_wait():
    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {}}})
    assisted = AssistedRolloutLLMServerClient(config=config, load_balancer_handle=None)
    baseline = FullyAsyncLLMServerClient(config=config, load_balancer_handle=None)
    assert assisted.abort_retry_wait_s == 0
    assert baseline.abort_retry_wait_s == 1


def test_replay_buffer_counts_only_finished_student_trajectories():
    buffer = ReplayBuffer.__new__(ReplayBuffer)
    buffer.partitions = {
        "train": {
            "done": {"student_gen_done_ts": 1.0},
            "partial": {},
            "pad": {"student_gen_done_ts": 2.0, "is_padding": True},
        }
    }
    buffer._sync_metadata_from_transfer_queue = lambda: None
    seen = []
    buffer.student_progress_hook = seen.append
    assert buffer.count_complete_student_trajectories("train") == 1
    buffer._notify_student_progress("train")
    buffer._notify_student_progress("val")
    assert seen == [1]


def _run_resume(client_cls, second_step):
    outputs = [
        TokenOutput(
            token_ids=[1, 2],
            log_probs=[-0.1, -0.2],
            stop_reason="aborted",
            extra_fields={
                "global_steps": 3,
                "rollout_server_id": "aux",
                "student_submit_ts": 10.0,
                "student_first_token_ts": 11.0,
                "engine_arrival_ts": 10.5,
            },
        ),
        TokenOutput(
            token_ids=[3],
            log_probs=[-0.3],
            stop_reason="completed",
            extra_fields={
                "global_steps": second_step,
                "student_submit_ts": 20.0,
                "student_first_token_ts": 21.0,
                "engine_arrival_ts": 20.5,
            },
        ),
    ]
    prompts = []

    async def fake(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        prompts.append(list(prompt_ids))
        return outputs[len(prompts) - 1]

    original = LLMServerClient.generate
    LLMServerClient.generate = fake
    try:
        client = client_cls(
            config=OmegaConf.create({"actor_rollout_ref": {"rollout": {"response_length": 8}}}),
            load_balancer_handle=None,
        )
        client.expected_weight_version = 3
        client.auxiliary_server_ids = {"aux"}
        result = asyncio.run(client.generate("r", prompt_ids=[7], sampling_params={"max_tokens": 4}))
    finally:
        LLMServerClient.generate = original
    return result, prompts


def test_resume_keeps_the_earliest_student_span_and_the_same_version():
    result, prompts = _run_resume(AssistedRolloutLLMServerClient, 3)
    assert result.token_ids == [1, 2, 3]
    assert result.extra_fields["student_submit_ts"] == 10.0
    assert result.extra_fields["student_first_token_ts"] == 11.0
    assert result.extra_fields["engine_arrival_ts"] == 10.5
    assert result.extra_fields["migration_count"] == 1
    assert result.extra_fields["migrated_prefix_tokens"] == 2
    assert result.extra_fields["aux_migrated_token_counts"] == [2]
    assert result.log_probs == [-0.1, -0.2, -0.3]
    assert prompts[1] == [7, 1, 2]
    with pytest.raises(RuntimeError, match="weight version"):
        _run_resume(AssistedRolloutLLMServerClient, 4)


def test_none_version_and_logprob_mismatch_are_rejected():
    original = LLMServerClient.generate

    async def no_version(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        return TokenOutput(token_ids=[1], log_probs=[-0.1], stop_reason="completed", extra_fields={})

    async def bad_logprobs(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        return TokenOutput(
            token_ids=[1, 2],
            log_probs=[-0.1],
            stop_reason="completed",
            extra_fields={"global_steps": 3},
        )

    client = AssistedRolloutLLMServerClient(
        config=OmegaConf.create({"actor_rollout_ref": {"rollout": {"response_length": 8}}}),
        load_balancer_handle=None,
    )
    client.expected_weight_version = 3
    try:
        LLMServerClient.generate = no_version
        with pytest.raises(RuntimeError, match="without a weight version"):
            asyncio.run(client.generate("r", prompt_ids=[7], sampling_params={"max_tokens": 4}))
        LLMServerClient.generate = bad_logprobs
        with pytest.raises(RuntimeError, match="logprobs do not match"):
            asyncio.run(client.generate("r", prompt_ids=[7], sampling_params={"max_tokens": 4}))
    finally:
        LLMServerClient.generate = original


def test_worker_uses_batch_version_after_serialization_without_sharing_request_state():
    from types import SimpleNamespace

    import cloudpickle
    from tensordict import NonTensorData, TensorDict

    import verl.trainer.ppo.v1.agent_loop_tq as worker_mod

    worker_cls = worker_mod.AgentLoopWorkerTQ.__ray_metadata__.modified_class
    driver_client = AssistedRolloutLLMServerClient(
        config=OmegaConf.create({"actor_rollout_ref": {"rollout": {}}}), load_balancer_handle=None
    )
    worker_client = cloudpickle.loads(cloudpickle.dumps(driver_client))
    driver_client.expected_weight_version = 99
    batch = cloudpickle.loads(cloudpickle.dumps(TensorDict({
        "expected_student_version": NonTensorData(3),
        "auxiliary_student_address": NonTensorData("aux"),
    }, batch_size=[1])))
    owner = SimpleNamespace(_teacher_follow_enabled=lambda *_: False)

    class Loop:
        server_manager = worker_client

        async def run(self, sampling_params, **kwargs):
            await asyncio.sleep(0)
            return self.server_manager.expected_weight_version, self.server_manager.auxiliary_server_ids

    async def run():
        first, second = Loop(), Loop()
        result = await asyncio.gather(
            worker_cls._invoke_agent_loop(owner, first, {}, {}, agent_name="single_turn_agent",
                **{key: value.data for key, value in batch.items()}),
            worker_cls._invoke_agent_loop(owner, second, {}, {}, agent_name="single_turn_agent",
                expected_student_version=4, auxiliary_student_address="other"),
        )
        assert result == [(3, {"aux"}), (4, {"other"})]
        assert first.server_manager is not second.server_manager
        assert getattr(worker_client, "expected_weight_version", None) is None
        with pytest.raises(RuntimeError, match="missing its expected"):
            await worker_cls._invoke_agent_loop(owner, Loop(), {}, {}, agent_name="single_turn_agent")

    asyncio.run(run())


def test_replay_buffer_skips_unpublished_tags_without_dropping(monkeypatch):
    from collections import defaultdict

    import verl.trainer.ppo.v1.replay_buffer as replay_mod

    buffer = ReplayBuffer.__new__(ReplayBuffer)
    buffer.partitions = defaultdict(dict)
    buffer.pending_keys = defaultdict(set)
    buffer.running_keys = defaultdict(set)
    buffer.finished_keys = defaultdict(set)
    buffer.failure_keys = defaultdict(set)
    buffer.prompt_global_steps = defaultdict(dict)
    cleared = []
    monkeypatch.setattr(replay_mod.tq, "kv_list", lambda: {
        "train": {
            "reserved": {},
            "prompt": {"is_prompt": True, "global_steps": 1},
            "ready": {"is_prompt": True, "global_steps": 1, "status": "running"},
            "traj": {"student_gen_done_ts": 1.0},
        }
    })
    monkeypatch.setattr(replay_mod.tq, "kv_clear", lambda **kwargs: cleared.append(kwargs))
    buffer._sync_metadata_from_transfer_queue()
    assert cleared == []
    assert "prompt" not in buffer.running_keys["train"]
    assert buffer.running_keys["train"] == {"ready"}
    assert buffer.partitions["train"]["traj"]["student_gen_done_ts"] == 1.0


def test_server_keeps_prefix_when_abort_terminal_is_empty_and_wake_releases_admission():
    import verl.workers.rollout.vllm_rollout.vllm_async_server as server_mod
    from verl.workers.rollout.replica import RolloutMode

    class LogProb:
        def __init__(self, logprob):
            self.logprob = logprob

    class Completion:
        def __init__(self, token_ids, logprobs, finish_reason):
            self.token_ids = token_ids
            self.logprobs = logprobs
            self.finish_reason = finish_reason

    class Request:
        def __init__(self, outputs):
            self.outputs = outputs
            self.metrics = None
            self.num_cached_tokens = 0

    def chunk(token_ids, finish_reason, logprobs=None):
        if logprobs is None:
            logprobs = [{token_id: LogProb(-0.1 * (index + 1))} for index, token_id in enumerate(token_ids)]
        return Request([Completion(token_ids, logprobs, finish_reason)])

    class Engine:
        def __init__(self):
            self.level = None
            self.woke = None
            self.entered = 0

        async def sleep(self, level):
            self.level = level

        async def wake_up(self, tags=None):
            self.woke = list(tags or [])

        async def reset_prefix_cache(self, reset_connector=False):
            return None

        def generate(self, **kwargs):
            self.entered += 1

            async def _gen():
                yield chunk([4, 5], None)
                yield chunk([4, 5, 6], None)
                yield Request([])

            return _gen()

    class Config:
        max_model_len = 32
        response_length = 8
        prompt_length = 8
        full_determinism = False
        free_cache_engine = True
        enable_rollout_routing_replay = False
        mtp = None

        def get(self, key, default=None):
            return default

    server = server_mod.vLLMHttpServer.__new__(server_mod.vLLMHttpServer)
    server.config = Config()
    server.model_config = type("Model", (), {"processor": None, "lora_rank": 0, "lora": {}})()
    server.replica_rank = 0
    server.node_rank = 0
    server.rollout_mode = RolloutMode.COLOCATED
    server.global_steps = 3
    server.engine = Engine()
    server._disaggregation_role = None
    server._pd_decode_peers = None
    server._migration_evacuated = False
    server._submission_paused = False
    server._serving = False
    server._serving_event = asyncio.Event()
    server._resume_event = asyncio.Event()
    server._admitting = 0
    server._get_wake_up_tags = lambda: ["kv_cache", "weights"]

    async def run():
        task = asyncio.create_task(
            server.generate(prompt_ids=[1], sampling_params={"logprobs": True}, request_id="r")
        )
        await asyncio.sleep(0)
        assert not task.done()
        assert server.engine.entered == 0
        await server.sleep()
        assert server.engine.level == 1
        assert not task.done()
        await server.wake_up()
        await server.set_serving(True)
        result = await task
        return result

    result = asyncio.run(run())
    assert result.token_ids == [4, 5, 6]
    assert result.log_probs == pytest.approx([-0.1, -0.2, -0.3])
    assert result.stop_reason == "aborted"
    assert server.engine.entered == 1
    assert server.engine.woke == ["kv_cache", "weights"]

    server._serving = True
    server._serving_event.set()
    server.engine.generate = lambda **kwargs: _missing_logprob_stream()

    async def _missing_logprob_stream():
        yield chunk([9], None, logprobs=[None])

    async def missing():
        return await server.generate(prompt_ids=[1], sampling_params={"logprobs": True}, request_id="r2")

    with pytest.raises(RuntimeError, match="no logprob"):
        asyncio.run(missing())
