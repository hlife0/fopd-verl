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

import threading
import time
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync


def _lite_trainer() -> PPOTrainerSync:
    trainer = PPOTrainerSync.__new__(PPOTrainerSync)
    trainer.early_actor_lite = True
    trainer.early_actor_stream_fb = False
    trainer.opd_no_task_reward_fast_path = False
    trainer.use_reference_policy = False
    trainer.use_critic = False
    trainer.global_steps = 1
    trainer._trace_actor_start_ts = None
    trainer._trace_actor_done_ts = None
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"n": 1, "temperature": 1.0}},
            "trainer": {"critic_warmup": 0, "v1": {"sync": {"early_actor_lite": True}}},
            "algorithm": {"filter_groups": {"enable": False}},
        }
    )
    return trainer


def test_early_actor_lite_sleeps_then_overlaps_old_log_prob_before_sample():
    trainer = _lite_trainer()
    order: list[str] = []

    class ReplayBufferStub:
        poll_interval = 2.0

        def wait_until_students_done(self, **kwargs):
            order.append("wait_students")

        def peek_student_ready(self, **kwargs):
            order.append("peek")
            return SimpleNamespace(extra_info={})

        def sample(self, **kwargs):
            order.append("sample")
            return SimpleNamespace(extra_info={}), {}

    trainer.replay_buffer = ReplayBufferStub()
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=object())
    trainer.on_sample_begin = lambda: order.append("sample_begin")
    trainer.on_sample_end = lambda: order.append("sleep")
    trainer._balance_batch = lambda batch, metrics: order.append("balance") or batch
    trainer._compute_old_log_prob = lambda batch, metrics: order.append("old_log_prob") or batch
    trainer._compute_advantage = lambda batch, metrics: order.append("adv") or batch
    trainer._update_actor = lambda batch, metrics: order.append("update_actor") or batch
    trainer._record_gen_split_timing = lambda *args, **kwargs: None

    batch = trainer._step_once({}, {}, sample_batch_size=2)

    assert order.index("sleep") < order.index("old_log_prob") < order.index("sample")
    assert order.count("sleep") == 1
    assert order.count("old_log_prob") == 1
    assert trainer.replay_buffer.poll_interval == 2.0
    assert batch.extra_info["temperature"] == 1.0


def test_early_actor_lite_streams_fb_on_teacher_ready_before_sample():
    trainer = _lite_trainer()
    trainer.opd_no_task_reward_fast_path = True
    trainer.early_actor_stream_fb = True
    order: list[str] = []

    class ReplayBufferStub:
        poll_interval = 2.0

        def __init__(self):
            self._ready_calls = 0

        def wait_until_students_done(self, **kwargs):
            order.append("wait_students")

        def peek_student_ready(self, **kwargs):
            order.append("peek")
            return SimpleNamespace(
                keys=["a", "b", "c", "d"],
                tags=[{"response_len": 2, "is_padding": False} for _ in range(4)],
                extra_info={},
                partition_id="train",
            )

        def peek_teacher_ready_keys(self, partition_id, traj_keys):
            self._ready_calls += 1
            if self._ready_calls == 1:
                return ["a", "b"]
            return list(traj_keys)

        def peek_trajectories(self, partition_id, traj_keys):
            order.append(f"chunk:{len(traj_keys)}")
            return SimpleNamespace(
                keys=list(traj_keys),
                tags=[{} for _ in traj_keys],
                extra_info={},
                partition_id=partition_id,
            )

        def sample(self, **kwargs):
            order.append("sample")
            return (
                SimpleNamespace(
                    keys=["a", "b", "c", "d"],
                    tags=[
                        {"is_padding": False, "teacher_done_ts": 10.0, "student_gen_done_ts": 1.0}
                        for _ in range(4)
                    ],
                    extra_info={},
                ),
                {},
            )

    class WorkerGroupStub:
        def begin_actor_accumulate(self):
            order.append("begin")

        def accumulate_actor(self, batch):
            order.append("accumulate")
            return {"metrics": {"mfu": 0.1, "chunk_fb_s": 0.01}}

        def finish_actor_accumulate(self):
            order.append("finish")
            return {"metrics": {"mfu": 0.1, "grad_norm": 1.0}}

        def abort_actor_accumulate(self):
            order.append("abort")

    trainer.replay_buffer = ReplayBufferStub()
    trainer.actor_rollout_wg = WorkerGroupStub()
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=object())
    trainer.on_sample_begin = lambda: order.append("sample_begin")
    trainer.on_sample_end = lambda: order.append("sleep")
    trainer._actor_dp_size = lambda: 2
    trainer._actor_update_extra_info = lambda: {}
    trainer._apply_actor_update_metrics = lambda output, metrics: order.append("metrics")
    trainer._balance_batch = lambda batch, metrics, **kwargs: order.append("balance") or batch
    trainer._record_gen_split_timing = lambda *args, **kwargs: None

    batch = trainer._step_once({}, {}, sample_batch_size=4)

    assert order.index("sleep") < order.index("begin") < order.index("accumulate")
    assert order.index("finish") < order.index("sample")
    assert "old_log_prob" not in order
    assert "abort" not in order
    assert order.count("accumulate") == 2
    assert "chunk:2" in order
    assert trainer.replay_buffer.poll_interval == 2.0
    assert batch.extra_info["temperature"] == 1.0


def test_early_actor_lite_rejects_group_filtering():
    trainer = _lite_trainer()
    trainer.config.algorithm.filter_groups.enable = True
    with pytest.raises(ValueError, match="group filtering"):
        trainer._configure_early_actor_lite()


def test_record_gen_split_timing_skips_missing_teacher_done():
    trainer = _lite_trainer()
    timing = {}
    batch = SimpleNamespace(
        tags=[
            {"is_padding": False, "student_gen_done_ts": 2.0, "teacher_done_ts": None, "student_gen_start_ts": 1.0},
            {"is_padding": False, "student_gen_done_ts": 3.0, "teacher_done_ts": 4.0, "student_gen_start_ts": 1.5},
        ]
    )
    trainer._record_gen_split_timing(batch, timing)
    assert timing["gen_student"] == 1.5
    assert timing["gen_teacher"] == 1.0


def test_disabled_early_actor_lite_uses_base_step(monkeypatch):
    trainer = _lite_trainer()
    trainer.early_actor_lite = False
    called = {}

    def fake_base(self, metrics, timing_raw, sample_batch_size):
        called["sample_batch_size"] = sample_batch_size
        return "base"

    monkeypatch.setattr("verl.trainer.ppo.v1.trainer_base.PPOTrainer._step_once", fake_base)
    assert trainer._step_once({}, {}, sample_batch_size=8) == "base"
    assert called["sample_batch_size"] == 8


@pytest.mark.parametrize("failure", [None, "sleep", "fb"])
def test_early_actor_plus_starts_after_students_and_joins_sleep_before_update(failure):
    trainer = _lite_trainer()
    trainer.early_actor_plus = True
    trainer.early_actor_stream_fb = True
    trainer.opd_no_task_reward_fast_path = True
    students_done, sleep_started, fb_started, sleep_done = [threading.Event() for _ in range(4)]
    order = []
    batch = SimpleNamespace(
        keys=["a", "b"],
        tags=[{"response_len": 2, "teacher_done_ts": 1.0} for _ in range(2)],
        extra_info={},
        partition_id="train",
    )

    def wait_students(**kwargs):
        assert not sleep_started.is_set() and not fb_started.is_set()
        students_done.set()

    def sleep():
        assert students_done.is_set()
        sleep_started.set()
        try:
            # A serial sleep-then-F/B implementation cannot satisfy this wait.
            assert fb_started.wait(3), "F/B did not start while sleep was pending"
            if failure == "sleep":
                raise RuntimeError("sleep failed")
        finally:
            order.append("sleep_done")
            sleep_done.set()

    class Workers:
        def begin_actor_accumulate(self):
            assert students_done.is_set()

        def accumulate_actor(self, chunk):
            assert sleep_started.wait(3)
            assert students_done.is_set() and not sleep_done.is_set()
            start = time.time()
            fb_started.set()
            assert sleep_done.wait(3)
            if failure == "fb":
                raise RuntimeError("fb failed")
            return {"metrics": {"chunk_fb_s": 0.001,
                                "chunk_fb_start_ts": [start], "chunk_fb_end_ts": [time.time()]}}

        def finish_actor_accumulate(self):
            assert sleep_done.is_set()
            order.append("finish")
            return {"metrics": {}}

        def abort_actor_accumulate(self):
            order.append("abort")

    trainer.replay_buffer = SimpleNamespace(
        poll_interval=2.0,
        wait_until_students_done=wait_students,
        peek_student_ready=lambda **kwargs: batch,
        peek_teacher_ready_keys=lambda *args: batch.keys,
        peek_trajectories=lambda *args: batch,
        sample=lambda **kwargs: (batch, {}),
    )
    trainer.actor_rollout_wg = Workers()
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=object())
    trainer.on_sample_begin = lambda: None
    trainer.on_sample_end = sleep
    trainer._actor_dp_size = lambda: 2
    trainer._actor_update_extra_info = lambda: {}
    trainer._balance_batch = lambda batch, **kwargs: batch
    trainer._apply_actor_update_metrics = lambda *args: None
    trainer._record_gen_split_timing = lambda *args: None
    trainer._dump_actor_timeline = lambda *args: None
    metrics = {}
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            trainer._step_once(metrics, {}, 2)
        assert "finish" not in order and "abort" in order
    else:
        assert trainer._step_once(metrics, {}, 2) is batch
        assert order.index("sleep_done") < order.index("finish")
        assert metrics["early_actor_plus/fb_sleep_overlap_s"] > 0
    assert sleep_done.is_set()
    assert trainer.replay_buffer.poll_interval == 2.0


def test_early_actor_plus_selects_streaming_path_and_rejects_missing_fast_path():
    trainer = _lite_trainer()
    trainer.config.trainer.v1.sync = {"early_actor_plus": True}
    trainer.config.data = {"train_batch_size": 2}
    trainer.config.actor_rollout_ref.actor = {"ppo_epochs": 1, "ppo_mini_batch_size": 2, "strategy": "fsdp"}
    with pytest.raises(ValueError, match="early_actor_plus requires"):
        trainer._configure_early_actor_lite()
    trainer.opd_no_task_reward_fast_path = True
    trainer._configure_early_actor_lite()
    assert trainer.early_actor_lite and trainer.early_actor_stream_fb and trainer.early_actor_plus
