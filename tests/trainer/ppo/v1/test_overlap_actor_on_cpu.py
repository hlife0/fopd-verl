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

"""Scheduling tests with a busy GPU worker and a separate CPU control actor."""

import threading
import time
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.overlap_actor import OverlapActorControl


@pytest.fixture
def trainer(monkeypatch):
    trainer = PPOTrainerSync.__new__(PPOTrainerSync)
    trainer.global_steps = 1
    trainer.overlap_actor = True
    trainer.early_actor_lite = False
    trainer.opd_no_task_reward_fast_path = True
    trainer.parameter_sync_step = 1
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 7, "max_response_length": 32},
            "actor_rollout_ref": {
                "actor": {"strategy": "fsdp", "loss_agg_mode": "token-mean", "ppo_epochs": 1, "ppo_mini_batch_size": 7},
                "rollout": {"temperature": 1.0},
            },
            "trainer": {"critic_warmup": 0, "v1": {"sync": {"overlap_actor": True, "overlap_actor_sm_fraction": 0.5}}},
        }
    )
    trainer._overlap_prompt_uids = [f"p{i}" for i in range(7)]
    trainer._actor_dp_size = lambda: 3
    trainer._actor_update_extra_info = lambda: {}
    trainer._apply_actor_update_metrics = lambda *args: None
    trainer._record_gen_split_timing = lambda *args: None
    trainer._dump_actor_timeline = lambda *args: None
    control = OverlapActorControl()
    control.begin(1)
    trainer.control = control
    trainer._overlap_control = SimpleNamespace(
        **{key: SimpleNamespace(remote=getattr(control, key)) for key in ("begin", "get", "full", "cancel")}
    )
    monkeypatch.setattr("verl.trainer.ppo.v1.trainer_sync.ray.get", lambda value: value)
    trainer.cleared = []
    monkeypatch.setattr("verl.trainer.ppo.v1.trainer_sync.tq.kv_clear", lambda **kw: trainer.cleared.extend(kw["keys"]))
    return trainer


def _wire(trainer, late_teacher=False, failure=None):
    order, trained = [], []
    busy = threading.Event()
    released = threading.Event()
    polls = [0]

    def snapshot(count):
        return KVBatchMeta(
            partition_id="train",
            keys=[f"p{i}_0_0" for i in range(count)],
            tags=[
                {"student_gen_done_ts": time.time(), "teacher_done_ts": time.time(), "seq_len": 4} for _ in range(count)
            ],
        )

    class Replay:
        poll_interval = 0.001

        def peek_actor_overlap_batch(self, *args):
            polls[0] += 1
            if busy.is_set():
                if failure == "poll":
                    raise RuntimeError("poll failed")
                batch = snapshot(7)
                return batch, batch.keys
            if late_teacher:
                batch = snapshot(7)
                return batch, [] if polls[0] == 1 else batch.keys
            batch = snapshot(min(polls[0], 3))
            return batch, batch.keys

        def materialize_actor_overlap_batch(self, *args):
            assert len(trained) == 7
            return snapshot(7)

    class Workers:
        def begin_overlap_actor(self, control, step, denominator):
            order.append("begin")
            assert denominator == 7 * 32
            if failure == "begin":
                raise RuntimeError("begin failed")

        def accumulate_actor(self, batch):
            order.append("fb")
            busy.set()
            if failure == "fb":
                raise RuntimeError("fb failed")
            # A chunk stays busy until the independent trainer/control path
            # signals full SMs or cancellation. Blocking trainer code deadlocks.
            deadline = time.monotonic() + 3
            while trainer.control.phase == OverlapActorControl.LIMITED:
                if time.monotonic() > deadline:
                    raise AssertionError("trainer did not signal while F/B was busy")
                time.sleep(0.001)
            if trainer.control.phase == OverlapActorControl.CANCELLED:
                raise RuntimeError("cancelled")
            released.set()
            real = [key for key, tag in zip(batch.keys, batch.tags, strict=True) if not tag.get("is_padding")]
            assert len(batch) % 3 == 0
            assert not set(real).intersection(trained)
            trained.extend(real)
            return {"metrics": {"chunk_fb_s": 0.001}}

        def finish_actor_accumulate(self, student_done_ts):
            order.append("finish")
            assert trainer.control.phase == OverlapActorControl.FULL
            assert student_done_ts > 0
            if failure == "finish":
                raise RuntimeError("finish failed")
            return {"metrics": {}}

        def abort_actor_accumulate(self):
            order.append("abort")
            assert trainer.control.phase == OverlapActorControl.CANCELLED

    def balance(batch, **kwargs):
        assert not kwargs["align_to_mini_batch"]
        n = (-len(batch)) % 3
        return KVBatchMeta(
            partition_id="train",
            keys=batch.keys + [f"pad{i}" for i in range(n)],
            tags=batch.tags + [{"is_padding": True}] * n,
            extra_info=batch.extra_info,
        )

    trainer.actor_rollout_wg = Workers()
    trainer.replay_buffer = Replay()
    trainer._balance_batch = balance
    trainer.on_sample_begin = lambda: None
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: order.append("sleep"))
    trainer.curr_step_profile = False
    return order, trained, released


def test_first_dp_group_starts_and_signal_interrupts_chunk_then_tail_pads(trainer):
    order, trained, released = _wire(trainer)
    batch = trainer._step_once({}, {}, 7)
    assert released.is_set()
    assert order.index("fb") < order.index("sleep") < order.index("finish")
    assert order.count("begin") == order.count("finish") == order.count("sleep") == 1
    assert len(trained) == len(set(trained)) == len(batch) == 7
    assert [event["n"] for event in trainer._trace_actor_chunk_events] == [3, 4]
    assert trainer.cleared == ["pad0", "pad1"]


def test_teacher_late_skips_limited_phase(trainer):
    order, trained, _ = _wire(trainer, late_teacher=True)
    trainer._step_once({}, {}, 7)
    assert order.index("sleep") < order.index("begin")
    assert len(trained) == 7
    assert len(trainer._trace_actor_chunk_events) == 1


@pytest.mark.parametrize("failure", ["begin", "fb", "poll", "finish"])
def test_failure_cancels_before_abort_and_never_updates(trainer, failure):
    order, _, _ = _wire(trainer, failure=failure)
    with pytest.raises(RuntimeError, match="failed"):
        trainer._step_once({}, {}, 7)
    assert trainer.control.phase == OverlapActorControl.CANCELLED
    assert order.count("abort") == 1
    assert order.count("finish") == int(failure == "finish")


def test_next_step_resets_control_before_submission(trainer):
    trainer.control.full(1)
    trainer.global_steps = 2
    trainer._next_train_batch = lambda: {"uid": ["new"]}
    calls = []
    trainer._submit_batch_to_rollout = lambda batch: calls.append(trainer.control.get(2))
    trainer.prepare_step()
    assert calls == [(OverlapActorControl.LIMITED, 0.0)]
    assert trainer._overlap_prompt_uids == ["new"]
    with pytest.raises(RuntimeError, match="different step"):
        trainer.control.full(1)


def test_full_signal_precedes_profile_teardown(trainer):
    trainer.curr_step_profile = True
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: None)
    phases = []
    trainer._stop_rollout_profiling = lambda: phases.append(trainer.control.get(1)[0])
    trainer.on_sample_end()
    assert phases == [OverlapActorControl.FULL]
    assert trainer._trace_sleep_end_ts <= trainer._trace_full_sm_signal_ts


@pytest.mark.parametrize("sync_config", [{}, {"overlap_actor": False, "overlap_actor_sm_fraction": 0}])
def test_disabled_overlap_keeps_early_actor_and_needs_no_green_or_mps(trainer, monkeypatch, sync_config):
    trainer.config.trainer.v1.sync = sync_config
    trainer.early_actor_lite = True
    trainer.opd_no_task_reward_fast_path = False

    def unexpected_setup(*args):
        raise AssertionError("disabled overlap must not initialize GPU resources or control")

    trainer.actor_rollout_wg = SimpleNamespace(configure_overlap_actor=unexpected_setup)
    monkeypatch.setattr("verl.utils.overlap_actor.create_overlap_control", unexpected_setup)
    trainer._configure_overlap_actor()
    assert trainer.overlap_actor is False
    assert trainer.early_actor_lite is True


def test_overlap_can_be_enabled_without_early_actor_or_teacher_follow(trainer, monkeypatch):
    fractions = []
    trainer.actor_rollout_wg = SimpleNamespace(configure_overlap_actor=fractions.append)
    handle = object()
    monkeypatch.setattr("verl.utils.overlap_actor.create_overlap_control", lambda: handle)
    trainer._configure_overlap_actor()
    assert trainer.overlap_actor is True
    assert trainer.early_actor_lite is False
    assert trainer._overlap_control is handle
    assert fractions == [0.5]


def test_overlap_rejects_simultaneous_early_actor(trainer):
    trainer.early_actor_lite = True
    with pytest.raises(ValueError, match="select either overlap_actor or early_actor_lite"):
        trainer._configure_overlap_actor()


@pytest.mark.parametrize("fraction", [0, 1, -1, float("nan")])
def test_invalid_sm_fraction_fails_before_worker_setup(trainer, fraction):
    trainer.config.trainer.v1.sync.overlap_actor_sm_fraction = fraction
    with pytest.raises(ValueError, match="between 0 and 1"):
        trainer._configure_overlap_actor()
