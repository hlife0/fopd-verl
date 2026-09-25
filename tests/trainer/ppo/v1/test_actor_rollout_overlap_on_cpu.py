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

import pytest
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync


@pytest.fixture(autouse=True)
def padding_cleanup(monkeypatch):
    cleared = []
    monkeypatch.setattr(
        "verl.trainer.ppo.v1.trainer_sync.tq.kv_clear", lambda **kwargs: cleared.extend(kwargs["keys"])
    )
    return cleared


def _trainer(batch_size=5):
    trainer = PPOTrainerSync.__new__(PPOTrainerSync)
    trainer.global_steps = 1
    trainer.parameter_sync_step = 1
    trainer.early_actor_lite = True
    trainer.early_actor_stream_fb = True
    trainer.actor_rollout_overlap = True
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": batch_size, "max_response_length": 32},
            "actor_rollout_ref": {
                "actor": {"loss_agg_mode": "token-mean"},
                "rollout": {"temperature": 1.0},
            },
            "trainer": {
                "critic_warmup": 0,
                "v1": {"sync": {"actor_rollout_overlap": True, "actor_rollout_overlap_chunk_size": 0}},
            },
        }
    )
    trainer._actor_overlap_prompt_uids = [f"p{i}" for i in range(batch_size)]
    trainer._actor_dp_size = lambda: 2
    trainer._actor_update_extra_info = lambda: {"opd_no_task_reward_fast_path": True}
    trainer._apply_actor_update_metrics = lambda output, metrics: None
    trainer._record_gen_split_timing = lambda *args: None
    trainer._dump_actor_timeline = lambda batch: None
    return trainer


def _wire_stream(trainer, failure=None, readiness=None):
    """Advance readiness after F/B, or through explicit polling snapshots."""
    order = []
    trained = []
    stage = [0]
    last_student = [None]
    total = len(trainer._actor_overlap_prompt_uids)

    polls = [0]

    def snapshot(count=None):
        count = (3 if stage[0] == 0 else total) if count is None else count
        if count == total and last_student[0] is None:
            last_student[0] = time.time()
        keys = [f"p{i}_0_0" for i in range(count)]
        tags = [
            {"student_gen_done_ts": last_student[0] or 0.0, "teacher_done_ts": time.time(), "seq_len": 4}
            for _ in keys
        ]
        return KVBatchMeta(partition_id="train", keys=keys, tags=tags)

    class Replay:
        poll_interval = 2.0

        def peek_actor_overlap_batch(self, partition_id, prompt_uids, global_steps):
            assert prompt_uids == trainer._actor_overlap_prompt_uids
            assert global_steps == trainer.global_steps
            if failure == "poll" and stage[0] == 1:
                raise RuntimeError("failed prompt group")
            if readiness is not None:
                students, teachers = readiness[min(polls[0], len(readiness) - 1)]
                polls[0] += 1
                batch = snapshot(students)
                return batch, batch.keys[:teachers]
            batch = snapshot()
            return batch, batch.keys

        def materialize_actor_overlap_batch(self, *args):
            order.append("materialize")
            assert len(trained) == total
            return snapshot()

    class Workers:
        def begin_actor_accumulate(self, loss_normalization_tokens):
            order.append("begin")
            assert loss_normalization_tokens == total * 32
            if failure == "begin":
                raise RuntimeError("begin failed")

        def accumulate_actor(self, batch):
            order.append("fb")
            assert len(batch) % 2 == 0
            assert batch.extra_info["opd_no_task_reward_fast_path"]
            real = [key for key, tag in zip(batch.keys, batch.tags, strict=True) if not tag.get("is_padding")]
            assert not set(real).intersection(trained)
            trained.extend(real)
            stage[0] += 1
            if failure == "fb" or (failure == "padded_fb" and len(real) < len(batch)):
                raise RuntimeError("fb failed")
            return {"metrics": {"chunk_fb_s": 0.001, "chunk_fb_start_ts": time.time()}}

        def finish_actor_accumulate(self):
            order.append("finish")
            assert len(trained) == total
            if failure == "finish":
                raise RuntimeError("finish failed")
            return {"metrics": {}}

        def abort_actor_accumulate(self):
            order.append("abort")

    def balance(batch, **kwargs):
        assert kwargs["align_to_mini_batch"] is False
        if len(batch) % 2:
            order.append("pad")
            batch = KVBatchMeta(
                partition_id=batch.partition_id,
                keys=batch.keys + ["padding"],
                tags=batch.tags + [{"is_padding": True}],
                extra_info=batch.extra_info,
            )
        return batch

    trainer.replay_buffer = Replay()
    trainer.actor_rollout_wg = Workers()
    trainer.on_sample_begin = lambda: order.append("sample_begin")
    trainer.on_sample_end = lambda: order.append("sleep")
    trainer._balance_batch = balance
    return order, trained


def test_overlap_dispatches_before_student_barrier_pads_partial_batch_and_updates_once(padding_cleanup):
    trainer = _trainer()
    order, trained = _wire_stream(trainer)
    metrics, timing = {}, {}
    batch = trainer._step_once(metrics, timing, 5)
    assert order.index("fb") < order.index("sleep") < order.index("finish")
    assert order.count("sleep") == order.count("finish") == order.count("pad") == 1
    assert order.count("fb") == 2
    assert "abort" not in order
    assert len(set(trained)) == len(batch) == metrics["actor_rollout_overlap/samples"] == 5
    assert metrics["actor_rollout_overlap/chunks_completed_before_student"] == 1
    assert metrics["actor_rollout_overlap/samples_completed_before_student"] == 3
    assert trainer.replay_buffer.poll_interval == 2.0
    assert timing["gen"] > 0 and timing["update_actor"] > 0
    assert padding_cleanup == ["padding"]


@pytest.mark.parametrize("failure", ["begin", "fb", "padded_fb", "poll", "finish"])
def test_overlap_aborts_held_training_and_restores_poll_on_error(failure, padding_cleanup):
    trainer = _trainer()
    order, _ = _wire_stream(trainer, failure)
    with pytest.raises(RuntimeError, match="failed"):
        trainer._step_once({}, {}, 5)
    assert order.count("abort") == 1
    assert order.count("finish") == int(failure == "finish")
    assert trainer.replay_buffer.poll_interval == 2.0
    assert padding_cleanup == ["padding"]


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("early_actor_stream_fb", False, "OPD fast path"),
        ("parameter_sync_step", 2, "parameter_sync_step"),
        ("config.actor_rollout_ref.actor.loss_agg_mode", "seq-mean-token-mean", "token-mean"),
        ("config.trainer.critic_warmup", 1, "critic_warmup"),
        ("config.trainer.v1.sync.actor_rollout_overlap_chunk_size", -2, "must be 0"),
        ("config.trainer.v1.sync.actor_rollout_overlap_chunk_size", 4, "must be 0"),
        ("config.trainer.v1.sync.actor_rollout_overlap_teacher_fraction", 0, "must be in"),
        ("config.trainer.v1.sync.actor_rollout_overlap_student_fraction", 1.1, "must be in"),
    ],
)
def test_overlap_rejects_unsupported_training_semantics(field, value, message):
    trainer = _trainer()
    obj = trainer
    for part in field.split(".")[:-1]:
        obj = getattr(obj, part)
    setattr(obj, field.split(".")[-1], value)
    with pytest.raises(ValueError, match=message):
        trainer._configure_actor_rollout_overlap()


@pytest.mark.parametrize(
    "readiness,expected_chunks,first_students",
    [
        ([(5, 3), (5, 4), (8, 8)], [4, 4], 5),
        ([(5, 3), (6, 3), (8, 8)], [3, 5], 6),
        ([(6, 3), (6, 4), (8, 8)], [3, 1, 4], 6),
        ([(8, 0), (8, 8)], [8], 8),
    ],
)
def test_overlap_waits_for_either_threshold_then_drains_every_ready_sample(
    readiness, expected_chunks, first_students
):
    trainer = _trainer(batch_size=8)
    order, trained = _wire_stream(trainer, readiness=readiness)
    trainer._step_once({}, {}, 8)
    events = trainer._trace_actor_chunk_events
    assert [event["n"] for event in events] == expected_chunks
    assert all(event["n"] == event["teacher_ready_when_dispatched"] for event in events)
    assert events[0]["students_done_when_dispatched"] == first_students
    assert len(trained) == len(set(trained)) == 8
    assert order.count("begin") == order.count("finish") == 1


def test_overlap_restores_poll_when_sample_begin_fails():
    trainer = _trainer()
    order, _ = _wire_stream(trainer)

    def fail():
        raise RuntimeError("sample begin failed")

    trainer.on_sample_begin = fail
    with pytest.raises(RuntimeError, match="sample begin failed"):
        trainer._step_once({}, {}, 5)
    assert trainer.replay_buffer.poll_interval == 2.0
    assert "begin" not in order and "abort" not in order


def test_overlap_snapshots_the_submitted_prompt_ids_before_dispatch():
    trainer = _trainer(batch_size=2)
    trainer._next_train_batch = lambda: {"uid": ["x", "y"]}
    calls = []

    def submit(batch):
        assert trainer._actor_overlap_prompt_uids == ["x", "y"]
        calls.append(batch)

    trainer._submit_batch_to_rollout = submit
    assert trainer.prepare_step() == {}
    assert len(calls) == 1
