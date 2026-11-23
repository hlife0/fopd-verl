"""Student-side local staging must not enter training or touch Teacher ranks."""

import time
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.workers.engine_workers import ActorRolloutRefWorker


def make_trainer():
    trainer = PPOTrainerSync.__new__(PPOTrainerSync)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(v1=SimpleNamespace(sync={"teacher_shared_actor_preload": True}), critic_warmup=0),
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=1, temperature=1.0)),
        distillation=SimpleNamespace(n_gpus_per_node=2),
    )
    trainer.teacher_shared_actor = True
    trainer.teacher_shared_actor_no_wait_sleep = True
    trainer.teacher_shared_actor_optimize = True
    trainer.early_actor_lite = trainer.overlap_actor = False
    trainer.curr_step_profile = False
    trainer.global_steps = 1
    trainer.timing_raw = {}
    trainer.opd_no_task_reward_fast_path = True
    trainer.use_critic = trainer.use_reference_policy = False
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=[])
    trainer._balance_batch = lambda batch, metrics: batch
    trainer._record_gen_split_timing = lambda *args: None
    return trainer


@pytest.mark.parametrize("rank", range(8))
def test_preload_moves_only_student_pool_ranks(monkeypatch, rank):
    import verl.workers.engine_workers as workers

    device = SimpleNamespace(synchronize=Mock())
    monkeypatch.setattr(workers, "get_torch_device", lambda: device)
    worker = SimpleNamespace(rank=rank, actor=SimpleNamespace(to=Mock()))
    result = ActorRolloutRefWorker.preload_shared_actor_student_ranks(worker, teacher_ranks=2)
    if rank < 2:
        assert result is None
        worker.actor.to.assert_not_called()
        device.synchronize.assert_not_called()
    else:
        worker.actor.to.assert_called_once_with("device", model=True, optimizer=True, grad=True)
        device.synchronize.assert_called_once_with()
        assert result["rank"] == rank and result["load_s"] >= 0 and result["ready_ts"] > 0


@pytest.mark.parametrize("ready_early", [True, False])
@pytest.mark.parametrize("no_wait_sleep", [True, False])
def test_preload_waits_for_student_then_overlaps_teacher_without_changing_batch(ready_early, no_wait_sleep):
    trainer = make_trainer()
    trainer.teacher_shared_actor_no_wait_sleep = no_wait_sleep
    events = []
    load_started, load_done, teacher_done, allow_load = Event(), Event(), Event(), Event()
    batch = SimpleNamespace(extra_info={}, tags=[])

    def student_barrier(**kwargs):
        assert kwargs == dict(partition_id="train", batch_size=96, sessions_per_prompt=1)
        events.append("student_done")

    def student_sleep():
        assert events == ["student_done"]
        events.append("student_sleep")

    def preload(teacher_ranks):
        assert teacher_ranks == 2 and "student_sleep" in events
        events.append("preload_start")
        load_started.set()
        if not ready_early:
            assert allow_load.wait(5)
        ready_ts = time.time()
        events.append("preload_done")
        load_done.set()
        return [None, None] + [dict(rank=rank, load_s=0.01, ready_ts=ready_ts) for rank in range(2, 8)]

    def sample(**kwargs):
        assert load_started.wait(5), "preload did not overlap score collection"
        if ready_early:
            assert load_done.wait(5)
        batch.tags = [dict(teacher_done_ts=time.time()) for _ in range(96)]
        teacher_done.set()
        events.append("teacher_done")
        allow_load.set()
        return batch, {}

    def teacher_sleep():
        assert teacher_done.is_set()
        events.append("teacher_sleep")

    def train(actual, metrics):
        assert actual is batch and len(actual.tags) == 96
        assert teacher_done.is_set() and load_done.is_set()
        events.append("train_full_batch")
        return actual

    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=Mock(side_effect=student_sleep))
    trainer.teacher_model_manager = SimpleNamespace(sleep=Mock(side_effect=teacher_sleep))
    trainer.actor_rollout_wg = SimpleNamespace(preload_shared_actor_student_ranks=Mock(side_effect=preload))
    trainer.replay_buffer = SimpleNamespace(poll_interval=2.0, wait_until_students_done=student_barrier, sample=sample)
    trainer._update_actor = Mock(side_effect=train)
    metrics = {}
    assert trainer._step_once(metrics, trainer.timing_raw, 96) is batch
    for future in getattr(trainer, "_pending_vllm_sleep", ()):
        future.result(timeout=5)
    trainer.checkpoint_manager.sleep_replicas.assert_called_once_with()
    trainer.teacher_model_manager.sleep.assert_called_once_with()
    trainer._update_actor.assert_called_once()
    assert events.index("student_sleep") < events.index("preload_start") < events.index("teacher_done")
    assert events.index("teacher_done") < events.index("train_full_batch")
    assert metrics["teacher_shared_actor/preload_ranks"] == 6
    assert metrics["teacher_shared_actor/preload_ready_at_teacher_done"] == int(ready_early)
    assert trainer.replay_buffer.poll_interval == 2.0
    assert trainer._shared_student_preload_future is None
    assert not trainer._shared_student_slept_for_preload
    assert "student_preload" in trainer.timing_raw and "student_preload_sleep" in trainer.timing_raw


@pytest.mark.parametrize("failed_stage", ["sleep", "load", "sample"])
def test_preload_failure_never_starts_training_and_joins_background_work(failed_stage):
    trainer = make_trainer()
    started, finished = Event(), Event()
    failure = RuntimeError(failed_stage + " failed")

    def student_sleep():
        started.set()
        if failed_stage == "sleep":
            finished.set()
            raise failure

    def preload(**kwargs):
        finished.set()
        if failed_stage == "load":
            raise failure
        return [None, None] + [dict(rank=rank, load_s=0.0, ready_ts=time.time()) for rank in range(2, 8)]

    def sample(**kwargs):
        assert started.wait(5)
        if failed_stage == "sample":
            raise failure
        return SimpleNamespace(extra_info={}, tags=[dict(teacher_done_ts=time.time())]), {}

    trainer.replay_buffer = SimpleNamespace(poll_interval=2.0, wait_until_students_done=Mock(), sample=sample)
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=student_sleep)
    trainer.teacher_model_manager = SimpleNamespace(sleep=Mock())
    trainer.actor_rollout_wg = SimpleNamespace(preload_shared_actor_student_ranks=preload)
    trainer._update_actor = Mock()
    with pytest.raises(RuntimeError) as error:
        trainer._step_once({}, trainer.timing_raw, 96)
    assert error.value is failure
    assert finished.is_set()
    trainer._update_actor.assert_not_called()
    trainer.teacher_model_manager.sleep.assert_not_called()
    assert trainer.replay_buffer.poll_interval == 2.0
