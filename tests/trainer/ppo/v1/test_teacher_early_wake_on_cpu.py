# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.workers.engine_workers import ActorRolloutRefWorker


@pytest.mark.parametrize("with_callback", [False, True])
def test_publication_starts_both_sides_before_actor_export_callback(monkeypatch, with_callback):
    import verl.checkpoint_engine.base as module

    events = []
    actor_refs, rollout_refs = [object(), object()], [object(), object()]
    actor_final, rollout_final = [object()], [object()]
    actor_result = [{"export_s": 1.0}, {}]
    rollout_result = [{}, {}]
    actor = SimpleNamespace(
        world_size=2,
        update_weights=lambda **kw: events.append("actor_start") or actor_refs,
        execute_checkpoint_engine=lambda commands: actor_final,
    )
    rollout = SimpleNamespace(
        world_size=2,
        update_weights=lambda **kw: events.append("rollout_start") or rollout_refs,
        execute_checkpoint_engine=lambda commands: rollout_final,
    )

    def get(refs):
        assert "actor_start" in events and "rollout_start" in events
        if refs == actor_refs:
            events.append("actor_done")
            return actor_result
        if refs == rollout_refs:
            assert events[-1] == "callback"
            events.append("rollout_done")
            return rollout_result
        if refs == actor_refs + rollout_refs:
            events.append("both_done")
            return actor_result + rollout_result
        assert refs == actor_final + rollout_final
        events.append("finalize")
        return []

    monkeypatch.setattr(module.ray, "get", get)
    monkeypatch.setattr(module, "RayWorkerGroup", lambda **kw: rollout)
    monkeypatch.setattr(module, "RayClassWithInitArgs", lambda **kw: None)
    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager.backend = "nccl"
    manager.actor_wg = actor
    manager.replicas = [SimpleNamespace(workers=[])]
    manager.abort_replicas = AsyncMock()
    manager.release_kv_cache_replicas = AsyncMock()
    manager.resume_kv_cache_replicas = AsyncMock()
    manager.resume_generation_replicas = AsyncMock()
    manager.build_process_group = Mock()

    def callback():
        assert events[-1] == "actor_done"
        events.append("callback")

    async def run():
        return await manager.update_weights(7, after_actor_export=callback if with_callback else None)

    assert asyncio.run(run()) == {"export_s": 1.0}
    expected = ["actor_start", "rollout_start"]
    expected += ["actor_done", "callback", "rollout_done"] if with_callback else ["both_done"]
    assert events == expected + ["finalize"]
    manager.resume_generation_replicas.assert_awaited_once()


@pytest.mark.parametrize("failure", [None, "publication", "student", "teacher"])
def test_early_teacher_wake_overlaps_receiver_tail_and_is_joined(failure):
    trainer = PPOTrainerSync.__new__(PPOTrainerSync)
    trainer.teacher_shared_actor = True
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(v1=SimpleNamespace(sync={
        "teacher_shared_actor_early_teacher_wake": True,
    })))
    trainer.global_steps = 3
    trainer.timing_raw = {}
    actor_exported, offloaded = Event(), Event()
    teacher_started, release_teacher, teacher_finished = Event(), Event(), Event()
    publication_returning = Event()
    error = RuntimeError(str(failure))

    def offload():
        assert actor_exported.is_set()
        assert not teacher_started.is_set()
        offloaded.set()

    def teacher_wake():
        assert offloaded.is_set() and not publication_returning.is_set()
        teacher_started.set()
        try:
            assert release_teacher.wait(5)
            if failure == "teacher":
                raise error
        finally:
            teacher_finished.set()

    def publication(step, *, after_actor_export):
        assert step == 3
        actor_exported.set()
        after_actor_export()
        assert teacher_started.wait(5)
        assert not teacher_finished.is_set()
        publication_returning.set()
        if failure == "publication":
            raise error
        return {"publication": 1}

    def student_wake(tags):
        if tags == ["weights"]:
            assert not actor_exported.is_set()
            return
        assert tags == ["kv_cache"] and publication_returning.is_set()
        if failure == "student":
            raise error

    trainer.actor_rollout_wg = SimpleNamespace(offload_actor_for_teacher_wake=Mock(side_effect=offload))
    trainer.teacher_model_manager = SimpleNamespace(wake_up=Mock(side_effect=teacher_wake))
    trainer.checkpoint_manager = SimpleNamespace(update_weights=publication, wake_up_replicas=student_wake)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(trainer._publish_actor_weights)
        try:
            assert publication_returning.wait(5)
            assert not result.done(), "publication returned before the Teacher wake finished"
        finally:
            release_teacher.set()
        if failure:
            with pytest.raises(RuntimeError) as exc:
                result.result(timeout=5)
            assert exc.value is error
        else:
            assert result.result(timeout=5) == {"publication": 1}
    assert teacher_finished.is_set()
    trainer.actor_rollout_wg.offload_actor_for_teacher_wake.assert_called_once()
    trainer.teacher_model_manager.wake_up.assert_called_once()


def test_teacher_early_wake_fences_and_releases_actor_memory(monkeypatch):
    import verl.workers.engine_workers as module

    events = []
    worker = SimpleNamespace(actor=SimpleNamespace(to=lambda **kw: events.append(("offload", kw))))
    device = SimpleNamespace(
        synchronize=lambda: events.append("sync"),
        empty_cache=lambda: events.append("empty_cache"),
    )
    monkeypatch.setattr(module, "get_torch_device", lambda: device)
    ActorRolloutRefWorker.offload_actor_for_teacher_wake(worker)
    assert events == [("offload", {"device": "cpu"}), "sync", "empty_cache"]
