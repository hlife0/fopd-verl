# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
from types import SimpleNamespace
from unittest.mock import Mock
from threading import Event

import pytest
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu


def make_trainer():
    t = PPOTrainerSync.__new__(PPOTrainerSync)
    t.config = OmegaConf.create({
        "trainer": {"total_epochs": 1, "v1": {"sync": {
            "teacher_prompt_lookahead": True,
            "teacher_shared_actor_early_teacher_wake": True}}},
        "distillation": {"teacher_follow": True},
        "actor_rollout_ref": {"rollout": {"agent": {"default_agent_loop": "single_turn_agent"}}},
    })
    t.teacher_shared_actor = True
    t.parameter_sync_step = 1
    t.global_steps = 3
    t.total_training_steps = 8
    t.steps_per_epoch = 8
    t.timing_raw = {}
    t._configure_teacher_prompt_lookahead()
    return t


def test_prefetched_batch_is_consumed_exactly_once_and_retagged():
    t = make_trainer()
    batch = TensorDict({}, batch_size=[2])
    tu.assign_non_tensor_data(batch, "global_steps", 3)
    t._next_train_batch = Mock(return_value=batch)
    t._submit_batch_to_rollout = Mock()
    t._prepare_teacher_prompt_lookahead()
    t._next_train_batch.assert_called_once_with()
    t._submit_batch_to_rollout.assert_not_called()
    assert t._teacher_lookahead_batch is batch
    assert batch["teacher_lookahead_batch_id"] == "4"
    t.global_steps = 4
    t.prepare_step()
    t._next_train_batch.assert_called_once_with()
    t._submit_batch_to_rollout.assert_called_once_with(batch)
    assert batch["global_steps"] == 4
    assert t._teacher_lookahead_batch is None


@pytest.mark.parametrize("step,total,per_epoch,epochs", [(8,8,8,1), (8,20,8,1), (4,20,4,1)])
def test_final_step_never_prefetches_unused_data(step,total,per_epoch,epochs):
    t = make_trainer()
    t.global_steps, t.total_training_steps, t.steps_per_epoch = step,total,per_epoch
    t.config.trainer.total_epochs = epochs
    t._next_train_batch = Mock()
    t._prepare_teacher_prompt_lookahead()
    t._next_train_batch.assert_not_called()
    assert t._teacher_lookahead_batch is None


def test_first_step_has_no_seed_and_disabled_path_never_fetches():
    t = make_trainer()
    t._next_train_batch = Mock(return_value=TensorDict({}, batch_size=[2]))
    t._submit_batch_to_rollout = Mock()
    t.global_steps = 1
    t.prepare_step()
    assert "teacher_lookahead_batch_id" not in t._submit_batch_to_rollout.call_args.args[0]
    t.teacher_prompt_lookahead = False
    t._next_train_batch.reset_mock()
    t._prepare_teacher_prompt_lookahead()
    t._next_train_batch.assert_not_called()


def test_lookahead_is_after_export_offload_and_teacher_wake():
    t = make_trainer()
    t._teacher_lookahead_batch = object()
    events = []
    primed = Event()
    def prime(batch, *, batch_id):
        assert batch is t._teacher_lookahead_batch and batch_id == "4"
        assert events[-1] == "teacher_wake"
        events.append("prime")
        primed.set()
        return {"primed_prompts": 2}
    def publication(step, *, after_actor_export):
        events.append("export_done")
        after_actor_export()
        assert primed.wait(5)
        events.append("publish_done")
        return {}
    t.actor_rollout_wg = SimpleNamespace(offload_actor_for_teacher_wake=lambda: events.append("offload"))
    t.teacher_model_manager = SimpleNamespace(wake_up=lambda: events.append("teacher_wake"))
    t.agent_loop_manager = SimpleNamespace(prime_teacher_prompts=prime)
    t.checkpoint_manager = SimpleNamespace(update_weights=publication, wake_up_replicas=lambda **kw: None)
    t._publish_actor_weights()
    assert events == ["export_done", "offload", "teacher_wake", "prime", "publish_done"]
    assert t._teacher_lookahead_metrics["primed_prompts"] == 2


def test_end_measurement_includes_prefetch_and_publication():
    t = make_trainer()
    t._trace_step_start_ts = 1.0
    events=[]
    t._prepare_teacher_prompt_lookahead = lambda: events.append("fetch")
    t._publish_actor_weights = lambda: events.append("publish_and_prime") or {}
    t.on_step_end()
    assert events == ["fetch", "publish_and_prime"]
    m=t._pending_sync_metrics
    assert m["lookahead/continuous_elapsed_s"] == m["lookahead/weights_done_ts"] - 1.0
    assert m["lookahead/weights_start_ts"] <= m["lookahead/weights_done_ts"]


def test_publication_without_started_step_does_not_invent_measurement():
    t = make_trainer()
    t._trace_step_start_ts = None
    t._prepare_teacher_prompt_lookahead = lambda: None
    t._publish_actor_weights = lambda: {"weight_publication/export_s": 0.5}
    t.on_step_end()
    assert t._pending_sync_metrics == {"weight_publication/export_s": 0.5}
    assert not hasattr(t, "_lookahead_measure_start_ts")
