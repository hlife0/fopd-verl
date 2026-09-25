# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import time
from types import SimpleNamespace

import pytest
from transfer_queue import KVBatchMeta

from test_actor_rollout_overlap_on_cpu import _trainer


def _wire(monkeypatch, failure=None):
    trainer = _trainer(batch_size=9)
    trainer.actor_rollout_migrate = True
    trainer.actor_rollout_overlap = False
    trainer._migration_dp = trainer._migration_chunk = 1
    trainer._migration_fraction = .5
    trainer._migration_units = ["root", "layer"]
    trainer._actor_dp_size = lambda: 3
    trainer.tokenizer = SimpleNamespace(eos_token_id=0)
    trainer._trace_actor_start_ts = None
    trainer._trace_actor_done_ts = None
    calls, early, main = [], [], []
    trainer.on_sample_begin = lambda: calls.append("sample")
    trainer.timing_raw = {}

    def remote(fn):
        return SimpleNamespace(remote=fn)

    async def sleep():
        calls.append("sleep")
    async def wake():
        calls.append("wake")
    def evacuate():
        calls.append("migrate")
        if failure == "migrate":
            raise RuntimeError("migration failed")
        return {"aborted_count": 2}
    server = SimpleNamespace(migrate_requests=remote(evacuate),
                             resume_generation=remote(lambda: calls.append("resume")))
    replicas = [SimpleNamespace(servers=[server], sleep=sleep, wake_up=wake) for _ in range(3)]
    trainer.llm_server_manager = SimpleNamespace(
        rollout_replicas=replicas, server_addresses=["a", "b", "c"], server_handles=[1, 2, 3],
        global_load_balancer=SimpleNamespace(
            remove_servers=remote(lambda ids: calls.append("remove")),
            add_servers=remote(lambda servers: calls.append("add")),
        ),
    )
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda *a: calls.append("publish"))
    monkeypatch.setattr("verl.trainer.ppo.v1.trainer_sync.ray.get", lambda value: value)
    cleared = []
    monkeypatch.setattr("verl.trainer.ppo.v1.trainer_sync.tq.kv_clear", lambda **kw: cleared.extend(kw["keys"]))

    def pad(batch, divisor, eos_token_id):
        missing = (-len(batch)) % divisor
        return KVBatchMeta(partition_id="train", keys=batch.keys + [f"pad{i}" for i in range(missing)],
                           tags=batch.tags + [{"is_padding": True} for _ in range(missing)], extra_info=batch.extra_info)
    monkeypatch.setattr("verl.trainer.ppo.padding_utils.upsample_batch_to_divisible_size", pad)

    def batch():
        count = 6 if len(early) < 2 else 9
        tags = [{"student_gen_done_ts": time.time(), "teacher_done_ts": time.time(),
                 "min_global_steps": 0, "max_global_steps": 1 if failure == "version" else 0,
                 "migration_count": int(i > 6), "migrated_prefix_tokens": 2 if i > 6 else 0} for i in range(count)]
        return KVBatchMeta(partition_id="train", keys=[f"p{i}_0_0" for i in range(count)], tags=tags)
    class Replay:
        poll_interval = 2.
        def peek_actor_overlap_batch(self, *args):
            result = batch()
            return result, result.keys
        def materialize_actor_overlap_batch(self, *args):
            assert len(set(early + main)) == 9
            return batch()
    trainer.replay_buffer = Replay()

    class Group:
        def __init__(self, name, trained):
            self.name, self.trained = name, trained
        def begin_actor_accumulate(self, loss_normalization_tokens):
            assert loss_normalization_tokens == 9 * 32
            calls.append(f"{self.name}:begin")
            if failure == f"{self.name}:begin":
                raise RuntimeError("begin failed")
        def accumulate_actor(self, batch):
            calls.append(f"{self.name}:fb")
            if failure == f"{self.name}:fb":
                raise RuntimeError("fb failed")
            take = [k for k, t in zip(batch.keys, batch.tags, strict=True) if not t.get("is_padding")]
            assert not set(take).intersection(early + main)
            self.trained.extend(take)
            return {"metrics": {"chunk_fb_s": .01}}
        def migration_loss_state(self):
            assert len(early) == 2
            return [{"tokens": 7., "outputs": []}]
        def send_migration_parameters(self, units):
            calls.append("send_weights")
            return ["send_weights"]
        def recv_migration_parameters(self, units):
            calls.append("recv_weights")
            return ["recv_weights"]
        def materialize_migration_grads(self):
            calls.append("materialize")
        def send_migration_gradients(self, units):
            assert tuple(units) == ("root", "layer")
            calls.append("send_migration_gradients")
            return ["send"]
        def recv_migration_gradients(self, units):
            assert tuple(units) == ("root", "layer")
            calls.append("recv_migration_gradients")
            return ["recv"]
        def add_migration_loss_state(self, state):
            assert state["tokens"] == 7
            calls.append("add_tokens")
        def abort_actor_accumulate(self):
            calls.append(f"{self.name}:abort")
        def finish_actor_accumulate(self):
            calls.append("finish")
            assert len(early + main) == 9
            return {"metrics": {}}
    trainer.migration_actor_wg = Group("early", early)
    trainer.actor_rollout_wg = Group("main", main)
    return trainer, calls, early, main, cleared


def test_migration_orders_abort_early_fb_main_fb_merge_and_single_update(monkeypatch):
    trainer, calls, early, main, cleared = _wire(monkeypatch)
    metrics = {}
    trainer._step_once(metrics, {}, 9)
    assert calls.index("remove") < calls.index("migrate") < calls.index("sleep") < calls.index("send_weights")
    assert calls.index("send_weights") < calls.index("early:fb")
    assert calls.index("main:fb") < calls.index("recv_migration_gradients") < calls.index("early:abort")
    assert calls.index("early:abort") < calls.index("finish")
    assert "to" not in calls
    assert calls.count("finish") == calls.count("add_tokens") == 1
    assert len(early) == 2 and len(main) == 7 and cleared == ["pad0", "pad1"]
    assert metrics["actor_rollout_migrate/early_loss_tokens"] == 7
    assert trainer.replay_buffer.poll_interval == 2.
    trainer.on_step_end()
    assert calls.index("publish") < calls.index("resume") < calls.index("add")


@pytest.mark.parametrize("failure", ["migrate", "early:begin", "early:fb", "main:begin", "main:fb", "version"])
def test_migration_failure_aborts_groups_and_restores_evacuated_replica(monkeypatch, failure):
    trainer, calls, _, _, _ = _wire(monkeypatch, failure)
    with pytest.raises(RuntimeError):
        trainer._step_once({}, {}, 9)
    assert "finish" not in calls
    assert calls[-3:] == ["wake", "resume", "add"]
    if "early:begin" in calls:
        assert "early:abort" in calls
    if "main:begin" in calls:
        assert "main:abort" in calls
    assert trainer.replay_buffer.poll_interval == 2.


@pytest.mark.parametrize("dtype", ["fp16", "float16"])
def test_migration_rejects_grad_scaler_before_allocating_workers(dtype):
    from omegaconf import OmegaConf
    trainer = _trainer()
    OmegaConf.update(trainer.config, "trainer.v1.sync.actor_rollout_migrate", True, force_add=True)
    OmegaConf.update(trainer.config, "actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype", dtype, force_add=True)
    with pytest.raises(ValueError, match="BF16/FP32.*FP16 GradScaler"):
        trainer._setup()


def test_parameter_sync_stays_off_the_step_start_path(monkeypatch):
    trainer, _, _, _, _ = _wire(monkeypatch)
    called = []
    trainer._sync_migration_weights = lambda: called.append("sync")
    trainer._next_train_batch = lambda: {"uid": [1]}
    trainer._submit_batch_to_rollout = lambda batch: None
    trainer.prepare_step()
    assert called == []
