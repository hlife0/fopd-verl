# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Compare separate early/main FSDP groups with one full-batch AdamW update."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, fully_shard

from test_held_train_normalization_on_cpu import _data, _make_worker, _local_gradients, _parameters
from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.migration import export_unit, import_unit, unit_names
from verl.workers.engine_workers import ActorRolloutRefWorker


class _NestedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(4, 8)
        self.second = torch.nn.Linear(8, 1)
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.second(torch.nn.functional.gelu(self.first(x))) * self.scale


def _model(strategy, mesh):
    torch.manual_seed(2026)
    model = _NestedModel()
    if strategy == "fsdp":
        kwargs = dict(device_id=torch.device("cpu"), use_orig_params=True,
                      sharding_strategy=ShardingStrategy.FULL_SHARD)
        model.first = FSDP(model.first, **kwargs)
        model.second = FSDP(model.second, **kwargs)
        return FSDP(model, **kwargs)
    fully_shard(model.first, mesh=mesh)
    fully_shard(model.second, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def _batch(indices):
    # Reuse real OPD fields from the held-training test with variable loss masks.
    rows = [tu.index_select_tensor_dict(_data(i % 2), [i % 4]) for i in indices]
    batch = tu.concat_tensordict(rows)
    tu.assign_non_tensor(batch, global_batch_size=24, opd_no_task_reward_fast_path=True)
    return batch


def _worker(rank, world, rendezvous, directory, stage):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world)
    original_reduce = dist.all_reduce
    def reduce(tensor, op=dist.ReduceOp.SUM, group=None, **kwargs):
        if op == dist.ReduceOp.AVG:
            original_reduce(tensor, op=dist.ReduceOp.SUM, group=group, **kwargs)
            tensor.div_(dist.get_world_size(group))
        else:
            original_reduce(tensor, op=op, group=group, **kwargs)
    memory = SimpleNamespace(max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0)
    try:
        with patch("torch.distributed.all_reduce", side_effect=reduce), \
             patch("verl.workers.engine_workers.get_torch_device", return_value=memory), \
             patch("verl.utils.fsdp_utils.get_device_id", return_value="cpu"):
            mesh = init_device_mesh("cpu", (world,))
            for strategy in ("fsdp", "fsdp2"):
                worker = _make_worker(_model(strategy, mesh))
                if stage == "early":
                    source = _model(strategy, mesh)
                    assert len(unit_names(source)) >= 3
                    with torch.no_grad():
                        for parameter in worker.engine.module.parameters():
                            parameter.zero_()
                    for unit in unit_names(source):
                        source_values = [export_unit(source, unit)]
                        dist.broadcast_object_list(source_values, src=0)
                        import_unit(worker.engine.module, unit, source_values[0])
                        actual_values = export_unit(worker.engine.module, unit)
                        if rank == 0:
                            assert actual_values.keys() == source_values[0].keys()
                            for name in actual_values:
                                torch.testing.assert_close(actual_values[name], source_values[0][name])
                    worker.engine.optimizer = None
                    worker.engine.optimizer_config = None
                    worker.engine.lr_scheduler = None
                    worker.begin_held_train(loss_normalization_tokens=128)
                    worker.accumulate_held_train(_batch(list(range(rank, 6, world))))
                    tokens = worker._held_loss_tokens.clone()
                    dist.all_reduce(tokens)
                    values = {unit: export_unit(worker.engine.module, unit, gradients=True)
                              for unit in unit_names(worker.engine.module)}
                    if rank == 0:
                        torch.save({"values": values, "tokens": float(tokens), "outputs": worker._held_output_lst}, Path(directory) / f"{strategy}.pt")
                    worker.abort_held_train()
                    assert worker.engine.optimizer is None
                else:
                    baseline = _make_worker(_model(strategy, mesh))
                    def run(w, indices, imported=None):
                        before = []
                        step = w.engine.optimizer_step
                        def record():
                            before.extend(_local_gradients(w.engine.module))
                            return step()
                        w.begin_held_train(loss_normalization_tokens=128)
                        w.accumulate_held_train(_batch(indices))
                        if imported:
                            for unit, values in imported["values"].items():
                                import_unit(w.engine.module, unit, values, gradients=True)
                            outer = SimpleNamespace(actor=w)
                            ActorRolloutRefWorker.add_migration_loss_state.__wrapped__(outer, imported)
                        with patch.object(w.engine, "optimizer_step", side_effect=record) as step_mock:
                            output = w.finish_held_train()
                            assert step_mock.call_count == 1
                        return before, tu.get(output, "metrics"), _parameters(w.engine.module)
                    expected = run(baseline, list(range(rank, 24, world)))
                    saved = torch.load(Path(directory) / f"{strategy}.pt", weights_only=False)
                    actual = run(worker, list(range(6 + rank, 24, world)), saved)
                    for expected_grads, actual_grads in zip(expected[0], actual[0], strict=True):
                        torch.testing.assert_close(actual_grads, expected_grads, rtol=3e-6, atol=1e-7)
                    assert actual[1]["loss_token_count"] == expected[1]["loss_token_count"]
                    assert actual[1]["loss"] == pytest.approx(expected[1]["loss"], rel=3e-6)
                    assert actual[1]["distillation/loss"] == pytest.approx(expected[1]["distillation/loss"], rel=3e-6)
                    assert actual[1]["grad_norm"] == pytest.approx(expected[1]["grad_norm"], rel=3e-6)
                    for key in ("distillation/loss_min", "distillation/loss_max"):
                        assert actual[1][key] == pytest.approx(expected[1][key], rel=3e-6)
                    for a, b in zip(actual[2], expected[2], strict=True):
                        torch.testing.assert_close(a, b, rtol=3e-6, atol=1e-7)
                    assert all(state["step"].item() == 1 for state in worker.engine.optimizer.state.values())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("early,main", [(1, 3), (2, 6)])
def test_migration_gradient_merge_matches_full_batch(tmp_path, early, main):
    mp.spawn(_worker, args=(early, str(tmp_path / "early_rendezvous"), str(tmp_path), "early"),
             nprocs=early, join=True)
    mp.spawn(_worker, args=(main, str(tmp_path / "main_rendezvous"), str(tmp_path), "main"),
             nprocs=main, join=True)


def test_migration_fb_only_model_does_not_construct_an_optimizer():
    from unittest.mock import Mock
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
    engine = object.__new__(FSDPEngine)
    engine.engine_config = SimpleNamespace(forward_only=False)
    engine.optimizer_config = None
    engine.model_config = SimpleNamespace(lora_rank=0)
    engine._is_lora = engine._qat_enabled = False
    engine.rank = 1
    module = torch.nn.Linear(3, 2)
    engine._build_module = lambda: module
    engine._build_fsdp_module = lambda value: value
    engine._build_optimizer = Mock(side_effect=AssertionError("Early worker must never construct AdamW"))
    engine._build_lr_scheduler = Mock(side_effect=AssertionError("Early worker must never construct a scheduler"))
    with patch("torch.distributed.barrier"), patch("verl.workers.engine.fsdp.transformer_impl.log_gpu_memory_usage"):
        engine._build_model_optimizer()
    assert engine.optimizer is None and engine.lr_scheduler is None
    assert all(parameter.requires_grad for parameter in engine.module.parameters())
    engine._build_optimizer.assert_not_called()
