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

"""Exercise held training with real OPD loss, FSDP shards, clipping and AdamW."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, fully_shard
from torch.distributed.tensor import DTensor

from verl.trainer.distillation.losses import distillation_ppo_loss
from verl.trainer.ppo.padding_utils import construct_minimal_padding_template
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig, DistillationLossConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker


def _opd_loss():
    return partial(
        distillation_ppo_loss,
        ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1, loss_agg_mode="token-mean"),
        SimpleNamespace(
            enabled=True,
            distillation_loss=DistillationLossConfig(
                loss_mode="k1", topk=1, use_task_rewards=False, use_policy_gradient=True
            ),
        ),
    )


def _make_worker(model, *, use_scaler=False):
    engine = object.__new__(FSDPEngine)
    engine.module = model
    engine.engine_config = SimpleNamespace(
        use_dynamic_bsz=False,
        max_token_len_per_gpu=128,
        micro_batch_size_per_gpu=1,
        use_fused_kernels=False,
        use_no_sync_for_gradient_accumulation=False,
        fsdp_size=1,
        forward_only=False,
    )
    engine.optimizer_config = SimpleNamespace(clip_grad=0.015)
    engine.optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.1)
    engine.lr_scheduler = torch.optim.lr_scheduler.StepLR(engine.optimizer, step_size=1, gamma=0.9)
    engine.scaler = torch.amp.GradScaler("cpu", init_scale=16) if use_scaler else None
    engine._qat_enabled = False
    engine._is_offload_param = False
    engine._is_offload_optimizer = False
    engine.ulysses_device_mesh = None
    engine.ulysses_sequence_parallel_size = 1
    engine.ulysses_parallel_group = None

    def forward_step(data, loss_function, forward_only):
        assert not forward_only
        output = model(data["features"]).flatten()
        loss, metrics = loss_function(model_output={"log_probs": output}, data=data)
        return loss, {"loss": loss.detach().item(), "metrics": metrics}

    engine.forward_step = forward_step
    worker = object.__new__(TrainingWorker)
    worker.engine = engine
    worker.engine_config = engine.engine_config
    worker.model_config = {}
    worker.device_name = "cpu"
    worker.flops_counter = None
    worker.profiler = Mock()
    worker._held_train_ctx = None
    worker._held_output_lst = []
    worker._held_loss_normalization_tokens = None
    worker._held_loss_tokens = None
    worker.loss_fn = _opd_loss()
    return worker


def _data(rank):
    # Uneven token counts both across ranks and across the three held chunks.
    # Rank 1's last sequence is a padding duplicate, with no loss contribution.
    counts = [4, 2, 1, 3] if rank == 0 else [1, 3, 4, 0]
    response_mask = torch.arange(4).unsqueeze(0) < torch.tensor(counts).unsqueeze(1)
    teacher = -torch.arange(20, dtype=torch.float32).view(4, 5, 1) / 100 - 0.4
    data = TensorDict(
        {
            "features": (torch.arange(80, dtype=torch.float32).reshape(4, 5, 4) + rank) / 200,
            "prompts": torch.ones(4, 1, dtype=torch.long),
            "responses": torch.ones(4, 4, dtype=torch.long),
            "attention_mask": torch.ones(4, 5, dtype=torch.long),
            "response_mask": response_mask,
            "loss_mask": torch.cat([torch.zeros(4, 1, dtype=torch.bool), response_mask], dim=1),
            "teacher_logprobs": torch.nested.as_nested_tensor(list(teacher), layout=torch.jagged),
            "rollout_log_probs": torch.full((4, 4), -0.2),
        },
        batch_size=[4],
    )
    tu.assign_non_tensor(data, global_batch_size=8, opd_no_task_reward_fast_path=True)
    return data


def _model(strategy, mesh):
    torch.manual_seed(2026)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 1))
    if strategy == "fsdp":
        return FSDP(model, device_id=torch.device("cpu"), sharding_strategy=ShardingStrategy.FULL_SHARD)
    fully_shard(model, mesh=mesh)
    return model


def _parameters(model):
    if isinstance(model, FSDP):
        with FSDP.summon_full_params(model):
            return [parameter.detach().clone() for parameter in model.parameters()]
    return [parameter.full_tensor().detach().clone() for parameter in model.parameters()]


def _local_gradients(model):
    return [
        (parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad).detach().clone()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]


def _run_step(worker, data, *, delayed):
    engine = worker.engine
    preclip_gradients = []
    original_step = engine.optimizer_step

    def record_before_clip():
        preclip_gradients.extend(_local_gradients(engine.module))
        return original_step()

    with patch.object(engine, "optimizer_step", side_effect=record_before_clip) as optimizer_step:
        with patch.object(engine, "lr_scheduler_step", wraps=engine.lr_scheduler_step) as scheduler_step:
            worker.begin_held_train(loss_normalization_tokens=128 if delayed else None)
            if delayed:
                chunks = tuple(tu.index_select_tensor_dict(data, indices) for indices in ([0], [1, 2], [3]))
            else:
                # Sequential reference knows the real full-batch denominator.
                tu.assign_non_tensor(data, batch_num_tokens_override=18)
                chunks = (data,)
            for chunk in chunks:
                output = worker.accumulate_held_train(chunk)
                metrics = tu.get(output, "metrics")
                assert metrics["chunk_fb_s"] >= 0
                assert metrics["chunk_fb_start_ts"] <= metrics["chunk_fb_end_ts"]
            output = worker.finish_held_train()
            assert optimizer_step.call_count == scheduler_step.call_count == 1
    assert worker._held_train_ctx is None
    assert worker._held_loss_tokens is None
    assert all(parameter.grad is None for parameter in engine.module.parameters())
    assert all(state["step"].item() == 1 for state in engine.optimizer.state.values())
    return preclip_gradients, tu.get(output, "metrics"), _parameters(engine.module)


def _distributed_worker(rank, world_size, rendezvous_file):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous_file}", rank=rank, world_size=world_size)
    original_all_reduce = dist.all_reduce
    token_count_reductions = [0]

    def gloo_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, **kwargs):
        if tensor.dtype == torch.float64 and tensor.numel() == 1:
            token_count_reductions[0] += 1
        # Gloo lacks AVG; preserve the production postprocessor's DP averaging.
        if op == dist.ReduceOp.AVG:
            original_all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, **kwargs)
            tensor.div_(dist.get_world_size(group))
        else:
            original_all_reduce(tensor, op=op, group=group, **kwargs)

    memory = SimpleNamespace(max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0)
    try:
        with (
            patch("verl.workers.engine_workers.get_torch_device", return_value=memory),
            patch("verl.utils.fsdp_utils.get_device_id", return_value="cpu"),
        ):
            with patch("torch.distributed.all_reduce", side_effect=gloo_all_reduce):
                mesh = init_device_mesh("cpu", (world_size,))
                for strategy in ("fsdp", "fsdp2"):
                    for use_scaler in (False, True):
                        baseline = _make_worker(_model(strategy, mesh), use_scaler=use_scaler)
                        delayed = _make_worker(_model(strategy, mesh), use_scaler=use_scaler)
                        base_grads, base_metrics, base_params = _run_step(baseline, _data(rank), delayed=False)
                        previous_count = token_count_reductions[0]
                        new_grads, new_metrics, new_params = _run_step(delayed, _data(rank), delayed=True)
                        assert token_count_reductions[0] == previous_count + 1
                        for expected, actual in zip(base_grads, new_grads, strict=True):
                            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
                        for expected, actual in zip(base_params, new_params, strict=True):
                            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
                        assert new_metrics["grad_norm"] > delayed.engine.optimizer_config.clip_grad
                        assert new_metrics["grad_norm"] == pytest.approx(base_metrics["grad_norm"], rel=2e-6)
                        assert new_metrics["loss_token_count"] == 18
                        assert new_metrics["loss"] == pytest.approx(base_metrics["loss"][0], rel=2e-6)
                        assert new_metrics["distillation/loss"] == pytest.approx(
                            base_metrics["distillation/loss"][0], rel=2e-6
                        )
                        assert torch.isfinite(torch.tensor(new_metrics["distillation/abs_loss"])).all()

                # Aborting and globally empty batches must neither update nor
                # leave a normalization denominator/gradients for the next step.
                worker = _make_worker(_model("fsdp2", mesh))
                before = _parameters(worker.engine.module)
                worker.begin_held_train(loss_normalization_tokens=32)
                worker.accumulate_held_train(tu.index_select_tensor_dict(_data(rank), [0]))
                worker.abort_held_train()
                assert worker._held_train_ctx is None
                assert worker._held_loss_tokens is None
                assert worker._held_loss_normalization_tokens is None
                assert all(parameter.grad is None for parameter in worker.engine.module.parameters())
                assert not worker.engine.optimizer.state
                empty = tu.index_select_tensor_dict(_data(rank), [0])
                empty["response_mask"].zero_()
                empty["loss_mask"].zero_()
                worker.begin_held_train(loss_normalization_tokens=32)
                worker.accumulate_held_train(empty)
                with pytest.raises(ValueError, match="positive finite total loss token count"):
                    worker.finish_held_train()
                assert worker._held_train_ctx is None
                assert worker._held_loss_tokens is None
                assert not worker.engine.optimizer.state
                assert worker.engine.lr_scheduler.last_epoch == 0
                for expected, actual in zip(before, _parameters(worker.engine.module), strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                worker.abort_held_train()  # idempotent after cleanup
    finally:
        dist.destroy_process_group()


def test_delayed_normalization_matches_full_batch_opd_update(tmp_path):
    mp.spawn(_distributed_worker, args=(2, str(tmp_path / "held_train_rdzv")), nprocs=2, join=True)


@pytest.mark.parametrize("denominator", [0, -1, float("inf"), float("nan")])
def test_invalid_fixed_denominator_does_not_enter_train_mode(denominator):
    worker = object.__new__(TrainingWorker)
    worker._held_train_ctx = None
    worker.engine = Mock()
    with pytest.raises(ValueError, match="finite and positive"):
        worker.begin_held_train(loss_normalization_tokens=denominator)
    worker.engine.train_mode.assert_not_called()


def test_actor_api_forwards_optional_denominator():
    worker = SimpleNamespace(actor=Mock())
    ActorRolloutRefWorker.begin_actor_accumulate(worker)
    worker.actor.begin_held_train.assert_called_once_with(loss_normalization_tokens=None)
    ActorRolloutRefWorker.begin_actor_accumulate(worker, loss_normalization_tokens=123)
    worker.actor.begin_held_train.assert_called_with(loss_normalization_tokens=123)


def test_real_synthetic_padding_has_zero_opd_loss_and_gradient():
    source_batch = _data(0)
    names = ("prompts", "responses", "response_mask", "loss_mask", "rollout_log_probs", "teacher_logprobs")
    source = {name: source_batch[name][0] for name in names}
    padding, tag = construct_minimal_padding_template(source, {"seq_len": 5}, eos_token_id=0)
    fields = {name: padding[name].unsqueeze(0) for name in names if name != "teacher_logprobs"}
    fields["attention_mask"] = padding["attention_mask"].unsqueeze(0)
    fields["teacher_logprobs"] = torch.nested.as_nested_tensor([padding["teacher_logprobs"]], layout=torch.jagged)
    data = TensorDict(fields, batch_size=[1])
    tu.assign_non_tensor(data, global_batch_size=8, batch_num_tokens=18, dp_size=2, opd_no_task_reward_fast_path=True)
    student_log_probs = torch.linspace(-0.4, -0.1, tag["seq_len"], requires_grad=True)
    loss, metrics = _opd_loss()(model_output={"log_probs": student_log_probs}, data=data)
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(student_log_probs.grad, torch.zeros_like(student_log_probs), rtol=0, atol=0)
    assert metrics["distillation/abs_loss"].aggregate() == 0
    assert metrics["distillation/loss_min"].aggregate() == float("inf")
    assert metrics["distillation/loss_max"].aggregate() == float("-inf")


def _worker_for_begin_failure():
    worker = object.__new__(TrainingWorker)
    worker._held_train_ctx = None
    worker._held_loss_normalization_tokens = None
    worker._held_loss_tokens = None
    worker.device_name = "cpu"
    worker.engine = Mock()
    worker.engine.is_mp_src_rank_with_outputs.return_value = False
    worker.engine.train_mode.return_value = MagicMock()
    return worker


def test_count_allocation_failure_does_not_enter_train_mode():
    worker = _worker_for_begin_failure()
    with patch("torch.zeros", side_effect=RuntimeError("allocation failed")):
        with pytest.raises(RuntimeError, match="allocation failed"):
            worker.begin_held_train(loss_normalization_tokens=32)
    worker.engine.train_mode.assert_not_called()
    assert worker._held_train_ctx is None


def test_enter_failure_does_not_exit_unentered_context():
    worker = _worker_for_begin_failure()
    ctx = worker.engine.train_mode.return_value
    ctx.__enter__.side_effect = RuntimeError("enter failed")
    with pytest.raises(RuntimeError, match="enter failed"):
        worker.begin_held_train(loss_normalization_tokens=32)
    ctx.__exit__.assert_not_called()
    assert worker._held_train_ctx is None
    assert worker._held_loss_normalization_tokens is None
    assert worker._held_loss_tokens is None


def test_zero_grad_failure_closes_entered_context_and_clears_state():
    worker = _worker_for_begin_failure()
    worker.engine.optimizer_zero_grad.side_effect = RuntimeError("zero grad failed")
    with pytest.raises(RuntimeError, match="zero grad failed"):
        worker.begin_held_train(loss_normalization_tokens=32)
    worker.engine.train_mode.return_value.__exit__.assert_called_once()
    worker.engine.optimizer_step.assert_not_called()
    assert worker._held_train_ctx is None
    assert worker._held_loss_normalization_tokens is None
    assert worker._held_loss_tokens is None
