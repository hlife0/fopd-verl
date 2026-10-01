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

"""Two-GPU, unbounded-MPS integration test of the production Green stream path."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_held_train_normalization_on_cpu import _data, _local_gradients, _make_worker, _parameters, _run_step
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import fully_shard

from verl.utils import tensordict_utils as tu
from verl.utils.overlap_actor import OverlapActorControl, stream_sm_count


def _model(strategy, mesh, rank):
    torch.manual_seed(2026)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 1)).cuda(rank)
    if strategy == "fsdp":
        return FSDP(model, device_id=rank)
    fully_shard(model, mesh=mesh)
    return model


def _gpu_worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cuda", (2,))
        for strategy in ("fsdp", "fsdp2"):
            for defer_sync in (False, True):
                _run_strategy(strategy, rank, mesh, defer_sync)
    finally:
        dist.destroy_process_group()


def _run_strategy(strategy, rank, mesh, defer_sync):
    baseline = _make_worker(_model(strategy, mesh, rank))
    baseline.device_name = "cuda"
    baseline.engine.engine_config.use_no_sync_for_gradient_accumulation = defer_sync
    reference = _run_step(baseline, _data(rank).cuda(), delayed=False)
    worker = _make_worker(_model(strategy, mesh, rank))
    worker.device_name = "cuda"
    engine = worker.engine
    engine.engine_config.use_no_sync_for_gradient_accumulation = defer_sync
    engine.configure_overlap_actor(0.5)
    execution = engine.overlap_actor_execution
    if strategy == "fsdp":
        streams = [
            getattr(engine.module, name) for name in ("_default_stream", "_unshard_stream", "_post_backward_stream")
        ]
    else:
        comm = engine.module._get_fsdp_state()._comm_ctx
        streams = [comm.all_gather_stream, comm.reduce_scatter_stream, comm.all_reduce_stream]
    assert all(stream_sm_count(stream) == execution.total_sms for stream in streams)
    control = OverlapActorControl()
    control.begin(1)
    handle = SimpleNamespace(get=SimpleNamespace(remote=control.get))
    execution.begin(handle, 1)
    observed = []
    backward_streams = []
    original_forward = engine.forward_step

    def forward(data, loss_function, forward_only):
        observed.append(stream_sm_count(torch.cuda.current_stream()))
        result = original_forward(data, loss_function, forward_only)
        result[0].register_hook(lambda grad: backward_streams.append(stream_sm_count(torch.cuda.current_stream())))
        # Signal while the first microbatch is executing, inside one
        # four-microbatch chunk. All ranks must change on the next MB.
        if rank == 0 and len(observed) == 1:
            control.full(1)
        return result

    grads = []
    original_step = engine.optimizer_step

    def optimizer_step():
        assert stream_sm_count(torch.cuda.current_stream()) == execution.total_sms
        grads.extend(_local_gradients(engine.module))
        return original_step()

    with (
        patch("verl.utils.overlap_actor.ray.get", side_effect=lambda value: value),
        patch.object(engine, "forward_step", side_effect=forward),
        patch.object(engine, "optimizer_step", side_effect=optimizer_step) as update,
    ):
        worker.begin_held_train(loss_normalization_tokens=128)
        worker.accumulate_held_train(_data(rank).cuda())
        execution.finish()
        output = worker.finish_held_train()
        assert update.call_count == 1
        # New step must forget the previous full phase; abort clears grads.
        control.begin(2)
        execution.begin(handle, 2)
        worker.begin_held_train(loss_normalization_tokens=128)
        worker.accumulate_held_train(tu.index_select_tensor_dict(_data(rank).cuda(), [0]))
        worker.abort_held_train()
        assert all(p.grad is None for p in engine.module.parameters())
        control.begin(3)
        execution.begin(handle, 3)
        worker.begin_held_train(loss_normalization_tokens=128)

        def cancel_during_forward(data, loss_function, forward_only):
            result = original_forward(data, loss_function, forward_only)
            if rank == 0:
                control.cancel(3)
            return result

        with patch.object(engine, "forward_step", side_effect=cancel_during_forward):
            with pytest.raises(RuntimeError, match="cancelled"):
                worker.accumulate_held_train(_data(rank).cuda())
        worker.abort_held_train()
        assert update.call_count == 1
        assert all(p.grad is None for p in engine.module.parameters())
    assert observed[:4] == [execution.limited_sms] + [execution.total_sms] * 3
    assert observed[4] == execution.limited_sms
    assert backward_streams == observed
    for expected, actual in zip(reference[0], grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
    for expected, actual in zip(reference[2], _parameters(engine.module), strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
    assert tu.get(output, "metrics")["grad_norm"] == pytest.approx(reference[1]["grad_norm"], rel=2e-5)
    print(f"{strategy} defer_sync={defer_sync} rank={rank}: SMs {observed}; gradient/update match", flush=True)


@pytest.mark.skipif(not os.environ.get("CUDA_MPS_PIPE_DIRECTORY"), reason="requires two idle GPUs under MPS")
def test_fsdp_green_to_full_within_one_chunk(tmp_path):
    if torch.cuda.device_count() < 2 or not hasattr(torch.cuda, "GreenContext"):
        pytest.skip("requires two GPUs and native GreenContext")
    mp.spawn(_gpu_worker, args=(str(tmp_path / "overlap_gpu_rdzv"),), nprocs=2, join=True)
