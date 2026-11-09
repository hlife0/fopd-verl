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

import os

import pytest
import torch
import torch.multiprocessing as mp

from verl.workers.engine.fsdp.swap_transfer import (
    TransferGroup,
    _overlaps,
    free_port,
    matching_numels,
    pull_main_params,
    pull_peer_grads,
    push_main_params,
    push_peer_grads,
    shard_range,
)

# FSDP units of very different sizes, including fewer elements than ranks.
NUMELS = [97, 1, 6, 4096, 13]


def _shard(full: torch.Tensor, world: int, rank: int) -> torch.Tensor:
    """What an FSDP1 rank holds: its torch.chunk piece, zero-padded to the first chunk's size."""
    chunk = -(-full.numel() // world)
    lo, hi = shard_range(full.numel(), world, rank)
    out = torch.zeros(chunk, dtype=full.dtype)
    out[: hi - lo] = full[lo:hi]
    return out


@pytest.mark.parametrize("numel", [1, 5, 6, 7, 97, 4096])
@pytest.mark.parametrize("main_world,peer_world", [(1, 1), (3, 1), (6, 2), (2, 3), (4, 4)])
def test_overlaps_cover_every_element_once(numel, main_world, peer_world):
    covered = torch.zeros(numel, dtype=torch.int64)
    for peer, main, lo, hi in _overlaps(numel, main_world, peer_world):
        assert shard_range(numel, peer_world, peer)[0] <= lo < hi <= shard_range(numel, peer_world, peer)[1]
        assert shard_range(numel, main_world, main)[0] <= lo < hi <= shard_range(numel, main_world, main)[1]
        covered[lo:hi] += 1
    assert torch.equal(covered, torch.ones_like(covered))


def test_shard_range_matches_torch_chunk():
    for numel in (1, 5, 6, 7, 97):
        for world in (1, 2, 3, 6, 8):
            chunks = torch.arange(numel).chunk(world)
            for rank in range(world):
                lo, hi = shard_range(numel, world, rank)
                expected = chunks[rank].tolist() if rank < len(chunks) else []
                assert list(range(lo, hi)) == expected


def test_matching_numels_allows_only_tail_padding():
    assert matching_numels([(10, "f"), (8, "f")], [(11, "f"), (8, "f")], 2) == [10, 8]
    with pytest.raises(ValueError):
        matching_numels([(10, "f")], [(13, "f")], 2)
    with pytest.raises(ValueError):
        matching_numels([(10, "f")], [(10, "g")], 2)
    with pytest.raises(ValueError):
        matching_numels([(10, "f")], [], 2)


def _worker(rank, main_world, peer_world, port, out_dir):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1234)
    # Every process derives the same full tensors.
    main_grads = [torch.randn(n, generator=gen, dtype=torch.float64) for n in NUMELS]
    peer_grads = [torch.randn(n, generator=gen, dtype=torch.float64) for n in NUMELS]
    new_params = [torch.randn(n, generator=gen, dtype=torch.float64) for n in NUMELS]
    group = TransferGroup.create("127.0.0.1", port, rank, main_world, peer_world, timeout_s=120)
    me = group.local_rank
    if group.is_main:
        params = []
        for grad, value in zip(main_grads, new_params, strict=True):
            p = torch.nn.Parameter(_shard(value, main_world, me))
            p.grad = _shard(grad, main_world, me)
            params.append(p)
        tokens = torch.tensor(5.0 + me, dtype=torch.float64)
        pull_peer_grads(params, group, NUMELS, tokens)
        push_main_params(params, group, NUMELS)
        torch.save({"grads": [p.grad for p in params], "tokens": tokens}, os.path.join(out_dir, f"main{me}.pt"))
    else:
        params = []
        for grad in peer_grads:
            p = torch.nn.Parameter(torch.zeros(-(-grad.numel() // peer_world), dtype=torch.float64))
            p.grad = _shard(grad, peer_world, me)
            params.append(p)
        push_peer_grads(params, group, NUMELS, torch.tensor(100.0 * (me + 1), dtype=torch.float64))
        pull_main_params(params, group, NUMELS)
        torch.save({"params": [p.data for p in params]}, os.path.join(out_dir, f"peer{me}.pt"))
    group.close()


@pytest.mark.parametrize("main_world,peer_world", [(1, 1), (3, 1), (6, 2), (2, 3)])
def test_transfer_merges_grads_and_copies_params(tmp_path, main_world, peer_world):
    port = free_port()
    mp.start_processes(
        _worker,
        args=(main_world, peer_world, port, str(tmp_path)),
        nprocs=main_world + peer_world,
        join=True,
        start_method="spawn",
    )
    gen = torch.Generator().manual_seed(1234)
    main_grads = [torch.randn(n, generator=gen, dtype=torch.float64) for n in NUMELS]
    peer_grads = [torch.randn(n, generator=gen, dtype=torch.float64) for n in NUMELS]
    new_params = [torch.randn(n, generator=gen, dtype=torch.float64) for n in NUMELS]
    peer_tokens = sum(100.0 * (p + 1) for p in range(peer_world))
    for me in range(main_world):
        got = torch.load(tmp_path / f"main{me}.pt")
        for unit, numel in enumerate(NUMELS):
            expected = _shard(main_grads[unit] + peer_grads[unit], main_world, me)
            assert torch.equal(got["grads"][unit], expected), (me, unit)
        assert got["tokens"].item() == 5.0 + me + (peer_tokens if me == 0 else 0.0)
    for me in range(peer_world):
        got = torch.load(tmp_path / f"peer{me}.pt")
        for unit in range(len(NUMELS)):
            assert torch.equal(got["params"][unit], _shard(new_params[unit], peer_world, me)), (me, unit)


def _fsdp_worker(rank, main_world, peer_world, ports, out_dir):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    from verl.workers.engine.fsdp.swap_transfer import flat_params, unit_layout

    is_main = rank < main_world
    local, world = (rank, main_world) if is_main else (rank - main_world, peer_world)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(ports[0] if is_main else ports[1]))
    dist.init_process_group("gloo", rank=local, world_size=world)
    model = _fsdp_model()
    fsdp = FSDP(model, device_id=torch.device("cpu"), auto_wrap_policy=ModuleWrapPolicy({torch.nn.Linear}))
    params = flat_params(fsdp)
    layout = unit_layout(params)
    group = TransferGroup.create("127.0.0.1", ports[2], rank, main_world, peer_world, timeout_s=120)

    inputs = _fsdp_inputs()
    samples = inputs[:_MAIN_SAMPLES] if is_main else inputs[_MAIN_SAMPLES:]
    mine = samples.chunk(world)[local]
    # verl's delayed normalization: local sum / C times the DP size, which FSDP averages away.
    loss = fsdp(mine).pow(2).sum() / inputs.shape[0] * world
    loss.backward()

    numels = [n for n, _ in layout]
    if is_main:
        pull_peer_grads(params, group, numels, None)
        torch.optim.SGD(fsdp.parameters(), lr=0.5).step()
        push_main_params(params, group, numels)
    else:
        push_peer_grads(params, group, numels, None)
        pull_main_params(params, group, numels)
    with FSDP.state_dict_type(fsdp, StateDictType.FULL_STATE_DICT, FullStateDictConfig(rank0_only=False)):
        state = {k: v.clone() for k, v in fsdp.state_dict().items()}
    if local == 0:
        torch.save(state, os.path.join(out_dir, "main.pt" if is_main else "peer.pt"))
    group.close()
    dist.destroy_process_group()


_MAIN_SAMPLES = 10


def _fsdp_model():
    torch.manual_seed(7)
    return torch.nn.Sequential(
        torch.nn.Linear(7, 11), torch.nn.Tanh(), torch.nn.Linear(11, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3)
    ).double()


def _fsdp_inputs():
    return torch.randn(16, 7, generator=torch.Generator().manual_seed(3), dtype=torch.float64)


@pytest.mark.parametrize("main_world,peer_world", [(1, 1), (3, 1), (2, 2), (5, 3)])
def test_fsdp1_groups_train_like_one_model(tmp_path, main_world, peer_world):
    ports = (free_port(), free_port(), free_port())
    mp.start_processes(
        _fsdp_worker,
        args=(main_world, peer_world, ports, str(tmp_path)),
        nprocs=main_world + peer_world,
        join=True,
        start_method="spawn",
    )
    reference = _fsdp_model()
    inputs = _fsdp_inputs()
    (reference(inputs).pow(2).sum() / inputs.shape[0]).backward()
    torch.optim.SGD(reference.parameters(), lr=0.5).step()
    main = torch.load(tmp_path / "main.pt")
    peer = torch.load(tmp_path / "peer.pt")
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(main[name], value, rtol=1e-12, atol=1e-12)
        assert torch.equal(peer[name], main[name]), name
