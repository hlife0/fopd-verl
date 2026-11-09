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

"""Move FSDP1 gradients and parameters between two independent FSDP groups of one model.

Swap placement trains one model in two FSDP groups with separate worlds: the main group
(on the Student GPUs, owns the optimizer) and a peer group (on the former Teacher GPUs,
forward/backward only). FSDP1 shards each FlatParameter with ``torch.chunk`` over its own
world size, so rank r of a W-rank group holds logical elements
``[r * c, min((r + 1) * c, N))`` of a unit's N-element flat vector, ``c = ceil(N / W)``,
followed by zero padding. Both groups wrap the model the same way, so units and their
layouts match and only ``c`` differs. A transfer pairs each peer shard with the main shards
it overlaps and moves only the overlapping slice, point to point over one communicator
spanning both groups: main ranks ``[0, M)``, peer ranks ``[M, M + P)``.

Every rank issues its sends/receives in the global order (unit, peer rank, main rank), so
the pairwise point-to-point operations cannot wait on each other in a cycle.
"""

import datetime
import socket
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._traversal_utils import _get_fsdp_handles

from verl.utils.device import is_cuda_available


def shard_range(numel: int, world_size: int, rank: int) -> tuple[int, int]:
    """Logical [lo, hi) of a unit held by `rank` (empty when torch.chunk yields fewer chunks)."""
    chunk = -(-numel // world_size)
    return min(rank * chunk, numel), min((rank + 1) * chunk, numel)


def flat_params(module: torch.nn.Module) -> list:
    if not isinstance(module, FSDP):
        raise NotImplementedError("swap placement transfers support FSDP1 (actor strategy 'fsdp') only")
    return [handle.flat_param for handle in _get_fsdp_handles(module)]


def unit_layout(params: list) -> list[tuple[int, str]]:
    """(N, dtype) per FlatParameter, in traversal order."""
    return [(int(p._unpadded_unsharded_size.numel()), str(p.dtype)) for p in params]


def matching_numels(main_layout: list, peer_layout: list, max_world_size: int) -> list[int]:
    """Common logical length per unit; differences may only be world-size tail padding."""
    if len(main_layout) != len(peer_layout):
        raise ValueError(f"FSDP unit count differs: main {len(main_layout)} vs peer {len(peer_layout)}")
    numels = []
    for i, ((n_main, d_main), (n_peer, d_peer)) in enumerate(zip(main_layout, peer_layout, strict=True)):
        if d_main != d_peer or abs(n_main - n_peer) >= max_world_size:
            raise ValueError(f"FSDP unit {i} differs: main ({n_main}, {d_main}) vs peer ({n_peer}, {d_peer})")
        numels.append(min(n_main, n_peer))
    return numels


def _overlaps(numel: int, main_world: int, peer_world: int):
    """(peer_rank, main_rank, lo, hi) for every non-empty overlap, in (peer, main) order."""
    for peer in range(peer_world):
        p_lo, p_hi = shard_range(numel, peer_world, peer)
        for main in range(main_world):
            m_lo, m_hi = shard_range(numel, main_world, main)
            lo, hi = max(p_lo, m_lo), min(p_hi, m_hi)
            if lo < hi:
                yield peer, main, lo, hi


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


@dataclass
class TransferGroup:
    """A process group across the main and peer FSDP groups, outside both default worlds."""

    rank: int
    main_world: int
    peer_world: int
    pg: object
    store: object

    @classmethod
    def create(cls, host: str, port: int, rank: int, main_world: int, peer_world: int, timeout_s: int = 600):
        # Every legitimate wait is one optimizer step or one transfer; a failed peer surfaces quickly.
        world = main_world + peer_world
        timeout = datetime.timedelta(seconds=timeout_s)
        store = dist.TCPStore(host, port, world, is_master=(rank == 0), timeout=timeout)
        prefix = dist.PrefixStore("swap_transfer", store)
        if is_cuda_available:
            options = dist.ProcessGroupNCCL.Options()
            options._timeout = timeout
            pg = dist.ProcessGroupNCCL(prefix, rank, world, options)
        else:
            pg = dist.ProcessGroupGloo(prefix, rank, world, timeout)
        return cls(rank=rank, main_world=main_world, peer_world=peer_world, pg=pg, store=store)

    @property
    def is_main(self) -> bool:
        return self.rank < self.main_world

    @property
    def local_rank(self) -> int:
        return self.rank if self.is_main else self.rank - self.main_world

    def send(self, tensor: torch.Tensor, dst: int):
        self.pg.send([tensor], dst, 0).wait()

    def recv(self, tensor: torch.Tensor, src: int):
        self.pg.recv([tensor], src, 0).wait()

    def close(self):
        shutdown = getattr(self.pg, "shutdown", None)
        if callable(shutdown):
            shutdown()


def _local(tensor: torch.Tensor, lo: int, hi: int, base: int) -> torch.Tensor:
    return tensor.view(-1)[lo - base : hi - base]


def push_peer_grads(params: list, group: TransferGroup, numels: list[int], loss_tokens: torch.Tensor | None) -> None:
    """Peer side: send this rank's gradient slices (and loss-token count) to the main ranks."""
    assert not group.is_main
    me = group.local_rank
    for param, numel in zip(params, numels, strict=True):
        base = shard_range(numel, group.peer_world, me)[0]
        grad = param.grad if param.grad is not None else torch.zeros_like(param)
        for peer, main, lo, hi in _overlaps(numel, group.main_world, group.peer_world):
            if peer == me:
                group.send(_local(grad, lo, hi, base).contiguous(), main)
    if loss_tokens is not None:
        group.send(loss_tokens.reshape(1).to(torch.float64), 0)


@torch.no_grad()
def pull_peer_grads(params: list, group: TransferGroup, numels: list[int], loss_tokens: torch.Tensor | None) -> None:
    """Main side: add the peer gradient slices into this rank's shards; rank 0 adds peer loss tokens."""
    assert group.is_main
    me = group.local_rank
    for param, numel in zip(params, numels, strict=True):
        base = shard_range(numel, group.main_world, me)[0]
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        for peer, main, lo, hi in _overlaps(numel, group.main_world, group.peer_world):
            if main == me:
                buf = torch.empty(hi - lo, dtype=param.grad.dtype, device=param.grad.device)
                group.recv(buf, group.main_world + peer)
                _local(param.grad, lo, hi, base).add_(buf)
    if loss_tokens is not None and me == 0:
        buf = torch.zeros(1, dtype=torch.float64, device=loss_tokens.device)
        for peer in range(group.peer_world):
            group.recv(buf, group.main_world + peer)
            loss_tokens.add_(buf[0].to(loss_tokens.dtype))


@torch.no_grad()
def push_main_params(params: list, group: TransferGroup, numels: list[int]) -> None:
    """Main side: send this rank's parameter slices to the peer ranks."""
    assert group.is_main
    me = group.local_rank
    for param, numel in zip(params, numels, strict=True):
        base = shard_range(numel, group.main_world, me)[0]
        for peer, main, lo, hi in _overlaps(numel, group.main_world, group.peer_world):
            if main == me:
                group.send(_local(param.data, lo, hi, base).contiguous(), group.main_world + peer)


@torch.no_grad()
def pull_main_params(params: list, group: TransferGroup, numels: list[int]) -> None:
    """Peer side: overwrite this rank's parameter shards with the main group's values."""
    assert not group.is_main
    me = group.local_rank
    for param, numel in zip(params, numels, strict=True):
        base = shard_range(numel, group.peer_world, me)[0]
        for peer, main, lo, hi in _overlaps(numel, group.main_world, group.peer_world):
            if peer == me:
                buf = torch.empty(hi - lo, dtype=param.dtype, device=param.device)
                group.recv(buf, main)
                _local(param.data, lo, hi, base).copy_(buf)
