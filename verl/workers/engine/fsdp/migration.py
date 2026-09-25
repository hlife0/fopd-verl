# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Bounded parameter/gradient exchange between the route-2 FSDP groups."""

from contextlib import contextmanager

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor, distribute_tensor


def _clean(name):
    return name.replace("_fsdp_wrapped_module.", "")


def unit_names(module):
    if isinstance(module, FSDP):
        return [name for name, child in module.named_modules() if isinstance(child, FSDP) and child._handle is not None]
    # One original parameter per transfer keeps FSDP2's peak bounded as well.
    return [name for name, _ in module.named_parameters()]


def _should_stage(storage):
    """NCCL cannot all-gather a CPU shard. Stage that one unit, never the whole model."""
    if storage.device.type != "cpu" or not torch.cuda.is_available():
        return False
    return not (torch.distributed.is_initialized() and torch.distributed.get_backend() == "gloo")


@contextmanager
def _stage_fsdp_handle(child):
    handle = child._handle
    flat = handle.flat_param
    if not _should_stage(flat):
        yield
        return
    device = torch.device("cuda", torch.cuda.current_device())
    grad = flat.grad
    handle.flat_param_to(device, non_blocking=False)
    flat._local_shard = flat.data
    if grad is not None:
        flat.grad = grad.to(device, non_blocking=False)
    try:
        yield
    finally:
        grad = flat.grad
        handle.flat_param_to(torch.device("cpu"), non_blocking=False)
        flat._local_shard = flat.data
        if grad is not None:
            flat.grad = grad.detach().to("cpu")
        torch.cuda.empty_cache()


def _parent_parameter(module, name):
    parent = module
    for part in name.split(".")[:-1]:
        parent = getattr(parent, part)
    return parent, name.rsplit(".", 1)[-1]


@contextmanager
def _stage_parameter(module, unit):
    parameter = module.get_parameter(unit)
    if not _should_stage(parameter):
        yield
        return
    parent, leaf = _parent_parameter(module, unit)
    device = torch.device("cuda", torch.cuda.current_device())
    parent._parameters[leaf] = parameter.to(device)
    try:
        yield
    finally:
        parent._parameters[leaf] = parent._parameters[leaf].to("cpu")
        torch.cuda.empty_cache()


@contextmanager
def _parameters(module, unit, *, gradients, writeback):
    if isinstance(module, FSDP):
        child = module.get_submodule(unit) if unit else module
        if not child._use_orig_params:
            raise ValueError("route-2 gradient transfer requires FSDP use_orig_params=true")
        owned = set(child._handle.flat_param._fqns)
        with _stage_fsdp_handle(child), FSDP.summon_full_params(
            child, recurse=False, writeback=writeback, with_grads=gradients
        ):
            yield {
                _clean(name): parameter for name, parameter in child.named_parameters()
                if _clean(name) in owned
            }
    else:
        with _stage_parameter(module, unit):
            yield {unit: module.get_parameter(unit)}


@torch.no_grad()
def export_unit(module, unit, *, gradients=False):
    result = {}
    with _parameters(module, unit, gradients=gradients, writeback=False) as parameters:
        for name, parameter in parameters.items():
            value = parameter.grad if gradients else parameter
            if value is None:
                raise RuntimeError(f"Missing accumulated migration gradient for {name}")
            full = value.full_tensor() if isinstance(value, DTensor) else value
            if torch.distributed.get_rank() == 0:
                result[name] = full.detach().to("cpu", copy=True)
    return result if torch.distributed.get_rank() == 0 else None


@torch.no_grad()
def import_unit(module, unit, values, *, gradients=False):
    with _parameters(module, unit, gradients=gradients, writeback=True) as parameters:
        if set(parameters) != set(values):
            raise ValueError(f"FSDP migration parameter names differ in unit {unit!r}")
        for name, parameter in parameters.items():
            target = parameter.grad if gradients else parameter
            if target is None:
                raise RuntimeError("Main Actor must accumulate before importing migration gradients")
            value = values[name].to(device=target.device, dtype=target.dtype)
            if isinstance(target, DTensor):
                value = distribute_tensor(value, device_mesh=target.device_mesh, placements=target.placements)
            if value.shape != target.shape:
                raise ValueError(f"Migration tensor shape mismatch for {name}")
            if gradients:
                # Each group's loss includes its own DP size before FSDP's
                # average. Both gradients therefore already represent sum/C.
                target.add_(value)
            else:
                target.copy_(value)


@torch.no_grad()
def export_units(module, units, *, gradients=False):
    """Export every unit inside one worker call so the driver is not on the path per unit."""
    merged = {}
    for unit in units:
        piece = export_unit(module, unit, gradients=gradients)
        if piece is not None:
            merged[unit] = piece
    return merged if torch.distributed.get_rank() == 0 else None


@torch.no_grad()
def import_units(module, units, values, *, gradients=False):
    if set(values) != set(units):
        raise ValueError("Migration payload units do not match the destination wrapping")
    for unit in units:
        import_unit(module, unit, values[unit], gradients=gradients)


@torch.no_grad()
def _full_grad(value):
    return value.full_tensor() if isinstance(value, DTensor) else value


def _ipc_spec(tensor):
    from torch.multiprocessing.reductions import reduce_tensor
    rebuild, args = reduce_tensor(tensor.detach().contiguous())
    return rebuild, args


def _ipc_tensor(spec):
    from torch.multiprocessing.reductions import rebuild_cuda_tensor
    rebuild, args = spec
    if rebuild is not rebuild_cuda_tensor:
        raise RuntimeError("Migration gradient IPC spec is not a CUDA tensor")
    return rebuild(*args)


@torch.no_grad()
def publish_gradients(module, units, store):
    """Keep full gradients on GPU and hand the colocated main rank a CUDA IPC handle."""
    import pickle
    held = []
    specs = []
    for unit in units:
        with _parameters(module, unit, gradients=True, writeback=False) as parameters:
            for name in sorted(parameters):
                value = parameters[name].grad
                if value is None:
                    raise RuntimeError(f"Missing accumulated migration gradient for {name}")
                full = _full_grad(value)
                if torch.distributed.get_rank() == 0:
                    cloned = full.detach().contiguous().clone()
                    held.append(cloned)
                    specs.append(_ipc_spec(cloned))
    if torch.distributed.get_rank() == 0:
        store.set("migration_grad_specs", pickle.dumps(specs))
        store.set("migration_grad_ready", b"1")
        store.wait(["migration_grad_done"])
        store.delete_key("migration_grad_ready")
        store.delete_key("migration_grad_done")
        store.delete_key("migration_grad_specs")
    del held


@torch.no_grad()
def consume_gradients(module, units, store, src_rank: int):
    """Map the IPC gradients on the colocated rank, then broadcast inside the main group."""
    import pickle
    rank = torch.distributed.get_rank()
    specs = None
    if rank == src_rank:
        store.wait(["migration_grad_ready"])
        specs = pickle.loads(store.get("migration_grad_specs"))
    count = torch.zeros(1, dtype=torch.int64, device="cuda")
    if rank == src_rank:
        count.fill_(len(specs))
    torch.distributed.broadcast(count, src=src_rank)
    cursor = 0
    for unit in units:
        with _parameters(module, unit, gradients=True, writeback=True) as parameters:
            for name in sorted(parameters):
                target = parameters[name].grad
                if target is None:
                    raise RuntimeError("Main Actor must accumulate before importing migration gradients")
                full = _full_grad(target)
                if rank == src_rank:
                    buf = _ipc_tensor(specs[cursor]).to(dtype=full.dtype)
                    if tuple(buf.shape) != tuple(full.shape):
                        raise RuntimeError(f"Migration gradient shape mismatch for {name}")
                else:
                    buf = torch.empty(tuple(full.shape), dtype=full.dtype, device="cuda")
                torch.distributed.broadcast(buf, src=src_rank)
                cursor += 1
                if isinstance(target, DTensor):
                    buf = distribute_tensor(buf, device_mesh=target.device_mesh, placements=target.placements)
                target.add_(buf)
    if rank == src_rank:
        if specs is None or cursor != len(specs):
            raise RuntimeError("Main Actor consumed a different gradient count than the early group published")
        store.set("migration_grad_done", b"1")


@torch.no_grad()
def publish_parameters(module, units, store, src_rank: int):
    """Share current parameters over CUDA IPC. The driver never sees the bytes."""
    import pickle
    held = []
    specs = []
    rank = torch.distributed.get_rank()
    for unit in units:
        with _parameters(module, unit, gradients=False, writeback=False) as parameters:
            for name in sorted(parameters):
                full = _full_grad(parameters[name])
                if rank == src_rank:
                    cloned = full.detach().contiguous().clone()
                    held.append(cloned)
                    specs.append(_ipc_spec(cloned))
    if rank == src_rank:
        store.set("migration_param_specs", pickle.dumps(specs))
        store.set("migration_param_ready", b"1")
        store.wait(["migration_param_done"])
        store.delete_key("migration_param_ready")
        store.delete_key("migration_param_done")
        store.delete_key("migration_param_specs")
    del held


@torch.no_grad()
def consume_parameters(module, units, store):
    """Copy IPC parameters into the early model, one unit at a time."""
    import pickle
    rank = torch.distributed.get_rank()
    specs = None
    if rank == 0:
        store.wait(["migration_param_ready"])
        specs = pickle.loads(store.get("migration_param_specs"))
    count = torch.zeros(1, dtype=torch.int64, device="cuda")
    if rank == 0:
        count.fill_(len(specs))
    if torch.distributed.get_world_size() > 1:
        torch.distributed.broadcast(count, src=0)
    cursor = 0
    for unit in units:
        with _parameters(module, unit, gradients=False, writeback=True) as parameters:
            for name in sorted(parameters):
                target = parameters[name]
                full = _full_grad(target)
                if rank == 0:
                    buf = _ipc_tensor(specs[cursor]).to(device=full.device, dtype=full.dtype)
                    if tuple(buf.shape) != tuple(full.shape):
                        raise RuntimeError(f"Migration parameter shape mismatch for {name}")
                else:
                    buf = torch.empty(tuple(full.shape), dtype=full.dtype, device=full.device)
                if torch.distributed.get_world_size() > 1:
                    torch.distributed.broadcast(buf, src=0)
                cursor += 1
                if isinstance(target, DTensor):
                    buf = distribute_tensor(buf, device_mesh=target.device_mesh, placements=target.placements)
                target.copy_(buf)
    if rank == 0:
        if specs is None or cursor != len(specs):
            raise RuntimeError("Early Actor consumed a different parameter count than the main group published")
        store.set("migration_param_done", b"1")
