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


@contextmanager
def _parameters(module, unit, *, gradients, writeback):
    if isinstance(module, FSDP):
        child = module.get_submodule(unit) if unit else module
        if not child._use_orig_params:
            raise ValueError("route-2 gradient transfer requires FSDP use_orig_params=true")
        owned = set(child._handle.flat_param._fqns)
        with FSDP.summon_full_params(child, recurse=False, writeback=writeback, with_grads=gradients):
            yield {
                _clean(name): parameter for name, parameter in child.named_parameters()
                if _clean(name) in owned
            }
    else:
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
