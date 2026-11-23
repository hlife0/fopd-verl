"""Diagnostic-only CUDA mm/addmm arguments; import in an isolated source.

Wrap an existing timeline span's body with ``mm_shape_scope(event, values)``.
FOPD_MM_SHAPE_DIR enables collection. No tensor values or CUDA synchronization
are read. TorchDispatchMode can change compilation behavior; do not use this run
for timing. Teacher spans retain their original step=-1 and request association.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from math import prod
import os
from pathlib import Path
import threading

_pid = None
_seen = {}
_writers = {}
_lock = threading.Lock()
_mode_type = None
_gate_logged = []


def _selected(event, fields):
    if str(fields.get("rank", os.getenv("RANK", ""))) != "0":
        return None
    if event == "actor.forward" and str(fields.get("step")) == "3":
        return "actor"
    if event == "teacher.execute_model":
        # In the current trace, counters 0/2 are warmups, 4 is empty and 5
        # is the first real execution. Keep the original diagnostic span ID.
        ident = str(fields.get("current_span_id", ""))
        suffix = ident.rsplit(":", 1)[-1]
        if suffix.isdigit() and int(suffix) >= 5 and fields.get("request_ids"):
            return "teacher"
    return None


def _tensor_metadata(torch, tensor):
    # vLLM ModelWeightParameter and FSDP FlatParameter are legitimate local
    # Tensor subclasses. Exclude fake/distributed wrappers, not all subclasses.
    if not isinstance(tensor, torch.Tensor):
        return None
    if hasattr(tensor, "fake_mode") or hasattr(tensor, "_local_tensor"):
        return None
    if tensor.device.type != "cuda" or tensor.layout != torch.strided:
        return None
    shape, stride = tuple(tensor.shape), tuple(tensor.stride())
    if not all(type(x) is int for x in shape + stride):
        return None
    try:
        offset, alignment = tensor.storage_offset(), tensor.data_ptr() % 256
    except (RuntimeError, NotImplementedError):
        return None  # No concrete local CUDA storage; do not invent an address.
    return {
        "shape": list(shape), "stride": list(stride),
        "dtype": str(tensor.dtype), "layout": str(tensor.layout),
        "tensor_type": type(tensor).__module__ + "." + type(tensor).__name__,
        "device": str(tensor.device), "storage_offset": offset,
        "pointer_alignment_mod_256": alignment,
    }


def _signature(op, tensors, alpha, beta):
    # A flat-buffer offset alone must not consume the sample budget separately
    # for every layer. Keep it in each actual record; compare matrix layouts and
    # pointer alignment so later real token packs can still contribute samples.
    return (op, alpha, beta, tuple(
        (tuple(t["shape"]), tuple(t["stride"]), t["dtype"], t["layout"],
         t["device"], t["pointer_alignment_mod_256"])
        for t in tensors
    ))


def _get_mode_type():
    global _mode_type
    if _mode_type is not None:
        return _mode_type
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class RealMMMode(TorchDispatchMode):
        def __init__(self, role, fields, out_dir, limit):
            super().__init__()
            self.role, self.fields = role, dict(fields)
            self.out_dir, self.limit = out_dir, limit

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            mm_ops = (torch.ops.aten.mm.default, torch.ops.aten.mm.out)
            addmm_ops = (torch.ops.aten.addmm.default, torch.ops.aten.addmm.out)
            linear_ops = (torch.ops.aten.linear.default,)
            if func not in mm_ops + addmm_ops + linear_ops:
                return func(*args, **kwargs)
            op = "mm" if func in mm_ops else "addmm" if func in addmm_ops else "linear"
            dispatch_op = str(func)
            count = 3 if op == "addmm" else 2
            if len(args) < count:
                return func(*args, **kwargs)
            input_args = list(args[:count])
            bias = args[2] if op == "linear" and len(args) > 2 else kwargs.get("bias")
            if op == "linear" and bias is not None:
                input_args.append(bias)
            tensors = [_tensor_metadata(torch, x) for x in input_args]
            if any(t is None for t in tensors):
                return func(*args, **kwargs)
            a, b = tensors[:2] if op == "linear" else tensors[-2:]
            if len(b["shape"]) != 2 or (len(a["shape"]) < 1 if op == "linear" else len(a["shape"]) != 2):
                return func(*args, **kwargs)
            alpha, beta = kwargs.get("alpha", 1), kwargs.get("beta", 1)
            if not isinstance(alpha, (int, float)) or not isinstance(beta, (int, float)):
                return func(*args, **kwargs)
            out_metadata = _tensor_metadata(torch, kwargs["out"]) if "out" in kwargs else None
            if "out" in kwargs and out_metadata is None:
                return func(*args, **kwargs)
            signature_inputs = tensors + ([out_metadata] if out_metadata is not None else [])
            signature = _signature(dispatch_op, signature_inputs, alpha, beta)
            with _lock:
                seen = _seen.setdefault(self.role, [])
                if signature in seen or len(seen) >= self.limit:
                    record = None
                else:
                    seen.append(signature)  # Tuple comparison, no generated hashes.
                    record = {
                        "record_id": f"{os.getpid()}:{self.role}:{len(seen) - 1}",
                        "role": self.role, "event": self.fields.get("event"),
                        "rank": self.fields.get("rank"),
                        "step": self.fields.get("step", -1),
                        "span_id": self.fields.get("current_span_id"),
                        "microbatch": self.fields.get("microbatch"),
                        "request_ids": self.fields.get("request_ids", []),
                        "scheduled_tokens": self.fields.get("scheduled_tokens", {}),
                        "op": op, "dispatch_op": dispatch_op,
                        "inputs": tensors, "alpha": alpha, "beta": beta,
                        "M": prod(a["shape"][:-1]) if op == "linear" else a["shape"][0],
                        "N": b["shape"][0 if op == "linear" else 1],
                        "K": a["shape"][-1] if op == "linear" else a["shape"][1],
                        "matmul_flags": {key: bool(getattr(torch.backends.cuda.matmul, key))
                            for key in ("allow_tf32", "allow_fp16_reduced_precision_reduction",
                                        "allow_bf16_reduced_precision_reduction")},
                    }
            tagged = record is not None and os.getenv("FOPD_MM_SHAPE_NVTX", "0") == "1"
            if tagged:
                torch.cuda.nvtx.range_push("opd_mm_shape|" + record["record_id"])
            try:
                result = func(*args, **kwargs)
            finally:
                if tagged:
                    torch.cuda.nvtx.range_pop()
            if record is not None:
                record["output"] = _tensor_metadata(torch, result)
                with _lock:
                    writer = _writers.get(self.role)
                    if writer is None:
                        path = Path(self.out_dir) / f"{self.role}-{os.getpid()}.jsonl"
                        # The operator creates a fresh shared output directory first.
                        writer = path.open("x", encoding="utf-8", buffering=1)
                        _writers[self.role] = writer
                    writer.write(json.dumps(record, separators=(",", ":")) + "\n")
            return result

    _mode_type = RealMMMode
    return _mode_type


@contextmanager
def mm_shape_scope(event, fields):
    global _pid, _seen, _writers
    out_dir = os.getenv("FOPD_MM_SHAPE_DIR", "")
    role = _selected(event, fields)
    if role is not None and (os.getpid(), role) not in _gate_logged:
        _gate_logged.append((os.getpid(), role))
        # Only the isolated diagnostic source imports this helper. A single CPU
        # event distinguishes a missing worker env gate from an unobserved op.
        from verl.utils import opd_timeline
        opd_timeline.emit("mm.scope_gate", role=role, rank=fields.get("rank"),
                          step=fields.get("step"), span_id=fields.get("current_span_id"),
                          mm_dir_present=bool(out_dir),
                          timeline_module=opd_timeline.__file__,
                          helper_module=__file__)
    if role is None or not out_dir:
        yield
        return
    if _pid != os.getpid():
        _pid, _seen, _writers = os.getpid(), {}, {}
    limit = int(os.getenv("FOPD_MM_SHAPE_MAX", "64"))
    if not 1 <= limit <= 128:
        raise ValueError("FOPD_MM_SHAPE_MAX must be 1..128")
    if len(_seen.get(role, [])) >= limit:
        yield
        return
    scope_fields = {**fields, "event": event}
    with _get_mode_type()(role, scope_fields, out_dir, limit):
        yield

