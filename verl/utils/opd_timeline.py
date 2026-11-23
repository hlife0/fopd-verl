"""Optional host events and NVTX launch ranges for OPD timelines.

These ranges describe host calls. GPU intervals must come from CUPTI, linked
through the CUDA launch correlation and the NVTX span ID. No CUDA fences are
introduced. FOPD_TIMELINE_DIR unset disables all recording.
"""

import atexit
import contextvars
import functools
import inspect
import itertools
import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_DIR = os.getenv("FOPD_TIMELINE_DIR", "")
_SCOPE = os.getenv("FOPD_TIMELINE_SCOPE", "full")
_STEPS = {int(x) for x in os.getenv("FOPD_TIMELINE_STEPS", "").split(",") if x.strip()}
_NVTX = os.getenv("FOPD_TIMELINE_NVTX", "0") == "1"
_context = contextvars.ContextVar("opd_timeline_context", default={})
_counter = itertools.count()
_writer = None
_lock = threading.RLock()
_nvtx_module = None
_nvtx_checked = False
_step_starts = {}


def enabled(step=None, coarse=False):
    return bool(_DIR) and (coarse or _SCOPE in {"full", "host"}) and (
        step is None or not _STEPS or int(step) in _STEPS
    )


def _plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if type(value).__name__ == "NonTensorData":
        return _plain(value.data)
    if hasattr(value, "device") and getattr(value.device, "type", "cpu") != "cpu":
        return None
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item"):
        return _plain(value.item())
    return str(value)


def _nvtx():
    global _nvtx_module, _nvtx_checked
    if _NVTX and not _nvtx_checked:
        _nvtx_checked = True
        try:
            import nvtx

            _nvtx_module = nvtx
        except ImportError:
            pass
    return _nvtx_module


def _get_writer():
    global _writer
    if _writer is None:
        Path(_DIR).mkdir(parents=True, exist_ok=True)
        host, pid = socket.gethostname(), os.getpid()
        _writer = open(Path(_DIR) / f"events-{host}-{pid}.jsonl", "a", buffering=65536)
        before = time.monotonic_ns()
        wall = time.time_ns()
        after = time.monotonic_ns()
        row = {
            "event": "clock.anchor", "boundary": "host", "host": host, "pid": pid,
            "tid": threading.get_native_id(), "mono_ns": (before + after) // 2,
            "wall_ns": wall, "mono_before_ns": before, "mono_after_ns": after,
            "nvtx_enabled": _nvtx() is not None,
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
            "rank": os.getenv("RANK", ""), "local_rank": os.getenv("LOCAL_RANK", ""),
            "replica_rank": os.getenv("VERL_REPLICA_RANK", ""),
        }
        _writer.write(json.dumps(row) + "\n")
        if _nvtx() is not None:
            marker_mono = time.monotonic_ns()
            _nvtx().mark(
                f"opd|clock|pid={pid}|mono_ns={marker_mono}|wall_ns={wall}",
                domain="opd",
            )
        _writer.flush()
    return _writer


def flush():
    with _lock:
        if _writer is not None:
            _writer.flush()


atexit.register(flush)


def emit(event, *, coarse=False, **fields):
    values = {**_context.get(), **fields}
    if not enabled(values.get("step"), coarse=coarse):
        return None
    now = time.monotonic_ns()
    row = {
        **_plain(values), "event": event, "boundary": "host", "mono_ns": now,
        "wall_ns": time.time_ns(), "host": socket.gethostname(), "pid": os.getpid(),
        "tid": threading.get_native_id(),
    }
    with _lock:
        _get_writer().write(json.dumps(row, separators=(",", ":")) + "\n")
    return row


@contextmanager
def span(event, *, coarse=False, async_range=False, **fields):
    values = {**_context.get(), **fields}
    if not enabled(values.get("step"), coarse=coarse):
        token = _context.set(values) if _DIR and _SCOPE in {"full", "host"} else None
        try:
            yield
        finally:
            if token is not None:
                _context.reset(token)
        return
    ident = f"{os.getpid()}:{next(_counter)}"
    values["parent_span_id"] = values.get("current_span_id")
    values["current_span_id"] = ident
    token = _context.set(values)
    emit(event + ".begin", coarse=coarse, span_id=ident)
    nv = _nvtx()
    handle = None
    label = (
        f"opd|{event}|{ident}|step={values.get('step', -1)}"
        f"|rank={values.get('rank', os.getenv('RANK', ''))}"
        f"|mono_ns={time.monotonic_ns()}"
    )
    if nv is not None:
        if async_range:
            handle = nv.start_range(label, domain="opd")
        else:
            nv.push_range(label, domain="opd")
    status = "ok"
    try:
        yield
    except BaseException as exc:
        status = type(exc).__name__
        raise
    finally:
        if nv is not None:
            if async_range:
                nv.end_range(handle)
            else:
                nv.pop_range(domain="opd")
        emit(event + ".end", coarse=coarse, span_id=ident, status=status)
        _context.reset(token)
        flush()


def context():
    return dict(_context.get()) if _DIR else {}


def _call_fields(arguments, role):
    flat = {**arguments.get("kwargs", {}), **arguments}
    obj = arguments.get("self")
    values = {**_context.get(), **flat.get("timeline_context", {})}
    batch = flat.get("batch")
    if batch is not None and isinstance(getattr(batch, "keys", None), list):
        values["members"] = list(batch.keys)
    data = flat.get("data")
    if data is not None:
        for key in ("uid", "session_id", "global_steps", "index"):
            item = _plain(data.get(key))
            if item is not None:
                values[key] = item
        denominator = _plain(data.get("batch_num_tokens_override"))
        if denominator is not None:
            values["loss_denominator"] = denominator
        chunk = _plain(data.get("timeline_chunk"))
        if chunk is not None:
            values["chunk"] = chunk
        uids = values.get("uid", [])
        if isinstance(uids, list):
            sessions = values.get("session_id", [0] * len(uids))
            if not isinstance(sessions, list):
                sessions = [sessions] * len(uids)
            values["members"] = [f"{u}_{s}_0" for u, s in zip(uids, sessions, strict=True)]
    for key in ("uid", "session_id", "index", "request_id", "priority", "follow", "global_steps"):
        if key in flat:
            values[key] = _plain(flat[key])
    if isinstance(values.get("uid"), str) and isinstance(values.get("session_id"), int):
        values["trajectory_id"] = f"{values['uid']}_{values['session_id']}_0"
    if "sequence_ids" in flat:
        values["prefix_end"] = len(flat["sequence_ids"])
        values["prefix_start"] = 0
    if "prompt_ids" in flat:
        values["prompt_tokens"] = len(flat["prompt_ids"])
    step = values.get("global_steps", getattr(obj, "global_steps", None))
    if isinstance(step, list):
        step = step[0] if step and len(set(step)) == 1 else None
    if step is not None:
        values["step"] = int(step)
    elif obj is not None and hasattr(obj, "_opd_timeline_step"):
        values["step"] = obj._opd_timeline_step
    if obj is not None and "step" in values:
        obj._opd_timeline_step = values["step"]
    values["role"] = role if role != "inherit" else values.get("role", "rollout")
    if obj is not None:
        values["replica_rank"] = getattr(obj, "replica_rank", os.getenv("VERL_REPLICA_RANK", ""))
        values["rank"] = getattr(obj, "rank", values.get("rank", os.getenv("RANK", "")))
    return values


def observed(event, role, *, coarse=False):
    def decorate(fn):
        signature = inspect.signature(fn)
        is_async = inspect.iscoroutinefunction(fn)

        @functools.wraps(fn)
        async def async_call(*args, **kwargs):
            if not enabled(coarse=coarse):
                return await fn(*args, **kwargs)
            values = _call_fields(signature.bind_partial(*args, **kwargs).arguments, role)
            with span(event, coarse=coarse, async_range=True, **values):
                result = await fn(*args, **kwargs)
                _result_event(event, result)
                return result

        @functools.wraps(fn)
        def sync_call(*args, **kwargs):
            if not enabled(coarse=coarse):
                return fn(*args, **kwargs)
            values = _call_fields(signature.bind_partial(*args, **kwargs).arguments, role)
            with span(event, coarse=coarse, **values):
                result = fn(*args, **kwargs)
                if event == "weights.ready":
                    weights_ready(values["step"])
                return result

        return async_call if is_async else sync_call

    return decorate


def _result_event(event, result):
    if event == "student.generate":
        response = getattr(result, "response_ids", None)
        mask = getattr(result, "response_mask", None)
        if response is not None:
            emit("student.metrics", response_tokens=len(response), effective_tokens=sum(mask) if mask is not None else None)
    if "teacher" in event or event == "server.request":
        if isinstance(result, tuple) and result and isinstance(result[-1], dict):
            result = result[-1]
        extra = getattr(result, "extra_fields", result if isinstance(result, dict) else {})
        if isinstance(extra, dict):
            emit(
                event + ".result",
                num_cached_tokens=extra.get("num_cached_tokens"),
                engine_queue_s=extra.get("engine_queue_s"),
                engine_prefill_s=extra.get("engine_prefill_s"),
                engine_decode_s=extra.get("engine_decode_s"),
            )


def rollout_submit(step, batch):
    if not enabled(step, coarse=True):
        return
    _step_starts[int(step)] = time.monotonic_ns()
    emit("step.rollout_start", coarse=True, role="driver", step=int(step), count=len(batch))
    if enabled(step):
        uids = _plain(batch["uid"])
        indices = _plain(batch.get("index"))
        for i, uid in enumerate(uids):
            emit(
                "prompt.submit", role="driver", step=int(step), uid=uid,
                index=indices[i] if isinstance(indices, list) else None, batch_position=i,
            )
    flush()


def weights_ready(step):
    if not enabled(step, coarse=True):
        return
    now = time.monotonic_ns()
    start = _step_starts.pop(int(step), None)
    elapsed = (now - start) / 1e9 if start is not None else None
    emit("step.weights_ready", coarse=True, role="driver", step=int(step), elapsed_s=elapsed)
    flush()
    if elapsed is not None:
        print(f"opd_e2e step={int(step)} rollout_to_weights_ready_s={elapsed:.9f}", flush=True)


def install_vllm_worker_observer():
    """Observe GPU-worker launch batches without modifying scheduling or outputs."""
    if not enabled() or _SCOPE == "host":
        return
    from vllm.v1.worker.gpu_worker import Worker

    if getattr(Worker, "_opd_timeline_installed", False):
        return
    Worker._opd_timeline_installed = True
    supported = []
    for method in ("execute_model", "sample_tokens"):
        original = getattr(Worker, method, None)
        if original is None:
            continue
        supported.append(method)

        def wrap(fn, name):
            @functools.wraps(fn)
            def call(self, *args, **kwargs):
                scheduler = kwargs.get("scheduler_output", args[0] if args else None)
                requests = getattr(scheduler, "num_scheduled_tokens", None)
                if isinstance(requests, dict):
                    self._opd_timeline_requests = list(requests)
                    self._opd_timeline_token_counts = dict(requests)
                role = os.getenv("FOPD_TIMELINE_ROLE", "vllm")
                with span(
                    role + "." + name, role=role, rank=getattr(self, "rank", ""),
                    request_ids=getattr(self, "_opd_timeline_requests", []),
                    scheduled_tokens=getattr(self, "_opd_timeline_token_counts", {}),
                    replica_rank=os.getenv("VERL_REPLICA_RANK", ""),
                ):
                    return fn(self, *args, **kwargs)

            return call

        setattr(Worker, method, wrap(original, method))
    emit(
        "process.vllm_observer", role=os.getenv("FOPD_TIMELINE_ROLE", "vllm"),
        methods=supported, replica_rank=os.getenv("VERL_REPLICA_RANK", ""),
    )
    flush()
