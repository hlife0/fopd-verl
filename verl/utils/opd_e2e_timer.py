"""Optional coarse rollout-to-publication timing; no per-sample recording."""

import functools
import json
import os
import socket
import time
from pathlib import Path

_directory = os.getenv("FOPD_E2E_DIR", "")
_starts = {}


def _write(event, step, now, **fields):
    Path(_directory).mkdir(parents=True, exist_ok=True)
    row = {
        "event": event, "step": int(step), "mono_ns": now, "wall_ns": time.time_ns(),
        "host": socket.gethostname(), "pid": os.getpid(), "boundary": "host",
        **fields,
    }
    path = Path(_directory) / f"e2e-{row['host']}-{row['pid']}.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def rollout_start(step, count):
    if _directory:
        now = time.monotonic_ns()
        _starts[int(step)] = now
        _write("rollout_start", step, now, count=int(count))


def finish(fn):
    @functools.wraps(fn)
    def call(self, *args, **kwargs):
        result = fn(self, *args, **kwargs)
        if _directory:
            now = time.monotonic_ns()
            step = int(self.global_steps)
            start = _starts.pop(step, None)
            seconds = (now - start) / 1e9 if start is not None else None
            _write("weights_ready", step, now, rollout_to_weights_ready_s=seconds)
            if seconds is not None:
                print(f"opd_e2e step={step} rollout_to_weights_ready_s={seconds:.9f}", flush=True)
        return result

    return call
