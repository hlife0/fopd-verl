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

"""Run a vLLM engine worker on part of the GPU's SMs with a CUDA green context.

VERL_GREEN_SMS=N
    The worker's streams run on the first N SMs, the group GreenContext.create(N) takes.
VERL_GREEN_EXCLUDE_SMS=N
    The worker's streams run on the SMs left after the first N.
FOPD_CUBLAS_SM_TARGET=N|green (off unless set)
    Once the green streams are active, cublasSetSmCountTarget(N) on the worker's cuBLAS
    handles, so cuBLAS sizes its grids for N SMs instead of the whole device; "green" uses
    the green context's SM count. cuBLAS handles are per thread: set on the activating
    thread and on each thread the first time it runs Worker.execute_model.

Applied in each vLLM GPU worker right after its distributed environment is up, so the
model runner's current stream and its CUDA graph capture streams are green streams;
graph nodes keep the context of the stream they were captured on. NCCL launches on
the issuing stream. Streams created before activation stay on the full device.
Processes on one GPU only run concurrently under MPS.
"""

import logging
import os

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_state: dict = {}
_hooked: list = []
_cublas: dict = {}


def _check(result):
    from cuda.bindings import driver

    err, *rest = result
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"green context: CUDA driver error {err}")
    return rest[0] if len(rest) == 1 else rest


def requested_green_sms() -> tuple[str, int] | None:
    first = os.environ.get("VERL_GREEN_SMS", "")
    rest = os.environ.get("VERL_GREEN_EXCLUDE_SMS", "")
    if first and rest:
        raise ValueError("set only one of VERL_GREEN_SMS and VERL_GREEN_EXCLUDE_SMS")
    if first and int(first) > 0:
        return "first", int(first)
    if rest and int(rest) > 0:
        return "rest", int(rest)
    return None


def stream_sm_count(stream) -> int:
    from cuda.bindings import driver

    ctx = _check(driver.cuStreamGetCtx(driver.CUstream(stream.cuda_stream)))
    res = _check(driver.cuCtxGetDevResource(ctx, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
    return int(res.sm.smCount)


def activate_green_streams(mode: str, count: int) -> int:
    """Make every later torch.cuda.Stream on the current device a green stream. Returns its SM count."""
    import torch
    from cuda.bindings import driver

    if _state:
        return _state["sms"]
    dev = torch.cuda.current_device()
    whole = _check(driver.cuDeviceGetDevResource(dev, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
    align = max(int(whole.sm.smCoscheduledAlignment), 1)
    count = count // align * align
    if not 0 < count < int(whole.sm.smCount):
        raise ValueError(f"green context needs 0 < SMs < {int(whole.sm.smCount)} in steps of {align}")
    groups, _, remaining = _check(driver.cuDevSmResourceSplitByCount(1, whole, 0, count))
    resource = groups[0] if mode == "first" else remaining
    desc = _check(driver.cuDevResourceGenerateDesc([resource], 1))
    gctx = _check(driver.cuGreenCtxCreate(desc, dev, driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))
    _state.update(gctx=gctx, dev=dev, sms=int(resource.sm.smCount))

    original = torch.cuda.Stream.__new__

    def green_new(cls, device=None, priority=0, **kwargs):
        if cls is torch.cuda.Stream and not ({"stream_id", "stream_ptr"} & kwargs.keys()):
            if device is None or (isinstance(device, torch.device) and device.index is None):
                index = torch.cuda.current_device()
            else:
                index = device.index if isinstance(device, torch.device) else int(device)
            if index == _state["dev"]:
                ptr = _check(
                    driver.cuGreenCtxStreamCreate(
                        _state["gctx"], driver.CUstream_flags.CU_STREAM_NON_BLOCKING, int(priority)
                    )
                )
                with torch.cuda.device(_state["dev"]):
                    return torch._C._CudaStreamBase.__new__(cls, priority=0, stream_ptr=int(ptr))
        return original(cls, device, priority, **kwargs)

    torch.cuda.Stream.__new__ = green_new
    stream = torch.cuda.Stream()
    if stream_sm_count(stream) != _state["sms"]:
        raise RuntimeError("green context: stream does not report the green SM count")
    torch.cuda.set_stream(stream)
    logger.warning(
        "green context: pid %s device %s runs on %s SMs (%s %s)", os.getpid(), dev, _state["sms"], mode, count
    )
    return _state["sms"]


def _set_cublas_sm_target(count: int, where: str) -> None:
    """cublasSetSmCountTarget(count) on the current thread's cuBLAS handle of the current stream."""
    import ctypes
    import threading

    import torch

    if "lib" not in _cublas:
        torch.cuda.current_blas_handle()
        with open("/proc/self/maps") as maps:
            _cublas["lib"] = ctypes.CDLL(next(line.split()[-1] for line in maps if "/libcublas.so" in line))
    lib = _cublas["lib"]
    handle = ctypes.c_void_p(torch.cuda.current_blas_handle())
    rc = lib.cublasSetSmCountTarget(handle, ctypes.c_int(count))
    got = ctypes.c_int(-1)
    lib.cublasGetSmCountTarget(handle, ctypes.byref(got))
    thread = threading.get_ident()
    logger.warning("cuBLAS SM target: pid %s %s thread %s rc=%s target=%s", os.getpid(), where, thread, rc, got.value)


def _install_cublas_sm_target(gpu_worker, count: int) -> None:
    import threading

    _set_cublas_sm_target(count, "activate")
    done = set()
    run = gpu_worker.Worker.execute_model

    def execute_model(self, *args, **kwargs):
        tid = threading.get_ident()
        if tid not in done:
            done.add(tid)
            _set_cublas_sm_target(count, "execute_model")
        return run(self, *args, **kwargs)

    gpu_worker.Worker.execute_model = execute_model


def install_vllm_green_hook() -> None:
    """In a vLLM GPU worker, activate green streams once its distributed environment is up."""
    request = requested_green_sms()
    if request is None or _hooked:
        return
    from vllm.v1.worker import gpu_worker

    original = gpu_worker.init_worker_distributed_environment

    def init_then_green(*args, **kwargs):
        result = original(*args, **kwargs)
        sms = activate_green_streams(*request)
        target = os.environ.get("FOPD_CUBLAS_SM_TARGET", "")
        if target:
            _install_cublas_sm_target(gpu_worker, sms if target == "green" else int(target))
        return result

    gpu_worker.init_worker_distributed_environment = init_then_green
    _hooked.append(True)
