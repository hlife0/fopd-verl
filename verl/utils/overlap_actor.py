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

"""Step-local control and a limited compute stream for rollout/Actor overlap.

FSDP and NCCL keep their primary-context streams. Only forward/backward compute
uses the Green stream, and no limited microbatch remains queued at a boundary.
"""

import math
import os
import subprocess
import time
from contextlib import contextmanager

import ray
import torch
import torch.distributed as dist


class OverlapActorControl:
    LIMITED, FULL, CANCELLED = range(3)

    def __init__(self):
        self.step = None
        self.phase = self.CANCELLED
        self.full_ts = 0.0

    def begin(self, step: int):
        if self.step is not None and step <= self.step:
            raise ValueError("overlap_actor step must advance")
        self.step, self.phase, self.full_ts = step, self.LIMITED, 0.0

    def get(self, step: int):
        if step != self.step:
            raise RuntimeError("overlap_actor control belongs to a different step")
        return self.phase, self.full_ts

    def full(self, step: int):
        self.get(step)
        if self.phase == self.CANCELLED:
            raise RuntimeError("overlap_actor step was cancelled")
        if self.phase != self.FULL:
            self.phase, self.full_ts = self.FULL, time.time()
        return self.full_ts

    def cancel(self, step: int):
        self.get(step)
        self.phase = self.CANCELLED


def create_overlap_control():
    return ray.remote(num_cpus=0)(OverlapActorControl).remote()


def _cuda_value(result):
    from cuda.bindings import driver

    if result[0] != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"overlap_actor CUDA resource query failed: {result[0]}")
    return result[1]


def stream_sm_count(stream) -> int:
    """Query the stream's actual resource descriptor, not device properties."""
    from cuda.bindings import driver

    context = _cuda_value(driver.cuStreamGetCtx(driver.CUstream(stream.cuda_stream)))
    resource = _cuda_value(driver.cuCtxGetDevResource(context, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
    return int(resource.sm.smCount)


def check_mps_client():
    if not os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
        raise RuntimeError("overlap_actor requires an active MPS service and CUDA_MPS_PIPE_DIRECTORY")
    for key in ("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "FOPD_MPS_ACTOR_PERCENT"):
        if os.environ.get(key) not in (None, "", "100"):
            raise ValueError(f"overlap_actor cannot restore full SMs with {key} set below 100")
    if os.environ.get("CUDA_MPS_SM_PARTITION"):
        raise ValueError("overlap_actor cannot restore full SMs inside a static MPS partition")

    def query(command):
        result = subprocess.run(
            ["nvidia-cuda-mps-control"],
            input=command + "\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()

    try:
        servers = [int(line) for line in query("get_server_list").splitlines() if line.strip().isdigit()]
        for server in servers:
            clients = {int(line) for line in query(f"get_client_list {server}").splitlines() if line.strip().isdigit()}
            if os.getpid() in clients:
                if float(query(f"get_active_thread_percentage {server}")) != 100:
                    raise ValueError("overlap_actor requires an MPS server without an active-thread percentage cap")
                return
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("overlap_actor could not query its MPS service") from exc
    raise RuntimeError("Actor CUDA context is not connected to the configured MPS service")


class OverlapActorExecution:
    def __init__(self, sm_fraction: float, group):
        if not math.isfinite(sm_fraction) or not 0 < sm_fraction < 1:
            raise ValueError("overlap_actor_sm_fraction must be between 0 and 1")
        if not hasattr(torch.cuda, "GreenContext"):
            raise RuntimeError("overlap_actor requires PyTorch with GreenContext support")
        from cuda.bindings import driver

        self.device = torch.cuda.current_device()
        self.primary = torch.cuda.current_stream()
        check_mps_client()
        resource = _cuda_value(
            driver.cuDeviceGetDevResource(
                self.device,
                driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
            )
        )
        self.total_sms = int(resource.sm.smCount)
        # Query the hardware granularity, including devices other than 3090.
        granularity = int(resource.sm.smCoscheduledAlignment)
        minimum = int(resource.sm.minSmPartitionSize)
        requested = math.floor(self.total_sms * sm_fraction / granularity) * granularity
        if requested < minimum:
            raise ValueError(f"overlap_actor SM budget is below the device minimum of {minimum}")
        self.green = torch.cuda.GreenContext.create(requested, self.device)
        self.limited = self.green.Stream()
        self.limited_sms = stream_sm_count(self.limited)
        if self.limited_sms > self.total_sms * sm_fraction or self.limited_sms >= self.total_sms:
            raise RuntimeError("Green stream did not honor the requested overlap_actor SM budget")
        if stream_sm_count(self.primary) != self.total_sms:
            raise RuntimeError("Actor primary stream cannot access all device SMs")
        self.group = group
        self.rank = dist.get_rank(group)
        self.source = dist.get_global_rank(group, 0) if group is not None else 0
        self.status = torch.zeros(2, dtype=torch.float64, device=self.device)
        # Initialize this process group's NCCL stream in the primary context.
        dist.all_reduce(self.status, group=group)
        torch.cuda.synchronize()
        self.control = None
        print(f"overlap_actor: limited={self.limited_sms} SM, full={self.total_sms} SM", flush=True)

    def begin(self, control, step):
        self.control, self.step = control, step
        self.full = False
        self.full_ts = None
        self.signal_ts = None
        self.limited_microbatches = 0
        self.full_microbatches = 0
        self.limited_completions = []
        self.early_samples = 0

    def _phase(self):
        # Rank 0 may finish a microbatch before its peers. Read control only
        # after every rank reaches this boundary, or a stale LIMITED decision
        # could enqueue one more MB after FULL arrived during the peer's F/B.
        dist.barrier(group=self.group, device_ids=[self.device])
        error = None
        if self.rank == 0:
            try:
                phase, timestamp = ray.get(self.control.get.remote(self.step))
            except Exception as exc:
                error = exc
                phase, timestamp = OverlapActorControl.CANCELLED, 0
            self.status[0], self.status[1] = phase, timestamp
        dist.broadcast(self.status, src=self.source, group=self.group)
        phase, timestamp = self.status.tolist()
        if int(phase) == OverlapActorControl.CANCELLED:
            raise RuntimeError("overlap_actor step cancelled or control unavailable") from error
        if int(phase) == OverlapActorControl.FULL and not self.full:
            self.primary.wait_stream(self.limited)
            if stream_sm_count(self.primary) != self.total_sms:
                raise RuntimeError("overlap_actor full phase still has an SM limit")
            self.full = True
            self.signal_ts, self.full_ts = timestamp, time.time()

    @contextmanager
    def microbatch(self, data=None):
        self._phase()
        if self.full:
            self.full_microbatches += 1
            with torch.cuda.stream(self.primary):
                yield
            return
        real_samples = 0 if data is None else sum(int(mask.any().item()) for mask in data["loss_mask"].unbind())
        self.limited.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(self.limited):
                yield
        finally:
            # FSDP's backward completion waits for its communication streams.
            # Synchronizing here bounds the outstanding limited work to one MB.
            self.limited.synchronize()
            self.primary.wait_stream(self.limited)
        self.limited_microbatches += 1
        self.limited_completions.append((time.time(), real_samples))

    def finish(self, student_done_ts=None):
        self._phase()
        if not self.full:
            raise RuntimeError("overlap_actor cannot update parameters before the full-SM phase")
        count = sum(n for timestamp, n in self.limited_completions if student_done_ts and timestamp < student_done_ts)
        completed = torch.tensor(count, device=self.device, dtype=torch.int64)
        dist.all_reduce(completed, group=self.group)
        self.early_samples = completed.item()
        switched = torch.tensor(self.full_ts, device=self.device, dtype=torch.float64)
        dist.all_reduce(switched, op=dist.ReduceOp.MAX, group=self.group)
        self.full_ts = switched.item()

    def metrics(self):
        return {
            "overlap_actor/limited_sms": self.limited_sms,
            "overlap_actor/full_sms": self.total_sms,
            "overlap_actor/limited_microbatches": self.limited_microbatches,
            "overlap_actor/full_microbatches": self.full_microbatches,
            "overlap_actor/full_switch_ts": self.full_ts,
            "overlap_actor/full_switch_delay_s": self.full_ts - self.signal_ts,
            "overlap_actor/samples_completed_before_student": self.early_samples,
        }
