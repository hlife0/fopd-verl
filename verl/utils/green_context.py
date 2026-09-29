"""Run a block of CUDA work on the stream of a CUDA green context limited to some SMs.

Unlike an MPS active-thread limit, which is fixed for the life of the process, this
only covers the work launched inside the block, so the same worker can train on part
of the GPU while colocated rollout runs and on all of it after. The green context is
never made the current CUDA context (NCCL, FSDP and Triton call cudaSetDevice, which
replaces it); a kernel launched on a green-context stream runs on that context's SMs.
"""

from contextlib import contextmanager

import torch

_contexts: dict[tuple[int, int], object] = {}


@contextmanager
def green_sms(num_sms: int):
    if not num_sms:
        yield
        return
    from torch.cuda.green_contexts import GreenContext

    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).multi_processor_count
    if not 0 < num_sms < total:
        raise ValueError(f"green context needs 0 < num_sms < {total}, got {num_sms}")
    ctx = _contexts.get((device, num_sms))
    if ctx is None:
        ctx = _contexts[(device, num_sms)] = GreenContext.create(num_sms, device)
    prev = torch.cuda.current_stream()
    # The two streams are ordered by full syncs; a chunk is seconds of work.
    prev.synchronize()
    torch.cuda.set_stream(ctx.Stream())
    try:
        yield
        torch.cuda.current_stream().synchronize()
    finally:
        torch.cuda.set_stream(prev)
