# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension as Extension
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


def _kv_only_server(request_states=None):
    calls = []

    def record(name):
        return AsyncMock(side_effect=lambda *args, **kwargs: calls.append((name, *args)))

    engine = SimpleNamespace(
        sleep=record("sleep"), wake_up=record("wake_up"), collective_rpc=record("collective_rpc"),
        reset_prefix_cache=record("reset_prefix_cache"), pause_generation=record("pause_generation"),
        resume_generation=record("resume_generation"),
        output_processor=SimpleNamespace(request_states=dict(request_states or {})),
    )
    server = SimpleNamespace(
        node_rank=0, rollout_mode=RolloutMode.COLOCATED,
        config=SimpleNamespace(free_cache_engine=True, sleep_keep_weights=True), engine=engine,
    )
    return server, calls


def test_server_kv_only_sleep_never_calls_full_sleep_or_weight_wake():
    server, calls = _kv_only_server()
    engine = server.engine
    asyncio.run(vLLMHttpServer.sleep(server))
    engine.collective_rpc.assert_awaited_once_with("sleep_kv_cache_only")
    asyncio.run(vLLMHttpServer.wake_up(server, tags=["weights"]))
    assert engine.collective_rpc.await_count == 1
    engine.resume_generation.assert_not_awaited()
    asyncio.run(vLLMHttpServer.wake_up(server))
    assert engine.collective_rpc.await_args.args == ("wake_kv_cache_only",)
    engine.sleep.assert_not_awaited()
    engine.wake_up.assert_not_awaited()
    assert [call[0] for call in calls] == [
        "pause_generation", "collective_rpc", "collective_rpc", "reset_prefix_cache", "resume_generation",
    ]


def test_server_kv_only_sleep_aborts_requests_before_freeing_kv():
    # A request still in flight at sleep, e.g. an overtaken follow hop, must be aborted
    # before the KV cache is unmapped, and new requests wait until the KV cache is back.
    server, calls = _kv_only_server(request_states={"orphan-hop": object()})
    asyncio.run(vLLMHttpServer.sleep(server))
    assert calls[0] == ("pause_generation",)
    server.engine.pause_generation.assert_awaited_once_with(wait_for_inflight_requests=False, clear_cache=True)
    assert calls[1] == ("collective_rpc", "sleep_kv_cache_only")


def test_kv_only_worker_maps_only_kv_and_retries_failed_mapping(monkeypatch):
    from vllm.device_allocator import cumem

    entries = {i: SimpleNamespace(tag=tag, handle=(0, 1024, i, i), cpu_backup_tensor=None)
               for i, tag in enumerate(["weights", "kv_cache", "kv_cache", "default"])}
    allocator = SimpleNamespace(pointer_to_data=entries)
    monkeypatch.setattr(cumem.CuMemAllocator, "get_instance", lambda: allocator)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (4096, 8192))
    unmapped, mapped = [], []
    monkeypatch.setattr(cumem, "unmap_and_release", lambda h: unmapped.append(h[2]))
    worker = SimpleNamespace(model_runner=SimpleNamespace(post_kv_cache_wake_up=Mock()))
    Extension.wake_kv_cache_only(worker)  # Initial wake is a no-op.
    Extension.sleep_kv_cache_only(worker)
    Extension.sleep_kv_cache_only(worker)
    assert unmapped == [1, 2]
    assert all(e.cpu_backup_tensor is None for e in entries.values())

    def failing_map(handle):
        if handle[2] == 1:
            raise RuntimeError("allocation failed")
        mapped.append(handle[2])

    monkeypatch.setattr(cumem, "create_and_map", failing_map)
    with pytest.raises(RuntimeError, match="allocation failed"):
        Extension.wake_kv_cache_only(worker)
    assert mapped == [2] and worker._kv_only_sleeping
    monkeypatch.setattr(cumem, "create_and_map", lambda h: mapped.append(h[2]))
    Extension.wake_kv_cache_only(worker)
    Extension.wake_kv_cache_only(worker)
    assert mapped == [2, 1]
    worker.model_runner.post_kv_cache_wake_up.assert_called_once()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_cumem_releases_kv_retains_weights_and_preserves_forward():
    from vllm.device_allocator.cumem import CuMemAllocator

    torch.cuda.set_device(0)
    allocator = CuMemAllocator.get_instance()
    worker = SimpleNamespace(model_runner=SimpleNamespace(post_kv_cache_wake_up=Mock()))
    with allocator.use_memory_pool("weights"):
        weights = torch.arange(1024 * 1024, device="cuda", dtype=torch.float32).reshape(1024, 1024) / 1024
    with allocator.use_memory_pool("kv_cache"):
        kv = torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    x = torch.ones(1024, device="cuda")
    expected = weights @ x
    weight_ptr = weights.data_ptr()
    try:
        for _ in range(3):
            kv.fill_(17)
            torch.cuda.synchronize()
            before = torch.cuda.mem_get_info()[0]
            Extension.sleep_kv_cache_only(worker)
            assert torch.cuda.mem_get_info()[0] - before >= 60 * 1024**2
            assert weights.data_ptr() == weight_ptr
            torch.testing.assert_close(weights @ x, expected, rtol=0, atol=0)
            assert all(e.cpu_backup_tensor is None for e in allocator.pointer_to_data.values())
            Extension.wake_kv_cache_only(worker)
            kv.fill_(23)
            assert kv[0].item() == kv[-1].item() == 23
            torch.testing.assert_close(weights @ x, expected, rtol=0, atol=0)
    finally:
        Extension.wake_kv_cache_only(worker)
    assert worker.model_runner.post_kv_cache_wake_up.call_count == 3
