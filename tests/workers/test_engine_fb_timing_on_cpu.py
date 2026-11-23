# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.utils.metric.utils import Metric
from verl.workers.engine.base import BaseEngine
from verl.workers.engine.fsdp import transformer_impl as impl


@pytest.mark.parametrize("enabled,forward_only", [(False, False), (True, False), (True, True)])
def test_fb_timer_excludes_preparation_postprocessing_and_optimizer(monkeypatch, enabled, forward_only):
    monkeypatch.setenv("FOPD_PROFILE_ACTOR_FB", "1" if enabled else "0")
    events = []
    data = TensorDict({"loss_mask": torch.ones(1)}, batch_size=[1])
    loss = torch.tensor(1.0, requires_grad=True)
    loss.register_hook(lambda grad: events.append("backward") or grad)

    def prepare(**kwargs):
        events.append("prepare")
        return [data, data], None

    def forward(*args, **kwargs):
        events.append("forward")
        return loss, {"metrics": {}, "loss": 1.0}

    def postprocess(**kwargs):
        events.append("postprocess")
        return {"metrics": {}, "loss": 2.0}

    ticks = iter([10.0, 13.5])

    def now():
        events.append("clock")
        return next(ticks)

    engine = SimpleNamespace(
        ulysses_sequence_parallel_size=1,
        get_data_parallel_group=lambda: None,
        get_data_parallel_size=lambda: 1,
        _gradient_sync_context=lambda **kwargs: nullcontext(),
        forward_step=forward,
        optimizer_zero_grad=lambda: events.append("zero"),
        optimizer_step=lambda: events.append("optimizer") or 1.0,
        is_mp_src_rank_with_outputs=lambda: True,
    )
    engine.forward_backward_batch = lambda *args, **kwargs: impl.FSDPEngine.forward_backward_batch(
        engine, *args, **kwargs
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(impl, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(impl, "prepare_micro_batches", prepare)
    monkeypatch.setattr(impl, "postprocess_batch_func", postprocess)
    monkeypatch.setattr(impl, "get_torch_device", lambda: SimpleNamespace(synchronize=lambda: events.append("sync")))
    monkeypatch.setattr(impl.time, "perf_counter", now)
    if forward_only:
        output = engine.forward_backward_batch(data, lambda: None, forward_only=True)
        assert events == ["prepare", "forward", "forward", "postprocess"]
    else:
        output = BaseEngine.train_batch(engine, data, lambda: None)
        bounds = ["sync", "clock"] if enabled else []
        assert events == [
            "zero", "prepare", *bounds, "forward", "backward", "forward", "backward",
            *bounds, "postprocess", "optimizer",
        ]
    metrics = output["metrics"]
    if enabled and not forward_only:
        assert metrics["perf/fb_time_max_s"].aggregate() == 3.5
        assert metrics["perf/fb_micro_batches_max"].aggregate() == 2
        # The worker must use the slowest DP rank, not silently average it away.
        assert Metric.aggregate_dp([metrics["perf/fb_time_max_s"], Metric("max", 4.0)]) == 4.0
    else:
        assert "perf/fb_time_max_s" not in metrics
