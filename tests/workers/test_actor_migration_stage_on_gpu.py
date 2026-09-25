# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Weight sync must stage one FSDP unit, not the whole early model."""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

from verl.utils.fsdp_utils import offload_fsdp_model_to_cpu
from verl.workers.engine.fsdp.migration import export_units, import_units, unit_names


class _Wide(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(1024, 1024, bias=False)
        self.second = torch.nn.Linear(1024, 1024, bias=False)
        self.third = torch.nn.Linear(1024, 1024, bias=False)

    def forward(self, x):
        return self.third(self.second(self.first(x)))


def _wrap(module):
    kwargs = dict(device_id=torch.device("cuda", 0), use_orig_params=True,
                  sharding_strategy=ShardingStrategy.FULL_SHARD)
    module.first = FSDP(module.first, **kwargs)
    module.second = FSDP(module.second, **kwargs)
    module.third = FSDP(module.third, **kwargs)
    return FSDP(module, **kwargs)


def test_import_keeps_only_one_unit_on_gpu(tmp_path):
    if not torch.cuda.is_available():
        return
    dist.init_process_group("nccl", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        torch.manual_seed(7)
        source = _wrap(_Wide().cuda())
        names = unit_names(source)
        assert len(names) >= 3
        source(torch.randn(2, 1024, device="cuda"))
        exported = export_units(source, names)
        offload_fsdp_model_to_cpu(source)
        del source
        destination = _wrap(_Wide().cuda())
        destination(torch.randn(2, 1024, device="cuda"))
        offload_fsdp_model_to_cpu(destination)
        torch.cuda.empty_cache()
        assert all(handle.flat_param.device.type == "cpu" for handle in destination._all_handles)
        resident = []

        def make_spy(handle):
            original = handle.flat_param_to

            def spy(device, *args, **kwargs):
                original(device, *args, **kwargs)
                if torch.device(device).type == "cuda":
                    resident.append(sum(
                        item.flat_param.device.type == "cuda" for item in destination._all_handles
                    ))

            return spy

        for handle in destination._all_handles:
            handle.flat_param_to = make_spy(handle)
        torch.cuda.reset_peak_memory_stats()
        import_units(destination, names, exported)
        assert resident and max(resident) == 1
        assert all(handle.flat_param.device.type == "cpu" for handle in destination._all_handles)
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu
        load_fsdp_model_to_gpu(destination)
        again = export_units(destination, names)
        for unit in names:
            for name in exported[unit]:
                torch.testing.assert_close(again[unit][name], exported[unit][name])
    finally:
        dist.destroy_process_group()
