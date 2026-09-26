# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""CPU checks for the student-decode-graph recipe on the installed vLLM 0.24.

The graph switch is a Student engine config. It does not change sampling
parameters, rejection sampling, or the requested rollout logprobs.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pytest

pytest.importorskip("vllm")

import vllm
from vllm.config.compilation import CUDAGraphMode
from vllm.config.speculative import EagleModelTypes, SpeculativeConfig
from vllm.config.vllm import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_mod
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator

_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "scripts"
_FAIR = Path("/csproject/fyp26_bl1/fopd/scripts/fair_compare/6gpu-0.6b-from-8b.sh")
_DISTILL = _REPO / "verl/trainer/config/distillation/distillation.yaml"
_SERVER = _REPO / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"


def _override_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text().splitlines():
        stripped = raw.strip().removesuffix("\\").strip()
        if stripped.startswith("actor_rollout_ref.") or stripped.startswith("distillation."):
            lines.append(stripped)
    return lines


def test_installed_vllm_is_0_24():
    assert vllm.__version__ == "0.24.0"


def test_launchers_differ_only_by_student_cudagraph_mode():
    full = _override_lines(_SCRIPTS / "student_decode_graph_4gpu.sh")
    piecewise = _override_lines(_SCRIPTS / "student_decode_piecewise_4gpu.sh")
    assert full == [
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.rollout.max_num_seqs=32",
        "distillation.teacher_models.teacher_model.inference.max_num_seqs=32",
        "actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_AND_PIECEWISE",
    ]
    assert piecewise[:-1] == full[:-1]
    assert piecewise[-1].endswith("cudagraph_mode=PIECEWISE")
    assert all("teacher" not in line or line.endswith("max_num_seqs=32") for line in full)
    for path in (
        _SCRIPTS / "student_decode_graph_4gpu.sh",
        _SCRIPTS / "student_decode_piecewise_4gpu.sh",
    ):
        text = path.read_text()
        assert "unset OPD_PUBLICATION_GC_FREEZE_STEP" in text
        assert "+ray_kwargs.ray_init.runtime_env.env_vars.OPD_PUBLICATION_GC_FREEZE_STEP='2'" in text
        assert "calculate_log_probs" not in text
        assert "rejection_sample_method" not in text
        assert "num_speculative_tokens" not in text
        assert "loss_mode" not in text


def test_fair_recipe_keeps_eagle3_k3_and_logprobs():
    text = _FAIR.read_text()
    assert "actor_rollout_ref.rollout.calculate_log_probs=True" in text
    assert "num_speculative_tokens:3" in text
    assert "rejection_sample_method:standard" in text
    assert "draft_sample_method:greedy" in text
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=PIECEWISE" in text
    assert "distillation.teacher_models.teacher_model.inference.engine_kwargs" not in text


def test_teacher_yaml_stays_eager_with_empty_engine_kwargs():
    text = _DISTILL.read_text()
    assert "enforce_eager: true" in text
    assert "engine_kwargs: {}" in text
    server = _SERVER.read_text()
    assert 'compilation_config.setdefault("cudagraph_mode", "FULL_AND_PIECEWISE")' in server


def test_graph_mode_is_not_a_sampling_field():
    params = SamplingParams(temperature=1.0, logprobs=1)
    assert params.logprobs == 1
    assert not hasattr(params, "cudagraph_mode")


def test_fixed_k3_does_not_force_piecewise():
    assert "eagle3" in get_args(EagleModelTypes)
    host = SimpleNamespace(num_speculative_tokens_per_batch_size=None, method="eagle3")
    assert SpeculativeConfig.uses_dynamic_speculative_decoding(host) is False
    assert SpeculativeConfig.use_eagle(host) is True

    compilation = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE)
    spec = SimpleNamespace(uses_dynamic_speculative_decoding=lambda: False)
    cfg = SimpleNamespace(speculative_config=spec, compilation_config=compilation)
    VllmConfig._maybe_override_dynamic_sd_cudagraph_mode(cfg)
    assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE


def test_piecewise_leaves_draft_decode_eager_and_full_captures_it():
    seen = []

    class Prefill:
        def __init__(self, vllm_config, device, cudagraph_mode, num_steps):
            seen.append(("prefill", cudagraph_mode, num_steps))

    class Decode:
        def __init__(self, vllm_config, device, cudagraph_mode, decode_query_len):
            seen.append(("decode", cudagraph_mode, decode_query_len))

    orig_prefill = spec_mod.PrefillSpeculatorCudaGraphManager
    orig_decode = spec_mod.DecodeSpeculatorCudaGraphManager
    spec_mod.PrefillSpeculatorCudaGraphManager = Prefill
    spec_mod.DecodeSpeculatorCudaGraphManager = Decode
    try:
        host = SimpleNamespace(vllm_config=object(), device="cpu", num_speculative_steps=3)
        AutoRegressiveSpeculator.init_cudagraph_manager(host, CUDAGraphMode.PIECEWISE)
        AutoRegressiveSpeculator.init_cudagraph_manager(host, CUDAGraphMode.FULL_AND_PIECEWISE)
    finally:
        spec_mod.PrefillSpeculatorCudaGraphManager = orig_prefill
        spec_mod.DecodeSpeculatorCudaGraphManager = orig_decode

    assert seen == [
        ("prefill", CUDAGraphMode.PIECEWISE, 4),
        ("decode", CUDAGraphMode.NONE, 1),
        ("prefill", CUDAGraphMode.FULL_AND_PIECEWISE, 4),
        ("decode", CUDAGraphMode.FULL_DECODE_ONLY, 1),
    ]
    assert seen[3][1].decode_mode() == CUDAGraphMode.FULL
    assert seen[1][1].decode_mode() == CUDAGraphMode.NONE


def _capture_descs(mode: CUDAGraphMode) -> dict:
    host = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[4, 8, 16, 32, 64, 96, 128]),
        cudagraph_mode=mode,
        max_num_reqs=32,
        decode_query_len=1,
        lora_capture_cases=[0],
        _candidates={},
        _capture_descs={},
    )
    CudaGraphManager._init_candidates(host)
    return host._capture_descs


def test_decode_full_capture_stops_at_max_num_seqs():
    full = _capture_descs(CUDAGraphMode.FULL_AND_PIECEWISE)
    assert [d.num_tokens for d in full[CUDAGraphMode.FULL]] == [32, 16, 8, 4]
    assert [d.num_tokens for d in full[CUDAGraphMode.PIECEWISE]] == [128, 96, 64, 32, 16, 8, 4]
    piecewise = _capture_descs(CUDAGraphMode.PIECEWISE)
    assert set(piecewise) == {CUDAGraphMode.PIECEWISE}
    assert [d.num_tokens for d in piecewise[CUDAGraphMode.PIECEWISE]] == [128, 96, 64, 32, 16, 8, 4]
