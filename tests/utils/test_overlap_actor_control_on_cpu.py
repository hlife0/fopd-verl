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

from types import SimpleNamespace

import pytest

from verl.utils.overlap_actor import OverlapActorControl, check_mps_client


def test_step_phase_is_monotonic_and_cancelled_steps_cannot_resume():
    control = OverlapActorControl()
    control.begin(1)
    assert control.get(1) == (control.LIMITED, 0.0)
    timestamp = control.full(1)
    assert control.full(1) == timestamp
    control.cancel(1)
    assert control.get(1)[0] == control.CANCELLED
    with pytest.raises(RuntimeError, match="cancelled"):
        control.full(1)
    with pytest.raises(ValueError, match="advance"):
        control.begin(1)
    control.begin(2)
    assert control.get(2) == (control.LIMITED, 0.0)
    with pytest.raises(RuntimeError, match="different step"):
        control.cancel(1)


def test_mps_requires_actual_client_and_uncapped_server(monkeypatch):
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/tmp/test-overlap-mps")
    for key in ("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "FOPD_MPS_ACTOR_PERCENT", "CUDA_MPS_SM_PARTITION"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("verl.utils.overlap_actor.os.getpid", lambda: 42)
    responses = {
        "get_server_list": "123\n",
        "get_client_list 123": "42\n",
        "get_active_thread_percentage 123": "100.0\n",
    }
    monkeypatch.setattr(
        "verl.utils.overlap_actor.subprocess.run",
        lambda *a, **kw: SimpleNamespace(stdout=responses[kw["input"].strip()]),
    )
    check_mps_client()
    responses["get_active_thread_percentage 123"] = "50.0\n"
    with pytest.raises(ValueError, match="percentage cap"):
        check_mps_client()
    responses["get_client_list 123"] = "99\n"
    with pytest.raises(RuntimeError, match="not connected"):
        check_mps_client()
    monkeypatch.setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "50")
    with pytest.raises(ValueError, match="below 100"):
        check_mps_client()
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY")
    with pytest.raises(RuntimeError, match="requires an active MPS"):
        check_mps_client()
