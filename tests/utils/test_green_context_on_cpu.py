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

import threading
from types import SimpleNamespace

from verl.utils import green_context


def test_cublas_sm_target_is_set_on_activation_and_once_per_execute_model_thread(monkeypatch):
    calls = []
    monkeypatch.setattr(green_context, "_set_cublas_sm_target", lambda count, where: calls.append((count, where)))

    class Worker:
        def execute_model(self, step):
            return step

    gpu_worker = SimpleNamespace(Worker=Worker)
    green_context._install_cublas_sm_target(gpu_worker, 64)
    assert calls == [(64, "activate")]
    worker = Worker()
    assert [worker.execute_model(step) for step in range(3)] == [0, 1, 2]
    thread = threading.Thread(target=worker.execute_model, args=(3,))
    thread.start()
    thread.join()
    assert calls == [(64, "activate"), (64, "execute_model"), (64, "execute_model")]
