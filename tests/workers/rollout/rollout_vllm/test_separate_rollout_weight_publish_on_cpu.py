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

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

pytest.importorskip("vllm")

from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter


def test_resident_standalone_publish_clears_kv_updates_version_without_sleep():
    async def run():
        events = []
        server = object.__new__(vLLMHttpServer)
        server.node_rank = 0
        server.rollout_mode = RolloutMode.STANDALONE
        server.config = SimpleNamespace(free_cache_engine=False)
        server.global_steps = 0
        server._submission_paused = False
        server._admitting = 0
        server._resume_event = asyncio.Event()
        server._resume_event.set()

        async def pause(**kwargs):
            assert kwargs == {"wait_for_inflight_requests": False, "clear_cache": True}
            assert server._submission_paused
            events.append("pause_and_clear")

        async def reset(**kwargs):
            assert kwargs == {"reset_connector": True}
            assert server._submission_paused
            events.append("clear_new_version_kv")

        async def resume():
            assert server.global_steps == 1
            events.append("resume")

        server.engine = SimpleNamespace(
            output_processor=SimpleNamespace(request_states={}),
            pause_generation=AsyncMock(side_effect=pause),
            resume_generation=AsyncMock(side_effect=resume),
            reset_prefix_cache=AsyncMock(side_effect=reset),
            reset_mm_cache=AsyncMock(), reset_encoder_cache=AsyncMock(),
            sleep=AsyncMock(), wake_up=AsyncMock(),
        )
        adapter = object.__new__(ServerAdapter)
        adapter.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=16))
        adapter._has_server = True
        adapter.replica_rank = adapter.rollout_rank = 0
        adapter.use_shm = False
        adapter.zmq_handle = "unused"
        adapter._execute_method = AsyncMock(return_value=None)
        adapter.server_handle = SimpleNamespace(
            clear_kv_cache=SimpleNamespace(remote=server.clear_kv_cache),
            set_global_steps=SimpleNamespace(remote=server.set_global_steps),
        )

        async def transfer(weights):
            assert server._submission_paused
            assert server.global_steps == 0
            events.append("load_weights")

        sender = Mock(async_send_weights=AsyncMock(side_effect=transfer))
        # The existing NCCL manager uses exactly this boundary sequence. The
        # receiver's real adapter performs a second cache reset after loading.
        result = await server.abort_all_requests()
        assert result["aborted_count"] == 0
        await server.release_kv_cache()
        with patch("verl.workers.rollout.vllm_rollout.vllm_rollout.BucketedWeightSender", return_value=sender):
            await adapter.update_weights(iter(()), global_steps=1)
        assert server.global_steps == 1
        await server.resume_kv_cache()
        await server.resume_generation()
        assert events == ["pause_and_clear", "load_weights", "clear_new_version_kv", "resume"]
        server.engine.sleep.assert_not_called()
        server.engine.wake_up.assert_not_called()
        assert not server._submission_paused and server._resume_event.is_set()

    asyncio.run(run())
