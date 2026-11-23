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
"""CPU coverage for fixed-Teacher next-batch prompt scoring reuse."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from verl.experimental.teacher_loop.teacher_follow import StudentTokenState, follow_submit_len
from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager


def _row(token):
    return [1000 + token, 2000 + token], [-0.1 * (token + 1), -0.2 * (token + 1)]


def _rows(sequence):
    rows = [_row(token) for token in sequence[:-1]]
    return [ids for ids, _ in rows] + [[0, 0]], [lps for _, lps in rows] + [[0.0, 0.0]]


def _k1_rows(sequence):
    return (
        [[token] for token in sequence[1:]] + [[0]],
        [[-0.01 * (previous * 7 + token)] for previous, token in zip(sequence[:-1], sequence[1:])] + [[0.0]],
    )


class _Teacher:
    """Immediate fixed Teacher; each sticky session has its own block cache."""

    def __init__(self):
        self.cached = {}
        self.calls = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        read_cache = sampling_params.get("skip_reading_prefix_cache") is False
        self.calls.append((request_id, list(prompt_ids), read_cache))
        blocks = self.cached.setdefault(request_id, set())
        cached = (len(prompt_ids) - 1) // 16 * 16 if read_cache else 0
        while cached and tuple(prompt_ids[:cached]) not in blocks:
            cached -= 16
        for end in range(16, len(prompt_ids) + 1, 16):
            blocks.add(tuple(prompt_ids[:end]))
        if sampling_params["prompt_logprobs"] == 0:
            ids, lps = _k1_rows(prompt_ids)
            # Deliberately different from the actual Student's next token.
            decode_ids, decode_lps = [99999], [-999.0]
        else:
            ids, lps = _rows(prompt_ids)
            decode_ids, decode_lps = _row(prompt_ids[-1])
        return SimpleNamespace(
            extra_fields={
                "prompt_ids": ids[cached:],
                "prompt_logprobs": lps[cached:],
                "num_cached_tokens": cached,
                "decode_topk_ids": decode_ids,
                "decode_topk_logprobs": decode_lps,
            }
        )


def _manager(teachers=None, *, use_topk=True):
    manager = object.__new__(AsyncTeacherLLMServerManager)
    manager.distillation_config = SimpleNamespace(
        teacher_follow_priority=False, teacher_follow_final_overtakes_mid=False
    )
    manager.distillation_loss_config = SimpleNamespace(topk=2, loss_settings=SimpleNamespace(use_topk=use_topk))
    manager.teacher_key = "teacher"
    manager.teacher_client = teachers or {"t": _Teacher()}
    manager.teacher_model_configs = {
        key: SimpleNamespace(inference=SimpleNamespace(temperature=1.0)) for key in manager.teacher_client
    }
    manager.follow_mid_gate = None
    manager._prompt_lookahead_full_determinism = False
    manager._prompt_lookahead_batch_id = None
    manager._prompt_lookahead_entries = {}
    return manager


def _state(prompt, response):
    state = StudentTokenState()
    state.set_prompt(prompt)
    state.update_response(response)
    state.mark_done()
    return state


def _assert_rows(ids, lps, sequence, *, use_topk=True):
    expected_ids, expected_lps = _rows(sequence) if use_topk else _k1_rows(sequence)
    assert torch.equal(ids, torch.tensor(expected_ids, dtype=torch.int32))
    torch.testing.assert_close(lps, torch.tensor(expected_lps, dtype=torch.float32))


@pytest.mark.parametrize("prompt_len", [16, 32, 33, 35])
@pytest.mark.parametrize("use_topk", [False, True])
def test_prompt_seed_matches_oneshot_without_prefix_repair(prompt_len, use_topk):
    async def run():
        teacher = _Teacher()
        manager = _manager({"t": teacher}, use_topk=use_topk)
        prompt, response = list(range(prompt_len)), [100, 101, 102, 103]
        manager.begin_prompt_lookahead("2")
        assert await manager.prime_prompt_lookahead(prompt, batch_id="2", request_id="prime")
        ids, lps, extra = await manager.compute_teacher_logprobs_follow(
            _state(prompt, response), "different-student-session", lookahead_batch_id="2"
        )
        _assert_rows(ids, lps, prompt + response, use_topk=use_topk)
        primed_len = prompt_len if use_topk else (follow_submit_len(prompt_len, False, 0) or min(prompt_len, 15))
        assert extra["teacher_prompt_lookahead_tokens"] == primed_len
        assert extra["teacher_last_cached_tokens"] == primed_len // 16 * 16
        if not use_topk:
            _, seed = manager._prompt_lookahead_entries[("t", tuple(prompt))]
            assert seed.pending_ids is None
            assert seed.pending_lps is None
            assert ids[prompt_len - 1].tolist() == [response[0]]
        # One prompt prefill and one suffix request; no missing-score repair.
        assert [(request, read_cache) for request, _, read_cache in teacher.calls] == [
            ("prime", False),
            ("prime", True),
        ]
        assert extra["teacher_num_requests"] == 1

    asyncio.run(run())


def test_prompt_seed_clones_scores_and_boundary_for_multiple_rollouts():
    async def run():
        teacher = _Teacher()
        manager = _manager({"t": teacher})
        prompt = list(range(32))
        manager.begin_prompt_lookahead("2")
        await manager.prime_prompt_lookahead(prompt, batch_id="2", request_id="prime")
        for response in ([100, 101, 102, 103], [200, 201, 202, 203, 204]):
            ids, lps, extra = await manager.compute_teacher_logprobs_follow(
                _state(prompt, response), "student-session", lookahead_batch_id="2"
            )
            _assert_rows(ids, lps, prompt + response)
            assert extra["teacher_prompt_lookahead_tokens"] == len(prompt)
        _, seed = manager._prompt_lookahead_entries[("t", tuple(prompt))]
        assert seed.scored_seq_len == 32
        assert seed.filled_real == 31
        assert seed.pending_ids == _row(prompt[-1])[0]
        assert len(teacher.calls) == 3

    asyncio.run(run())


def test_prompt_seed_requires_exact_batch_teacher_and_prompt():
    async def run():
        manager = _manager({"a": _Teacher(), "b": _Teacher()})
        prompt = list(range(32))
        manager.begin_prompt_lookahead("2")
        await manager.prime_prompt_lookahead(prompt, batch_id="2", request_id="prime", routing_key="a")
        state = _state(prompt, [100])
        assert manager._prompt_lookahead_seed(state, "2", "a") is not None
        assert manager._prompt_lookahead_seed(state, "3", "a") is None
        assert manager._prompt_lookahead_seed(state, "2", "b") is None
        assert manager._prompt_lookahead_seed(_state(prompt[:-1] + [99], [100]), "2", "a") is None
        state.multi_modal_data = {"images": [object()]}
        assert manager._prompt_lookahead_seed(state, "2", "a") is None
        state.multi_modal_data = None
        state.mm_processor_kwargs = {"setting": "different"}
        assert manager._prompt_lookahead_seed(state, "2", "a") is None
        manager.begin_prompt_lookahead("3")
        assert manager._prompt_lookahead_entries == {}
        assert manager._prompt_lookahead_seed(_state(prompt, [100]), "3", "a") is None

    asyncio.run(run())


def test_unmarked_follow_keeps_baseline_behavior():
    async def run():
        teacher = _Teacher()
        manager = _manager({"t": teacher})
        prompt, response = list(range(32)), [100, 101]
        manager.begin_prompt_lookahead("2")
        await manager.prime_prompt_lookahead(prompt, batch_id="2", request_id="prime")
        ids, lps, extra = await manager.compute_teacher_logprobs_follow(_state(prompt, response), "ordinary")
        _assert_rows(ids, lps, prompt + response)
        assert "teacher_prompt_lookahead_tokens" not in extra
        assert teacher.calls[-1][0] == "ordinary"
        assert extra["teacher_last_cached_tokens"] == 0

    asyncio.run(run())


def test_replaced_batch_does_not_receive_late_prefill_result():
    async def run():
        teacher = _Teacher()
        started, release = asyncio.Event(), asyncio.Event()
        original = teacher.generate

        async def delayed(*args, **kwargs):
            started.set()
            await release.wait()
            return await original(*args, **kwargs)

        teacher.generate = delayed
        manager = _manager({"t": teacher})
        manager.begin_prompt_lookahead("2")
        task = asyncio.create_task(
            manager.prime_prompt_lookahead(list(range(32)), batch_id="2", request_id="prime")
        )
        await started.wait()
        manager.begin_prompt_lookahead("3")
        release.set()
        assert await task is False
        assert manager._prompt_lookahead_entries == {}

    asyncio.run(run())


def test_prompt_lookahead_rejects_nonunique_engine_request_ids():
    manager = _manager()
    manager._prompt_lookahead_full_determinism = True
    with pytest.raises(ValueError, match="full_determinism"):
        manager.begin_prompt_lookahead("2")


def test_prompt_lookahead_rejects_missing_boundary_topk():
    async def run():
        teacher = _Teacher()
        original = teacher.generate

        async def missing_boundary(*args, **kwargs):
            result = await original(*args, **kwargs)
            result.extra_fields.pop("decode_topk_ids")
            return result

        teacher.generate = missing_boundary
        manager = _manager({"t": teacher})
        manager.begin_prompt_lookahead("2")
        with pytest.raises(RuntimeError, match="boundary top-k"):
            await manager.prime_prompt_lookahead(list(range(32)), batch_id="2", request_id="prime")
        assert manager._prompt_lookahead_entries == {}

    asyncio.run(run())
