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
"""CPU tests for Teacher follow stitching."""

from typing import Any

import torch

from verl.experimental.teacher_loop.teacher_follow import (
    TeacherFollowAccumulator,
    TeacherFollowGapError,
    _should_submit_follow,
    follow_submit_len,
    _valid_teacher_rows,
)
from verl.experimental.teacher_loop.teacher_manager import _get_teacher_sampling_params


def teacher_alignment_report(
    ids_a: torch.Tensor,
    lps_a: torch.Tensor,
    ids_b: torch.Tensor,
    lps_b: torch.Tensor,
) -> dict[str, Any]:
    report = {
        "shape_ok": tuple(ids_a.shape) == tuple(ids_b.shape) and tuple(lps_a.shape) == tuple(lps_b.shape),
        "top1_mismatch": 0,
        "set_mismatch": 0,
        "n": int(ids_a.shape[0]) if ids_a.ndim >= 1 else 0,
    }
    if not report["shape_ok"]:
        return report
    for i in range(ids_a.shape[0]):
        if int(ids_a[i, 0]) != int(ids_b[i, 0]):
            report["top1_mismatch"] += 1
        if set(int(t) for t in ids_a[i].tolist()) != set(int(t) for t in ids_b[i].tolist()):
            report["set_mismatch"] += 1
    return report


def teacher_topk_equivalent(
    ids_a: torch.Tensor,
    lps_a: torch.Tensor,
    ids_b: torch.Tensor,
    lps_b: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> tuple[bool, str]:
    if ids_a.shape != ids_b.shape or lps_a.shape != lps_b.shape:
        return False, f"shape mismatch {tuple(ids_a.shape)}/{tuple(lps_a.shape)} vs {tuple(ids_b.shape)}/{tuple(lps_b.shape)}"
    if ids_a.shape[0] != lps_a.shape[0]:
        return False, "ids/logprobs length mismatch"
    for i in range(ids_a.shape[0]):
        map_a = {int(t): float(p) for t, p in zip(ids_a[i].tolist(), lps_a[i].tolist())}
        map_b = {int(t): float(p) for t, p in zip(ids_b[i].tolist(), lps_b[i].tolist())}
        if set(map_a) != set(map_b):
            return False, f"row {i} top-k set {sorted(map_a)} != {sorted(map_b)}"
        for token_id, lp_a in map_a.items():
            lp_b = map_b[token_id]
            if abs(lp_a - lp_b) > atol + rtol * abs(lp_b):
                return False, f"row {i} token {token_id} logprob {lp_a} vs {lp_b}"
    return True, ""


def _oneshot_rows(seq_len: int, width: int) -> tuple[list[list[int]], list[list[float]]]:
    ids, lps = [], []
    for i in range(seq_len - 1):
        ids.append([1000 + i, 2000 + i][:width] + [0] * max(0, width - 2))
        lps.append([-0.1 * (i + 1), -0.2 * (i + 1)][:width] + [0.0] * max(0, width - 2))
        if width >= 2:
            ids[-1] = [1000 + i, 2000 + i]
            lps[-1] = [-0.1 * (i + 1), -0.2 * (i + 1)]
    ids.append([0] * width)
    lps.append([0.0] * width)
    return ids, lps


def test_valid_rows_full_extract_uses_cached_offset():
    ids, lps = _oneshot_rows(6, 2)
    start, real_ids, real_lps = _valid_teacher_rows(6, num_cached=4, extracted_ids=ids, extracted_lps=lps)
    assert start == 4
    assert real_ids == ids[4:5]
    assert real_lps == lps[4:5]


def test_valid_rows_suffix_extract():
    ids = [[5, 6], [0, 0]]
    lps = [[-0.5, -0.6], [0.0, 0.0]]
    start, real_ids, real_lps = _valid_teacher_rows(6, num_cached=4, extracted_ids=ids, extracted_lps=lps)
    assert start == 4
    assert real_ids == [[5, 6]]
    assert real_lps == [[-0.5, -0.6]]


def test_accumulator_exact_cache_uses_pending_boundary():
    width = 2
    oneshot_ids, oneshot_lps = _oneshot_rows(6, width)
    acc = TeacherFollowAccumulator(width)
    first_ids, first_lps = _oneshot_rows(4, width)
    acc.apply(
        seq_len=4,
        num_cached=0,
        extracted_ids=first_ids,
        extracted_lps=first_lps,
        decode_ids=oneshot_ids[3],
        decode_lps=oneshot_lps[3],
    )
    suffix_ids = oneshot_ids[4:]  # row 4 real + dummy, if treated as suffix of len 2
    suffix_lps = oneshot_lps[4:]
    acc.apply(
        seq_len=6,
        num_cached=4,
        extracted_ids=suffix_ids,
        extracted_lps=suffix_lps,
        decode_ids=[9, 10],
        decode_lps=[-0.9, -1.0],
    )
    got_ids, got_lps = acc.finalize()
    ok, msg = teacher_topk_equivalent(
        got_ids, got_lps, torch.tensor(oneshot_ids), torch.tensor(oneshot_lps)
    )
    assert ok, msg


def test_accumulator_partial_cache_overwrites_overlap():
    width = 2
    oneshot_ids, oneshot_lps = _oneshot_rows(6, width)
    acc = TeacherFollowAccumulator(width)
    first_ids, first_lps = _oneshot_rows(4, width)
    acc.apply(4, 0, first_ids, first_lps, oneshot_ids[3], oneshot_lps[3])
    acc.apply(6, 2, oneshot_ids, oneshot_lps, [9, 10], [-0.9, -1.0])
    got_ids, got_lps = acc.finalize()
    ok, msg = teacher_topk_equivalent(
        got_ids, got_lps, torch.tensor(oneshot_ids), torch.tensor(oneshot_lps)
    )
    assert ok, msg


def test_should_submit_follow_aligns_to_kv_block():
    # Wait until the live prefix fills at least one KV block.
    assert _should_submit_follow(10, student_done=False, scored_seq_len=0) is False
    assert _should_submit_follow(16, student_done=False, scored_seq_len=0) is True
    # scored=20, seq=30 → aligned 16, nothing new.
    assert _should_submit_follow(10, student_done=False, scored_seq_len=20) is False
    # scored=20, seq=32 → aligned 32.
    assert _should_submit_follow(12, student_done=False, scored_seq_len=20) is True
    # leftover 5 tokens stay until Student finishes.
    assert _should_submit_follow(5, student_done=False, scored_seq_len=32) is False
    assert _should_submit_follow(5, student_done=True, scored_seq_len=32) is True
    assert _should_submit_follow(0, student_done=True, scored_seq_len=32) is False
    assert follow_submit_len(45, student_done=False, scored_seq_len=0) == 32
    assert follow_submit_len(48, student_done=False, scored_seq_len=32) == 48
    assert follow_submit_len(53, student_done=False, scored_seq_len=48) == 0
    assert follow_submit_len(53, student_done=True, scored_seq_len=48) == 53


def test_follow_sampling_params_enable_prefix_read():
    teacher_cfg = type("T", (), {"inference": type("I", (), {"temperature": 1.0})()})()
    loss_cfg = type(
        "L",
        (),
        {"topk": 64, "loss_settings": type("S", (), {"use_topk": True})()},
    )()
    baseline = _get_teacher_sampling_params(teacher_cfg, loss_cfg, follow=False)
    follow = _get_teacher_sampling_params(teacher_cfg, loss_cfg, follow=True)
    assert "skip_reading_prefix_cache" not in baseline
    assert follow["skip_reading_prefix_cache"] is False
    assert follow["logprobs"] == 64
    assert "logprobs" not in baseline
    gap_fill = _get_teacher_sampling_params(teacher_cfg, loss_cfg, follow=False, need_decode_topk=True)
    assert "skip_reading_prefix_cache" not in gap_fill
    assert gap_fill["logprobs"] == 64
    k1_cfg = type(
        "L",
        (),
        {"topk": 64, "loss_settings": type("S", (), {"use_topk": False})()},
    )()
    k1_follow = _get_teacher_sampling_params(teacher_cfg, k1_cfg, follow=True)
    assert k1_follow["prompt_logprobs"] == 0
    assert k1_follow["logprobs"] == 1
    assert k1_follow["skip_reading_prefix_cache"] is False


def test_accumulator_adjacent_cache_without_pending_keeps_length():
    acc = TeacherFollowAccumulator(1)
    first_ids, first_lps = _oneshot_rows(4, 1)
    acc.apply(4, 0, first_ids, first_lps, None, None)
    suffix_ids = [[5], [0]]
    suffix_lps = [[-0.5], [0.0]]
    acc.apply(6, 4, suffix_ids, suffix_lps, None, None)
    got_ids, _ = acc.finalize()
    assert got_ids.shape[0] == 6


def test_accumulator_raises_on_cache_gap_without_full_extract():
    acc = TeacherFollowAccumulator(2)
    first_ids, first_lps = _oneshot_rows(4, 2)
    acc.apply(4, 0, first_ids, first_lps, [7, 8], [-0.7, -0.8])
    suffix_ids = [[9, 10], [0, 0]]
    suffix_lps = [[-0.9, -1.0], [0.0, 0.0]]
    try:
        acc.apply(8, 6, suffix_ids, suffix_lps, [11, 12], [-1.1, -1.2])
    except TeacherFollowGapError as exc:
        assert exc.compute_start == 6
        return
    raise AssertionError("expected TeacherFollowGapError")


def test_gap_hole_fill_then_suffix_matches_oneshot():
    width = 2
    oneshot_ids, oneshot_lps = _oneshot_rows(8, width)
    acc = TeacherFollowAccumulator(width)
    first_ids, first_lps = _oneshot_rows(4, width)
    acc.apply(4, 0, first_ids, first_lps, oneshot_ids[3], oneshot_lps[3])
    suffix_ids = oneshot_ids[6:]
    suffix_lps = oneshot_lps[6:]
    try:
        acc.apply(8, 6, suffix_ids, suffix_lps, [11, 12], [-1.1, -1.2])
    except TeacherFollowGapError as exc:
        assert exc.compute_start == 6
    else:
        raise AssertionError("expected TeacherFollowGapError")
    hole_ids, hole_lps = _oneshot_rows(6, width)
    acc.apply(6, 0, hole_ids, hole_lps, oneshot_ids[5], oneshot_lps[5])
    acc.apply(8, 6, suffix_ids, suffix_lps, [11, 12], [-1.1, -1.2])
    got_ids, got_lps = acc.finalize()
    ok, msg = teacher_topk_equivalent(
        got_ids, got_lps, torch.tensor(oneshot_ids), torch.tensor(oneshot_lps)
    )
    assert ok, msg


def test_student_token_state_ignores_shorter_update():
    import asyncio

    from verl.experimental.teacher_loop.teacher_follow import StudentTokenState

    async def _run():
        state = StudentTokenState()
        state.set_prompt([1, 2, 3])
        state.update_response([10, 11, 12])
        state.update_response([10])
        assert state.response_ids == [10, 11, 12]
        state.update_response([10, 11, 12, 13])
        assert state.response_ids == [10, 11, 12, 13]

    asyncio.run(_run())


def test_teacher_topk_equivalent_detects_shift():
    ids_a = torch.tensor([[1, 2], [3, 4], [0, 0]])
    lps_a = torch.tensor([[-0.1, -0.2], [-0.3, -0.4], [0.0, 0.0]])
    ids_b = torch.tensor([[3, 4], [1, 2], [0, 0]])
    lps_b = torch.tensor([[-0.3, -0.4], [-0.1, -0.2], [0.0, 0.0]])
    ok, _ = teacher_topk_equivalent(ids_a, lps_a, ids_b, lps_b)
    assert not ok


def test_teacher_topk_equivalent_rejects_set_mismatch_with_same_top1():
    ids_a = torch.tensor([[1, 2, 3], [0, 0, 0]])
    lps_a = torch.tensor([[-0.1, -0.2, -0.3], [0.0, 0.0, 0.0]])
    ids_b = torch.tensor([[1, 2, 4], [0, 0, 0]])
    lps_b = torch.tensor([[-0.1, -0.2, -0.4], [0.0, 0.0, 0.0]])
    ok, msg = teacher_topk_equivalent(ids_a, lps_a, ids_b, lps_b)
    assert not ok
    assert "top-k set" in msg
    report = teacher_alignment_report(ids_a, lps_a, ids_b, lps_b)
    assert report["shape_ok"] and report["top1_mismatch"] == 0 and report["set_mismatch"] == 1


def test_follow_request_priority_puts_finished_first():
    from verl.experimental.teacher_loop.teacher_follow import FOLLOW_MID_PRIORITY, follow_request_priority

    assert FOLLOW_MID_PRIORITY > 0
    assert follow_request_priority(student_done=False, prioritize_done=True) == FOLLOW_MID_PRIORITY
    assert follow_request_priority(student_done=True, prioritize_done=True) == 0
    assert follow_request_priority(student_done=False, prioritize_done=False) == 0
    assert follow_request_priority(student_done=True, prioritize_done=False) == 0


def _run_follow_with_recorded_priorities(prioritize_done: bool) -> tuple[list[tuple[int, int]], torch.Tensor]:
    import asyncio
    from types import SimpleNamespace

    from verl.experimental.teacher_loop.teacher_follow import StudentTokenState
    from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager

    calls: list[tuple[int, int]] = []

    class FakeClient:
        async def generate(self, request_id, *, prompt_ids, sampling_params, priority=0, **kwargs):
            calls.append((len(prompt_ids), priority))
            ids, lps = _oneshot_rows(len(prompt_ids), 1)
            extra = {
                "prompt_ids": ids + [[0]],
                "prompt_logprobs": lps + [[0.0]],
                "num_cached_tokens": 0,
            }
            return SimpleNamespace(extra_fields=extra)

    manager = object.__new__(AsyncTeacherLLMServerManager)
    manager.distillation_config = SimpleNamespace(
        teacher_follow_priority=prioritize_done, teacher_follow_final_overtakes_mid=False
    )
    manager.distillation_loss_config = SimpleNamespace(topk=0, loss_settings=SimpleNamespace(use_topk=False))
    manager.teacher_model_configs = {"t": SimpleNamespace(inference=SimpleNamespace(temperature=1.0))}
    manager.teacher_client = {"t": FakeClient()}
    manager.follow_mid_gate = None

    async def _run():
        state = StudentTokenState()
        state.set_prompt(list(range(8)))
        state.update_response(list(range(100, 116)))
        task = asyncio.create_task(manager.compute_teacher_logprobs_follow(state, request_id="r"))
        while not calls:
            await asyncio.sleep(0)
        # Let the mid-follow hop return, then finish the Student sequence.
        for _ in range(5):
            await asyncio.sleep(0)
        state.mark_done()
        teacher_ids, _, _ = await task
        return teacher_ids

    teacher_ids = asyncio.run(_run())
    return calls, teacher_ids


def test_follow_loop_sends_mid_hops_at_lower_priority_when_enabled():
    calls, teacher_ids = _run_follow_with_recorded_priorities(prioritize_done=True)
    # 24 tokens: the mid hop sends the block-aligned 16, the final request the full 24.
    assert calls == [(16, 1), (24, 0)]
    assert teacher_ids.shape[0] == 24

    calls, teacher_ids = _run_follow_with_recorded_priorities(prioritize_done=False)
    assert calls == [(16, 0), (24, 0)]
    assert teacher_ids.shape[0] == 24


def test_configure_teacher_follow_replicas_sets_priority_policy():
    from types import SimpleNamespace

    from verl.experimental.teacher_loop.teacher_model import _configure_teacher_follow_replicas

    def _configure(prioritize_done: bool) -> dict:
        rollout_config = SimpleNamespace(engine_kwargs={}, gpu_memory_utilization=0.9)
        distillation_config = SimpleNamespace(
            teacher_follow_min_gpu_memory_utilization=0.75, teacher_follow_priority=prioritize_done
        )
        _configure_teacher_follow_replicas([SimpleNamespace()], rollout_config, distillation_config)
        return rollout_config.engine_kwargs["vllm"]

    assert _configure(True)["scheduling_policy"] == "priority"
    assert "scheduling_policy" not in _configure(False)


def test_follow_mid_gate_caps_requests_and_tokens():
    from verl.experimental.teacher_loop.teacher_follow import FollowMidGate

    gate = FollowMidGate(max_requests=2, max_tokens=100)
    # One hop is always admitted, even above the token cap.
    assert gate.try_acquire(500)
    assert not gate.try_acquire(1)
    gate.release(500)
    assert gate.try_acquire(60)
    assert not gate.try_acquire(41)
    assert gate.try_acquire(40)
    assert not gate.try_acquire(0)
    gate.release(60)
    gate.release(40)
    assert (gate.requests, gate.tokens) == (0, 0)

    unlimited = FollowMidGate()
    assert all(unlimited.try_acquire(10_000) for _ in range(100))


def test_follow_mid_gate_wakes_on_release_or_event():
    import asyncio

    from verl.experimental.teacher_loop.teacher_follow import FollowMidGate

    async def _run():
        gate = FollowMidGate(max_requests=1)
        assert gate.try_acquire(16)
        event = asyncio.Event()
        waiter = asyncio.create_task(gate.wait_release_or(event))
        await asyncio.sleep(0)
        assert not waiter.done()
        gate.release(16)
        await asyncio.wait_for(waiter, 1.0)

        assert gate.try_acquire(16)
        waiter = asyncio.create_task(gate.wait_release_or(event))
        await asyncio.sleep(0)
        event.set()
        await asyncio.wait_for(waiter, 1.0)
        assert gate._waiters == []

    asyncio.run(_run())


def test_make_follow_mid_gate_splits_over_workers():
    from types import SimpleNamespace

    from verl.experimental.teacher_loop.teacher_manager import _make_follow_mid_gate

    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(agent=SimpleNamespace(num_workers=8)))
    )

    def _dc(follow=True, requests=0, tokens=0):
        return SimpleNamespace(
            teacher_follow=follow, teacher_follow_mid_max_requests=requests, teacher_follow_mid_max_tokens=tokens
        )

    assert _make_follow_mid_gate(_dc(), config) is None
    assert _make_follow_mid_gate(_dc(follow=False, requests=8), config) is None
    gate = _make_follow_mid_gate(_dc(requests=12, tokens=1000), config)
    assert (gate.max_requests, gate.max_tokens) == (2, 125)
    gate = _make_follow_mid_gate(_dc(requests=4), config)
    assert (gate.max_requests, gate.max_tokens) == (1, 0)


class _CachingTeacher:
    """Fake Teacher with a block prefix cache; each request waits until released.

    compute() runs a pending request (reads and fills the cache) without
    returning it yet, like a hop whose step finished before its reply arrives.
    """

    def __init__(self, width: int, block: int = 16):
        self.width = width
        self.block = block
        self.cached: set[tuple[int, ...]] = set()
        self.pending: dict[str, list] = {}
        self.calls: list[tuple[str, int, int]] = []

    def _num_cached(self, prompt_ids: list[int]) -> int:
        k = (len(prompt_ids) - 1) // self.block * self.block
        while k > 0 and tuple(prompt_ids[:k]) not in self.cached:
            k -= self.block
        return k

    async def generate(self, request_id, *, prompt_ids, sampling_params, priority=0, **kwargs):
        import asyncio
        from types import SimpleNamespace

        entry = [list(prompt_ids), asyncio.Event(), None]
        self.calls.append((request_id, len(prompt_ids), priority))
        self.pending.setdefault(request_id, []).append(entry)
        await entry[1].wait()
        return SimpleNamespace(extra_fields=entry[2])

    def _entry(self, request_id: str, payload_len: int):
        for entry in self.pending.get(request_id, []):
            if len(entry[0]) == payload_len:
                return entry
        raise AssertionError(f"no pending request {request_id}/{payload_len}")

    def compute(self, request_id: str, payload_len: int) -> None:
        entry = self._entry(request_id, payload_len)
        if entry[2] is not None:
            return
        prompt_ids = entry[0]
        n = len(prompt_ids)
        cached = self._num_cached(prompt_ids)
        ids, lps = _oneshot_rows(n + 1, self.width)
        for k in range(self.block, n + 1, self.block):
            self.cached.add(tuple(prompt_ids[:k]))
        entry[2] = {
            "prompt_ids": ids[cached : n - 1] + [[0] * self.width],
            "prompt_logprobs": lps[cached : n - 1] + [[0.0] * self.width],
            "num_cached_tokens": cached,
            "decode_topk_ids": ids[n - 1],
            "decode_topk_logprobs": lps[n - 1],
        }

    def release(self, request_id: str, payload_len: int) -> None:
        self.compute(request_id, payload_len)
        entry = self._entry(request_id, payload_len)
        self.pending[request_id].remove(entry)
        entry[1].set()


def _follow_manager(teacher, *, width: int, gate=None, overtake: bool = False):
    from types import SimpleNamespace

    from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager

    manager = object.__new__(AsyncTeacherLLMServerManager)
    manager.distillation_config = SimpleNamespace(
        teacher_follow_priority=True, teacher_follow_final_overtakes_mid=overtake
    )
    manager.distillation_loss_config = SimpleNamespace(topk=width, loss_settings=SimpleNamespace(use_topk=True))
    manager.teacher_model_configs = {"t": SimpleNamespace(inference=SimpleNamespace(temperature=1.0))}
    manager.teacher_client = {"t": teacher}
    manager.follow_mid_gate = gate
    return manager


async def _settle():
    import asyncio

    for _ in range(20):
        await asyncio.sleep(0)


def _assert_matches_oneshot(ids, lps, seq_len: int, width: int) -> None:
    want_ids, want_lps = _oneshot_rows(seq_len, width)
    ok, msg = teacher_topk_equivalent(ids, lps, torch.tensor(want_ids), torch.tensor(want_lps))
    assert ok, msg


def test_follow_mid_gate_holds_mid_hops_but_not_finished_requests():
    import asyncio

    from verl.experimental.teacher_loop.teacher_follow import FollowMidGate, StudentTokenState

    width = 2
    teacher = _CachingTeacher(width)
    manager = _follow_manager(teacher, width=width, gate=FollowMidGate(max_requests=1))

    async def _run():
        a, b = StudentTokenState(), StudentTokenState()
        a.set_prompt(list(range(8)))
        b.set_prompt(list(range(50, 58)))
        a.update_response(list(range(100, 132)))
        b.update_response(list(range(200, 232)))
        task_a = asyncio.create_task(manager.compute_teacher_logprobs_follow(a, request_id="a"))
        task_b = asyncio.create_task(manager.compute_teacher_logprobs_follow(b, request_id="b"))
        await _settle()
        # A's block-aligned mid hop holds the only slot; B's hop waits at the gate.
        assert teacher.calls == [("a", 32, 1)]
        # B finishes while A's hop is still in flight: its final request goes out at once.
        b.mark_done()
        await _settle()
        assert teacher.calls == [("a", 32, 1), ("b", 40, 0)]
        teacher.release("b", 40)
        ids_b, lps_b, _ = await asyncio.wait_for(task_b, 1.0)
        # A keeps decoding; its next hop waits for its own in-flight hop, then reuses the cache.
        a.update_response(list(range(100, 148)))
        teacher.release("a", 32)
        await _settle()
        assert teacher.calls[-1] == ("a", 48, 1)
        a.mark_done()
        teacher.release("a", 48)
        await _settle()
        assert teacher.calls[-1] == ("a", 56, 0)
        teacher.release("a", 56)
        ids_a, lps_a, _ = await asyncio.wait_for(task_a, 1.0)
        assert (manager.follow_mid_gate.requests, manager.follow_mid_gate.tokens) == (0, 0)
        return (ids_a, lps_a, 56), (ids_b, lps_b, 40)

    for ids, lps, seq_len in asyncio.run(_run()):
        _assert_matches_oneshot(ids, lps, seq_len, width)


def _run_finish_during_hop(overtake: bool, hop_computed_first: bool):
    import asyncio
    import json

    from verl.experimental.teacher_loop.teacher_follow import StudentTokenState

    width = 2
    teacher = _CachingTeacher(width)
    manager = _follow_manager(teacher, width=width, overtake=overtake)

    async def _run():
        state = StudentTokenState()
        state.set_prompt(list(range(8)))
        state.update_response(list(range(100, 124)))
        task = asyncio.create_task(manager.compute_teacher_logprobs_follow(state, request_id="r"))
        await _settle()
        teacher.release("r", 32)
        state.update_response(list(range(100, 140)))
        await _settle()
        assert teacher.calls[-1] == ("r", 48, 1)
        if hop_computed_first:
            teacher.compute("r", 48)
        state.update_response(list(range(100, 145)))
        state.mark_done()
        await _settle()
        if not overtake:
            # The final request waits for the in-flight hop.
            assert teacher.calls[-1] == ("r", 48, 1)
            teacher.release("r", 48)
            await _settle()
        assert teacher.calls[-1] == ("r", 53, 0)
        teacher.release("r", 53)
        await _settle()
        if overtake and hop_computed_first:
            # The final request read the hop's blocks, so it waits for the hop's rows.
            assert not task.done()
            teacher.release("r", 48)
        ids, lps, extra = await asyncio.wait_for(task, 1.0)
        return ids, lps, json.loads(extra["teacher_requests"])

    ids, lps, rows = asyncio.run(_run())
    _assert_matches_oneshot(ids, lps, 53, width)
    return teacher, rows


def test_follow_final_waits_for_hop_without_overtake():
    _, rows = _run_finish_during_hop(overtake=False, hop_computed_first=False)
    assert [(r["payload_len"], r["overtaken"]) for r in rows] == [(32, False), (48, False), (53, False)]


def test_follow_final_overtakes_queued_hop_and_drops_it():
    teacher, rows = _run_finish_during_hop(overtake=True, hop_computed_first=False)
    # The final request recomputed from the stitched prefix, so the hop was dropped.
    assert [(r["payload_len"], r["overtaken"]) for r in rows] == [(32, False), (48, True), (53, False)]
    assert rows[-1]["num_cached"] == 32


def test_follow_final_overtakes_computed_hop_and_stitches_it_first():
    _, rows = _run_finish_during_hop(overtake=True, hop_computed_first=True)
    assert [(r["payload_len"], r["overtaken"]) for r in rows] == [(32, False), (48, False), (53, False)]
    assert rows[-1]["num_cached"] == 48
