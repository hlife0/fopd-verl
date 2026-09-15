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


def test_should_submit_follow_merges_short_increments():
    assert _should_submit_follow(10, student_done=False, scored_seq_len=0) is True
    assert _should_submit_follow(10, student_done=False, scored_seq_len=20) is False
    assert _should_submit_follow(127, student_done=False, scored_seq_len=20) is False
    assert _should_submit_follow(128, student_done=False, scored_seq_len=20) is True
    assert _should_submit_follow(1, student_done=True, scored_seq_len=20) is True
    assert _should_submit_follow(0, student_done=True, scored_seq_len=20) is False


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
