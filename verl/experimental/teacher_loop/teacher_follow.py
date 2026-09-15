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
"""Stitch suffix-only Teacher prompt_logprobs onto a growing Student prefix.

vLLM `prompt_logprobs` is shift-1 (row i scores token i+1) and always appends a
dummy last row. With `skip_reading_prefix_cache=False`, cached prefix rows are
omitted. This module maps each extract onto `teacher[0:seq_len]` so Actor still
sees the same packed fields as a one-shot Teacher call.

Idle signal is this sequence's previous Teacher request finishing — not a
whole-GPU barrier. Payload is always the full prefix; compute should land on
the uncached suffix.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import torch

# While Student is still decoding, wait until this many new tokens before the
# next Teacher request so one prefill is not split into many tiny ones.
FOLLOW_MIN_NEW_TOKENS = 128

# Follow-only fields written onto extra_fields / TransferQueue tags.
TEACHER_FOLLOW_TRACE_KEYS = (
    "teacher_num_requests",
    "teacher_last_cached_tokens",
    "teacher_last_prefill_s",
    "teacher_first_prefill_s",
)

# Engine timings already produced by the vLLM server under student_* names.
_ENGINE_TIMING_MAP = (
    ("teacher_submit_ts", "student_submit_ts"),
    ("teacher_first_token_ts", "student_first_token_ts"),
    ("teacher_last_token_ts", "student_last_token_ts"),
    ("teacher_engine_queue_s", "engine_queue_s"),
    ("teacher_engine_prefill_s", "engine_prefill_s"),
    ("teacher_engine_decode_s", "engine_decode_s"),
)

TEACHER_EXTRA_KEYS = (
    "teacher_start_ts",
    "teacher_done_ts",
    *(dst for dst, _ in _ENGINE_TIMING_MAP),
    *TEACHER_FOLLOW_TRACE_KEYS,
)


def copy_teacher_engine_timings(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Copy vLLM server timings onto Teacher extra_fields (student_* → teacher_*)."""
    for dst_key, src_key in _ENGINE_TIMING_MAP:
        dst[dst_key] = src.get(src_key)


def merge_teacher_extra(dst: dict[str, Any], src: Optional[dict[str, Any]]) -> None:
    """Copy Teacher extra keys that are present on the follow/one-shot result."""
    if not src:
        return
    for key in TEACHER_EXTRA_KEYS:
        if key in src:
            dst[key] = src[key]


def _as_row_lists(rows: Any) -> list[list[Any]]:
    if rows is None:
        return []
    if isinstance(rows, torch.Tensor):
        return rows.tolist()
    return [list(row) for row in rows]


def unpack_teacher_extract(extra: dict[str, Any]) -> tuple[list[list[Any]], list[list[Any]], Any, Any, int]:
    """Read prompt_logprobs extract + optional decode top-k from server extra_fields."""
    extracted_ids = _as_row_lists(extra.get("prompt_ids"))
    extracted_lps = _as_row_lists(extra.get("prompt_logprobs"))
    decode_ids = extra.get("decode_topk_ids")
    decode_lps = extra.get("decode_topk_logprobs")
    if decode_ids is not None:
        decode_ids = list(decode_ids)
    if decode_lps is not None:
        decode_lps = list(decode_lps)
    num_cached = int(extra.get("num_cached_tokens") or 0)
    return extracted_ids, extracted_lps, decode_ids, decode_lps, num_cached


def valid_teacher_rows(
    seq_len: int,
    num_cached: int,
    extracted_ids: list[list[int]],
    extracted_lps: list[list[float]],
) -> tuple[int, list[list[int]], list[list[float]]]:
    """Map an engine extract to (start, real_ids, real_lps) for teacher[start:seq_len-1].

    Three shapes show up:
    - full-length extract (`n == seq_len`): prefix rows may be zeros; skip `num_cached`
    - suffix-only extract (`n == seq_len - cached`): drop the dummy last row
    - otherwise: treat as dummy-terminated suffix aligned to the end
    """
    n = len(extracted_ids)
    cached = max(int(num_cached or 0), 0)
    if n == 0:
        return min(cached, max(seq_len - 1, 0)), [], []
    if n == seq_len:
        start = min(cached, max(seq_len - 1, 0))
        return start, extracted_ids[start : seq_len - 1], extracted_lps[start : seq_len - 1]
    if cached and n == seq_len - cached:
        return cached, extracted_ids[:-1], extracted_lps[:-1]
    real_ids = extracted_ids[:-1]
    real_lps = extracted_lps[:-1]
    start = max(seq_len - 1 - len(real_ids), 0)
    return start, real_ids, real_lps


# Keep the previous private name so existing tests can import either.
_valid_teacher_rows = valid_teacher_rows


def should_submit_follow(
    new_tokens: int,
    student_done: bool,
    scored_seq_len: int,
    min_new_tokens: int = FOLLOW_MIN_NEW_TOKENS,
) -> bool:
    """Whether to submit the next Teacher request for this sequence.

    First snapshot and the tail after Student finishes go out immediately.
    While Student is still decoding, merge short increments.
    """
    if new_tokens <= 0:
        return False
    if student_done or scored_seq_len == 0:
        return True
    return new_tokens >= min_new_tokens


_should_submit_follow = should_submit_follow


class TeacherFollowGapError(RuntimeError):
    """Prefix cache hid rows this trajectory has not scored yet."""

    def __init__(self, message: str, compute_start: int = 0):
        super().__init__(message)
        self.compute_start = int(compute_start)


class TeacherFollowAccumulator:
    """Stitch suffix-only Teacher rows onto a growing sequence.

    `filled_real` is the first index that is still the dummy last-row (or
    unfilled). Decode top-k from request N fills that boundary when request
    N+1 caches the entire previous prefix.
    """

    def __init__(self, width: int):
        self.width = int(width)
        self.ids: list[list[int]] = []
        self.lps: list[list[float]] = []
        self.filled_real = 0
        self.pending_ids: Optional[list[int]] = None
        self.pending_lps: Optional[list[float]] = None
        self.scored_seq_len = 0

    def _resize(self, seq_len: int) -> None:
        pad_ids = [0] * self.width
        pad_lps = [0.0] * self.width
        while len(self.ids) < seq_len:
            self.ids.append(list(pad_ids))
            self.lps.append(list(pad_lps))

    def _use_full_extract_if_needed(
        self,
        start: int,
        seq_len: int,
        extracted_ids: list[list[int]],
        extracted_lps: list[list[float]],
        real_ids: list[list[int]],
        real_lps: list[list[float]],
    ) -> tuple[int, list[list[int]], list[list[float]]]:
        # Some requests still return a full-length extract. If cache jumped past
        # filled_real, take rows from there instead of leaving a hole.
        if start > self.filled_real + 1 and len(extracted_ids) == seq_len:
            start = self.filled_real
            return start, extracted_ids[start : seq_len - 1], extracted_lps[start : seq_len - 1]
        return start, real_ids, real_lps

    def _close_adjacent_boundary(self, start: int, seq_len: int, num_cached: int, extract_rows: int) -> None:
        if start <= self.filled_real:
            return
        if start == self.filled_real + 1:
            # Dummy last row of the previous request. Prefer decode top-k;
            # otherwise leave zeros rather than treating this as an unscored hole.
            if self.pending_ids is not None and self.pending_lps is not None:
                self.ids[self.filled_real] = list(self.pending_ids)
                self.lps[self.filled_real] = list(self.pending_lps)
            self.filled_real += 1
            return
        raise TeacherFollowGapError(
            f"Teacher follow gap: filled_real={self.filled_real} compute_start={start} "
            f"seq_len={seq_len} num_cached={num_cached} pending={self.pending_ids is not None} "
            f"extract_rows={extract_rows}",
            compute_start=start,
        )

    def apply(
        self,
        seq_len: int,
        num_cached: int,
        extracted_ids: list[list[int]],
        extracted_lps: list[list[float]],
        decode_ids: Optional[list[int]],
        decode_lps: Optional[list[float]],
    ) -> None:
        if seq_len <= 0:
            return
        start, real_ids, real_lps = valid_teacher_rows(seq_len, num_cached, extracted_ids, extracted_lps)
        self._resize(seq_len)
        start, real_ids, real_lps = self._use_full_extract_if_needed(
            start, seq_len, extracted_ids, extracted_lps, real_ids, real_lps
        )
        self._close_adjacent_boundary(start, seq_len, num_cached, len(extracted_ids))
        for offset, (row_ids, row_lps) in enumerate(zip(real_ids, real_lps, strict=False)):
            pos = start + offset
            if pos >= seq_len - 1:
                break
            self.ids[pos] = list(row_ids)
            self.lps[pos] = list(row_lps)
        self.filled_real = max(self.filled_real, min(start + len(real_ids), seq_len - 1))
        self.ids[seq_len - 1] = [0] * self.width
        self.lps[seq_len - 1] = [0.0] * self.width
        if decode_ids is not None and decode_lps is not None:
            self.pending_ids = list(decode_ids)
            self.pending_lps = list(decode_lps)
        self.scored_seq_len = seq_len

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.ids:
            width = self.width
            return (
                torch.zeros((0, width), dtype=torch.int32),
                torch.zeros((0, width), dtype=torch.float32),
            )
        return (
            torch.tensor(self.ids, dtype=torch.int32),
            torch.tensor(self.lps, dtype=torch.float32),
        )


class StudentTokenState:
    """In-process snapshot shared by Student generate() and the Teacher follow loop."""

    def __init__(self):
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.multi_modal_data: Optional[dict[str, Any]] = None
        self.mm_processor_kwargs: Optional[dict[str, Any]] = None
        self.student_done = False
        self.event = asyncio.Event()
        self.ready = asyncio.Event()
        self.loop = asyncio.get_running_loop()

    def set_prompt(
        self,
        prompt_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.prompt_ids = list(prompt_ids)
        self.multi_modal_data = multi_modal_data
        self.mm_processor_kwargs = mm_processor_kwargs
        self.ready.set()
        self.event.set()

    def update_response(self, token_ids: list[int]) -> None:
        # Token-stream RPCs can arrive late or out of order. Never shrink;
        # generate()'s in-process result is the longest list.
        if len(token_ids) < len(self.response_ids):
            return
        self.response_ids = list(token_ids)
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self.loop:
            self.event.set()
        else:
            self.loop.call_soon_threadsafe(self.event.set)

    def mark_done(self) -> None:
        self.student_done = True
        self.event.set()

    def snapshot(self) -> list[int]:
        return list(self.prompt_ids) + list(self.response_ids)

    async def wait_until_submittable(self, scored_seq_len: int) -> None:
        """Wait for more tokens or Student done. Do not busy-loop on a short increment."""
        self.event.clear()
        new_tokens = len(self.snapshot()) - scored_seq_len
        if should_submit_follow(new_tokens, self.student_done, scored_seq_len):
            return
        if self.student_done:
            return
        await self.event.wait()
