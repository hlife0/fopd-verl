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
import logging
import os
import time
from typing import Any, Optional
from uuid import uuid4

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from verl.experimental.teacher_loop.teacher_follow import (
    StudentTokenState,
    TeacherFollowAccumulator,
    TeacherFollowGapError,
    _should_submit_follow,
    _valid_teacher_rows,
    copy_teacher_engine_timings,
    should_submit_follow,
    unpack_teacher_extract,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Re-export follow types so existing imports from this module keep working.
__all__ = [
    "AsyncTeacherLLMServerManager",
    "StudentTokenState",
    "TeacherFollowAccumulator",
    "TeacherFollowGapError",
    "_get_teacher_sampling_params",
    "_should_submit_follow",
    "_valid_teacher_rows",
]


def _teacher_topk_width(distillation_loss_config: DistillationLossConfig) -> int:
    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    return max(int(num_logprobs or 0), 1)


def _teacher_num_logprobs(distillation_loss_config: DistillationLossConfig) -> int:
    return distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
    follow: bool = False,
    need_decode_topk: bool = False,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    # Temperature has no effect on prompt_logprobs: the teacher performs a forward pass over
    # existing tokens (no sampling). Always use temperature=1.0 regardless of the config value.
    # The default distillation.yaml copies the student rollout temperature via Hydra interpolation
    # (temperature: ${oc.select:actor_rollout_ref.rollout.temperature}), which causes a spurious
    # crash when rollout.temperature != 1.0.
    if teacher_model_config.inference.temperature != 1.0:
        logger.warning(
            "Teacher inference temperature is set to %.1f, but temperature has no effect "
            "on prompt_logprobs (forward pass only). Using temperature=1.0.",
            teacher_model_config.inference.temperature,
        )
    num_logprobs = _teacher_num_logprobs(distillation_loss_config)
    params: dict[str, Any] = {
        "max_tokens": 1,
        "temperature": 1.0,
        "prompt_logprobs": num_logprobs,
        "detokenize": False,
    }
    if follow:
        # vLLM 0.24 defaults skip_reading_prefix_cache=True whenever prompt_logprobs
        # is set, which recomputes the whole prefix. Follow must read the cache.
        params["skip_reading_prefix_cache"] = False
    if follow or need_decode_topk:
        # max_tokens=1 decode row fills the dummy last position of this request
        # so the next cache-hit request does not leave a one-token hole.
        # Needed even when prompt_logprobs=0 (loss_mode without top-k).
        params["logprobs"] = max(int(num_logprobs or 0), 1)
    return params


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        teacher_client: dict[str, LLMServerClient],
    ):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.teacher_key: str = self.distillation_config.teacher_key

        self.teacher_model_configs: dict[str, DistillationTeacherModelConfig] = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(teacher_client.keys()) != expected:
            raise ValueError(
                f"teacher client keys {sorted(teacher_client.keys())} "
                f"do not match teacher routing keys {sorted(expected)}."
            )
        self.teacher_client: dict[str, LLMServerClient] = teacher_client

    def _resolve_teacher_key(self, routing_key: Optional[str]) -> str:
        if len(self.teacher_model_configs) == 1:
            # Single-teacher path: route everything to the one teacher regardless of the sample's key.
            return next(iter(self.teacher_model_configs))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in self.teacher_model_configs:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_key

    async def _teacher_forward(
        self,
        sequence_ids: list[int],
        *,
        request_id: str,
        follow: bool,
        need_decode_topk: bool = False,
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
    ) -> dict:
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        sampling_params = _get_teacher_sampling_params(
            teacher_model_config,
            self.distillation_loss_config,
            follow=follow,
            need_decode_topk=need_decode_topk,
        )
        teacher_output = await client.generate(
            request_id=request_id,
            prompt_ids=sequence_ids,
            sampling_params=dict(sampling_params),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        extra = teacher_output.extra_fields
        if not follow:
            teacher_ids = torch.tensor(extra["prompt_ids"], dtype=torch.int32)
            teacher_logprobs = torch.tensor(extra["prompt_logprobs"])
            assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)
        return extra

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        extra = await self._teacher_forward(
            sequence_ids,
            request_id=uuid4().hex,
            follow=False,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            routing_key=routing_key,
        )
        return (
            torch.tensor(extra["prompt_ids"], dtype=torch.int32),
            torch.tensor(extra["prompt_logprobs"]),
            extra,
        )

    async def _fill_unscored_prefix(
        self,
        acc: TeacherFollowAccumulator,
        seq: list[int],
        compute_start: int,
        request_id: str,
        routing_key: Optional[str],
        state: StudentTokenState,
    ) -> None:
        """Recompute only the jumped prefix (shared-prompt / block-aligned cache)."""
        hole_end = min(max(compute_start, acc.filled_real + 1), len(seq))
        hole_seq = seq[:hole_end]
        hole_extra = await self._teacher_forward(
            hole_seq,
            request_id=request_id,
            follow=False,
            need_decode_topk=True,
            multi_modal_data=state.multi_modal_data,
            mm_processor_kwargs=state.mm_processor_kwargs,
            routing_key=routing_key,
        )
        hole_ids, hole_lps, hole_dec_ids, hole_dec_lps, _ = unpack_teacher_extract(hole_extra)
        acc.apply(
            seq_len=len(hole_seq),
            num_cached=0,
            extracted_ids=hole_ids,
            extracted_lps=hole_lps,
            decode_ids=hole_dec_ids,
            decode_lps=hole_dec_lps,
        )

    async def _apply_follow_extract(
        self,
        acc: TeacherFollowAccumulator,
        seq: list[int],
        extra: dict[str, Any],
        request_id: str,
        routing_key: Optional[str],
        state: StudentTokenState,
    ) -> int:
        """Apply a follow extract. Returns 1 if a hole-fill request was issued."""
        extracted_ids, extracted_lps, decode_ids, decode_lps, num_cached = unpack_teacher_extract(extra)
        try:
            acc.apply(
                seq_len=len(seq),
                num_cached=num_cached,
                extracted_ids=extracted_ids,
                extracted_lps=extracted_lps,
                decode_ids=decode_ids,
                decode_lps=decode_lps,
            )
            return 0
        except TeacherFollowGapError as exc:
            await self._fill_unscored_prefix(acc, seq, exc.compute_start, request_id, routing_key, state)
        try:
            acc.apply(
                seq_len=len(seq),
                num_cached=num_cached,
                extracted_ids=extracted_ids,
                extracted_lps=extracted_lps,
                decode_ids=decode_ids,
                decode_lps=decode_lps,
            )
        except TeacherFollowGapError:
            # Prefix is scored; the next loop iteration submits the remaining suffix.
            logger.warning(
                "Teacher follow hole fill left a suffix gap: filled_real=%s seq_len=%s num_cached=%s",
                acc.filled_real,
                len(seq),
                num_cached,
            )
        return 1

    async def compute_teacher_logprobs_follow(
        self,
        state: StudentTokenState,
        request_id: str,
        routing_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Follow one Student sequence: at most one in-flight Teacher request.

        Idle signal is this sequence's previous Teacher request finishing.
        Payload is always the full current prefix; compute is only the new suffix.
        """
        await state.ready.wait()
        acc = TeacherFollowAccumulator(_teacher_topk_width(self.distillation_loss_config))
        extra_out: dict[str, Any] = {}
        teacher_start_ts = None
        num_requests = 0
        first_prefill_s = None
        last_cached = None
        last_prefill_s = None

        while True:
            seq = state.snapshot()
            new_tokens = len(seq) - acc.scored_seq_len
            if should_submit_follow(new_tokens, state.student_done, acc.scored_seq_len):
                if teacher_start_ts is None:
                    teacher_start_ts = time.time()
                extra = await self._teacher_forward(
                    seq,
                    request_id=request_id,
                    follow=True,
                    multi_modal_data=state.multi_modal_data,
                    mm_processor_kwargs=state.mm_processor_kwargs,
                    routing_key=routing_key,
                )
                extra_calls = await self._apply_follow_extract(acc, seq, extra, request_id, routing_key, state)
                num_requests += 1 + extra_calls
                last_cached = int(extra.get("num_cached_tokens") or 0)
                last_prefill_s = extra.get("engine_prefill_s")
                if first_prefill_s is None:
                    first_prefill_s = last_prefill_s
                extra_out = extra
                copy_teacher_engine_timings(extra_out, extra)
            elif state.student_done:
                break
            else:
                await state.wait_until_submittable(acc.scored_seq_len)

        teacher_ids, teacher_logprobs = acc.finalize()
        extra_out["teacher_start_ts"] = teacher_start_ts
        extra_out["teacher_done_ts"] = time.time()
        extra_out["teacher_num_requests"] = num_requests
        extra_out["teacher_last_cached_tokens"] = last_cached
        extra_out["teacher_last_prefill_s"] = last_prefill_s
        extra_out["teacher_first_prefill_s"] = first_prefill_s
        return teacher_ids, teacher_logprobs, extra_out
