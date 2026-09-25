#!/usr/bin/env python3
"""GPU probe: vLLM 0.24 prompt_logprobs prefix reuse vs one-shot.

Uses one GPU (default CUDA_VISIBLE_DEVICES=1). Prints cache hits and the
max absolute difference of the scored-token logprob. Does not record
environment, secrets, or checksums.
"""

from __future__ import annotations

import asyncio
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from vllm import SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM


def _params(skip: bool | None) -> SamplingParams:
    kwargs = dict(
        max_tokens=1,
        temperature=1.0,
        prompt_logprobs=1,
        logprobs=1,
        detokenize=False,
    )
    if skip is not None:
        kwargs["skip_reading_prefix_cache"] = skip
    return SamplingParams(**kwargs)


async def _one(engine: AsyncLLM, token_ids: list[int], request_id: str, skip: bool | None):
    params = _params(skip)
    final = None
    async for out in engine.generate(
        TokensPrompt(prompt_token_ids=token_ids),
        params,
        request_id,
    ):
        final = out
    assert final is not None and final.prompt_logprobs is not None
    nonempty = 0
    chosen: dict[int, float] = {}
    for i, row in enumerate(final.prompt_logprobs):
        if not row:
            continue
        nonempty += 1
        tok = token_ids[i]
        item = row.get(tok)
        if item is not None:
            chosen[i] = float(item.logprob)
    return {
        "id": request_id,
        "prompt_len": len(token_ids),
        "skip": params.skip_reading_prefix_cache,
        "cached": int(final.num_cached_tokens or 0),
        "logprob_rows": len(final.prompt_logprobs),
        "nonempty": nonempty,
        "chosen": chosen,
        "rows": final.prompt_logprobs,
    }


def _report(row: dict) -> None:
    print(
        f"req id={row['id']} prompt_len={row['prompt_len']} skip={row['skip']} "
        f"cached={row['cached']} logprob_rows={row['logprob_rows']} nonempty={row['nonempty']}",
        flush=True,
    )


def _chosen_abs(token_ids: list[int], rows: list, shift: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for i, row in enumerate(rows):
        if not row:
            continue
        pos = i + shift
        if 0 <= pos < len(token_ids) and token_ids[pos] in row:
            out[pos] = float(row[token_ids[pos]].logprob)
    return out


def _aligned_diff(hop: dict[int, float], ref: dict[int, float], shift: int) -> tuple[int, float, float]:
    diffs = [abs(lp - ref[i + shift]) for i, lp in hop.items() if (i + shift) in ref]
    if not diffs:
        return 0, float("nan"), float("nan")
    diffs.sort()
    mid = diffs[len(diffs) // 2]
    return len(diffs), mid, diffs[-1]


def _match_formula(token_ids: list[int], rows: list, cached: int) -> None:
    """Count which absolute token index each returned row actually scores."""
    formulas = {
        "i": lambda i: i,
        "cached+i": lambda i: cached + i,
        "cached+i-1": lambda i: cached + i - 1,
        "end+i": lambda i: len(token_ids) - len(rows) + i,
    }
    counts = {name: 0 for name in formulas}
    nonempty = 0
    for i, row in enumerate(rows):
        if not row:
            continue
        nonempty += 1
        for name, fn in formulas.items():
            pos = fn(i)
            if 0 <= pos < len(token_ids) and token_ids[pos] in row:
                counts[name] += 1
    print(f"row_match nonempty={nonempty} " + " ".join(f"{k}={v}" for k, v in counts.items()), flush=True)


async def main() -> None:
    args = AsyncEngineArgs(
        model="Qwen/Qwen3-0.6B",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.30,
        max_model_len=256,
        enforce_eager=True,
        max_logprobs=5,
        disable_log_stats=True,
    )
    engine = AsyncLLM.from_engine_args(args, usage_context=UsageContext.ENGINE_CONTEXT)
    try:
        tok = engine.get_tokenizer()
        ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 30)[:96]
        assert len(ids) == 96 and len(ids) % 16 == 0

        # Semantic reference first, then drop what it wrote into the cache.
        oneshot = await _one(engine, ids, "oneshot", True)
        _report(oneshot)
        reset_ok = await engine.reset_prefix_cache()
        print(f"reset_prefix_cache={reset_ok}", flush=True)

        hops = []
        for end, req in ((32, "hop32"), (64, "hop64"), (96, "hop96")):
            row = await _one(engine, ids[:end], req, False)
            _report(row)
            _match_formula(ids[:end], row["rows"], row["cached"])
            hops.append(row)
            abs_map = _chosen_abs(ids[:end], row["rows"], row["cached"])
            n, med, worst = _aligned_diff(abs_map, oneshot["chosen"], 0)
            worst_pos = None
            worst_val = -1.0
            for pos, lp in abs_map.items():
                if pos not in oneshot["chosen"]:
                    continue
                diff = abs(lp - oneshot["chosen"][pos])
                if diff >= worst_val:
                    worst_val = diff
                    worst_pos = pos
            print(
                f"align id={row['id']} formula=cached+i positions={n} "
                f"median_abs={med:.6g} max_abs={worst:.6g} max_pos={worst_pos}",
                flush=True,
            )

        # Default flag is True whenever prompt_logprobs is set, so this must miss.
        forced = await _one(engine, ids, "default-skip", None)
        _report(forced)
        print(f"default_skip_resolved={forced['skip'] is None}", flush=True)

        await engine.reset_prefix_cache()
        for end in (32, 64, 96):
            row = await _one(engine, ids[:end], "same-id", False)
            _report(row)
            abs_map = _chosen_abs(ids[:end], row["rows"], row["cached"])
            n, med, worst = _aligned_diff(abs_map, oneshot["chosen"], 0)
            print(
                f"align id=same-id-{end} formula=cached+i positions={n} "
                f"median_abs={med:.6g} max_abs={worst:.6g}",
                flush=True,
            )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
