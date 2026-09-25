"""One Student replica: PIECEWISE vs FULL_AND_PIECEWISE with EAGLE3 k=3.

Batch 16, Qwen3-0.6B, the recipe draft. Component window is 128 new tokens,
not the training cap of 2048. Prints whether draft-decode graphs are captured.
"""

import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")

MODE = sys.argv[1]
DRAFT = "GavinLucky/SGLang-EAGLE3-Qwen3-0.6B-SpecForge"
PROMPTS = [f"Question {i}: what is {i}+{i}? Answer briefly.\n" for i in range(16)]


def main() -> None:
    from vllm import LLM, SamplingParams

    print(f"MODE_BEGIN {MODE}", flush=True)
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=16,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=True,
        speculative_config={
            "method": "eagle3",
            "model": DRAFT,
            "num_speculative_tokens": 3,
            "draft_tensor_parallel_size": 1,
            "rejection_sample_method": "standard",
            "draft_sample_method": "greedy",
        },
        compilation_config={
            "cudagraph_mode": MODE,
            "cudagraph_capture_sizes": [4, 8, 16],
        },
    )
    params = SamplingParams(temperature=0, max_tokens=128, seed=1, logprobs=1)
    for _ in range(1):
        llm.generate(PROMPTS[:1], params)
    torch_sync = __import__("torch").cuda.synchronize
    torch_sync()
    t0 = time.perf_counter()
    outputs = llm.generate(PROMPTS, params)
    torch_sync()
    elapsed = time.perf_counter() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outputs)
    ids = [list(o.outputs[0].token_ids) for o in outputs]
    print(
        f"MODE_RESULT {MODE} elapsed_s={elapsed:.3f} tokens={ntok} "
        f"tok_per_s={ntok / elapsed:.1f} lens={[len(x) for x in ids]}",
        flush=True,
    )
    # Compact id fingerprint so two modes can be compared without dumping text.
    out_path = f"/tmp/sdg-ids-{MODE}.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in ids:
            fh.write(",".join(str(token) for token in row) + "\n")
    print("MODE_IDS_FILE", out_path, flush=True)


if __name__ == "__main__":
    main()
