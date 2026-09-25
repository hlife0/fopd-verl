"""Time the k1 teacher score output path and the logprob kernels.

CPU section rebuilds one step of prompt-logprob objects the way vLLM 0.24 does
when detokenize is off and prompt_logprobs is 0, then runs verl's extract.
GPU section times the LM-head-sized gemm against the logsumexp and rank scans.
It does not load the 8B weights.
"""

import argparse
import pickle
import time

import torch

from verl.workers.rollout.vllm_rollout.utils import extract_prompt_logprobs


def _cpu_step(n_seq: int, seq_len: int) -> None:
    from vllm.logprobs import create_prompt_logprobs
    from vllm.v1.engine.logprobs import LogprobsProcessor
    from vllm.v1.outputs import LogprobsTensors

    positions = seq_len - 1
    token_ids = torch.arange(1, n_seq * positions + 1, dtype=torch.int32).reshape(n_seq, positions)
    logprobs = -torch.rand(n_seq, positions, dtype=torch.float32)
    ranks = torch.ones(n_seq, positions, dtype=torch.int32)

    def one(flat: bool, index: int):
        proc = LogprobsProcessor(
            tokenizer=None,
            logprobs=None,
            prompt_logprobs=create_prompt_logprobs(flat),
            cumulative_logprob=None,
            num_logprobs=None,
            num_prompt_logprobs=0,
        )
        tensors = LogprobsTensors(
            logprob_token_ids=token_ids[index].unsqueeze(1),
            logprobs=logprobs[index].unsqueeze(1),
            selected_token_ranks=ranks[index],
        )
        proc._update_prompt_logprobs(tensors)
        return proc.prompt_logprobs

    t0 = time.perf_counter()
    built = [one(False, i) for i in range(n_seq)]
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    flat_built = [one(True, i) for i in range(n_seq)]
    flat_build_s = time.perf_counter() - t0

    class _Out:
        def __init__(self, prompt_logprobs):
            self.prompt_logprobs = prompt_logprobs

    t0 = time.perf_counter()
    extracted = []
    for rows in built:
        result = {}
        extract_prompt_logprobs(_Out(rows), num_prompt_logprobs=0, result_dict=result)
        extracted.append(
            (
                torch.tensor(result["prompt_ids"], dtype=torch.int32),
                torch.tensor(result["prompt_logprobs"]),
            )
        )
    extract_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for rows in flat_built:
        result = {}
        extract_prompt_logprobs(_Out(rows), num_prompt_logprobs=0, result_dict=result)
        torch.tensor(result["prompt_ids"], dtype=torch.int32)
        torch.tensor(result["prompt_logprobs"])
    flat_extract_s = time.perf_counter() - t0

    payload = [{"prompt_ids": ids.tolist(), "prompt_logprobs": lps.tolist()} for ids, lps in extracted]
    t0 = time.perf_counter()
    blob = pickle.dumps(payload)
    pickle_s = time.perf_counter() - t0
    tensor_payload = extracted
    t0 = time.perf_counter()
    tensor_blob = pickle.dumps(tensor_payload)
    tensor_pickle_s = time.perf_counter() - t0

    ids0, lps0 = extracted[0]
    print(
        f"cpu n_seq={n_seq} seq_len={seq_len} positions={n_seq * seq_len} "
        f"dict_build_s={build_s:.4f} flat_build_s={flat_build_s:.4f} "
        f"extract_tensor_s={extract_s:.4f} flat_extract_s={flat_extract_s:.4f} "
        f"pickle_lists_s={pickle_s:.4f} pickle_lists_mb={len(blob) / 1e6:.2f} "
        f"pickle_tensors_s={tensor_pickle_s:.4f} pickle_tensors_mb={len(tensor_blob) / 1e6:.2f} "
        f"row_dtype={lps0.dtype} row_shape={tuple(lps0.shape)} "
        f"chosen_id_ok={int(ids0[0, 0]) == int(token_ids[0, 0])} "
        f"eos_row_ok={int(ids0[-2, 0]) == int(token_ids[0, -1])} "
        f"dummy_ok={int(ids0[-1, 0]) == 0 and float(lps0[-1, 0]) == 0.0}"
    )


def _gpu_kernels(tokens: int, hidden: int, vocab: int, chunk: int) -> None:
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs, _ranks_kernel

    device = torch.device("cuda")
    weight = torch.randn(vocab, hidden, device=device, dtype=torch.bfloat16)
    hidden_states = torch.randn(chunk, hidden, device=device, dtype=torch.bfloat16)
    chosen = torch.randint(0, vocab, (chunk,), device=device, dtype=torch.int64)

    def gemm():
        return torch.nn.functional.linear(hidden_states, weight)

    def logsumexp(logits):
        return compute_token_logprobs(logits, chosen.unsqueeze(1))

    def ranks(logits):
        out = torch.empty(chunk, dtype=torch.int64, device=device)
        _ranks_kernel[(chunk,)](
            out,
            logits,
            logits.stride(0),
            chosen,
            vocab,
            BLOCK_SIZE=8192,
        )
        return out

    logits = gemm()
    for fn in (gemm, lambda: logsumexp(logits), lambda: ranks(logits)):
        fn()
    torch.cuda.synchronize()

    n_chunk = max(tokens // chunk, 1)

    def bench(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_chunk):
            fn()
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    gemm_s = bench(gemm)
    logits = gemm()
    torch.cuda.synchronize()
    lse_s = bench(lambda: logsumexp(logits))
    rank_s = bench(lambda: ranks(logits))
    covered = n_chunk * chunk
    print(
        f"gpu tokens={covered} chunk={chunk} hidden={hidden} vocab={vocab} "
        f"gemm_s={gemm_s:.4f} logsumexp_s={lse_s:.4f} rank_s={rank_s:.4f} "
        f"gemm_us_per_tok={gemm_s / covered * 1e6:.2f} "
        f"logsumexp_us_per_tok={lse_s / covered * 1e6:.2f} "
        f"rank_us_per_tok={rank_s / covered * 1e6:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    # Step 5 of the published 4-GPU baseline: 48 sequences, 58245 prompt+response tokens.
    _cpu_step(48, 1213)
    if args.gpu:
        _gpu_kernels(58245, hidden=4096, vocab=151936, chunk=1024)


if __name__ == "__main__":
    main()
