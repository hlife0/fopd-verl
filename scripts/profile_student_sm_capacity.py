"""Measure one TP2 Student SD replica at a caller-selected MPS limit.

Run in a fresh process for each limit. Uses fixed prompts and greedy decoding;
no Teacher or Actor is active. Two warmups, three measured repeats by default.
"""
import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--fixed-length', type=int, default=0, help='Force this response length for a capacity-only control')
    args = parser.parse_args()
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = 'Qwen/Qwen3-0.6B'
    tokenizer = AutoTokenizer.from_pretrained(model)
    rows = pq.read_table(args.data).slice(0, args.batch_size).to_pylist()
    prompts = [{'prompt_token_ids': tokenizer.apply_chat_template(
        row['prompt'], tokenize=True, add_generation_prompt=True, return_dict=False
    )} for row in rows]
    llm = LLM(
        model=model, dtype='bfloat16', tensor_parallel_size=2,
        gpu_memory_utilization=0.4, max_model_len=3073, max_num_seqs=32,
        enable_prefix_caching=False, seed=42,
        speculative_config=dict(method='eagle3', model='GavinLucky/SGLang-EAGLE3-Qwen3-0.6B-SpecForge',
                                num_speculative_tokens=3, draft_tensor_parallel_size=2,
                                rejection_sample_method='standard', draft_sample_method='greedy'),
        compilation_config=dict(cudagraph_mode='PIECEWISE', cudagraph_capture_sizes=[4,8,16,32,64,96,128]),
    )
    params = SamplingParams(temperature=0, max_tokens=args.fixed_length or 2048,
                            min_tokens=args.fixed_length, ignore_eos=bool(args.fixed_length), seed=42)
    results = []
    reference = None
    for step in range(5):
        start = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        seconds = time.perf_counter() - start
        tokens = [o.outputs[0].token_ids for o in outputs]
        if reference is None:
            reference = tokens
        row = dict(step=step+1, seconds=seconds, tokens=sum(map(len, tokens)),
                   same_tokens_as_first=tokens == reference)
        results.append(row)
        print(json.dumps(row), flush=True)
    Path(args.output).write_text(json.dumps(dict(
        fixed_length=args.fixed_length,
        active_thread_percentage=os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', '100'),
        visible_sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
        steps=results, token_ids=reference,
    ), indent=2))


if __name__ == '__main__':
    main()
