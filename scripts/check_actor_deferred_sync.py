"""CPU check that cross-chunk deferred gradient sync gives the same FSDP1 gradients and AdamW step.

Three gloo ranks wrap Qwen3-0.6B (real weights) the way FSDPEngine does for the actor (FSDP1,
FULL_SHARD, use_orig_params=False, bf16 param / fp32 reduce / fp32 buffer, one unit per decoder
layer, gradient checkpointing) and run one streamed optimizer step through the real
FSDPEngine.forward_backward_batch / optimizer_step (clip + AdamW), in three modes:

  ref    every chunk reduce-scatters (current path, use_no_sync_for_gradient_accumulation=False)
  defer  defer_grad_sync=True on every chunk but the last (the candidate), after an aborted
         deferred chunk whose gradients must be dropped by optimizer_zero_grad
  bf16   defer without the fp32 upcast, i.e. plain FSDP1 no_sync accumulation, for scale

Chunks have uneven per-rank lengths, one or several micro-batches, and a one-sample-per-rank
tail. Reported per mode vs ref: sharded fp32 grads before clipping, grad_norm, and parameters
after the step (as a fraction of the step's own update), plus the unsharded grad dtype seen
between chunks and the reduce-scatter count per chunk on every rank. No GPU is used; CPU bf16
matmuls are not CUDA's, so this checks the accumulation path, not GPU kernel numerics.
--device cuda runs the same comparison on three GPUs with NCCL and also reports each mode's peak
allocated memory during the step (after the aborted chunk).

Usage: CUDA_VISIBLE_DEVICES= python scripts/check_actor_deferred_sync.py [--model Qwen/Qwen3-0.6B]
       CUDA_VISIBLE_DEVICES=4,5,6 python scripts/check_actor_deferred_sync.py --device cuda
"""

import argparse
import os
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

WORLD = 3
# Per-rank micro-batch groups (sample indices) of each chunk; the last chunk closes the step.
SCENARIOS = {
    "tail1": [[[0, 1]], [[0]], [[0], [1, 2]], [[0]]],
    "tail_multi_mb": [[[0]], [[0, 1]], [[0], [1]]],
}
ABORTED_CHUNK = [[0, 1]]


def make_chunk(groups, rank, seed, vocab, dev):
    """One rank's chunk: uneven lengths per rank and sample, right padded, loss on the tail half."""
    g = torch.Generator().manual_seed(seed * 100 + rank)
    n = sum(len(x) for x in groups)
    lens = torch.randint(12, 48, (n,), generator=g)
    width = int(lens.max())
    ids = torch.randint(0, vocab, (n, width), generator=g)
    attn = torch.zeros(n, width, dtype=torch.long)
    loss_mask = torch.zeros(n, width)
    for i, length in enumerate(lens.tolist()):
        attn[i, :length] = 1
        loss_mask[i, length // 2 : length - 1] = 1
    return TensorDict(
        {"input_ids": ids, "attention_mask": attn, "loss_mask": loss_mask}, batch_size=[n]
    ).to(dev), groups


def rs_counter():
    count = [0]
    orig = dist.reduce_scatter_tensor

    def counted(*args, **kwargs):
        count[0] += 1
        return orig(*args, **kwargs)

    dist.reduce_scatter_tensor = counted
    return count


def worker(rank, model_path, device):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29651", RANK=str(rank), WORLD_SIZE=str(WORLD))
    cuda = device == "cuda"
    if cuda:
        torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank) if cuda else torch.device("cpu")
    dist.init_process_group("nccl" if cuda else "gloo", rank=rank, world_size=WORLD)
    torch.manual_seed(0)

    from transformers import AutoModelForCausalLM

    import verl.workers.engine.fsdp.transformer_impl as impl
    from verl.utils import tensordict_utils as tu
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy

    hf = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vocab = hf.config.vocab_size
    mesh = init_device_mesh(device, (WORLD,), mesh_dim_names=("fsdp",))
    model = FSDP(
        hf,
        auto_wrap_policy=get_fsdp_wrap_policy(module=hf, config={"min_num_params": 0}, is_lora=False),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        ),
        device_id=dev,
        device_mesh=mesh,
        use_orig_params=False,
    )
    handles = [m._handle for m in FSDP.fsdp_modules(model) if m._handle is not None]
    init_params = [h.flat_param.detach().clone() for h in handles]

    # Micro-batch split comes from the chunk's own groups (same count on every rank, as
    # prepare_micro_batches(same_micro_num_in_dp=True) guarantees).
    def split(data, dp_group, same_micro_num_in_dp):
        out = []
        for idx in mb_groups[0]:
            mb = TensorDict({k: data[k][idx] for k in ("input_ids", "attention_mask", "loss_mask")}, [len(idx)])
            tu.assign_non_tensor(mb, batch_num_tokens=tu.get(data, "batch_num_tokens"))
            out.append(mb)
        return out, None

    mb_groups = [None]  # the chunk being run; a list value would become a per-sample NonTensorStack
    impl.prepare_micro_batches = split
    impl.postprocess_batch_func = lambda output_lst, indices, data: output_lst
    rs_count = rs_counter()

    class Engine(impl.FSDPEngine):
        def __init__(self, upcast):
            self.module = model
            self.ulysses_sequence_parallel_size = 1
            self._is_offload_param = False
            self.engine_config = types.SimpleNamespace(use_no_sync_for_gradient_accumulation=False)
            self.scaler = None
            self._qat_enabled = False
            self.optimizer_config = types.SimpleNamespace(clip_grad=1.0)
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            self.upcast = upcast

        def get_data_parallel_group(self):
            return dist.group.WORLD

        def get_data_parallel_size(self):
            return WORLD

        def _upcast_deferred_grads(self):
            if self.upcast:
                super()._upcast_deferred_grads()

        def forward_step(self, micro_batch, loss_function, forward_only):
            ids, attn, mask = micro_batch["input_ids"], micro_batch["attention_mask"], micro_batch["loss_mask"]
            logits = self.module(input_ids=ids, attention_mask=attn, use_cache=False).logits.float()
            tok = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="none"
            )
            # Global-token normalization like batch_num_tokens_override; x dp for FSDP's grad average.
            loss = (tok * mask[:, :-1].reshape(-1)).sum() / tu.get(micro_batch, "batch_num_tokens") * WORLD
            return loss, {"loss": loss.detach()}

    def total_tokens(chunks):
        local = sum(float(td["loss_mask"][:, :-1].sum()) for td, _ in chunks)
        t = torch.tensor([local], dtype=torch.float64, device=dev)
        dist.all_reduce(t)
        return float(t)

    def grad_kind(handle):
        grad = handle.flat_param.grad
        if grad is None:
            return "none"
        shard = "unsharded" if grad.numel() > handle.flat_param.numel() else "sharded"
        return f"{str(grad.dtype)[6:]}/{shard}"

    def run(mode, chunk_groups):
        for h, p in zip(handles, init_params, strict=True):
            h.flat_param.data.copy_(p)
        engine = Engine(upcast=mode != "bf16")
        engine.optimizer.zero_grad()
        info = {"rs_per_chunk": [], "between_chunk_grad": set()}
        if mode != "ref":
            # A deferred chunk that is then aborted: abort_held_train zeroes grads before closing.
            td, groups = make_chunk(ABORTED_CHUNK, rank, 99, vocab, dev)
            mb_groups[0] = groups
            tu.assign_non_tensor(td, batch_num_tokens_override=1000, defer_grad_sync=True)
            engine.forward_backward_batch(td, loss_function=None)
            engine.optimizer_zero_grad()
            assert all(h.flat_param.grad is None for h in handles), "abort left gradients"
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        chunks = [make_chunk(g, rank, i, vocab, dev) for i, g in enumerate(chunk_groups)]
        n_tok = total_tokens(chunks)
        for i, (td, groups) in enumerate(chunks):
            is_last = i == len(chunks) - 1
            mb_groups[0] = groups
            tu.assign_non_tensor(td, batch_num_tokens_override=n_tok)
            if mode != "ref":
                tu.assign_non_tensor(td, defer_grad_sync=not is_last)
            before = rs_count[0]
            engine.forward_backward_batch(td, loss_function=None)
            info["rs_per_chunk"].append(rs_count[0] - before)
            if not is_last:
                info["between_chunk_grad"] |= {grad_kind(h) for h in handles}
        info["final_grad"] = {grad_kind(h) for h in handles}
        grads = [h.flat_param.grad.detach().clone() for h in handles]
        if cuda:
            info["fb_peak_gb"] = round((torch.cuda.max_memory_allocated() - base) / 2**30, 3)
        grad_norm = float(engine.optimizer_step())
        params = [h.flat_param.detach().clone() for h in handles]
        info["rs_per_chunk"] = tuple(info["rs_per_chunk"])
        info["between_chunk_grad"] = sorted(info["between_chunk_grad"])
        info["final_grad"] = sorted(info["final_grad"])
        return grads, grad_norm, params, info

    def diff(a_list, b_list, scale_list=None):
        """Max abs diff, and max abs diff over max abs of the reference (or of the given scale)."""
        mx, ref_mx, sq, ref_sq = 0.0, 0.0, 0.0, 0.0
        for i, (a, b) in enumerate(zip(a_list, b_list, strict=True)):
            d = (a.double() - b.double())
            s = (b if scale_list is None else scale_list[i]).double()
            mx, ref_mx = max(mx, float(d.abs().max())), max(ref_mx, float(s.abs().max()))
            sq, ref_sq = sq + float((d * d).sum()), ref_sq + float((s * s).sum())
        t = torch.tensor([mx, ref_mx, sq, ref_sq], dtype=torch.float64, device=dev)
        dist.all_reduce(t[:2], op=dist.ReduceOp.MAX)
        dist.all_reduce(t[2:])
        return {"max_abs": float(t[0]), "max_abs_rel": float(t[0] / t[1]), "l2_rel": float((t[2] / t[3]).sqrt())}

    report = {}
    for name, chunk_groups in SCENARIOS.items():
        ref = run("ref", chunk_groups)
        update = [p - p0 for p, p0 in zip(ref[2], init_params, strict=True)]
        # ref_again: the path's own run-to-run noise (0 on CPU; GPU kernels may not be deterministic).
        for mode in ("ref_again", "defer", "bf16"):
            out = run("ref" if mode == "ref_again" else mode, chunk_groups)
            report[(name, mode)] = {
                "grad": diff(out[0], ref[0]),
                "grad_norm": (out[1], ref[1]),
                "param_vs_update": diff(out[2], ref[2], update),
                "info": out[3],
                "ref_info": ref[3],
            }

    infos = [None] * WORLD
    dist.all_gather_object(infos, {k: (v["info"], v["ref_info"]) for k, v in report.items()})
    if rank == 0:
        for key, r in report.items():
            print(f"== {key[0]} / {key[1]} vs ref")
            print(f"  grad (sharded fp32, pre-clip): {r['grad']}")
            print(f"  grad_norm: {r['grad_norm'][0]:.9g} vs {r['grad_norm'][1]:.9g}")
            print(f"  params after AdamW, diff / |update|: {r['param_vs_update']}")
            for rk, per_rank in enumerate(infos):
                mine, ref_i = per_rank[key]
                print(f"  rank{rk}: rs/chunk {mine['rs_per_chunk']} (ref {ref_i['rs_per_chunk']}), "
                      f"between chunks {mine['between_chunk_grad']}, final {mine['final_grad']}")
                if cuda:
                    print(f"    F/B peak above step start {mine['fb_peak_gb']} GB (ref {ref_i['fb_peak_gb']} GB)")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()
    mp.spawn(worker, args=(args.model, args.device), nprocs=WORLD)
