"""CPU check that FSDP1 publication with one FULL_STATE_DICT gather sends the same weights.

Three gloo ranks wrap Qwen3-0.6B (real weights, tied embeddings) the way FSDPEngine does for the
actor (FSDP1, FULL_SHARD, use_orig_params=False, bf16 param / fp32 reduce, one unit per decoder
layer, SHARDED_STATE_DICT default), take an AdamW step, and then compare the old path
(SHARDED state_dict + DTensor.full_tensor) with FSDPEngine.get_per_tensor_param: names and order,
dtype, device, shape and exact values, plus that parameters, optimizer state, the module's
state_dict type and the resharded flat params are left as they were. No GPU is used.

Usage: CUDA_VISIBLE_DEVICES= python scripts/check_publication_single_gather.py [--model Qwen/Qwen3-0.6B]
"""

import argparse
import os
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType
from torch.distributed.tensor import DTensor

WORLD = 3


def old_per_tensor_param(module):
    """The pre-change FSDP1 path: SHARDED state_dict, then full_tensor() per DTensor."""
    from verl.utils.model import convert_weight_keys

    params = module.state_dict()
    params = convert_weight_keys(params, getattr(module, "_fsdp_wrapped_module", module))
    return (
        (name, param.to("cpu", non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
        for name, param in params.items()
    )


def snapshot(model, optim):
    flat = [h.flat_param.detach().clone() for h in model._all_handles]
    state = [{k: v.clone() for k, v in s.items()} for s in optim.state.values()]
    return flat, state


def same_snapshot(a, b):
    return all(torch.equal(x, y) for x, y in zip(a[0], b[0], strict=True)) and all(
        s1.keys() == s2.keys() and all(torch.equal(s1[k], s2[k]) for k in s1)
        for s1, s2 in zip(a[1], b[1], strict=True)
    )


def worker(rank, model_path):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29631", RANK=str(rank), WORLD_SIZE=str(WORLD))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    torch.manual_seed(0)

    from transformers import AutoModelForCausalLM

    import verl.workers.engine.fsdp.transformer_impl as impl
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy

    # The engine stages FSDP1 params on the accelerator; on CPU there is nothing to move.
    impl.load_fsdp_model_to_gpu = lambda module: None
    impl.get_device_id = lambda: "cpu"
    impl.log_gpu_memory_usage = lambda *a, **k: None

    hf = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    tied = hf.config.tie_word_embeddings
    mesh = init_device_mesh("cpu", (WORLD,), mesh_dim_names=("fsdp",))
    model = FSDP(
        hf,
        auto_wrap_policy=get_fsdp_wrap_policy(module=hf, config={"min_num_params": 0}, is_lora=False),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        ),
        device_id=torch.device("cpu"),
        device_mesh=mesh,
        use_orig_params=False,
    )
    FSDP.set_state_dict_type(model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig())
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ids = torch.randint(0, hf.config.vocab_size, (1, 64), generator=torch.Generator().manual_seed(rank))
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    optim.step()
    optim.zero_grad(set_to_none=True)

    stub = types.SimpleNamespace(
        module=model,
        _uses_fsdp2_cpu_offload_policy=False,
        _is_offload_param=False,
        _qat_enabled=False,
        model_config=types.SimpleNamespace(lora={}, hf_config=hf.config),
    )
    with torch.no_grad():
        before = snapshot(model, optim)
        eval_ids = torch.randint(0, hf.config.vocab_size, (1, 32), generator=torch.Generator().manual_seed(7))
        logits_before = model(input_ids=eval_ids).logits

        old = list(old_per_tensor_param(model))
        new_gen, peft_cfg = impl.FSDPEngine.get_per_tensor_param(stub)
        new = list(new_gen)

        after = snapshot(model, optim)
        logits_after = model(input_ids=eval_ids).logits

    names_old, names_new = [n for n, _ in old], [n for n, _ in new]
    bad = [
        n
        for (n, a), (_, b) in zip(old, new, strict=True)
        if not (a.dtype == b.dtype and a.device == b.device and a.shape == b.shape and torch.equal(a, b))
    ]
    new_d = dict(new)
    res = {
        "rank": rank,
        "tensors": len(new),
        "names_and_order_equal": names_old == names_new,
        "value_dtype_shape_mismatch": bad,
        "dtypes": sorted({str(t.dtype) for _, t in new}),
        "any_dtensor_in_new": any(isinstance(t, DTensor) for _, t in new),
        "tie_word_embeddings": tied,
        "lm_head_in_both": ("lm_head.weight" in names_old, "lm_head.weight" in names_new),
        "lm_head_equals_embed": (
            torch.equal(new_d["lm_head.weight"], new_d["model.embed_tokens.weight"])
            if "lm_head.weight" in new_d
            else None
        ),
        "params_and_optim_unchanged": same_snapshot(before, after),
        "forward_unchanged": torch.equal(logits_before, logits_after),
        "state_dict_type_restored": FSDP.get_state_dict_type(model).state_dict_type == StateDictType.SHARDED_STATE_DICT,
        "flat_params_resharded": all(
            h.flat_param.data_ptr() == h.flat_param._local_shard.data_ptr() for h in model._all_handles
        ),
        "full_mb": round(sum(t.numel() * t.element_size() for _, t in new) / 2**20, 1),
        "peft_config": peft_cfg,
    }
    out = [None] * WORLD
    dist.all_gather_object(out, res)
    if rank == 0:
        for r in out:
            print(r)
        ok = all(
            r["names_and_order_equal"]
            and not r["value_dtype_shape_mismatch"]
            and not r["any_dtensor_in_new"]
            and r["params_and_optim_unchanged"]
            and r["forward_unchanged"]
            and r["state_dict_type_restored"]
            and r["flat_params_resharded"]
            for r in out
        )
        print("PASS" if ok else "FAIL")
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()
    mp.spawn(worker, args=(args.model,), nprocs=WORLD, join=True)
