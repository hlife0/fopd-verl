"""One captured FSDP6 batch, three resident replays, then one normal update.

Imported only by an isolated source copy. GPU setup stays inside the allocation.
The existing OverlapActorExecution owns limited compute and primary NCCL streams.
"""
import json
import math
import os
from pathlib import Path
import random
import time


def tensor_comparison(reference, current):
    """All local gradient elements; bounded temporary CPU memory."""
    import torch
    if reference.keys() != current.keys():
        raise ValueError("gradient coverage changed")
    ref2 = delta2 = cur2 = 0.0
    max_abs = 0.0
    finite = True
    count = 0
    for key, ref in reference.items():
        cur = current[key]
        if ref.shape != cur.shape:
            raise ValueError("gradient shape changed")
        a, b = ref.reshape(-1), cur.reshape(-1)
        for begin in range(0, a.numel(), 1048576):
            x = a[begin:begin + 1048576].double()
            y = b[begin:begin + 1048576].double()
            finite = finite and bool(torch.isfinite(x).all() and torch.isfinite(y).all())
            delta = y - x
            ref2 += float(x.square().sum())
            cur2 += float(y.square().sum())
            delta2 += float(delta.square().sum())
            if delta.numel():
                max_abs = max(max_abs, float(delta.abs().max()))
            count += x.numel()
    return dict(reference_sq=ref2, current_sq=cur2, difference_sq=delta2,
                max_abs=max_abs, finite=finite, elements=count)


def _rng():
    import numpy as np
    import torch
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state())


def _set_rng(state):
    import numpy as np
    import torch
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state(state[3])


def capture(worker, data):
    if not os.environ.get("OPD_ACTOR_QUOTA_DIR") or getattr(worker, "_quota_replaying", False):
        return
    if getattr(worker, "_quota_completed", False):
        raise RuntimeError("quota measurement allows one optimizer batch only")
    if not hasattr(worker, "_quota_chunks"):
        worker._quota_chunks = []
    # Preserve actual TensorDict and non-tensor metadata after train defaults.
    worker._quota_chunks.append((data.detach().clone().cpu(), _rng()))


def _local_tensor(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _parameters(engine):
    return {name: _local_tensor(p.detach()).cpu().clone()
            for name, p in engine.module.named_parameters()}


def _gradients(engine):
    values, missing = {}, []
    for name, parameter in engine.module.named_parameters():
        if not parameter.requires_grad or not _local_tensor(parameter).numel():
            continue
        if parameter.grad is None:
            missing.append(name)
        else:
            values[name] = _local_tensor(parameter.grad.detach()).cpu().clone()
    return values, missing


def _loss(value):
    if isinstance(value, (list, tuple)):
        return sum(_loss(x) for x in value)
    if hasattr(value, "aggregate"):
        value = value.aggregate()
    return float(value)


def replay(worker):
    output_dir = os.environ.get("OPD_ACTOR_QUOTA_DIR")
    if not output_dir:
        return
    import torch
    import torch.distributed as dist
    import ray
    from verl.utils import tensordict_utils as tu
    from verl.utils.overlap_actor import create_overlap_control, stream_sm_count

    engine = worker.engine
    group = engine.get_data_parallel_group()
    rank = dist.get_rank(group)
    device = torch.cuda.current_device()

    def agree(ok, reason):
        flag = torch.tensor(int(bool(ok)), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
        if not flag.item():
            raise RuntimeError(reason)

    agree(dist.get_world_size(group) == 6, "requires original FSDP6")
    agree(worker._held_train_ctx is not None and worker._held_loss_normalization_tokens is None,
          "requires held train with explicit full-batch token denominator")
    agree(not engine._is_offload_param and not engine._is_offload_optimizer,
          "resident parameters and optimizer required; offload recovery unmeasured")
    agree(getattr(engine, "scaler", None) is None, "GradScaler replay not supported")
    agree(getattr(engine, "overlap_actor_execution", None) is None,
          "existing overlap execution cannot be replaced")
    chunks = worker._quota_chunks
    agree(bool(chunks), "no captured Actor inputs")
    denominators = [float(tu.get(data, "batch_num_tokens_override", default=0)) for data, _ in chunks]
    agree(len(set(denominators)) == 1 and denominators[0] > 0, "invalid token denominator")
    denominator = denominators[0]
    counts = torch.tensor([
        sum(float(data["loss_mask"].sum()) for data, _ in chunks),
        sum(sum(bool(mask.any()) for mask in data["loss_mask"].unbind()) for data, _ in chunks),
    ], device=device, dtype=torch.float64)
    dist.all_reduce(counts, group=group)
    agree(counts[0].item() == denominator and counts[1].item() == 96, "requires complete B96 loss masks")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / f"rank-{rank}.json"
    if manifest_path.exists():
        raise RuntimeError("refusing to overwrite quota result")
    manifest = dict(schema="actor-quota-v1", rank=rank, world_size=6, batch_size=96,
                    tokens=denominator, local_chunks=len(chunks), target_sms=64,
                    work_counts=dict(capture_full=1, replay_full=2, replay_limited=1, optimizer=0),
                    scope="resident hot service; compute limited; FSDP/NCCL primary streams",
                    offload_recovery_measured=False, phases=[], status="prepared")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    # Same-process weights remain live and unchanged through all quotas. Persist
    # only new real inputs/RNG; these self-produced files are not loaded here.
    torch.save(chunks, target / f"rank-{rank}-chunks.pt")
    weights = _parameters(engine)
    final_rng = _rng()
    reference = None
    worker._quota_replaying = True
    execution = None
    control = None
    try:
        setup_start = time.perf_counter()
        total = torch.cuda.get_device_properties(device).multi_processor_count
        agree(total == 132, "requires the requested H800 132-SM device")
        engine.configure_overlap_actor(64 / total)
        execution = engine.overlap_actor_execution
        agree(execution.limited_sms == 64 and stream_sm_count(execution.primary) == 132,
              "actual context resource query must report limited64 / primary132")
        control = create_overlap_control() if rank == 0 else None
        manifest.update(actual_limited_sms=execution.limited_sms, primary_sms=stream_sm_count(execution.primary),
                        setup_seconds=time.perf_counter() - setup_start)
        for index, phase in enumerate(("full_before", "limited64", "full_after")):
            engine.overlap_actor_execution = execution if phase == "limited64" else None
            if phase == "limited64":
                if rank == 0:
                    ray.get(control.begin.remote(index))
                execution.begin(control, index)
            engine.optimizer_zero_grad()
            worker._held_output_lst = []
            torch.cuda.synchronize()
            dist.barrier(group=group)
            torch.cuda.reset_peak_memory_stats()
            seconds = 0.0
            losses = []
            for data, rng in chunks:
                # Transfers/cloning/RNG reset are outside hot service timing.
                replay_data = data.clone().to(device)
                _set_rng(rng)
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = worker.accumulate_held_train(replay_data)
                torch.cuda.synchronize()
                seconds += time.perf_counter() - start
                losses.append(_loss(tu.get(result, "metrics")["loss"]))
                del replay_data, result
            restore_seconds = 0.0
            if phase == "limited64":
                start = time.perf_counter()
                if rank == 0:
                    ray.get(control.full.remote(index))
                execution.finish()
                engine.overlap_actor_execution = None
                torch.cuda.synchronize()
                restore_seconds = time.perf_counter() - start
            current, missing = _gradients(engine)
            agree(not missing and bool(current), "missing trainable local gradients")
            if reference is None:
                reference = current
            stats = tensor_comparison(reference, current)
            aggregate = torch.tensor([stats["reference_sq"], stats["difference_sq"], stats["current_sq"]],
                                     dtype=torch.float64, device=device)
            dist.all_reduce(aggregate, group=group)
            norm2, delta2, current2 = aggregate.tolist()
            relative_l2 = math.sqrt(delta2 / norm2) if norm2 else (0.0 if delta2 == 0 else math.inf)
            agree(stats["finite"] and all(math.isfinite(x) for x in losses), "nonfinite loss or gradient")
            # A strict engineering acceptance threshold, not a training-quality claim.
            agree(relative_l2 <= 0.001, "gradient relative L2 exceeds 1e-3")
            unchanged = all(torch.equal(weights[name], _local_tensor(p.detach()).cpu())
                            for name, p in engine.module.named_parameters())
            agree(unchanged, "parameters changed before the single optimizer update")
            timing = torch.tensor([seconds, restore_seconds], dtype=torch.float64, device=device)
            dist.all_reduce(timing, op=dist.ReduceOp.MAX, group=group)
            phase_info = dict(name=phase, local_service_seconds=seconds,
                              max_rank_service_seconds=timing[0].item(),
                              max_rank_hot_switch_seconds=timing[1].item(),
                              gradient_relative_l2=relative_l2, gradient_norm=math.sqrt(current2),
                              local_gradient_elements=stats["elements"], local_max_abs=stats["max_abs"],
                              losses=losses, parameters_unchanged=unchanged,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved())
            manifest["phases"].append(phase_info)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            if current is not reference:
                del current
        manifest["status"] = "replays_passed_pending_single_update"
        worker._quota_completed = True
    except BaseException as exc:
        manifest.update(status="failed_no_update", error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        engine.overlap_actor_execution = None
        worker._quota_replaying = False
        _set_rng(final_rng)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        # Final full-resource gradients and outputs remain for normal finish.
        worker._quota_manifest = (manifest_path, manifest)
        worker._quota_chunks = []


def updated(worker):
    if not os.environ.get("OPD_ACTOR_QUOTA_DIR"):
        return
    path, manifest = worker._quota_manifest
    manifest["work_counts"]["optimizer"] = 1
    manifest["status"] = "completed_one_update"
    path.write_text(json.dumps(manifest, indent=2))
