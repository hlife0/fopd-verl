"""Single diagnostic executor. No optimizer, quota, Green, or generator control.

Import is CPU-safe. Actual execution requires the frozen verl runtime and real
TensorDict payloads; no substitute tensor implementation is provided.
"""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import time
import socket


def persist(path, value):
    def safe(x):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {str(k): safe(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [safe(v) for v in x]
        return x
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(safe(value), indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def preflight():
    missing = [x for x in ('torch', 'tensordict') if importlib.util.find_spec(x) is None]
    return dict(status='blocked_missing_dependencies' if missing else 'dependencies_present_unverified',
                missing=missing, real_payload_validated=False, gpu_validated=False)


class Normalization:
    """One gradient normalization before comparison; abort does not normalize."""
    def __init__(self, denominator, expected_tokens):
        if not all(math.isfinite(x) and x > 0 for x in (denominator, expected_tokens)):
            raise ValueError('positive finite denominator and token count required')
        self.denominator, self.expected = denominator, expected_tokens
        self.applied = False

    def apply(self, observed_tokens, scale_gradients):
        if self.applied:
            raise RuntimeError('normalization already applied')
        if observed_tokens != self.expected:
            raise ValueError('global loss token count differs from fixed input')
        # Mark before the call: a failed/incomplete scale cannot be retried.
        self.applied = True
        scale = self.denominator / observed_tokens
        scale_gradients(scale)
        return scale


def make_groups(records, members, step):
    if len(members) != 96 or len(set(members)) != 96:
        raise ValueError('exactly 96 unique real trajectories required')
    if set(records) != set(members):
        raise ValueError('eligibility mapping must match this captured batch exactly')
    for key in members:
        r = records[key]
        if r['step'] != step or not math.isfinite(r['first_eligible_seen']):
            raise ValueError('wrong step or invalid eligibility clock')
    ordered = sorted(members, key=lambda k: (records[k]['first_eligible_seen'], k))
    return [ordered[i:i+6] for i in range(0, 96, 6)]


def eligible_snapshot(buffer, now):
    """Read-only observation after the existing metadata snapshot is complete."""
    result = {}
    entries = {}
    for key, tag in buffer.partitions['train'].items():
        entries.setdefault(key.split('_')[0], []).append((key, tag))
    for uid in buffer.finished_keys['train']:
        rows = entries.get(uid, [])
        if len(rows) != 1 or uid in buffer.failure_keys['train']:
            continue
        key, tag = rows[0]
        step = buffer.prompt_global_steps['train'].get(uid)
        if (step is None or tag.get('global_steps') != step or tag.get('is_padding', False)
                or tag.get('student_gen_done_ts') is None or tag.get('teacher_done_ts') is None):
            continue
        result[key] = dict(step=step, first_eligible_seen=now)
    return result


def observe(buffer):
    root = os.environ.get('OPD_READINESS_DIR')
    if not root:
        return
    seen = getattr(buffer, '_rb_seen', {})
    now, wall = time.perf_counter(), time.time()
    for key, value in eligible_snapshot(buffer, now).items():
        identity = (value['step'], key)
        if identity not in seen:
            value['first_eligible_seen_wall'] = wall
            seen[identity] = value
            with (Path(root) / 'eligibility.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(trajectory_id=key, **value)) + '\n')
    buffer._rb_seen = seen
    student_tags = {}
    for key, tag in buffer.partitions['train'].items():
        uid = key.split('_')[0]
        step = tag.get('global_steps')
        done = tag.get('student_gen_done_ts')
        if (done is not None and not tag.get('is_padding', False)
                and buffer.prompt_global_steps['train'].get(uid) == step):
            student_tags.setdefault(step, {})[uid] = float(done)
    snapshots = dict(snapshot_start=getattr(buffer, '_rb_snapshot_start', now),
                     snapshot_end=now, wall=wall, driver_host=socket.gethostname(),
                     students=[dict(step=step, count=len(values),
                                    max_student_done_tag_wall=max(values.values()))
                               for step, values in student_tags.items()])
    with (Path(root)/'metadata-snapshots.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(snapshots)+'\n')


def snapshot_begin(buffer):
    buffer._rb_snapshot_start = time.perf_counter()


def student_barrier(buffer, ready):
    root = os.environ.get('OPD_READINESS_DIR')
    if root:
        with (Path(root)/'student-barrier.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(driver_seen_all_students=time.perf_counter(),
                                    wall=time.time(), prompt_ids=sorted(ready)))+'\n')


def start_capture(trainer):
    if not os.environ.get('OPD_READINESS_DIR'):
        raise RuntimeError('isolated measurement requires OPD_READINESS_DIR')
    if getattr(trainer, '_rb_captured', False):
        raise RuntimeError('only one fresh logical batch permitted')
    cfg = trainer.config
    if (cfg.trainer.v1.sync.overlap_actor or not cfg.trainer.v1.sync.early_actor_lite
            or not cfg.trainer.v1.sync.opd_no_task_reward_fast_path
            or cfg.data.train_batch_size != 96 or cfg.actor_rollout_ref.actor.ppo_epochs != 1
            or cfg.actor_rollout_ref.actor.ppo_mini_batch_size != 96
            or cfg.actor_rollout_ref.actor.loss_agg_mode != 'token-mean'):
        raise RuntimeError('requires early_actor_lite, B96, one full-batch epoch, overlap disabled')
    trainer._rb_captured = True
    trainer._rb_chunks = []
    trainer._rb_capture = dict(status='started', calls_started=0, calls_completed=0,
                               optimizer_updates=0, observer_blocked_intervals=[])
    trainer._rb_capture_path = Path(os.environ['OPD_READINESS_DIR']) / 'capture.json'
    if trainer._rb_capture_path.exists():
        raise RuntimeError('refusing to overwrite capture')
    persist(trainer._rb_capture_path, trainer._rb_capture)
    try:
        trainer.actor_rollout_wg.begin_actor_accumulate()
    except BaseException as exc:
        trainer._rb_capture.update(status='begin_failed_no_update', error=repr(exc), abort_attempted=True)
        persist(trainer._rb_capture_path, trainer._rb_capture)
        try:
            trainer.actor_rollout_wg.abort_actor_accumulate()
            trainer._rb_capture['abort_rpc_returned'] = True
        except BaseException as cleanup_exc:
            trainer._rb_capture['abort_error'] = repr(cleanup_exc)
        persist(trainer._rb_capture_path, trainer._rb_capture)
        raise


def capture_group(trainer, chunk):
    # Preserve the original metadata and rank order. The existing caller still
    # performs consumed.update(take); this function never clears or consumes TQ.
    trainer._rb_chunks.append(copy.deepcopy(chunk))
    trainer._rb_capture['calls_started'] += 1
    persist(trainer._rb_capture_path, trainer._rb_capture)
    interval = dict(start=time.perf_counter(), end=None, reason='synchronous original accumulate RPC')
    trainer._rb_capture['observer_blocked_intervals'].append(interval)
    persist(trainer._rb_capture_path, trainer._rb_capture)
    try:
        # Preserve natural original grouping, including the wait for preceding
        # original F/B; this warm capture pass is separate from measured A/B/C.
        output = trainer.actor_rollout_wg.accumulate_actor(chunk)
        trainer._rb_capture['calls_completed'] += 1
        persist(trainer._rb_capture_path, trainer._rb_capture)
        return output
    except BaseException as exc:
        trainer._rb_capture.update(status='failed_no_update', error=repr(exc))
        persist(trainer._rb_capture_path, trainer._rb_capture)
        raise
    finally:
        interval['end'] = time.perf_counter()
        persist(trainer._rb_capture_path, trainer._rb_capture)


def consumed(trainer, take, dispatched):
    with (Path(os.environ['OPD_READINESS_DIR']) / 'consumption.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(step=trainer.global_steps, consumed=list(take),
                                dispatched=list(dispatched), seen=time.perf_counter())) + '\n')


def execute(trainer):
    """Driver phase timers include real RPC, TQ conversion and group calls."""
    root = Path(os.environ['OPD_READINESS_DIR'])
    path = root / 'driver.json'
    if path.exists():
        raise RuntimeError('refusing to overwrite driver results')
    state = dict(status='preparing', phases=[], optimizer_updates=0,
                 grouping='fresh natural capture original chunks; not historical 650063 packing',
                 candidate_order='dispatcher first-observation order; blocked-tail true order unknown',
                 timing_scope='controlled aligned worker service; raw RPC includes probes and added alignment')
    persist(path, state)
    try:
        original = trainer._rb_chunks
        members = [key for chunk in original for key in chunk.keys]
        if len(members) != 96 or len(set(members)) != 96:
            raise ValueError('padding, repeated or incomplete original members unsupported')
        # Generation has ended at this early_actor_lite entry. Wait for every
        # terminal group before any measured A/B/candidate service.
        deadline = time.monotonic() + 60
        while True:
            _, ready = trainer.replay_buffer.peek_actor_overlap_batch(
                'train', [key.split('_')[0] for key in members], trainer.global_steps)
            if set(ready) == set(members):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('complete successful groups unavailable')
            time.sleep(trainer.replay_buffer.poll_interval)
        records = {key: trainer.replay_buffer._rb_seen[(trainer.global_steps, key)] for key in members}
        groups = make_groups(records, members, trainer.global_steps)
        trainer.actor_rollout_wg.abort_actor_accumulate()
        trainer._rb_capture.update(status='completed_no_update', held_context_closed=True)
        persist(trainer._rb_capture_path, trainer._rb_capture)
        n = int(original[0].extra_info['batch_num_tokens_override'])
        c = 96 * int(trainer.config.data.max_response_length)
        if any(int(x.extra_info['batch_num_tokens_override']) != n for x in original):
            raise ValueError('inconsistent original denominators')
        candidate = []
        for keys in groups:
            # Reuse the existing TQ->TensorDict bridge; do not manually concatenate
            # jagged tensors or invent sample metadata. Six ordered rows dispatch
            # one per DP rank. No length repartition is necessary for this witness.
            chunk = trainer.replay_buffer.peek_trajectories('train', keys)
            chunk.extra_info = copy.deepcopy(original[0].extra_info)
            candidate.append(chunk)
        state.update(step=trainer.global_steps, tokens=n, C=c, readiness=records,
                     original_groups=[list(x.keys) for x in original], candidate_groups=groups)
        persist(path, state)
        result = None
        for name, chunks, denom in [('A', original, n), ('B', original, n), ('candidate', candidate, c)]:
            phase = dict(name=name, calls_started=0, calls_completed=0, calls=[])
            state['phases'].append(phase)
            persist(path, state)
            start = time.perf_counter()
            trainer.actor_rollout_wg.readiness_begin(name, n, denom)
            phase['begin_rpc_s'] = time.perf_counter()-start
            for i, chunk in enumerate(chunks):
                chunk = copy.deepcopy(chunk)
                chunk.extra_info.update(rb_members=list(chunk.keys), rb_chunk=i,
                                        batch_num_tokens_override=denom)
                phase['calls_started'] += 1
                persist(path, state)
                t = time.perf_counter()
                trainer.actor_rollout_wg.accumulate_actor(chunk)
                phase['calls'].append(dict(group=i, rpc_s=time.perf_counter()-t))
                phase['calls_completed'] += 1
                persist(path, state)
            phase['through_last_group_s'] = time.perf_counter()-start
            t = time.perf_counter()
            result = trainer.actor_rollout_wg.readiness_close()
            phase['close_rpc_including_diagnostics_s'] = time.perf_counter()-t
            phase['total_including_diagnostics_s'] = time.perf_counter()-start
            phase['completed'] = True
            persist(path, state)
        state['status'] = 'passed_no_update'
        persist(path, state)
        return result
    except BaseException as exc:
        state.update(status='failed_no_update', error=repr(exc))
        persist(path, state)
        raise


def _rng(state=None):
    import numpy as np
    import torch
    if state is None:
        return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state()
    random.setstate(state[0]); np.random.set_state(state[1])
    torch.set_rng_state(state[2]); torch.cuda.set_rng_state(state[3])


def _local(x):
    return x.to_local() if hasattr(x, 'to_local') else x


def _snapshot(engine, gradients):
    result, missing = {}, []
    for key, p in engine.module.named_parameters():
        if gradients and (not p.requires_grad or not _local(p).numel()):
            continue
        value = p.grad if gradients else p
        if value is None:
            missing.append(key)
        else:
            result[key] = _local(value.detach()).cpu().clone()
    return result, missing


def _agree(worker, ok, message):
    import torch
    import torch.distributed as dist
    flag = torch.tensor(int(bool(ok)), device=worker.device_name)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=worker.engine.get_data_parallel_group())
    if not flag.item():
        raise RuntimeError(message)


def worker_begin(worker, name, n, denominator):
    import torch
    import torch.distributed as dist
    engine = worker.engine
    rank = dist.get_rank(engine.get_data_parallel_group())
    if not hasattr(worker, '_rb_state'):
        worker._rb_path = Path(os.environ['OPD_READINESS_DIR']) / f'rank-{rank}.json'
        if worker._rb_path.exists():
            raise RuntimeError('refusing to overwrite rank output')
        worker._rb_state = dict(phases=[], optimizer_updates=0, status='preparing')
    state = worker._rb_state
    phase = dict(name=name, started=True, calls_started=0, calls_completed=0,
                 microbatch_indices=[], normalization_calls=0, members=[], losses=[])
    state['phases'].append(phase)
    worker._rb_phase = phase
    engine._rb_phase = phase
    persist(worker._rb_path, state)
    _agree(worker, name == ('A', 'B', 'candidate')[len(state['phases'])-1], 'invalid phase order')
    _agree(worker, dist.get_world_size(engine.get_data_parallel_group()) == 6, 'requires DP6')
    _agree(worker, getattr(engine, 'overlap_actor_execution', None) is None and
           getattr(engine, 'scaler', None) is None and not engine._is_offload_param and
           not engine._is_offload_optimizer and engine.ulysses_sequence_parallel_size == 1,
           'requires resident SP1 without scaler or overlap execution')
    _agree(worker, worker.engine_config.use_dynamic_bsz and
           not worker.engine_config.use_no_sync_for_gradient_accumulation,
           'requires dynamic batches and actual per-microbatch gradient synchronization')
    _agree(worker, not any(os.environ.get(key) for key in
           ('OPD_ACTOR_QUOTA_DIR', 'CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')),
           'quota/MPS environment must be absent')
    _agree(worker, not any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in engine.module.modules()),
           'active dropout unsupported for regrouping correctness test')
    if name == 'A':
        worker._rb_initial_rng = _rng()
        worker._rb_weights, _ = _snapshot(engine, False)
        worker._rb_inputs = []
    _rng(worker._rb_initial_rng)
    worker._rb_norm = Normalization(denominator, n)
    worker._rb_local_tokens = 0
    worker._rb_rows = {}
    worker._rb_denominator = denominator
    phase['normalization_denominator'] = denominator
    t = time.perf_counter()
    worker.begin_held_train(loss_normalization_tokens=denominator if name == 'candidate' else None)
    torch.cuda.synchronize()
    phase['begin_state_s'] = time.perf_counter()-t
    persist(worker._rb_path, state)


def before_accumulate(worker, data):
    if getattr(worker, '_rb_phase', None) is None:
        return data  # Natural capture F/B before any measured replay phase.
    import torch
    import torch.distributed as dist
    from verl.utils import tensordict_utils as tu
    phase = worker._rb_phase
    started = time.perf_counter()
    phase['calls_started'] += 1
    persist(worker._rb_path, worker._rb_state)
    all_members = tu.get(data, 'rb_members', default=None)
    rank = dist.get_rank(worker.engine.get_data_parallel_group())
    size = len(data)
    _agree(worker, isinstance(all_members, list) and len(all_members) == size*6,
           'rank-local row mapping does not match actual TensorDict batch')
    members = all_members[rank*size:(rank+1)*size]
    actual = tu.get(data, 'rb_actual_keys', default=None)
    _agree(worker, actual == members and tu.get(data, 'rb_rows_independently_checked', default=False),
           'actual TQ key/index/single-key-row mapping differs or was not checked')
    phase.setdefault('actual_key_index_mapping', []).append(dict(
        keys=actual, global_indexes=tu.get(data, 'rb_global_indexes', default=None)))
    for i, key in enumerate(members):
        if key in worker._rb_rows:
            raise ValueError('duplicate actual row')
        # Check every tensor field, including labels, lengths, masks and positions.
        # Per-row hashing avoids concatenating jagged inputs across ranks.
        worker._rb_rows[key] = tensor_row_digest(data, i)
    phase['members'].extend(members)
    worker._rb_local_tokens += float(data['loss_mask'].sum())
    if phase['name'] == 'A':
        worker._rb_inputs.append((data.detach().clone().cpu(), _rng(), members))
        torch.save(worker._rb_inputs, worker._rb_path.with_name(worker._rb_path.stem+'-inputs.pt'))
    elif phase['name'] == 'B':
        _rng(worker._rb_inputs[phase['calls_started']-1][1])
    phase.setdefault('input_probe_s', []).append(time.perf_counter()-started)
    alignment = time.perf_counter()
    torch.cuda.synchronize()
    dist.barrier(group=worker.engine.get_data_parallel_group())
    torch.cuda.synchronize()
    phase.setdefault('added_alignment_s', []).append(time.perf_counter()-alignment)
    worker._rb_service_start = time.perf_counter()
    return data


def after_accumulate(worker, output):
    if getattr(worker, '_rb_phase', None) is None:
        return
    import torch
    from verl.utils import tensordict_utils as tu
    phase = worker._rb_phase
    torch.cuda.synchronize()
    phase.setdefault('controlled_accumulate_service_s', []).append(
        time.perf_counter()-worker._rb_service_start)
    phase['calls_completed'] += 1
    if output is not None:
        value = tu.get(output, 'metrics', default={}).get('loss')
        def scalar(x):
            if isinstance(x, (list, tuple)):
                return sum(scalar(y) for y in x)
            if hasattr(x, 'aggregate'):
                x = x.aggregate()
            return float(x)
        if value is not None:
            phase['losses'].append(scalar(value))
    persist(worker._rb_path, worker._rb_state)


def record_microbatches(engine, indices):
    phase = getattr(engine, '_rb_phase', None)
    if phase is not None:
        if indices is None:
            raise RuntimeError('actual dynamic microbatch indices required')
        phase['microbatch_indices'].append([
            [int(i) for i in idx] for idx in indices])


def worker_close(worker):
    import torch
    import torch.distributed as dist
    engine, phase, state = worker.engine, worker._rb_phase, worker._rb_state
    group = engine.get_data_parallel_group()
    error = None
    output = None
    try:
        tokens = torch.tensor(worker._rb_local_tokens, dtype=torch.float64, device=worker.device_name)
        dist.all_reduce(tokens, group=group)
        phase.update(observed_global_N=tokens.item(), expected_N=worker._rb_norm.expected,
                     held_global_N=None)
        if phase['name'] == 'candidate':
            held = worker._held_loss_tokens.clone()
            dist.all_reduce(held, group=group)
            phase['held_global_N'] = held.item()
        persist(worker._rb_path, state)
        _agree(worker, tokens.item() == worker._rb_norm.expected, 'token coverage mismatch')
        if phase['name'] == 'candidate':
            _agree(worker, held.item() == tokens.item(), 'held token accounting mismatch')
        t = time.perf_counter()
        phase['normalization_attempts'] = 1
        persist(worker._rb_path, state)
        phase['normalization_scale'] = worker._rb_norm.apply(tokens.item(), engine.scale_gradients)
        phase['normalization_calls'] = 1
        torch.cuda.synchronize()
        phase['normalization_s'] = time.perf_counter()-t
        persist(worker._rb_path, state)
        t = time.perf_counter()
        rows = [None]*6
        dist.all_gather_object(rows, worker._rb_rows, group=group)
        merged = {}
        for row in rows:
            if set(merged) & set(row):
                raise ValueError('duplicate member across ranks')
            merged.update(row)
        if phase['name'] == 'A':
            worker._rb_reference_rows = merged
        phase['input_coverage_ok'] = len(merged) == 96 and merged == worker._rb_reference_rows
        current, missing = _snapshot(engine, True)
        phase['missing_gradients'] = missing
        phase['parameters_unchanged'] = all(
            torch.equal(_local(p.detach()).cpu(), worker._rb_weights[key])
            for key, p in engine.module.named_parameters())
        reference = current if phase['name'] == 'A' else worker._rb_reference
        ref2 = delta2 = cur2 = 0.0
        coverage = bool(current) and not missing and current.keys() == reference.keys()
        finite = True
        if coverage:
            for key, ref in reference.items():
                if ref.shape != current[key].shape:
                    coverage = False
                    break
                a, b = ref.reshape(-1), current[key].reshape(-1)
                for offset in range(0, a.numel(), 1048576):
                    x, y = a[offset:offset+1048576].double(), b[offset:offset+1048576].double()
                    finite = finite and bool(torch.isfinite(x).all() and torch.isfinite(y).all())
                    ref2 += float(x.square().sum()); cur2 += float(y.square().sum())
                    delta2 += float((x-y).square().sum())
        values = torch.tensor([ref2, delta2, cur2], dtype=torch.float64, device=worker.device_name)
        dist.all_reduce(values, group=group)
        ref2, delta2, cur2 = values.tolist()
        relative = math.sqrt(delta2/ref2) if ref2 > 0 else (0.0 if delta2 == 0 else math.inf)
        phase.update(reference='self_reference_establishment' if phase['name']=='A' else 'A',
                     reference_sq=ref2, difference_sq=delta2, current_sq=cur2,
                     relative_l2=relative, finite_gradients=finite, gradient_coverage_ok=coverage,
                     finite_losses=bool(phase['losses']) and all(math.isfinite(x) for x in phase['losses']),
                     normalized_loss_sum=sum(phase['losses'])*phase['normalization_scale'],
                     synchronized_backward_rounds=sum(len(x) for x in phase['microbatch_indices']))
        if phase['name'] == 'A':
            worker._rb_reference = current
            torch.save(worker._rb_inputs, worker._rb_path.with_name(worker._rb_path.stem+'-inputs.pt'))
        phase['diagnostics_and_capture_s'] = time.perf_counter()-t
        persist(worker._rb_path, state)  # Always before threshold rejection.
        _agree(worker, coverage and finite and phase['finite_losses'] and phase['input_coverage_ok']
               and phase['parameters_unchanged'] and math.isfinite(relative) and relative <= 1e-3,
               'diagnostic rejection; recorded before rejecting; threshold remains 1e-3')
        phase['accepted'] = True
    except BaseException as exc:
        error = exc
        state.update(status='failed_no_update', error=repr(exc))
        persist(worker._rb_path, state)
    finally:
        t = time.perf_counter()
        try:
            # step=False skips both normalization and optimizer. Scaling above is
            # therefore required exactly once before capturing/comparing gradients.
            output = worker._close_held_train(step=False)
            engine.optimizer_zero_grad()
            torch.cuda.synchronize()
            phase['context_closed'] = True
        except BaseException as exc:
            state.update(status='failed_no_update', close_error=repr(exc))
            error = error or exc
        phase['close_state_s'] = time.perf_counter()-t
        phase['completed'] = error is None
        if error is None:
            state['status'] = 'phase_completed_no_update'
        persist(worker._rb_path, state)
        worker._rb_phase = None
        engine._rb_phase = None
    if error:
        raise error
    return output


def worker_failure(worker, exc):
    if hasattr(worker, '_rb_state'):
        worker._rb_state.update(status='failed_no_update', error=repr(exc))
        persist(worker._rb_path, worker._rb_state)


def tensor_row_digest(data, index):
    import torch
    from verl.utils import tensordict_utils as tu
    row = tu.index_select_tensor_dict(data.detach().cpu(), [index])
    digest = hashlib.sha256()
    for field in sorted(row.keys()):
        value = row[field]
        if isinstance(value, torch.Tensor):
            if value.is_nested:
                value = value.unbind()[0]
            value = value.detach().cpu().contiguous()
            digest.update(str((field, value.dtype, tuple(value.shape))).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


async def verify_tq_rows(client, meta, data):
    """Real reverse key lookup and independent singleton reads, before timed F/B."""
    if not meta.extra_info or 'rb_members' not in meta.extra_info:
        return  # Natural capture; measured phases carry the explicit mapping.
    from verl.utils import tensordict_utils as tu
    if not meta.partition_ids or len(set(meta.partition_ids)) != 1:
        raise ValueError('mixed or missing actual TQ partitions')
    partition = meta.partition_ids[0]
    indexes = [int(x) for x in meta.global_indexes]
    keys = list(await client.async_kv_retrieve_keys(global_indexes=indexes, partition_id=partition))
    if len(keys) != len(data) or len(set(keys)) != len(keys):
        raise ValueError('actual TQ row/key count mismatch')
    for i, key in enumerate(keys):
        single = await client.async_kv_retrieve_meta(keys=[key], partition_id=partition, create=False)
        if list(single.global_indexes) != [indexes[i]]:
            raise ValueError('key/index roundtrip mismatch')
        single = single.select_fields(meta.field_names)
        reference = await client.async_get_data(single)
        if len(reference) != 1 or tensor_row_digest(data, i) != tensor_row_digest(reference, 0):
            raise ValueError('actual batched row differs from independent single-key read')
    tu.assign_non_tensor_data(data, 'rb_actual_keys', keys)
    tu.assign_non_tensor_data(data, 'rb_global_indexes', indexes)
    tu.assign_non_tensor_data(data, 'rb_rows_independently_checked', True)


if __name__ == '__main__':
    print(json.dumps(preflight(), indent=2))
