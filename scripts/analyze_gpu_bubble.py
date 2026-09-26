"""Summarize one gpu_bubble_profile_4gpu.sh run: replica drain, Teacher tail, Actor chunks, publication.

Usage: python analyze_gpu_bubble.py RUN_DIR [--steps 3 4 5]
Replica r -> GPU 4+r, Teacher -> GPU 7 (checked in gpu_pid_map.txt for the run).
"""

import argparse
import csv
import gzip
import json
import os
import re
from collections import defaultdict

REPLICA_GPU = {0: 4, 1: 5, 2: 6}
TEACHER_GPU = 7
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def load_busy(path):
    busy = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            busy[int(row["gpu"])].append((float(row["ts"]), int(row["util"])))
    return busy


def busy_in(busy, gpu, a, b):
    """Busy seconds of `gpu` in [a, b] from NVML samples (each covers the preceding ~200 ms)."""
    if b <= a:
        return 0.0, 0.0
    total = 0.0
    prev = None
    for ts, util in busy.get(gpu, []):
        if prev is not None:
            lo, hi = max(prev, a), min(ts, b)
            if hi > lo:
                total += (hi - lo) * util / 100.0
        prev = ts
    return total, b - a


def parse_kv(line, prefix):
    line = ANSI.sub("", line)
    i = line.find(prefix)
    if i < 0:
        return None
    out = {}
    for tok in line[i + len(prefix) :].split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def trace_summary(path):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as fh:
        data = json.load(fh)
    events = data.get("traceEvents", data)
    gpu = []
    labels = defaultdict(float)
    cats = defaultdict(float)
    n_kernels = 0
    nccl = 0.0
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            s, d = float(e["ts"]), float(e.get("dur", 0))
            gpu.append((s, s + d))
            cats[cat] += d
            if cat == "kernel":
                n_kernels += 1
                if "nccl" in e.get("name", "").lower():
                    nccl += d
        elif str(e.get("name", "")).startswith("pub/") or e.get("cat") == "user_annotation":
            labels[e["name"]] += float(e.get("dur", 0))
    if not gpu:
        return {"path": os.path.basename(path), "kernels": 0}
    gpu.sort()
    union = 0.0
    cur_s, cur_e = gpu[0]
    gaps = []
    for s, e in gpu[1:]:
        if s > cur_e:
            union += cur_e - cur_s
            gaps.append(s - cur_e)
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    union += cur_e - cur_s
    span = gpu[-1][1] - gpu[0][0]
    return {
        "path": os.path.basename(path),
        "gpu_span_ms": round(span / 1e3, 2),
        "gpu_busy_ms": round(union / 1e3, 2),
        "busy_frac": round(union / span, 3) if span else None,
        "kernels": n_kernels,
        "kernel_ms": round(cats["kernel"] / 1e3, 2),
        "memcpy_ms": round(cats["gpu_memcpy"] / 1e3, 2),
        "nccl_ms": round(nccl / 1e3, 2),
        "gaps_gt_1ms": sum(1 for g in gaps if g > 1e3),
        "gap_ms_gt_1ms": round(sum(g for g in gaps if g > 1e3) / 1e3, 2),
        "gap_ms_le_1ms": round(sum(g for g in gaps if g <= 1e3) / 1e3, 2),
        "labels_ms": {k: round(v / 1e3, 2) for k, v in labels.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--steps", type=int, nargs="+", default=[3, 4, 5])
    args = ap.parse_args()
    run = args.run

    # timers.log: the PUBLICATION_*/ACTOR_* lines copied from the Ray session worker logs;
    # train.log drops per-rank copies ("[repeated Nx across cluster]").
    log = os.path.join(run, "timers.log")
    if not os.path.exists(log):
        log = os.path.join(run, "train.log")
    lines = open(log, errors="replace").read().splitlines()
    stages = [x for x in (parse_kv(l, "PUBLICATION_STAGES ") for l in lines) if x]
    sends = [x for x in (parse_kv(l, "PUBLICATION_SEND ") for l in lines) if x]
    recvs = [x for x in (parse_kv(l, "PUBLICATION_RECV ") for l in lines) if x]
    chunks = [x for x in (parse_kv(l, "ACTOR_CHUNK ") for l in lines) if x]
    busy_path = os.path.join(run, "gpu_busy.csv")
    busy = load_busy(busy_path) if os.path.exists(busy_path) else {}

    report = {}
    for step in args.steps:
        sp = json.load(open(os.path.join(run, f"step_{step}_samples.json")))
        tl = json.load(open(os.path.join(run, f"step_{step}_actor_timeline.json")))
        t0 = tl["chunks"][0]["dispatch_ts"] - tl["chunks"][0]["dispatch_s"]
        samples = sp["samples"]
        last_student = max(s["student_last_token_ts"] for s in samples)
        last_teacher = max(s["teacher_done_ts"] for s in samples)
        first_submit = min(s["student_submit_ts"] for s in samples)
        rel = lambda ts: round(ts - t0, 3)  # noqa: E731

        # replica drain
        per_rep = defaultdict(list)
        for s in samples:
            per_rep[s.get("student_replica_rank")].append(s)
        reps = {}
        drain_idle = 0.0
        for r, ss in sorted(per_rep.items(), key=lambda kv: str(kv[0])):
            r_last = max(s["student_last_token_ts"] for s in ss)
            r_first = min(s["student_submit_ts"] for s in ss)
            gpu = REPLICA_GPU.get(r)
            idle = last_student - r_last
            drain_idle += idle
            b_run, w_run = busy_in(busy, gpu, r_first, r_last) if gpu else (None, None)
            b_idle, w_idle = busy_in(busy, gpu, r_last, last_student) if gpu else (None, None)
            ends = sorted(s["student_last_token_ts"] for s in ss)
            reps[str(r)] = {
                "n": len(ss),
                "resp_tokens": sum(s["response_len"] for s in ss),
                "max_resp": max(s["response_len"] for s in ss),
                "n_2048": sum(1 for s in ss if s["response_len"] >= 2048),
                "first_submit_s": rel(r_first),
                "last_token_s": rel(r_last),
                "idle_before_barrier_s": round(idle, 3),
                "gpu_busy_frac_running": round(b_run / w_run, 3) if w_run else None,
                "gpu_busy_frac_after_done": round(b_idle / w_idle, 3) if w_idle else None,
                "active_last_2s": sum(1 for e in ends if e > r_last - 2.0),
                "tail_single_req_s": round(ends[-1] - ends[-2], 3) if len(ends) > 1 else None,
            }

        # Teacher tail
        tail = [s for s in samples if s["teacher_done_ts"] > last_student]
        tb, tw = busy_in(busy, TEACHER_GPU, last_student, last_teacher)
        tb_gen, tw_gen = busy_in(busy, TEACHER_GPU, first_submit, last_student)
        teacher = {
            "tail_s": round(last_teacher - last_student, 3),
            "n_done_after_last_student": len(tail),
            "n_submitted_after_last_student": sum(
                1 for s in samples if (s.get("teacher_submit_ts") or 0) > last_student
            ),
            "tail_queue_s_max": max((s.get("teacher_engine_queue_s") or 0) for s in tail) if tail else None,
            "tail_prefill_s_max": max((s.get("teacher_engine_prefill_s") or 0) for s in tail) if tail else None,
            "gpu7_busy_frac_tail": round(tb / tw, 3) if tw else None,
            "gpu7_busy_frac_during_student": round(tb_gen / tw_gen, 3) if tw_gen else None,
            "last_teacher_s": rel(last_teacher),
            "last_student_s": rel(last_student),
        }
        # Actor GPUs while Teacher tail runs
        actor_tail_busy = {}
        for r, g in REPLICA_GPU.items():
            b, w = busy_in(busy, g, last_student, last_teacher)
            actor_tail_busy[g] = round(b / w, 3) if w else None
        teacher["actor_gpus_busy_frac_during_tail"] = actor_tail_busy

        # Actor chunks
        a0, a1 = tl["update_actor_start_s"] + t0, tl["update_actor_end_s"] + t0
        rank_chunks = [c for c in chunks if a0 - 0.5 <= c["t0"] <= a1 + 0.5]
        per_chunk = []
        for ev in tl["chunks"]:
            mine = [c for c in rank_chunks if ev["dispatch_ts"] - 0.05 <= c["t0"] <= ev["return_ts"]]
            call = ev["return_ts"] - ev["dispatch_ts"]
            per_chunk.append(
                {
                    "n": ev["n"],
                    "dispatch_s": ev["dispatch_s"],
                    "controller_s": round(call, 3),
                    # fb_wall_s / fb_gpu_span_s exist only on the CUDA-timed trace step.
                    "fb_wall_s_max": max((c["fb_wall_s"] for c in mine if "fb_wall_s" in c), default=None),
                    "fb_gpu_span_s": [round(c["fb_gpu_span_s"], 3) for c in mine if "fb_gpu_span_s" in c],
                    "fb_launch_s": [round(c["fb_launch_s"], 3) for c in mine],
                    "token_gather_s": [round(c["token_gather_s"], 3) for c in mine],
                    "tokens": mine[0]["tokens"] if mine else None,
                    "overhead_outside_fb_s": round(
                        call - max((c.get("fb_wall_s", c["fb_launch_s"]) for c in mine), default=0), 3
                    ),
                }
            )
        gaps = []
        prev_ret = tl["fsdp_load_end_s"]
        for ev in tl["chunks"]:
            gaps.append(round(ev["dispatch_s"] - prev_ret, 3))
            prev_ret = ev["return_s"]
        actor = {
            "start_s": tl["update_actor_start_s"],
            "end_s": tl["update_actor_end_s"],
            "student_last_to_first_dispatch_s": round(tl["chunks"][0]["dispatch_s"] - rel(last_student), 3),
            "chunks": per_chunk,
            "wait_before_each_chunk_s": gaps,
            "optimizer_s": round(tl["optimizer_end_s"] - tl["optimizer_start_s"], 3),
            "after_last_teacher_s": round(tl["update_actor_end_s"] - rel(last_teacher), 3),
        }
        actor_busy = {}
        for r, g in REPLICA_GPU.items():
            b, w = busy_in(busy, g, a0, a1)
            actor_busy[g] = round(b / w, 3) if w else None
        actor["gpu_busy_frac_actor_window"] = actor_busy

        # publication
        w0, w1 = sp["weights_start_ts"], sp["weights_done_ts"]
        pub = {
            "controller_s": round(w1 - w0, 3),
            "stages": [c for c in stages if w0 - 0.1 <= c["t0"] <= w1],
            "send": [c for c in sends if w0 - 0.1 <= c["t0"] <= w1],
            "recv": [c for c in recvs if w0 - 0.1 <= c["t0"] <= w1],
        }
        pub_busy = {}
        for g in (4, 5, 6, 7):
            b, w = busy_in(busy, g, w0, w1)
            pub_busy[g] = round(b / w, 3) if w else None
        pub["gpu_busy_frac"] = pub_busy

        report[step] = {
            "step_s": sp["timing_raw"].get("step"),
            "gen_student_s": sp["timing_raw"].get("gen_student"),
            "gen_teacher_s": sp["timing_raw"].get("gen_teacher"),
            "update_actor_s": sp["timing_raw"].get("update_actor"),
            "update_weights_s": sp["timing_raw"].get("update_weights"),
            "resp_tokens": sum(s["response_len"] for s in samples),
            "first_submit_s": rel(first_submit),
            "student_barrier_s": tl["student_barrier_s"],
            "sleep_s": round(tl["sleep_end_s"] - tl["sleep_start_s"], 3),
            "replicas": reps,
            "replica_drain_idle_gpu_s": round(drain_idle, 3),
            "teacher": teacher,
            "actor": actor,
            "publication": pub,
        }

    trace_dir = os.path.join(run, "window_trace")
    if os.path.isdir(trace_dir):
        report["traces"] = [trace_summary(os.path.join(trace_dir, f)) for f in sorted(os.listdir(trace_dir))]
    out = os.path.join(run, "bubble_report.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
