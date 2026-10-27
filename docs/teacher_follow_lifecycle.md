# Teacher follow request lifecycle

Last updated: 2026-10-27

This note describes the lifecycle inside `AsyncTeacherLLMServerManager.compute_teacher_logprobs_follow` for one student sequence. The reviewed base is `dd58cf07b0c400247014d7eaf9ddce9d96751849`; the current `codex/teacher-follow-lifecycle` change adds cancellation regressions and updates local task cleanup. It records implementation behavior and test coverage without prescribing new interfaces or scheduling policies.

## Objects and ownership

| Object or operation | Owner and lifetime | Relevant source |
|---|---|---|
| `StudentTokenState` | The AgentLoop caller creates it, registers it by sequence key, and supplies it to generation and the follow call. It holds copied prompt/response tokens, completion state, and `ready`, `event`, and `done` notifications. It does not own Teacher request tasks. | [`StudentTokenState`](../verl/experimental/teacher_loop/teacher_follow.py); [`_invoke_agent_loop`](../verl/trainer/ppo/v1/agent_loop_tq.py) |
| Follow call and accumulator | The caller owns `follow_task`. Each `compute_teacher_logprobs_follow` invocation owns its `TeacherFollowAccumulator`, request records, and submission loop. | [`compute_teacher_logprobs_follow`](../verl/experimental/teacher_loop/teacher_manager.py) |
| `hop` and `done_wait` | The follow invocation creates `hop` for `_follow_forward`. When final overtaking is enabled for a mid request, it also creates `done_wait` for `state.done.wait()`. | [`compute_teacher_logprobs_follow`](../verl/experimental/teacher_loop/teacher_manager.py) |
| Overtaking final request and gap-fill request | These coroutines are awaited directly by the follow invocation; neither introduces another explicitly created request task in this function. | [`_follow_forward`, `_apply_follow_extract`, `_fill_unscored_prefix`](../verl/experimental/teacher_loop/teacher_manager.py) |
| Mid-request gate | The manager owns a `FollowMidGate`, shared by sequences handled by that AgentLoop worker. The submitting loop acquires a request/token allowance and attaches a completion callback to `hop`. That callback releases the captured allowance when the task terminates, including cancellation before its coroutine starts. | [`_make_follow_mid_gate`, `compute_teacher_logprobs_follow`](../verl/experimental/teacher_loop/teacher_manager.py); [`FollowMidGate`](../verl/experimental/teacher_loop/teacher_follow.py) |
| Teacher RPC | `_teacher_forward` calls `LLMServerClient.generate`. The client selects a server and calls `server.generate.remote`; its `finally` invokes `_release_server`. | [`_teacher_forward`](../verl/experimental/teacher_loop/teacher_manager.py); [`LLMServerClient.generate`](../verl/workers/rollout/llm_server.py) |

`scored_seq_len` includes the request's final dummy row. `filled_real` is the boundary used by the accumulator for gap detection; pending decode rows can populate an adjacent cache boundary when available. These values have different meanings in submission and supervision alignment.

The gate caps mid-follow requests from one AgentLoop worker. A completed student sequence bypasses this gate. Its counters do not measure GPU idleness or prove that backend work has stopped.

## Ordinary request

1. Wait for `state.ready`, then snapshot the visible tokens and completion state. `follow_submit_len` selects whole KV blocks plus one token during generation, or the complete remaining prefix after generation finishes.
2. If no new request is eligible, `_wait_for_follow_progress` clears the old notification, rechecks progress and completion, and waits only when still necessary. The outer loop rechecks eligibility after notification.
3. For a mid request, acquire the optional gate allowance. If acquisition fails, wait for a release or a state notification and reevaluate the sequence. Completion can make the final request eligible without acquiring a mid-request allowance.
4. Create and await `hop`. Without final overtaking, an in-flight mid request finishes before the final request is submitted. The `hop` completion callback releases any acquired allowance after the request returns, raises, or completes cancellation cleanup.
5. Apply the returned extract through `_apply_follow_extract`, record the request, and inspect the latest generation progress. Once the student is done and no further prefix is eligible, finalize the accumulated tensors.

Within a sequence, ordinary requests are serialized through their result application. Generation continues concurrently. Gate release occurs before result application, so another sequence can acquire an allowance while this sequence aligns supervision or fills a gap.

## Final request overtaking a mid request

With `teacher_follow_final_overtakes_mid` enabled, the invocation waits for either `hop` or `state.done`. If the student finishes while `hop` is unfinished, it submits the complete final sequence immediately. The mid and final requests can then coexist. The final request does not acquire a mid-request allowance.

After receiving the final extract, the code compares its `num_cached_tokens` with the accumulator's current `scored_seq_len`:

- If `num_cached_tokens <= scored_seq_len`, it cancels `hop` and awaits it with `asyncio.gather(..., return_exceptions=True)`. The mid result is dropped and its request record is marked `overtaken`.
- Otherwise, it awaits the mid result and applies that extract first. The final extract may depend on cached positions whose supervision still has to be accumulated from the mid result.

In both cases, final rows are applied last. Overtaking does not permit applying the two results in arbitrary arrival order. The mid allowance is released by the `hop` completion callback, including when that task is cancelled. If the mid request completes while the final request is still pending, its allowance is released without waiting for the final result.

## Supervision gap fill

`TeacherFollowAccumulator.apply` interprets the cached offset and returned rows. A gap beyond the adjacent boundary raises `TeacherFollowGapError`.

`_apply_follow_extract` then awaits `_fill_unscored_prefix`, which requests a prefix extending one token past the gap where possible. It uses `_teacher_forward(follow=False, need_decode_topk=True)` so the fill has an appropriate boundary row. It applies the fill first, then retries the original extract. If that retry still reports a gap, the code logs a warning and leaves further suffix submission to the outer loop.

This fill is part of the same follow invocation and updates the same accumulator. It is directly awaited, uses no additional mid-request gate allowance, and must complete before the interrupted extract can be retried. The original mid request has already released its allowance when this stage begins.

## Cancellation of the complete follow call

The boundary here is cancellation of `compute_teacher_logprobs_follow`. Cancellation of the surrounding Student generation or the complete AgentLoop pipeline is outside the tested boundary of this note.

- While waiting for readiness or token progress, no Teacher request has been created by that wait.
- While waiting for gate admission, this sequence has not acquired an allowance. `FollowMidGate.wait_release_or` removes its waiter and requests cancellation of its event-wait task in `finally`.
- In the ordinary `await hop` path, cancellation propagates into the awaited request task. The acquired allowance is released by the task-completion callback after the request's cleanup completes.
- While awaiting an overtaking final request or a gap fill, cancellation propagates into that directly awaited coroutine. Any separate mid-request task remains the follow invocation's cleanup responsibility.

The per-request `finally` now cancels and gathers unfinished `hop` and `done_wait` tasks before the invocation exits that block. The normal overtaking branch that deliberately drops a mid result also awaits that cancelled `hop`. Gate release is attached to task completion, so cancellation before the request coroutine starts can release an already acquired allowance. The cancellation tests below exercise one outer cancellation with event-controlled local client cleanup.

The caller's `_invoke_agent_loop` creates the follow task, publishes Student fields before awaiting the Teacher fields, and removes the state registration in `finally`. Its `except Exception` path cancels and gathers the follow task. Those source observations do not establish full AgentLoop cancellation coverage.

Cancelling a Python coroutine, releasing a local gate allowance, or executing client server-release bookkeeping does not prove that a remote Teacher request or GPU kernel has stopped. Backend abort semantics and resource reclamation require separate validation.

## CPU evidence and remaining scope

The base implementation's real package imports and 32 CPU cases passed in the [main-branch CPU workflow](https://github.com/Liu-KM/seamless-opd-verl/actions/runs/40459810845). The tests below use synthetic rows and controllable Teacher clients; they provide evidence for Python control flow and supervision bookkeeping.

Existing tests in [`test_teacher_follow_on_cpu.py`](../tests/workers/test_teacher_follow_on_cpu.py):

| Path or property | Actual test names | Coverage boundary |
|---|---|---|
| Ordinary requests and gate ownership | `test_follow_mid_gate_holds_mid_hops_but_not_finished_requests`; `test_follow_final_waits_for_hop_without_overtake` | Submission order, final gate bypass, eventual zero gate counters, and synthetic output alignment |
| Gate wakeups and limits | `test_follow_mid_gate_caps_requests_and_tokens`; `test_follow_mid_gate_wakes_on_release_or_event`; `test_make_follow_mid_gate_splits_over_workers` | Local accounting and notification behavior; no GPU scheduling measurements |
| Final overtaking | `test_follow_final_overtakes_queued_hop_and_drops_it`; `test_follow_final_overtakes_computed_hop_and_stitches_it_first` | Both result-order branches with a controllable fake prefix cache |
| Gap detection and alignment | `test_accumulator_raises_on_cache_gap_without_full_extract`; `test_gap_hole_fill_then_suffix_matches_oneshot` | Direct accumulator calls and synthetic rows; these tests do not exercise the manager's gap-fill RPC lifecycle |
| Follow output | `test_follow_matches_oneshot_without_using_decode_rows` | Agreement with synthetic one-shot rows across multiple mid requests |

Existing tests in [`test_teacher_follow_wait_on_cpu.py`](../tests/workers/test_teacher_follow_wait_on_cpu.py) cover already-visible progress, stale notifications, short EOS tails, insufficient updates, progress arriving during notification clearing, and cancellation of an individual progress wait. In particular, `test_cancelled_wait_does_not_consume_future_progress` covers the wait helper; it does not cancel the complete follow invocation.

New tests in `test_teacher_follow_on_cpu.py`:

- `test_follow_cancellation_waits_for_requests_and_releases_gate`, parameterized by `mid`, `overtake_wait`, and `overtake_final`: the outer call waits for event-controlled client cleanup; its mid task and completion waiter finish, gate counters return to zero, and a later mid request can run.
- `test_follow_cancelled_before_request_starts_releases_gate`: cancellation before the client is called still releases the acquired allowance.
- `test_follow_mid_gate_releases_before_overtaking_final_finishes`: a completed mid request releases its allowance while the final request remains pending, then the returned rows match the synthetic one-shot reference.

Before the cleanup fix, test-only commit `29e0056fe04e819a1bc8227187e84d470febb7f7` passed real imports but produced [3 failed and 33 passed CPU cases](https://github.com/Liu-KM/seamless-opd-verl/actions/runs/40461251401/job/122046152781). Both overtaking cancellation cases ended the outer call before mid-request cleanup finished; pre-start cancellation left the gate at `(1, 17)`. The code now uses task-completion accounting and awaits pending children. For post-fix validation, consult the CI records for the corresponding commit in [PR #7](https://github.com/Liu-KM/seamless-opd-verl/pull/7).

To run the same real-import and CPU checks from the repository:

```bash
bash scripts/ci/check_teacher_follow_cpu.sh
```

The script uses the frozen CPU dependency configuration, imports the actual follow modules, and runs both test files listed above.

GPU execution, actual prefix-cache behavior, backend cancellation/abort semantics, complete AgentLoop cancellation, end-to-end timing, and training quality have not been validated by these CPU tests.
