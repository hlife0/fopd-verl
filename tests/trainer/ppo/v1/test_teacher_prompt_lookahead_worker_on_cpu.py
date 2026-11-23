"""Teacher prompt lookahead preserves CT tokens, worker affinity, and baseline dispatch."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from tensordict import NonTensorData, TensorDict

from verl.trainer.ppo.v1 import agent_loop_tq
from verl.utils.tensordict_utils import assign_non_tensor_stack


Worker = agent_loop_tq.AgentLoopWorkerTQ.__ray_metadata__.modified_class


def make_batch(count=2):
    batch = TensorDict({"teacher": torch.arange(count)}, batch_size=[count])
    assign_non_tensor_stack(batch, "uid", [f"u{i}" for i in range(count)])
    assign_non_tensor_stack(batch, "raw_prompt", [[{"role": "user", "content": f"prompt {i}"}] for i in range(count)])
    batch["teacher_lookahead_batch_id"] = NonTensorData("next-batch")
    return batch


def make_worker():
    worker = Worker.__new__(Worker)
    worker.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(agent=SimpleNamespace(default_agent_loop="single_turn_agent"))
        ),
        data=SimpleNamespace(),
    )
    worker.distillation_enabled = True
    worker.teacher_key = "teacher"
    worker.teacher_server_manager = SimpleNamespace(
        distillation_config=SimpleNamespace(teacher_follow=True),
        begin_prompt_lookahead=Mock(),
        prime_prompt_lookahead=AsyncMock(return_value=True),
    )
    worker.llm_client = SimpleNamespace(generate=AsyncMock())
    worker.tokenizer = object()
    worker.processor = None
    worker.hf_model_type = "qwen3"
    worker.dataset_cls = object()
    worker.tools = []
    worker._student_token_states = {}
    return worker


def test_worker_primes_exact_ct_tokens_once_per_prompt_without_student(monkeypatch):
    worker = make_worker()
    batch = make_batch()
    loops, constructors = [], []

    def instantiate(**kwargs):
        constructors.append(kwargs)
        loop = SimpleNamespace(
            process_multi_modal_info=AsyncMock(return_value={}),
            _assert_mm_supported=Mock(),
            ct_build_initial_tokens=AsyncMock(return_value=[11, 12, 13]),
            run=AsyncMock(),
        )
        loops.append(loop)
        return loop

    monkeypatch.setattr(agent_loop_tq.hydra.utils, "instantiate", instantiate)
    result = asyncio.run(worker.prime_teacher_prompts(batch, "next-batch"))
    worker.teacher_server_manager.begin_prompt_lookahead.assert_called_once_with("next-batch")
    assert result["prompts"] == result["primed_prompts"] == 2
    assert result["prompt_tokens"] == 6 and result["elapsed_s"] >= 0
    for i, (loop, constructor, call) in enumerate(
        zip(loops, constructors, worker.teacher_server_manager.prime_prompt_lookahead.await_args_list, strict=True)
    ):
        messages = [{"role": "user", "content": f"prompt {i}"}]
        loop.process_multi_modal_info.assert_awaited_once_with(messages)
        loop._assert_mm_supported.assert_called_once_with(False)
        loop.ct_build_initial_tokens.assert_awaited_once_with(messages, images=None, videos=None, audios=None)
        loop.run.assert_not_called()
        assert constructor["config"] is agent_loop_tq._agent_loop_registry["single_turn_agent"]
        assert constructor["trainer_config"].config is worker.config
        assert constructor["data_config"].config is worker.config.data
        assert constructor["server_manager"] is worker.llm_client
        assert constructor["tokenizer"] is worker.tokenizer
        assert constructor["processor"] is worker.processor
        assert constructor["hf_model_type"] == worker.hf_model_type
        assert constructor["dataset_cls"] is worker.dataset_cls
        assert constructor["tools"].tools is worker.tools
        assert call.args == ([11, 12, 13],)
        assert call.kwargs == dict(
            batch_id="next-batch", request_id=f"teacher-lookahead-next-batch-u{i}", routing_key=i
        )
    worker.llm_client.generate.assert_not_called()
    assert "agent_name" not in batch
    assert batch["teacher_lookahead_batch_id"] == "next-batch"


@pytest.mark.parametrize("unsupported", ["validation", "multi_turn", "multimodal"])
def test_worker_rejects_unsupported_lookahead(monkeypatch, unsupported):
    worker = make_worker()
    batch = make_batch(1)
    if unsupported == "validation":
        batch["validate"] = NonTensorData(True)
    elif unsupported == "multi_turn":
        batch["agent_name"] = NonTensorData("tool_agent")
    loop = SimpleNamespace(process_multi_modal_info=AsyncMock(return_value={"images": [object()]}))
    monkeypatch.setattr(agent_loop_tq.hydra.utils, "instantiate", Mock(return_value=loop))
    with pytest.raises(ValueError, match="single-turn Teacher follow|text-only"):
        asyncio.run(worker.prime_teacher_prompts(batch, "next-batch"))
    worker.teacher_server_manager.prime_prompt_lookahead.assert_not_called()
    worker.llm_client.generate.assert_not_called()


def test_worker_waits_for_other_prompts_before_raising():
    worker = make_worker()

    async def run():
        sibling_started, sibling_done = asyncio.Event(), asyncio.Event()

        async def prime(prompt, batch_id):
            if prompt["uid"] == "u0":
                await sibling_started.wait()
                raise RuntimeError("prefill failed")
            sibling_started.set()
            await asyncio.sleep(0)
            sibling_done.set()
            return True, 3

        worker._prime_teacher_prompt = prime
        with pytest.raises(RuntimeError, match="prefill failed"):
            await worker.prime_teacher_prompts(make_batch(), "next-batch")
        assert sibling_done.is_set()

    asyncio.run(run())


def test_manager_uses_generation_worker_split_and_waits_for_all(monkeypatch):
    manager = agent_loop_tq.AgentLoopManagerTQ.__new__(agent_loop_tq.AgentLoopManagerTQ)
    prime_chunks, generation_chunks = [], []

    def prime(index, chunk, batch_id):
        assert batch_id == "next-batch"
        prime_chunks.append((index, [Worker._prompt_from_batch(chunk, i)["uid"] for i in range(len(chunk))]))
        return dict(prompts=len(chunk), primed_prompts=len(chunk), prompt_tokens=3 * len(chunk), elapsed_s=0.25)

    def generate(index, chunk):
        generation_chunks.append((index, [Worker._prompt_from_batch(chunk, i)["uid"] for i in range(len(chunk))]))

    manager.agent_loop_workers = [
        SimpleNamespace(
            prime_teacher_prompts=SimpleNamespace(remote=lambda chunk, batch_id, i=i: prime(i, chunk, batch_id)),
            generate_sequences=SimpleNamespace(remote=lambda chunk, i=i: generate(i, chunk)),
        )
        for i in range(2)
    ]
    monkeypatch.setattr(agent_loop_tq.ray, "get", lambda result: result)
    batch = make_batch(5)
    result = manager.prime_teacher_prompts(batch, "next-batch")
    manager.generate_sequences(batch)
    assert prime_chunks == generation_chunks == [(0, ["u0", "u1", "u2"]), (1, ["u3", "u4"])]
    assert result["prompts"] == result["primed_prompts"] == 5
    assert result["prompt_tokens"] == 15 and result["worker_elapsed_s"] == 0.5
    assert result["elapsed_s"] >= 0


def test_manager_waits_for_other_workers_before_raising(monkeypatch):
    manager = agent_loop_tq.AgentLoopManagerTQ.__new__(agent_loop_tq.AgentLoopManagerTQ)
    refs = [object(), object()]
    manager.agent_loop_workers = [
        SimpleNamespace(prime_teacher_prompts=SimpleNamespace(remote=Mock(return_value=ref))) for ref in refs
    ]
    waited = []

    def get(ref):
        waited.append(ref)
        if ref is refs[0]:
            raise RuntimeError("worker failed")
        return dict(prompts=1, primed_prompts=1, prompt_tokens=3, elapsed_s=0.25)

    monkeypatch.setattr(agent_loop_tq.ray, "get", get)
    with pytest.raises(RuntimeError, match="worker failed"):
        manager.prime_teacher_prompts(make_batch(), "next-batch")
    assert waited == refs


@pytest.mark.parametrize("lookahead", [False, True])
def test_follow_receives_batch_id_only_when_present(lookahead):
    worker = make_worker()
    worker.teacher_server_manager.compute_teacher_logprobs_follow = AsyncMock(return_value=(object(), object(), {}))
    worker._publish_agent_outputs = AsyncMock()
    output = SimpleNamespace(extra_fields={})
    loop = SimpleNamespace(run=AsyncMock(return_value=output))
    kwargs = dict(uid="u0", session_id=0, teacher=torch.tensor(2))
    if lookahead:
        kwargs["teacher_lookahead_batch_id"] = "next-batch"
    assert asyncio.run(
        worker._invoke_agent_loop(loop, {}, {"validate": False}, agent_name="single_turn_agent", **kwargs)
    ) is output
    call = worker.teacher_server_manager.compute_teacher_logprobs_follow.await_args
    expected = dict(request_id="teacher-u0_0", routing_key=2)
    if lookahead:
        expected["lookahead_batch_id"] = "next-batch"
    assert call.kwargs == expected
    worker._publish_agent_outputs.assert_awaited_once_with([output], False, **kwargs)
    assert not worker._student_token_states
