"""k1 teacher score contract: one chosen-token logprob per position, current alignment."""

import torch

from verl.experimental.teacher_loop.teacher_manager import _get_teacher_sampling_params
from verl.workers.rollout.vllm_rollout.utils import extract_prompt_logprobs


class _LogProb:
    def __init__(self, logprob, rank=1):
        self.logprob = logprob
        self.rank = rank


def _k1_loss():
    return type("L", (), {"topk": 64, "loss_settings": type("S", (), {"use_topk": False})()})()


def _teacher_cfg():
    return type("T", (), {"inference": type("I", (), {"temperature": 1.0})()})()


def test_k1_oneshot_requests_chosen_token_only():
    params = _get_teacher_sampling_params(_teacher_cfg(), _k1_loss(), follow=False)
    assert params["prompt_logprobs"] == 0
    assert params["detokenize"] is False
    assert params["max_tokens"] == 1
    assert "logprobs" not in params
    assert "skip_reading_prefix_cache" not in params


def test_extract_matches_sequence_index_dtype_and_eos():
    eos = 151645
    seq = [11, 22, 33, eos]
    # vLLM prompt_logprobs[0] is None. Row i>=1 is the logprob of token i.
    rows = [None]
    for token_id in seq[1:]:
        rows.append({token_id: _LogProb(-(token_id / 10.0), rank=3)})
    output = type("O", (), {"prompt_logprobs": rows})()
    result = {}
    extract_prompt_logprobs(output, num_prompt_logprobs=0, result_dict=result)

    ids = torch.tensor(result["prompt_ids"], dtype=torch.int32)
    logprobs = torch.tensor(result["prompt_logprobs"])
    assert ids.shape == logprobs.shape == (len(seq), 1)
    assert ids.dtype == torch.int32
    assert logprobs.dtype == torch.float32
    # Stored row i is the logprob of token i+1. The final row is the dummy pad.
    for index, token_id in enumerate(seq[1:]):
        assert int(ids[index, 0]) == token_id
        expected = torch.tensor([-(token_id / 10.0)], dtype=torch.float32)
        assert torch.equal(logprobs[index], expected)
    assert int(ids[-1, 0]) == 0
    assert float(logprobs[-1, 0]) == 0.0
    assert int(ids[-2, 0]) == eos
