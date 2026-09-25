import pytest
from verl.utils.mps import role_mps_env


def test_role_limits_are_separate_and_opt_in(monkeypatch):
    for role in ("ACTOR", "STUDENT", "TEACHER"):
        monkeypatch.delenv(f"FOPD_MPS_{role}_PERCENT", raising=False)
    assert role_mps_env("actor") == {}
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/tmp/test-mps")
    monkeypatch.setenv("FOPD_MPS_ACTOR_PERCENT", "25")
    monkeypatch.setenv("FOPD_MPS_ACTOR_PRIORITY", "1")
    monkeypatch.setenv("FOPD_MPS_STUDENT_PERCENT", "75")
    assert role_mps_env("actor")["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "25"
    assert role_mps_env("actor")["CUDA_MPS_CLIENT_PRIORITY"] == "1"
    assert role_mps_env("student")["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "75"
    assert role_mps_env("student")["CUDA_MPS_CLIENT_PRIORITY"] == "0"
    assert role_mps_env("teacher") == {}


@pytest.mark.parametrize("percent", ["0", "101", "bad"])
def test_invalid_limit_rejected(monkeypatch, percent):
    monkeypatch.setenv("FOPD_MPS_ACTOR_PERCENT", percent)
    with pytest.raises(ValueError):
        role_mps_env("actor")


def test_limit_requires_mps_pipe(monkeypatch):
    monkeypatch.setenv("FOPD_MPS_ACTOR_PERCENT", "25")
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    with pytest.raises(ValueError, match="requires"):
        role_mps_env("actor")
