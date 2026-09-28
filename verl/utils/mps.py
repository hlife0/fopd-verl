"""Opt-in role-specific MPS limits, applied before worker CUDA initialization."""
import os


def role_mps_env(role: str) -> dict[str, str]:
    key = f"FOPD_MPS_{role.upper()}_PERCENT"
    value = os.environ.get(key)
    if value is None:
        return {}
    percent = int(value)
    if not 1 <= percent <= 100:
        raise ValueError(f"{key} must be in [1, 100]")
    pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    if not pipe:
        raise ValueError(f"{key} requires CUDA_MPS_PIPE_DIRECTORY")
    env = {"CUDA_MPS_PIPE_DIRECTORY": pipe, "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(percent)}
    if role == "actor":
        priority = os.environ.get("FOPD_MPS_ACTOR_PRIORITY", "0")
        if priority not in ("0", "1"):
            raise ValueError("FOPD_MPS_ACTOR_PRIORITY must be 0 or 1")
        env["CUDA_MPS_CLIENT_PRIORITY"] = priority
    else:
        env["CUDA_MPS_CLIENT_PRIORITY"] = "0"
    return env
