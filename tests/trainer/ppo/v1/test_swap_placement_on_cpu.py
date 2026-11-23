# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Swap placement: scheduling, finish order, and configuration checks without Ray or GPUs."""

import threading
import time

import pytest
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync, swap_assign

BATCH = 8
DP = {"main": 3, "peer": 2}


# --- swap_assign ---------------------------------------------------------------------------


def test_assign_before_barrier_gives_peer_everything_once_it_has_a_chunk():
    rates = {"main": 3.0, "peer": 1.0}
    assert swap_assign(["a"], {"a": 5}, ["peer"], {}, rates, DP, 0.0, tail=False) == {}
    plan = swap_assign(["a", "b", "c"], {"a": 5, "b": 1, "c": 3}, ["peer"], {}, rates, DP, 0.0, tail=False)
    assert sorted(plan["peer"]) == ["a", "b", "c"]


def test_assign_splits_by_throughput_and_counts_busy_work():
    tokens = {f"k{i}": 10 for i in range(8)}
    rates = {"main": 30.0, "peer": 10.0}
    plan = swap_assign(list(tokens), tokens, ["main", "peer"], {}, rates, DP, 0.0, tail=True)
    assert len(plan["main"]) == 6 and len(plan["peer"]) == 2
    # A busy main that frees soon still takes most of the burst; the idle peer only gets
    # what it can finish no later.
    plan = swap_assign(list(tokens), tokens, ["peer"], {"main": 0.5}, rates, DP, 0.0, tail=True)
    assert len(plan["peer"]) == 2 and "main" not in plan


def test_assign_leaves_the_last_short_samples_to_the_faster_group():
    tokens = {"long": 100, "s1": 5, "s2": 5}
    rates = {"main": 30.0, "peer": 10.0}
    # Main frees up soon enough to finish both before the idle peer would.
    assert swap_assign(["s1", "s2"], tokens, ["peer"], {"main": 0.1}, rates, DP, 0.0, tail=True) == {}
    # Main stays busy long enough that the peer finishes them first.
    plan = swap_assign(["s1", "s2"], tokens, ["peer"], {"main": 1.0}, rates, DP, 0.0, tail=True)
    assert sorted(plan["peer"]) == ["s1", "s2"]
    plan = swap_assign(["long", "s1", "s2"], tokens, ["main", "peer"], {}, rates, DP, 0.0, tail=True)
    assert plan["main"][0] == "long"


def test_assign_requires_a_dp_sized_chunk_until_the_tail():
    tokens = {"a": 4}
    rates = {"main": 3.0, "peer": 1.0}
    assert swap_assign(["a"], tokens, ["main"], {}, rates, DP, 0.0, tail=False) == {}
    assert swap_assign(["a"], tokens, ["main"], {}, rates, DP, 0.0, tail=True) == {"main": ["a"]}


# --- _stream_swap_placement --------------------------------------------------------------


class Group:
    def __init__(self, name, log, trained, world, failure=None):
        self.name, self.log, self.trained, self.world_size, self.failure = name, log, trained, world, failure

    def begin_actor_accumulate(self, denominator):
        assert denominator == BATCH * 32
        self.log.append(f"{self.name}:begin")

    def accumulate_actor(self, batch):
        real = [key for key, tag in zip(batch.keys, batch.tags, strict=True) if not tag.get("is_padding")]
        assert len(batch) % DP[self.name] == 0
        assert not set(real).intersection(k for k, _ in self.trained)
        if self.failure == "fb":
            raise RuntimeError("fb failed")
        time.sleep(0.002)
        self.trained.extend((key, self.name) for key in real)
        self.log.append(f"{self.name}:fb")
        return {"metrics": {"chunk_fb_s": 0.001}}

    def swap_peer_push_grads(self):
        self.log.append("peer:push_grads")
        return [["peer-chunk-outputs"]]

    def swap_main_pull_grads(self):
        self.log.append("main:pull_grads")
        return [None] * self.world_size

    def swap_main_finish(self, peer_outputs):
        self.log.append(f"main:finish:{peer_outputs}")
        if self.failure == "finish":
            raise RuntimeError("finish failed")
        return [{"metrics": {"loss": 1.0}}] + [None] * (self.world_size - 1)

    def swap_peer_finish(self):
        self.log.append("peer:finish")
        return [None] * self.world_size

    def abort_actor_accumulate(self):
        self.log.append(f"{self.name}:abort")


@pytest.fixture
def trainer(monkeypatch):
    trainer = PPOTrainerSync.__new__(PPOTrainerSync)
    trainer.global_steps = 1
    trainer.swap_placement = True
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": BATCH, "max_response_length": 32},
            "actor_rollout_ref": {"rollout": {"temperature": 1.0}},
            "trainer": {"v1": {"sync": {"swap_placement": True}}},
        }
    )
    trainer._swap_prompt_uids = [f"p{i}" for i in range(BATCH)]
    trainer._actor_dp_size = lambda wg=None: DP[wg.name]
    trainer._actor_update_extra_info = lambda: {}
    trainer.applied = []
    trainer._apply_actor_update_metrics = lambda output, metrics: trainer.applied.append(output)
    trainer._record_gen_split_timing = lambda *args: None
    trainer._dump_actor_timeline = lambda *args: None
    trainer.on_sample_begin = lambda: None
    trainer.curr_step_profile = False
    monkeypatch.setattr("verl.trainer.ppo.v1.trainer_sync.ray.get", lambda value: value)
    trainer.cleared = []
    monkeypatch.setattr("verl.trainer.ppo.v1.trainer_sync.tq.kv_clear", lambda **kw: trainer.cleared.extend(kw["keys"]))
    return trainer


def _wire(trainer, students_done_after=4, failure=None, scores_after_barrier=False):
    """Scores arrive one per poll; the Student barrier falls after `students_done_after` polls."""
    log, trained = [], []
    polls = [0]
    lock = threading.Lock()

    def snapshot(count):
        return KVBatchMeta(
            partition_id="train",
            keys=[f"p{i}_0_0" for i in range(count)],
            tags=[{"seq_len": 10 + i} for i in range(count)],
        )

    class Replay:
        poll_interval = 0.001

        def peek_actor_overlap_batch(self, *args):
            with lock:
                polls[0] += 1
                n = polls[0]
            done = BATCH if n > students_done_after else min(n, BATCH - 1)
            scored = 0 if scores_after_barrier and n <= students_done_after else min(n, done)
            batch = snapshot(done)
            return batch, batch.keys[:scored]

        def materialize_actor_overlap_batch(self, *args):
            assert sorted(k for k, _ in trained) == sorted(f"p{i}_0_0" for i in range(BATCH))
            return snapshot(BATCH)

    def balance(batch, worker_group=None, **kwargs):
        assert not kwargs["align_to_mini_batch"]
        n = (-len(batch)) % DP[worker_group.name]
        return KVBatchMeta(
            partition_id="train",
            keys=batch.keys + [f"pad{worker_group.name}{len(trainer.cleared)}{i}" for i in range(n)],
            tags=batch.tags + [{"is_padding": True}] * n,
            extra_info=batch.extra_info,
        )

    trainer.actor_rollout_wg = Group("main", log, trained, 3, failure if failure == "finish" else None)
    trainer.swap_peer_wg = Group("peer", log, trained, 2, failure if failure == "fb" else None)
    trainer.replay_buffer = Replay()
    trainer._balance_batch = balance
    trainer.on_sample_end = lambda: log.append("student:sleep")
    return log, trained


def test_peer_trains_during_rollout_and_both_groups_finish_once(trainer):
    log, trained = _wire(trainer, students_done_after=6)
    metrics, timing = {}, {}
    batch = trainer._stream_swap_placement(metrics, timing, BATCH)

    assert len(batch) == BATCH
    sleep = log.index("student:sleep")
    assert log[0] == "peer:begin"
    assert "peer:fb" in log[:sleep], "the peer must train before the Student barrier"
    assert log.index("main:begin") > sleep
    groups = {name for _, name in trained}
    assert groups == {"main", "peer"}
    tail = log[log.index("peer:push_grads") :]
    assert tail == ["peer:push_grads", "main:pull_grads", "main:finish:['peer-chunk-outputs']", "peer:finish"]
    assert trainer.applied == [{"metrics": {"loss": 1.0}}]
    assert metrics["swap_placement/peer_samples"] + metrics["swap_placement/main_samples"] == BATCH
    assert all(key.startswith("pad") for key in trainer.cleared)
    assert "gen" in timing and "swap_finish" in timing


def test_unused_peer_skips_the_gradient_merge_but_still_gets_parameters(trainer):
    log, trained = _wire(trainer, students_done_after=2, scores_after_barrier=True)
    trainer._swap_rates = {"main": 1e9, "peer": 1.0}
    trainer._swap_rates_measured = {"main", "peer"}
    trainer._stream_swap_placement({}, {}, BATCH)
    assert {name for _, name in trained} == {"main"}
    assert "peer:push_grads" not in log and "main:pull_grads" not in log
    assert log[-2:] == ["main:finish:None", "peer:finish"]


def test_failures_abort_both_groups_and_clear_padding(trainer):
    log, _ = _wire(trainer, students_done_after=6, failure="fb")
    with pytest.raises(RuntimeError, match="fb failed"):
        trainer._stream_swap_placement({}, {}, BATCH)
    assert "peer:abort" in log
    assert "main:finish:None" not in log

    log, _ = _wire(trainer, students_done_after=6, failure="finish")
    with pytest.raises(RuntimeError, match="finish failed"):
        trainer._stream_swap_placement({}, {}, BATCH)
    assert log[-2:] == ["main:abort", "peer:abort"]


# --- configuration -----------------------------------------------------------------------


def _swap_config(**sync):
    import os

    from hydra import compose, initialize_config_dir

    from verl.trainer.config import __file__ as config_init

    with initialize_config_dir(config_dir=os.path.dirname(config_init), version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "trainer.n_gpus_per_node=3",
                "trainer.nnodes=1",
                "data.train_batch_size=8",
                "actor_rollout_ref.actor.ppo_mini_batch_size=8",
                "actor_rollout_ref.actor.strategy=fsdp",
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
                "distillation.enabled=true",
                "distillation.n_gpus_per_node=1",
                "distillation.nnodes=1",
                "distillation.teacher_placement=student",
                "distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-4B",
                "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1",
                "distillation.teacher_models.teacher_model.inference.pipeline_model_parallel_size=3",
                "algorithm.adv_estimator=grpo",
                "trainer.v1.sync.swap_placement=true",
                "trainer.v1.sync.opd_no_task_reward_fast_path=true",
            ]
            + [f"trainer.v1.sync.{key}={value}" for key, value in sync.items()],
        )
    return config


def _validator(config):
    trainer = PPOTrainerSync.__new__(PPOTrainerSync)
    trainer.config = config
    trainer.use_critic = trainer.use_reference_policy = False
    return trainer


def test_swap_placement_pools_put_the_teacher_on_student_gpus():
    from verl.trainer.ppo.utils import Role

    trainer = _validator(_swap_config())
    trainer._init_resource_pool_mgr()
    spec = trainer.resource_pool_manager.resource_pool_spec
    assert spec == {"global_pool": [3], "swap_peer_pool": [1]}
    assert trainer.mapping[Role.TeacherModel] == "global_pool"


@pytest.mark.parametrize(
    "override,message",
    [
        ({"early_actor_lite": "true"}, "replaces"),
        ({"opd_no_task_reward_fast_path": "false"}, "fast path"),
    ],
)
def test_swap_placement_rejects_conflicting_schedules(override, message):
    trainer = _validator(_swap_config(**override))
    with pytest.raises(ValueError, match=message):
        trainer._validate_swap_placement()


def test_swap_placement_requires_the_teacher_to_span_the_student_gpus():
    config = _swap_config()
    config.distillation.teacher_models.teacher_model.inference.pipeline_model_parallel_size = 2
    with pytest.raises(ValueError, match="TP \\* DP \\* PP"):
        _validator(config)._validate_swap_placement()
    config = _swap_config()
    config.actor_rollout_ref.actor.strategy = "fsdp2"
    with pytest.raises(ValueError, match="FSDP1"):
        _validator(config)._validate_swap_placement()


def test_fused_inference_uses_the_trainer_gpus_only():
    from verl.trainer.ppo.utils import Role

    config = _swap_config(swap_placement="false", fused_inference="true")
    trainer = _validator(config)
    trainer._init_resource_pool_mgr()
    assert trainer.resource_pool_manager.resource_pool_spec == {"global_pool": [3]}
    assert trainer.mapping[Role.TeacherModel] == "global_pool"
    trainer.actor_rollout_wg = type("WG", (), {"world_size": 3})()
    trainer._configure_early_actor_lite()
    assert trainer.early_actor_lite is True
    assert trainer.early_actor_stream_fb is False


def test_fused_inference_rejects_a_teacher_that_does_not_span_the_node():
    config = _swap_config(swap_placement="false", fused_inference="true")
    config.distillation.teacher_models.teacher_model.inference.pipeline_model_parallel_size = 2
    with pytest.raises(ValueError, match="TP \\* DP \\* PP"):
        _validator(config)._init_resource_pool_mgr()
    config = _swap_config(swap_placement="false")
    with pytest.raises(ValueError, match="teacher_placement=student requires"):
        _validator(config)._init_resource_pool_mgr()


def test_distillation_config_for_student_placement():
    from verl.utils.config import omega_conf_to_dataclass

    config = _swap_config()
    distillation = omega_conf_to_dataclass(config.distillation)
    teacher = distillation.teacher_models["default"]
    assert teacher.num_replicas == 1 and teacher.per_replica_world_size == 3
    config.distillation.teacher_placement = "dedicated"
    config.distillation.teacher_green_sms = 16
    # Hydra instantiation wraps the ValueError.
    with pytest.raises(Exception, match="teacher_green_sms needs teacher_placement"):
        omega_conf_to_dataclass(config.distillation)
