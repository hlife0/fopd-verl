"""CPU checks of identity propagation, async ranges, and observation-only hooks."""

import asyncio
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[2] / "verl/utils/opd_timeline.py"


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="opd-timeline-test-")
        self.modules = []

    def tearDown(self):
        for module in self.modules:
            if module._writer is not None:
                module._writer.close()
                module._writer = None
        self.temp.cleanup()

    def load(self, **env):
        values = {
            "FOPD_TIMELINE_DIR": self.temp.name, "FOPD_TIMELINE_SCOPE": "full",
            "FOPD_TIMELINE_STEPS": "3", "FOPD_TIMELINE_NVTX": "0", **env,
        }
        spec = importlib.util.spec_from_file_location(f"timeline_test_{len(self.modules)}", SOURCE)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, values):
            spec.loader.exec_module(module)
        self.modules.append(module)
        return module

    def rows(self):
        return [json.loads(line) for p in Path(self.temp.name).glob("*.jsonl") for line in p.read_text().splitlines()]

    def test_disabled_does_not_write_or_change_results(self):
        module = self.load(FOPD_TIMELINE_DIR="")

        @module.observed("actor.batch", "actor")
        def work(data):
            return data

        marker = object()
        self.assertIs(work(marker), marker)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_96_trajectories_keep_identity_through_server_and_teacher(self):
        module = self.load()

        class Worker:
            @module.observed("trajectory", "agent")
            async def run(self, global_steps, uid, index, session_id=0):
                with module.span("student.generate", role="student"):
                    await self.server(request_id=f"request-{uid}", timeline_context=module.context())
                with module.span("teacher.ready", role="teacher"):
                    await asyncio.sleep(0)
                return uid

            @module.observed("server.request", "inherit")
            async def server(self, request_id, timeline_context=None):
                await asyncio.sleep(0)
                return {"num_cached_tokens": 0}

        async def batch():
            worker = Worker()
            return await asyncio.gather(*[worker.run(3, f"uid-{i}", i) for i in range(96)])

        result = asyncio.run(batch())
        self.assertEqual(result, [f"uid-{i}" for i in range(96)])
        rows = self.rows()
        for event in ("student.generate.end", "teacher.ready.end", "server.request.end"):
            selected = [r for r in rows if r["event"] == event]
            self.assertEqual(len(selected), 96)
            self.assertEqual({r["uid"] for r in selected}, set(result))
            self.assertEqual({r["step"] for r in selected}, {3})
        self.assertTrue(all(r["boundary"] == "host" for r in rows))

    def test_filtered_step_context_still_reaches_remote_metadata(self):
        module = self.load()
        with module.span("trajectory", step=2, global_steps=2, uid="other-step", role="student"):
            self.assertEqual(module.context()["uid"], "other-step")
            module.emit("server.request")
        module.flush()
        self.assertEqual(self.rows(), [])

    def test_shared_microbatch_records_exact_members(self):
        module = self.load()
        members = [f"u{i}_0_0" for i in range(4)]
        data = {"uid": [f"u{i}" for i in range(4)], "session_id": [0] * 4, "global_steps": [3] * 4}
        fields = module._call_fields({"data": data}, "actor")
        with module.span("actor.forward", microbatch=0, **fields):
            pass
        begin = next(r for r in self.rows() if r["event"] == "actor.forward.begin")
        self.assertEqual(begin["members"], members)
        self.assertEqual(begin["boundary"], "host")
        self.assertNotIn("gpu_duration", begin)

    def test_async_nvtx_ranges_are_independent(self):
        calls, serial = [], itertools.count()
        fake = types.ModuleType("nvtx")
        fake.mark = lambda msg, **kw: calls.append(("mark", msg))
        fake.start_range = lambda msg, **kw: (calls.append(("start", msg)), (next(serial), "domain-handle"))[1]
        fake.end_range = lambda handle: calls.append(("end", handle))
        fake.push_range = lambda msg, **kw: calls.append(("push", msg))
        fake.pop_range = lambda **kw: calls.append(("pop",))
        with patch.dict(sys.modules, {"nvtx": fake}):
            module = self.load(FOPD_TIMELINE_NVTX="1")

            @module.observed("teacher.score", "teacher")
            async def score(global_steps, uid):
                await asyncio.sleep(0)

            async def concurrent():
                await asyncio.gather(score(3, "a"), score(3, "b"))

            asyncio.run(concurrent())
        self.assertEqual(sum(c[0] == "start" for c in calls), 2)
        self.assertEqual(sum(c[0] == "end" for c in calls), 2)
        self.assertFalse(any(c[0] in ("push", "pop") for c in calls))
        self.assertTrue(all("mono_ns=" in c[1] for c in calls if c[0] == "start"))

    def test_gpu_worker_hook_preserves_outputs_and_shared_request_ids(self):
        module = self.load()
        marker = object()

        class Worker:
            rank = 0

            def execute_model(self, scheduler_output):
                return marker

            def sample_tokens(self):
                return marker

        fake = types.ModuleType("vllm.v1.worker.gpu_worker")
        fake.Worker = Worker
        parents = {name: types.ModuleType(name) for name in ("vllm", "vllm.v1", "vllm.v1.worker")}
        with patch.dict(sys.modules, {**parents, fake.__name__: fake}), patch.dict(os.environ, {"FOPD_TIMELINE_ROLE": "student"}):
            module.install_vllm_worker_observer()
            worker = Worker()
            scheduler = types.SimpleNamespace(num_scheduled_tokens={"request-a": 3, "request-b": 3})
            self.assertIs(worker.execute_model(scheduler), marker)
            self.assertIs(worker.sample_tokens(), marker)
        begin = next(r for r in self.rows() if r["event"] == "student.execute_model.begin")
        self.assertEqual(begin["request_ids"], ["request-a", "request-b"])
        self.assertEqual(begin["rank"], 0)
        self.assertEqual(begin["scheduled_tokens"], scheduler.num_scheduled_tokens)

    def test_coarse_scope_excludes_sample_records(self):
        module = self.load(FOPD_TIMELINE_SCOPE="step", FOPD_TIMELINE_STEPS="")
        module.rollout_submit(3, {"uid": ["unused"] * 96})
        module.emit("teacher.hop", step=3)
        module.weights_ready(3)
        rows = self.rows()
        self.assertEqual([r["event"] for r in rows if r["event"] != "clock.anchor"], ["step.rollout_start", "step.weights_ready"])

    def test_same_host_monotonic_clock_is_comparable_across_processes(self):
        before = time.monotonic_ns()
        child = int(subprocess.check_output([sys.executable, "-c", "import time; print(time.monotonic_ns())"], text=True))
        after = time.monotonic_ns()
        self.assertLessEqual(before, child)
        self.assertLessEqual(child, after)


if __name__ == "__main__":
    unittest.main()
