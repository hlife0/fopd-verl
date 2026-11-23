"""CPU checks of diagnostic selection and existing worker field propagation."""
import ast
import copy
import importlib.util
import itertools
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MMShapeScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="opd-mm-scope-test-")
        self.env = patch.dict(os.environ, {
            "FOPD_TIMELINE_DIR": self.temp.name, "FOPD_TIMELINE_SCOPE": "full",
            "FOPD_TIMELINE_STEPS": "3", "FOPD_TIMELINE_NVTX": "0",
            "FOPD_MM_SHAPE_DIR": self.temp.name, "FOPD_MM_SHAPE_MAX": "64",
        })
        self.env.start()
        self.helper = load("mm_helper_cpu", ROOT / "verl/utils/opd_mm_shapes.py")
        root_package = types.ModuleType("verl")
        root_package.__path__ = []
        utils_package = types.ModuleType("verl.utils")
        utils_package.__path__ = []
        utils_package.opd_mm_shapes = self.helper
        root_package.utils = utils_package
        self.imports = patch.dict(sys.modules, {
            "verl": root_package, "verl.utils": utils_package,
            "verl.utils.opd_mm_shapes": self.helper,
        })
        self.imports.start()
        self.timeline = load("mm_timeline_cpu", ROOT / "verl/utils/opd_timeline.py")
        utils_package.opd_timeline = self.timeline
        self.entered, self.exited = [], []

        @contextmanager
        def mode(role, fields, out_dir, limit):
            self.entered.append((role, dict(fields)))
            try:
                yield
            finally:
                self.exited.append(role)
        self.helper._mode_type = mode  # No tensor fixture or torch import.

    def tearDown(self):
        if self.timeline._writer is not None:
            self.timeline._writer.close()
            self.timeline._writer = None
        self.imports.stop()
        self.env.stop()
        self.temp.cleanup()

    def test_actor_worker_rank_and_step_enter_scope_without_changing_return(self):
        timeline, marker = self.timeline, object()
        class Actor:
            rank = 0
            @timeline.observed("actor.forward", "actor")
            def forward(self, data):
                return marker
        actor = Actor()
        self.assertIs(actor.forward({"global_steps": [3], "uid": ["member"], "session_id": [0]}), marker)
        self.assertEqual([role for role, _ in self.entered], ["actor"])
        self.assertEqual(self.entered[0][1]["rank"], 0)
        self.assertEqual(self.entered[0][1]["step"], 3)

    def test_other_actor_steps_ranks_and_backward_do_not_enter(self):
        for event, fields in (("actor.forward", {"rank": 0, "step": 2}),
                              ("actor.forward", {"rank": 1, "step": 3}),
                              ("actor.backward", {"rank": 0, "step": 3})):
            with self.timeline.span(event, **fields):
                pass
        self.assertEqual(self.entered, [])

    def test_teacher_real_request_pack_preserves_unassigned_step(self):
        self.timeline._counter = itertools.count(4)
        with self.timeline.span("teacher.execute_model", rank=0, request_ids=["request-a"]):
            pass  # Known warmup counter is excluded.
        with self.timeline.span("teacher.execute_model", rank=0, request_ids=[], scheduled_tokens={}):
            pass
        pack = {"request-a": 9, "request-b": 3}
        with self.timeline.span("teacher.execute_model", rank=0, request_ids=list(pack), scheduled_tokens=pack):
            pass
        self.assertEqual([role for role, _ in self.entered], ["teacher"])
        fields = self.entered[0][1]
        self.assertEqual(fields.get("step", -1), -1)
        self.assertEqual(fields["request_ids"], list(pack))
        self.assertEqual(fields["scheduled_tokens"], pack)
        self.assertTrue(fields["current_span_id"].endswith(":6"))

    def test_exception_and_context_cleanup_are_preserved(self):
        error = ValueError("training marker")
        with self.assertRaises(ValueError) as caught:
            with self.timeline.span("actor.forward", rank=0, step=3):
                raise error
        self.assertIs(caught.exception, error)
        self.assertEqual(self.exited, ["actor"])
        self.assertEqual(self.timeline.context(), {})

    def test_env_forwarding_preserves_unique_server_names(self):
        source = ROOT / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
        keys = ("FOPD_MM_SHAPE_DIR", "FOPD_MM_SHAPE_MAX", "FOPD_MM_SHAPE_NVTX")
        loops = [node for node in ast.walk(ast.parse(source.read_text()))
                 if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple)
                 and all(isinstance(value, ast.Constant) for value in node.iter.elts)
                 and tuple(value.value for value in node.iter.elts) == keys]
        self.assertEqual(len(loops), 1)
        loop_module = ast.Module(body=[loops[0]], type_ignores=[])
        code = compile(ast.fix_missing_locations(loop_module), str(source), "exec")
        names = ["rollout_server_0_0", "rollout_server_1_0", "teacher_server_0_0"]
        actual_names = []
        with patch.dict(os.environ, {keys[0]: self.temp.name, keys[1]: "64", keys[2]: "0"}):
            for server_name in names:
                namespace = {"os": os, "name": server_name, "env_vars": {"existing": "kept"}}
                exec(code, namespace)
                actual_names.append(namespace["name"])
                self.assertEqual(namespace["env_vars"], {
                    "existing": "kept", keys[0]: self.temp.name, keys[1]: "64", keys[2]: "0"})
        self.assertEqual(actual_names, names)
        self.assertEqual(len(set(actual_names)), 3)

    def test_teacher_missing_dir_gate_is_visible_once(self):
        import json
        self.timeline._counter = itertools.count(5)
        with patch.dict(os.environ, {"FOPD_MM_SHAPE_DIR": ""}):
            for _ in range(2):
                with self.timeline.span("teacher.execute_model", rank=0,
                                        request_ids=["request-a"], scheduled_tokens={"request-a": 49}):
                    pass
        self.timeline.flush()
        rows = [json.loads(line) for p in Path(self.temp.name).glob("*.jsonl")
                for line in p.read_text().splitlines()]
        gates = [r for r in rows if r["event"] == "mm.scope_gate"]
        self.assertEqual(len(gates), 1)
        self.assertFalse(gates[0]["mm_dir_present"])
        self.assertEqual(gates[0]["rank"], 0)
        self.assertIsNone(gates[0]["step"])
        self.assertEqual(self.entered, [])

    def tensor_fixture(self):
        # Metadata-only CPU fixtures; no fixture is written as a CUDA record.
        class Device:
            type = "cuda"
            def __str__(self):
                return "cuda:0"
        class Tensor:
            device, layout, dtype = Device(), "torch.strided", "torch.bfloat16"
            shape = (8, 16)
            def stride(self):
                return (16, 1)
            def storage_offset(self):
                return 128
            def data_ptr(self):
                return 0x1010
            def item(self):
                raise AssertionError("tensor values must not be read")
            def cpu(self):
                raise AssertionError("tensor values must not be transferred")
        return types.SimpleNamespace(Tensor=Tensor, strided="torch.strided"), Tensor

    def test_parameter_subclass_metadata_gate(self):
        torch, Tensor = self.tensor_fixture()
        class ModelWeightParameter(Tensor):
            pass
        class FlatParameter(Tensor):
            pass
        for cls in (ModelWeightParameter, FlatParameter):
            meta = self.helper._tensor_metadata(torch, cls())
            self.assertIsNotNone(meta)
            self.assertEqual(meta["shape"], [8, 16])
            self.assertEqual(meta["stride"], [16, 1])
            self.assertEqual(meta["storage_offset"], 128)
            self.assertEqual(meta["pointer_alignment_mod_256"], 16)
            self.assertTrue(meta["tensor_type"].endswith("." + cls.__name__))

    def test_fake_distributed_meta_symbolic_and_no_storage_skip(self):
        torch, Tensor = self.tensor_fixture()
        fixtures = []
        fake = Tensor(); fake.fake_mode = object(); fixtures.append(fake)
        distributed = Tensor(); distributed._local_tensor = Tensor(); fixtures.append(distributed)
        meta = Tensor(); meta.device = types.SimpleNamespace(type="meta"); fixtures.append(meta)
        symbolic = Tensor(); symbolic.shape = (8, object()); fixtures.append(symbolic)
        class NoStorage(Tensor):
            def storage_offset(self):
                raise RuntimeError("no concrete storage")
        class NoPointer(Tensor):
            def data_ptr(self):
                raise NotImplementedError("no concrete pointer")
        fixtures.extend([NoStorage(), NoPointer(), object()])
        for fixture in fixtures:
            self.assertIsNone(self.helper._tensor_metadata(torch, fixture))

    def test_flat_offset_does_not_consume_budget_but_layout_changes_do(self):
        metadata = {"shape": [8, 16], "stride": [16, 1], "dtype": "torch.bfloat16",
                    "layout": "torch.strided", "device": "cuda:0", "storage_offset": 0,
                    "pointer_alignment_mod_256": 0}
        changed = copy.deepcopy(metadata)
        changed["storage_offset"] = 10000
        self.assertEqual(self.helper._signature("mm", [metadata], 1, 1),
                         self.helper._signature("mm", [changed], 1, 1))
        changed["stride"] = [1, 8]
        self.assertNotEqual(self.helper._signature("mm", [metadata], 1, 1),
                            self.helper._signature("mm", [changed], 1, 1))


if __name__ == "__main__":
    unittest.main()
