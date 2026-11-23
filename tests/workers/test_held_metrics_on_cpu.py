"""Exercise the held-train metric reducer without importing GPU dependencies."""

import ast
import itertools
import sys
import types
import unittest
from pathlib import Path


class Metric:
    def __init__(self, aggregation, value):
        self.aggregation = types.SimpleNamespace(value=aggregation)
        self.value = value

    def aggregate(self):
        return self.value

    @staticmethod
    def aggregate_dp(values):
        return sum(v.value for v in values) / len(values)


class Output(dict):
    def cpu(self):
        return self


def append_to_dict(target, source):
    for key, value in source.items():
        target.setdefault(key, []).append(value)


def method():
    source = Path(__file__).resolve().parents[2] / "verl/workers/engine_workers.py"
    node = next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n, ast.FunctionDef) and n.name == "_close_held_train")
    namespace = {
        "sys": sys, "Metric": Metric, "chain": itertools.chain,
        "append_to_dict": append_to_dict,
        "tu": types.SimpleNamespace(
            get=lambda output, key: output[key],
            get_tensordict=lambda tensor_dict, non_tensor_dict: Output(non_tensor_dict),
        ),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "held-metric-reducer", "exec"), namespace)
    return namespace["_close_held_train"]


class HeldMetricTests(unittest.TestCase):
    def check_chunks(self, contributions):
        calls = []
        engine = types.SimpleNamespace(
            optimizer_step=lambda: (calls.append("optimizer"), 7.0)[1],
            lr_scheduler_step=lambda: (calls.append("lr"), 1e-6)[1],
            is_mp_src_rank_with_outputs=lambda: True,
        )
        worker = types.SimpleNamespace(
            engine=engine,
            _held_train_ctx=types.SimpleNamespace(__exit__=lambda *args: calls.append("exit")),
            _held_loss_normalization_tokens=None, _held_loss_tokens=None,
            _held_output_lst=[
                {"metrics": {
                    "loss": value,
                    "distillation/loss": [Metric("sum", value)],
                    "distillation/abs_loss": [Metric("mean", .19)],
                }} for value in contributions
            ],
        )
        result = method()(worker, step=True)["metrics"]
        self.assertAlmostEqual(result["loss"], sum(contributions))
        self.assertAlmostEqual(result["distillation/loss"], sum(contributions))
        self.assertEqual(result["distillation/abs_loss"], [.19] * len(contributions))
        self.assertEqual(calls, ["optimizer", "lr", "exit"])
        self.assertEqual(result["grad_norm"], 7.0)
        self.assertIsNone(worker._held_train_ctx)

    def test_two_pre_normalized_chunks_are_summed(self):
        self.check_chunks([.08, .04])

    def test_three_pre_normalized_chunks_are_summed(self):
        self.check_chunks([.07, .03, .02])


if __name__ == "__main__":
    unittest.main()
