from __future__ import annotations

import ast
import inspect
import json
import unittest
from pathlib import Path

from pmldl_llm.gemma2_experiment import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    E2_EXPERIMENT_ID,
    full_finetune_adamw_memory_lower_bound_gib,
    full_finetune_preflight,
    run_gemma2_experiment,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "E20260913115603636696"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
NOTEBOOK_PATH = (
    ROOT
    / "output"
    / "jupyter-notebook"
    / f"{EXPERIMENT_ID}__sprint-2-gemma-2-9b-lora.ipynb"
)
LAUNCHER_PATH = ROOT / "output" / "kaggle" / "run_gemma2_experiment.ipynb"


class Gemma2ExperimentTests(unittest.TestCase):
    def test_full_finetune_preflight_rejects_two_t4s(self) -> None:
        required = full_finetune_adamw_memory_lower_bound_gib(9_000_000_000)
        self.assertGreater(required, 130.0)
        report = full_finetune_preflight(
            parameter_count=9_000_000_000,
            total_gpu_memory_gib=30.0,
        )
        self.assertEqual(report["status"], "skipped_resource_limit")
        self.assertGreater(
            report["required_memory_lower_bound_gib"],
            report["usable_gpu_memory_gib"],
        )

    def test_full_finetune_preflight_validates_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "parameter_count"):
            full_finetune_adamw_memory_lower_bound_gib(0)
        with self.assertRaisesRegex(ValueError, "total_gpu_memory"):
            full_finetune_preflight(
                parameter_count=9_000_000_000,
                total_gpu_memory_gib=0.0,
            )

    def test_runner_releases_arm_gpu_references_before_empty_cache(self) -> None:
        source = inspect.getsource(run_gemma2_experiment)
        finally_block = source.split("        finally:\n", maxsplit=1)[1]
        empty_cache_offset = finally_block.index("torch.cuda.empty_cache()")
        for reference in (
            "model",
            "optimizer",
            "scheduler",
            "scaler",
            "train_loader",
            "batch",
            "inputs",
            "logits",
            "loss",
            "labels",
        ):
            self.assertLess(
                finally_block.index(f"{reference} = None"),
                empty_cache_offset,
            )

    def test_config_repeats_e2_protocol_and_matches_peft_arms(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        training = config["training"]
        self.assertEqual(config["parent_experiment_id"], E2_EXPERIMENT_ID)
        self.assertEqual(config["model"]["name"], DEFAULT_MODEL_NAME)
        self.assertEqual(config["model"]["revision"], DEFAULT_MODEL_REVISION)
        self.assertEqual(config["evaluation_role"], "selection")
        self.assertEqual(training["epoch_checkpoints"], [1, 2, 3])
        self.assertEqual(training["effective_batch_size"], 32)
        self.assertEqual(training["max_length"], 512)
        self.assertEqual(
            training["preprocessing"]["random_ab_swap_probability"], 0.5
        )
        self.assertTrue(
            training["preprocessing"]["validation_swap_averaging"]
        )
        self.assertFalse(training["preprocessing"]["model_identity_features"])
        self.assertEqual(training["arms"]["qlora"]["micro_batch_size"], 1)
        self.assertEqual(training["arms"]["qlora"]["evaluation_batch_size"], 1)
        for key in ("r", "alpha", "dropout", "target_modules", "modules_to_save"):
            self.assertEqual(
                training["arms"]["lora"][key], training["arms"]["qlora"][key]
            )
        self.assertEqual(training["control"]["experiment_id"], E2_EXPERIMENT_ID)

    def test_managed_notebook_is_clean_and_bound(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn(EXPERIMENT_ID, source)
        self.assertIn("run_gemma2_experiment", source)
        self.assertIn("run_notebook_experiment", source)
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])
                ast.parse("".join(cell.get("source", [])))

    def test_launcher_is_clean_when_present(self) -> None:
        if not LAUNCHER_PATH.exists():
            self.skipTest("Launcher is pinned after the implementation commit.")
        notebook = json.loads(LAUNCHER_PATH.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn(EXPERIMENT_ID, source)
        self.assertIn("UserSecretsClient", source)
        self.assertIn("gemma2-run-output.zip", source)
        self.assertIn(
            "/kaggle/input/models/google/gemma-2/transformers/gemma-2-9b-it/2",
            source,
        )
        self.assertIn("if len(discovered) == 1", source)
        self.assertNotIn("kaggle.json", source)
        self.assertTrue(notebook["metadata"]["kaggle"]["isGpuEnabled"])
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])
                ast.parse("".join(cell.get("source", [])))


if __name__ == "__main__":
    unittest.main()
