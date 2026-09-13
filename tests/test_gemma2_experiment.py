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
    ProjectedRuntimeLimit,
    full_finetune_adamw_memory_lower_bound_gib,
    full_finetune_preflight,
    projected_training_runtime_seconds,
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
PREFLIGHT_PATH = (
    ROOT / "results" / "preflights" / f"{EXPERIMENT_ID}__t4x2.json"
)


class Gemma2ExperimentTests(unittest.TestCase):
    def test_checked_t4x2_preflight_preserves_decision_boundary(self) -> None:
        report = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(report["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(report["status"], "resource_limited")
        self.assertFalse(report["conclusion"]["selection_result_available"])
        self.assertFalse(report["conclusion"]["candidate_comparison_allowed"])
        self.assertEqual(report["protocol"]["selection_fold"], 7)
        self.assertEqual(report["protocol"]["unopened_folds"], [8, 9])
        gate = report["peft_runtime_gate"]
        self.assertEqual(gate["probe_micro_batches"], 4)
        self.assertEqual(gate["total_micro_batches"], 120_705)
        for arm in ("lora", "qlora"):
            self.assertEqual(gate["arms"][arm]["status"], "skipped_runtime_limit")
            self.assertGreater(
                gate["arms"][arm]["projected_training_runtime_seconds"],
                26 * gate["limit_seconds"],
            )
        self.assertLess(
            gate["arms"]["qlora"]["peak_gpu_memory_mb_by_device"][1],
            100,
        )

    def test_projected_runtime_limit_keeps_machine_readable_evidence(self) -> None:
        error = ProjectedRuntimeLimit(
            arm_name="qlora",
            observed_seconds=12.5,
            observed_micro_batches=4,
            total_micro_batches=120_705,
            projected_seconds=377_203.125,
            limit_seconds=39_600.0,
        )
        self.assertEqual(error.arm_name, "qlora")
        self.assertEqual(error.observed_micro_batches, 4)
        self.assertEqual(error.total_micro_batches, 120_705)
        self.assertEqual(error.projected_seconds, 377_203.125)
        self.assertIn("4 of 120705 micro-batches", str(error))

    def test_runtime_projection_uses_micro_batches(self) -> None:
        self.assertEqual(
            projected_training_runtime_seconds(
                elapsed_seconds=8.0,
                observed_micro_batches=4,
                total_micro_batches=100,
            ),
            200.0,
        )
        with self.assertRaisesRegex(ValueError, "observed_micro_batches"):
            projected_training_runtime_seconds(
                elapsed_seconds=8.0,
                observed_micro_batches=0,
                total_micro_batches=100,
            )

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

    def test_runner_persists_no_arm_runtime_preflight(self) -> None:
        source = inspect.getsource(run_gemma2_experiment)
        no_arm_block = source.split("    if not completed_arms:\n", maxsplit=1)[1]
        before_selection = no_arm_block.split("    best_arm = min(\n", maxsplit=1)[0]
        self.assertIn('run.artifact_path("runtime_preflight.json")', before_selection)
        self.assertIn('run.log_artifact("runtime_preflight.json"', before_selection)
        self.assertIn('"arms": arm_results', before_selection)
        self.assertIn("json.dumps(arm_results", before_selection)

    def test_runner_resets_peak_memory_for_every_gpu_arm(self) -> None:
        source = inspect.getsource(run_gemma2_experiment)
        self.assertIn(
            "for device_index in range(torch.cuda.device_count()):\n"
            "                torch.cuda.reset_peak_memory_stats(device_index)",
            source,
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
        self.assertEqual(training["runtime_probe_micro_batches"], 4)
        self.assertNotIn("runtime_probe_optimizer_steps", training)
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
