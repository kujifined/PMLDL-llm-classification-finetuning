from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "E20260905212645934620"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
LAUNCHER_PATH = ROOT / "output" / "kaggle" / "run_managed_experiment.ipynb"
INFERENCE_PATH = ROOT / "output" / "kaggle" / "offline_inference.ipynb"


def load_notebook(path: Path) -> dict[str, object]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    return notebook


class E2DeliveryTests(unittest.TestCase):
    def test_managed_e2_encodes_the_frozen_comparison(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["evaluation_role"], "selection")
        self.assertIsInstance(config["smoke_test"], bool)
        self.assertEqual(config["model"]["name"], "microsoft/deberta-v3-base")
        self.assertEqual(
            config["model"]["revision"],
            "8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
        )
        self.assertEqual(config["training"]["effective_batch_size"], 32)
        self.assertEqual(config["training"]["max_length"], 512)
        self.assertEqual(config["training"]["preprocessing"]["budget_weights"], [1, 2, 2])
        self.assertFalse(
            config["training"]["preprocessing"]["model_identity_features"]
        )
        self.assertEqual(
            config["training"]["lora_screening"]["optimizer_steps"], 500
        )

        notebook = load_notebook(ROOT / config["training"]["notebook"])
        implementation = "".join(notebook["cells"][5]["source"])
        required_fragments = (
            "screening_steps = 500",
            "from contextlib import nullcontext",
            "else nullcontext()",
            "use_fast=False",
            '("rank_8", lora_settings(rank=8))',
            '("dropout_0", lora_settings(dropout=0.0))',
            '("learning_rate_2e_4", lora_settings(learning_rate=2e-4))',
            'target_modules=["query_proj", "value_proj"]',
            'bnb_4bit_quant_type="nf4"',
            "bnb_4bit_use_double_quant=True",
            '"full_ft"',
            '"lora"',
            '"qlora"',
            "budget_weights=(1, 2, 2)",
            'content = "null_response"',
            'return "\\n<turn_boundary>\\n".join(rendered)',
            "training_swap_flags = swap_rng.random(len(train_part)) < 0.5",
            "swap_probability_columns(swapped_probability)",
            '"model_identity_features_used": False',
        )
        for fragment in required_fragments:
            self.assertIn(fragment, implementation)

    def test_launcher_is_bound_to_the_managed_notebook(self) -> None:
        notebook = load_notebook(LAUNCHER_PATH)
        source = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn(EXPERIMENT_ID, source)
        self.assertIn("NotebookClient", source)
        self.assertIn('resources={"metadata": {"path": str(REPOSITORY_DIR)}}', source)
        self.assertIn("UserSecretsClient", source)
        self.assertIn('get_secret(secret_name)', source)
        self.assertIn('"CLEARML_API_ACCESS_KEY"', source)
        self.assertIn('"CLEARML_API_SECRET_KEY"', source)
        self.assertTrue(notebook["metadata"]["kaggle"]["isInternetEnabled"])

    def test_offline_notebook_enforces_submission_contract(self) -> None:
        notebook = load_notebook(INFERENCE_PATH)
        source = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertFalse(notebook["metadata"]["kaggle"]["isInternetEnabled"])
        self.assertIn('os.environ["HF_HUB_OFFLINE"] = "1"', source)
        self.assertIn("local_files_only=True", source)
        self.assertIn(
            'expected_columns = ["id", "winner_model_a", "winner_model_b", "winner_tie"]',
            source,
        )
        self.assertIn("np.isfinite(values).all()", source)
        self.assertIn("values.sum(axis=1)", source)
        self.assertIn('Path("/kaggle/working/submission.csv")', source)
        self.assertIn("elapsed > 9 * 60 * 60", source)


if __name__ == "__main__":
    unittest.main()
