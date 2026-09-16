#!/usr/bin/env python3
"""Run frozen seed-42 QLoRA inference for fold 8 or the final holdout.

The default exports calibration-fold predictions. Once ``final_model.json`` is
frozen, ``--target-fold 9`` exports holdout and Kaggle-test probabilities from
the same verified adapter. No mode trains or selects a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter-archive", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--target-fold", type=int, default=8)
    parser.add_argument(
        "--include-kaggle-test",
        action="store_true",
        help="Also write inference_predictions.csv in Kaggle submission schema.",
    )
    parser.add_argument("--manifest-name", default="model_manifest.json")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    if str(root / "src") not in os.sys.path:
        os.sys.path.insert(0, str(root / "src"))

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from peft import PeftModel, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    from pmldl_llm.config import validate_fold_roles
    from pmldl_llm.constants import TARGET_COLUMNS
    from pmldl_llm.data import (
        decode_turns,
        load_checksum_manifest,
        load_competition_data,
        swap_probability_columns,
        target_indices,
        verify_competition_data_dir,
    )
    from pmldl_llm.evaluation import evaluate_probabilities, normalize_probabilities
    from pmldl_llm.split import load_frozen_folds
    from pmldl_llm.truncation import balanced_head_tail_truncate

    config_path = args.config if args.config.is_absolute() else root / args.config
    adapter_archive = args.adapter_archive.resolve()
    source_manifest_path = args.source_manifest.resolve()
    output_dir = args.output_dir.resolve()
    for path in (config_path, adapter_archive, source_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    training = config["training"]
    candidate = training["qlora"]
    if int(config["seed"]) != 42:
        raise ValueError("Inference handoff is frozen to the selected seed 42.")
    if source_manifest["run_id"] != (
        "E20260914010000000000__s42__e76c1fcb__20260913T215932089199Z"
    ):
        raise ValueError("Unexpected source run: " + str(source_manifest["run_id"]))
    expected_adapter_hash = source_manifest["artifacts"]["adapter"]["sha256"]
    actual_adapter_hash = sha256(adapter_archive)
    if actual_adapter_hash != expected_adapter_hash:
        raise ValueError(
            f"Adapter SHA256 mismatch: {actual_adapter_hash} != {expected_adapter_hash}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA inference requires CUDA.")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    if "H100" not in gpu_name.upper():
        raise RuntimeError(f"Expected H100, got {gpu_name!r}")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    max_length = int(training["max_length"])

    data_candidates = [
        Path(os.environ[name])
        for name in ("PMLDL_DATA_DIR",)
        if os.environ.get(name)
    ]
    data_candidates += [
        root / "data" / "llm-classification-finetuning",
        Path("/kaggle/input/llm-classification-finetuning"),
    ]
    data_dir = next(
        (path for path in data_candidates if (path / "train.csv").is_file()),
        None,
    )
    if data_dir is None:
        raise FileNotFoundError(
            "Competition data not found: " + ", ".join(map(str, data_candidates))
        )

    hashes = load_checksum_manifest(root / "data" / "checksums.sha256")
    verified_hashes = verify_competition_data_dir(data_dir, hashes)
    train_frame, test_frame = load_competition_data(data_dir)
    split_path = root / "configs" / "split.json"
    roles = validate_fold_roles(json.loads(split_path.read_text(encoding="utf-8")))
    if roles.calibration_fold != 8 or roles.final_holdout_fold != 9:
        raise ValueError("Expected calibration fold 8 and final holdout fold 9.")
    target_fold = int(args.target_fold)
    allowed_folds = {roles.calibration_fold, roles.final_holdout_fold}
    if target_fold not in allowed_folds:
        raise ValueError(f"target fold must be one of {sorted(allowed_folds)}")
    folds = load_frozen_folds(
        train_frame,
        root / "data" / "splits" / "folds.csv",
        split_path,
        n_splits=roles.n_splits,
        metadata_path=root / "data" / "splits" / "metadata.json",
        dataset_hashes=hashes,
    )
    fold_values = folds["fold"].to_numpy()
    target_indices_in_train = np.flatnonzero(fold_values == target_fold)
    evaluation_part = train_frame.iloc[target_indices_in_train].reset_index(drop=True)
    targets = target_indices(evaluation_part)

    with tempfile.TemporaryDirectory(prefix="pmldl-qlora-adapter-") as temporary:
        adapter_dir = Path(temporary) / "adapter"
        adapter_dir.mkdir()
        with ZipFile(adapter_archive) as archive:
            archive.extractall(adapter_dir)

        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
        cls_id = tokenizer.cls_token_id or tokenizer.bos_token_id
        sep_id = tokenizer.sep_token_id or tokenizer.eos_token_id
        if cls_id is None or sep_id is None or tokenizer.pad_token_id is None:
            raise RuntimeError("Tokenizer needs CLS/BOS, SEP/EOS and PAD tokens.")

        def render(value: object) -> str:
            turns: list[str] = []
            for index, turn in enumerate(decode_turns(value)):
                if turn is None:
                    text = "null_response"
                elif isinstance(turn, str):
                    text = turn
                else:
                    text = json.dumps(
                        turn,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                turns.append(f"<turn_{index}> {text}")
            return "\n<turn_boundary>\n".join(turns)

        def encode(frame: pd.DataFrame, swap: bool) -> list[list[int]]:
            sequences: list[list[int]] = []
            for row in frame.itertuples(index=False):
                response_a, response_b = (
                    (row.response_b, row.response_a)
                    if swap
                    else (row.response_a, row.response_b)
                )
                prompt_ids = tokenizer.encode(
                    render(row.prompt), add_special_tokens=False
                )
                response_a_ids = tokenizer.encode(
                    render(response_a), add_special_tokens=False
                )
                response_b_ids = tokenizer.encode(
                    render(response_b), add_special_tokens=False
                )
                prompt_ids, response_a_ids, response_b_ids, _ = (
                    balanced_head_tail_truncate(
                        prompt_ids,
                        response_a_ids,
                        response_b_ids,
                        max_length=max_length,
                        special_tokens=4,
                        budget_weights=(1, 2, 2),
                    )
                )
                sequences.append(
                    [
                        cls_id,
                        *prompt_ids,
                        sep_id,
                        *response_a_ids,
                        sep_id,
                        *response_b_ids,
                        sep_id,
                    ]
                )
            return sequences

        class PreferenceDataset(Dataset):
            def __init__(self, sequences: list[list[int]]) -> None:
                self.sequences = sequences

            def __len__(self) -> int:
                return len(self.sequences)

            def __getitem__(self, index: int) -> dict[str, object]:
                return {"input_ids": self.sequences[index]}

        def collate(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
            longest = min(
                max_length,
                int(
                    math.ceil(max(len(item["input_ids"]) for item in batch) / 8)
                    * 8
                ),
            )
            input_ids = torch.full(
                (len(batch), longest),
                int(tokenizer.pad_token_id),
                dtype=torch.long,
            )
            attention = torch.zeros((len(batch), longest), dtype=torch.long)
            for index, item in enumerate(batch):
                values = item["input_ids"]
                input_ids[index, : len(values)] = torch.tensor(
                    values, dtype=torch.long
                )
                attention[index, : len(values)] = 1
            return {"input_ids": input_ids, "attention_mask": attention}

        model = AutoModelForSequenceClassification.from_pretrained(
            config["model"]["name"],
            revision=config["model"]["revision"],
            num_labels=3,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=str(candidate["quantization"]),
                bnb_4bit_use_double_quant=bool(candidate["double_quantization"]),
                bnb_4bit_compute_dtype=compute_dtype,
                llm_int8_skip_modules=["pooler", "classifier"],
            ),
            device_map={"": 0},
        )
        model.config.pad_token_id = int(tokenizer.pad_token_id)
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False
        )
        model = PeftModel.from_pretrained(
            model, adapter_dir, is_trainable=False
        )

        def cast_quantized_input(
            module: object, hook_args: tuple[object, ...]
        ) -> tuple[object, ...] | None:
            del module
            if (
                hook_args
                and torch.is_tensor(hook_args[0])
                and hook_args[0].dtype != compute_dtype
            ):
                return (hook_args[0].to(compute_dtype), *hook_args[1:])
            return None

        for module in model.modules():
            if type(module).__name__ == "Linear4bit":
                module.register_forward_pre_hook(cast_quantized_input)

        def predict(sequences: list[list[int]]) -> np.ndarray:
            loader = DataLoader(
                PreferenceDataset(sequences),
                batch_size=64,
                shuffle=False,
                collate_fn=collate,
                pin_memory=True,
            )
            values: list[np.ndarray] = []
            model.eval()
            with torch.inference_mode():
                for batch in loader:
                    batch = {
                        key: value.to(device, non_blocking=True)
                        for key, value in batch.items()
                    }
                    with torch.autocast(device_type="cuda", dtype=compute_dtype):
                        logits = model(**batch).logits
                    values.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
            return normalize_probabilities(np.concatenate(values))

        def predict_frame(frame: pd.DataFrame) -> np.ndarray:
            original = predict(encode(frame, swap=False))
            swapped_back = swap_probability_columns(predict(encode(frame, swap=True)))
            return normalize_probabilities(0.5 * (original + swapped_back))

        averaged = predict_frame(evaluation_part)
        kaggle_probabilities = (
            predict_frame(test_frame) if args.include_kaggle_test else None
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"fold{target_fold}_predictions.csv"
    prediction_frame = pd.DataFrame(
        {
            "id": evaluation_part["id"].to_numpy(),
            "target": targets,
            **{
                name: averaged[:, index]
                for index, name in enumerate(TARGET_COLUMNS)
            },
        }
    )
    expected_columns = ["id", "target", *TARGET_COLUMNS]
    if list(prediction_frame.columns) != expected_columns:
        raise RuntimeError("Prediction columns drifted from the frozen schema.")
    if prediction_frame.empty or prediction_frame["id"].duplicated().any():
        raise RuntimeError("Prediction ids must be non-empty and unique.")
    probabilities = prediction_frame.loc[:, list(TARGET_COLUMNS)].to_numpy()
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise RuntimeError("Probabilities must be finite and non-negative.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7, rtol=0.0):
        raise RuntimeError("Probabilities must sum to one.")
    prediction_frame.to_csv(prediction_path, index=False)

    kaggle_path: Path | None = None
    if kaggle_probabilities is not None:
        kaggle_frame = pd.DataFrame(
            {
                "id": test_frame["id"].to_numpy(),
                **{
                    name: kaggle_probabilities[:, index]
                    for index, name in enumerate(TARGET_COLUMNS)
                },
            }
        )
        kaggle_columns = ["id", *TARGET_COLUMNS]
        if list(kaggle_frame.columns) != kaggle_columns:
            raise RuntimeError("Kaggle prediction columns drifted from the schema.")
        if kaggle_frame.empty or kaggle_frame["id"].duplicated().any():
            raise RuntimeError("Kaggle prediction ids must be non-empty and unique.")
        kaggle_values = kaggle_frame.loc[:, list(TARGET_COLUMNS)].to_numpy()
        if not np.isfinite(kaggle_values).all() or (kaggle_values < 0).any():
            raise RuntimeError("Kaggle probabilities must be finite and non-negative.")
        if not np.allclose(kaggle_values.sum(axis=1), 1.0, atol=1e-7, rtol=0.0):
            raise RuntimeError("Kaggle probabilities must sum to one.")
        sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
        if not np.array_equal(
            kaggle_frame["id"].to_numpy(), sample_submission["id"].to_numpy()
        ):
            raise RuntimeError("Kaggle prediction ids differ from sample_submission.csv.")
        kaggle_path = output_dir / "inference_predictions.csv"
        kaggle_frame.to_csv(kaggle_path, index=False)

    is_final_holdout = target_fold == roles.final_holdout_fold
    artifact_name = f"fold{target_fold}_predictions"
    protocol = {
        **source_manifest["protocol"],
        "calibration_fold": roles.calibration_fold,
        "final_holdout_fold": roles.final_holdout_fold,
        "target_fold": target_fold,
        "final_holdout_fold_opened": is_final_holdout,
    }
    if not is_final_holdout:
        protocol["final_holdout_fold_unopened"] = roles.final_holdout_fold
    artifacts = {
        **source_manifest["artifacts"],
        artifact_name: {
            "path": prediction_path.name,
            "sha256": sha256(prediction_path),
            "bytes": prediction_path.stat().st_size,
            "rows": len(prediction_frame),
            "columns": expected_columns,
        },
    }
    if kaggle_path is not None:
        artifacts["kaggle_test_predictions"] = {
            "path": kaggle_path.name,
            "sha256": sha256(kaggle_path),
            "bytes": kaggle_path.stat().st_size,
            "rows": len(test_frame),
            "columns": ["id", *TARGET_COLUMNS],
        }
    inference_metadata: dict[str, object] = {
        "code_commit": git_commit(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_adapter_sha256": actual_adapter_hash,
        "data_hashes": verified_hashes,
        "gpu": gpu_name,
        "compute_dtype": str(compute_dtype),
        "swap_averaged_inference": True,
        "fold9_accessed": is_final_holdout,
    }
    if is_final_holdout:
        metrics = evaluate_probabilities(targets, averaged)
        one_hot = np.eye(3, dtype=np.float64)[targets]
        metrics["brier_score"] = float(
            np.square(averaged - one_hot).sum(axis=1).mean()
        )
        inference_metadata["fold9_metrics"] = metrics

    manifest = {
        **source_manifest,
        "handoff_role": "calibration" if not is_final_holdout else "final_holdout",
        "protocol": protocol,
        "artifacts": artifacts,
        "fold8_inference" if not is_final_holdout else "final_inference": inference_metadata,
    }
    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "source_run_id": source_manifest["run_id"],
                "target_fold": target_fold,
                "final_holdout_fold_accessed": is_final_holdout,
                "rows": len(prediction_frame),
                "prediction_path": str(prediction_path),
                "prediction_sha256": sha256(prediction_path),
                "kaggle_prediction_path": str(kaggle_path) if kaggle_path else None,
                "kaggle_prediction_sha256": sha256(kaggle_path) if kaggle_path else None,
                "manifest_path": str(manifest_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
