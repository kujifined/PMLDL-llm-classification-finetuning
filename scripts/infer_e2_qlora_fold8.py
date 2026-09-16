#!/usr/bin/env python3
"""Generate the frozen seed-42 QLoRA probabilities for calibration fold 8.

This program is inference-only. It accepts the selected adapter from the
completed fold-7 shortlist, reads only the configured calibration fold, writes
the exact handoff schema requested by the final-ensemble pipeline, and records
the output SHA256 in an updated manifest. The final holdout fold is never read.
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
    from pmldl_llm.evaluation import normalize_probabilities
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
        raise ValueError("Fold-8 handoff is frozen to the selected seed 42.")
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
        raise RuntimeError("Fold-8 QLoRA inference requires CUDA.")
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
    train_frame, _ = load_competition_data(data_dir)
    split_path = root / "configs" / "split.json"
    roles = validate_fold_roles(json.loads(split_path.read_text(encoding="utf-8")))
    if roles.calibration_fold != 8 or roles.final_holdout_fold != 9:
        raise ValueError("Expected calibration fold 8 and final holdout fold 9.")
    folds = load_frozen_folds(
        train_frame,
        root / "data" / "splits" / "folds.csv",
        split_path,
        n_splits=roles.n_splits,
        metadata_path=root / "data" / "splits" / "metadata.json",
        dataset_hashes=hashes,
    )
    fold_values = folds["fold"].to_numpy()
    calibration_indices = np.flatnonzero(fold_values == roles.calibration_fold)
    calibration_part = train_frame.iloc[calibration_indices].reset_index(drop=True)
    targets = target_indices(calibration_part)

    with tempfile.TemporaryDirectory(prefix="pmldl-fold8-adapter-") as temporary:
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

        original_sequences = encode(calibration_part, swap=False)
        swapped_sequences = encode(calibration_part, swap=True)

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

        original = predict(original_sequences)
        swapped_back = swap_probability_columns(predict(swapped_sequences))
        averaged = normalize_probabilities(0.5 * (original + swapped_back))

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "fold8_predictions.csv"
    prediction_frame = pd.DataFrame(
        {
            "id": calibration_part["id"].to_numpy(),
            "target": targets,
            **{
                name: averaged[:, index]
                for index, name in enumerate(TARGET_COLUMNS)
            },
        }
    )
    expected_columns = ["id", "target", *TARGET_COLUMNS]
    if list(prediction_frame.columns) != expected_columns:
        raise RuntimeError("Fold-8 handoff columns drifted from the frozen schema.")
    if prediction_frame.empty or prediction_frame["id"].duplicated().any():
        raise RuntimeError("Fold-8 handoff ids must be non-empty and unique.")
    probabilities = prediction_frame.loc[:, list(TARGET_COLUMNS)].to_numpy()
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise RuntimeError("Fold-8 probabilities must be finite and non-negative.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7, rtol=0.0):
        raise RuntimeError("Fold-8 probabilities must sum to one.")
    prediction_frame.to_csv(prediction_path, index=False)

    manifest = {
        **source_manifest,
        "handoff_role": "calibration",
        "protocol": {
            **source_manifest["protocol"],
            "calibration_fold": roles.calibration_fold,
            "final_holdout_fold_unopened": roles.final_holdout_fold,
        },
        "artifacts": {
            **source_manifest["artifacts"],
            "fold8_predictions": {
                "path": "fold8_predictions.csv",
                "sha256": sha256(prediction_path),
                "bytes": prediction_path.stat().st_size,
                "rows": len(prediction_frame),
                "columns": expected_columns,
            },
        },
        "fold8_inference": {
            "code_commit": git_commit(root),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_adapter_sha256": actual_adapter_hash,
            "data_hashes": verified_hashes,
            "gpu": gpu_name,
            "compute_dtype": str(compute_dtype),
            "swap_averaged_inference": True,
            "fold9_accessed": False,
        },
    }
    manifest_path = output_dir / "model_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "source_run_id": source_manifest["run_id"],
                "calibration_fold": roles.calibration_fold,
                "final_holdout_fold_accessed": False,
                "rows": len(prediction_frame),
                "prediction_path": str(prediction_path),
                "prediction_sha256": sha256(prediction_path),
                "manifest_path": str(manifest_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
