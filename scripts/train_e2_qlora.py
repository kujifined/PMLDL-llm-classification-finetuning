#!/usr/bin/env python3
"""Train one frozen Sprint 2 E2 QLoRA seed and export fold-7 evidence.

The winning rank, learning rate, dropout, epoch count and scheduler horizon
come from the completed seed-42 HPO.  Sprint 2 changes only the training seed.
Fold 8, fold 9 and Kaggle are never read by this program.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    if str(root / "src") not in os.sys.path:
        os.sys.path.insert(0, str(root / "src"))

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
        get_linear_schedule_with_warmup,
        set_seed,
    )

    from pmldl_llm.config import validate_fold_roles
    from pmldl_llm.constants import TARGET_COLUMNS
    from pmldl_llm.data import (
        decode_turns,
        load_checksum_manifest,
        load_competition_data,
        swap_probability_columns,
        swap_target_indices,
        target_indices,
        verify_competition_data_dir,
    )
    from pmldl_llm.evaluation import (
        evaluate_experiment_probabilities,
        normalize_probabilities,
    )
    from pmldl_llm.experiment import ExperimentRun
    from pmldl_llm.run_validation import validate_run_directory
    from pmldl_llm.split import load_frozen_folds
    from pmldl_llm.truncation import balanced_head_tail_truncate

    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training = config["training"]
    candidate = training["qlora"]
    seed = int(config["seed"])
    smoke = bool(config["smoke_test"])
    max_length = 128 if smoke else int(training["max_length"])
    epochs = 1 if smoke else int(training["epochs"])
    scheduler_horizon = (
        epochs if smoke else int(training["scheduler_horizon_epochs"])
    )
    if epochs < 1 or scheduler_horizon < epochs:
        raise ValueError("scheduler_horizon_epochs must be at least epochs.")

    if not torch.cuda.is_available():
        raise RuntimeError("Sprint 2 QLoRA training requires CUDA.")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    if "H100" not in gpu_name.upper():
        raise RuntimeError(f"Expected H100, got {gpu_name!r}")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    effective_batch_size = int(training["effective_batch_size"])
    micro_batch_size = min(effective_batch_size, 32)
    grad_accumulation = math.ceil(effective_batch_size / micro_batch_size)

    def reset_seed(offset: int = 0) -> None:
        value = seed + offset
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)
        set_seed(value)

    reset_seed()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")

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
    folds = load_frozen_folds(
        train_frame,
        root / "data" / "splits" / "folds.csv",
        split_path,
        n_splits=roles.n_splits,
        metadata_path=root / "data" / "splits" / "metadata.json",
        dataset_hashes=hashes,
    )
    fold_values = folds["fold"].to_numpy()
    train_mask = np.isin(fold_values, roles.training_folds)
    validation_mask = fold_values == roles.validation_fold
    all_targets = target_indices(train_frame)

    if smoke:
        def balanced_indices(mask: np.ndarray, per_class: int) -> np.ndarray:
            chosen: list[int] = []
            local_rng = np.random.default_rng(seed)
            for class_index in range(3):
                values = np.flatnonzero(mask & (all_targets == class_index))
                local_rng.shuffle(values)
                chosen.extend(values[:per_class])
            return np.asarray(sorted(chosen), dtype=np.int64)

        train_indices = balanced_indices(train_mask, 24)
        validation_indices = balanced_indices(validation_mask, 12)
    else:
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)

    train_part = train_frame.iloc[train_indices].reset_index(drop=True)
    validation_part = train_frame.iloc[validation_indices].reset_index(drop=True)
    y_train = target_indices(train_part)
    y_validation = target_indices(validation_part)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        revision=config["model"]["revision"],
        use_fast=False,
    )
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

    def encode(
        frame: pd.DataFrame,
        swaps: np.ndarray,
    ) -> tuple[list[list[int]], np.ndarray]:
        labels = target_indices(frame).copy()
        sequences: list[list[int]] = []
        for index, row in enumerate(frame.itertuples(index=False)):
            response_a, response_b = (
                (row.response_b, row.response_a)
                if swaps[index]
                else (row.response_a, row.response_b)
            )
            prompt_ids = tokenizer.encode(render(row.prompt), add_special_tokens=False)
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
        labels[swaps] = swap_target_indices(labels[swaps])
        return sequences, labels

    class PreferenceDataset(Dataset):
        def __init__(
            self,
            sequences: list[list[int]],
            labels: np.ndarray | None = None,
        ) -> None:
            self.sequences = sequences
            self.labels = labels

        def __len__(self) -> int:
            return len(self.sequences)

        def __getitem__(self, index: int) -> dict[str, object]:
            item: dict[str, object] = {"input_ids": self.sequences[index]}
            if self.labels is not None:
                item["labels"] = int(self.labels[index])
            return item

    def collate(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        longest = min(
            max_length,
            int(
                math.ceil(max(len(item["input_ids"]) for item in batch) / 8) * 8
            ),
        )
        input_ids = torch.full(
            (len(batch), longest),
            int(tokenizer.pad_token_id),
            dtype=torch.long,
        )
        attention = torch.zeros((len(batch), longest), dtype=torch.long)
        labels: list[int] = []
        for index, item in enumerate(batch):
            values = item["input_ids"]
            input_ids[index, : len(values)] = torch.tensor(values, dtype=torch.long)
            attention[index, : len(values)] = 1
            if "labels" in item:
                labels.append(int(item["labels"]))
        output = {"input_ids": input_ids, "attention_mask": attention}
        if labels:
            output["labels"] = torch.tensor(labels, dtype=torch.long)
        return output

    augmentation_rng = np.random.default_rng(seed)
    train_sequences, encoded_y_train = encode(
        train_part,
        augmentation_rng.random(len(train_part))
        < float(training["augmentation"]["random_ab_swap_probability"]),
    )
    validation_sequences, _ = encode(
        validation_part, np.zeros(len(validation_part), dtype=bool)
    )
    swapped_validation_sequences, _ = encode(
        validation_part, np.ones(len(validation_part), dtype=bool)
    )

    load_kwargs: dict[str, object] = {
        "revision": config["model"]["revision"],
        "num_labels": 3,
        "ignore_mismatched_sizes": True,
        "low_cpu_mem_usage": True,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(candidate["quantization"]),
            bnb_4bit_use_double_quant=bool(candidate["double_quantization"]),
            bnb_4bit_compute_dtype=compute_dtype,
            llm_int8_skip_modules=["pooler", "classifier"],
        ),
        "device_map": {"": 0},
    }
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["name"], **load_kwargs
    )
    model.config.pad_token_id = int(tokenizer.pad_token_id)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(candidate["r"]),
            lora_alpha=int(candidate["alpha"]),
            lora_dropout=float(candidate["dropout"]),
            target_modules=list(candidate["target_modules"]),
            modules_to_save=list(candidate["modules_to_save"]),
            bias="none",
        ),
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

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    batches_per_epoch = math.ceil(len(train_sequences) / micro_batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / grad_accumulation)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(candidate["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(
            updates_per_epoch * scheduler_horizon * float(training["warmup_ratio"])
        ),
        num_training_steps=updates_per_epoch * scheduler_horizon,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=compute_dtype == torch.float16)

    def predict(sequences: list[list[int]]) -> np.ndarray:
        loader = DataLoader(
            PreferenceDataset(sequences),
            batch_size=64,
            shuffle=False,
            collate_fn=collate,
            pin_memory=True,
        )
        result: list[np.ndarray] = []
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch.items()
                }
                with torch.autocast(device_type="cuda", dtype=compute_dtype):
                    logits = model(**batch).logits
                result.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
        return normalize_probabilities(np.concatenate(result))

    def save_checkpoint(run: ExperimentRun) -> Path:
        base = run.artifact_path("models/qlora_adapter")
        with tempfile.TemporaryDirectory(
            dir=run.artifact_dir, prefix="qlora-adapter-"
        ) as temporary:
            directory = Path(temporary) / "checkpoint"
            model.save_pretrained(directory, safe_serialization=True)
            tokenizer.save_pretrained(directory)
            return Path(shutil.make_archive(str(base), "zip", root_dir=directory))

    torch.cuda.reset_peak_memory_stats()
    with ExperimentRun(config_path, project_root=root) as run:
        run.log_metrics(
            {
                "rank": int(candidate["r"]),
                "alpha": int(candidate["alpha"]),
                "dropout": float(candidate["dropout"]),
                "learning_rate": float(candidate["learning_rate"]),
                "epochs": epochs,
                "scheduler_horizon_epochs": scheduler_horizon,
                "seed": seed,
            },
            namespace="training_parameters",
        )
        history: list[dict[str, object]] = []
        global_step = 0
        training_started = time.perf_counter()
        original: np.ndarray | None = None
        swapped_back: np.ndarray | None = None
        final_metrics: dict[str, float] | None = None

        for epoch in range(1, epochs + 1):
            reset_seed(epoch)
            loader = DataLoader(
                PreferenceDataset(train_sequences, encoded_y_train),
                batch_size=micro_batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed + epoch),
                collate_fn=collate,
                pin_memory=True,
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            loss_sum = 0.0
            for batch_index, batch in enumerate(loader, start=1):
                batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch.items()
                }
                with torch.autocast(device_type="cuda", dtype=compute_dtype):
                    loss = model(**batch).loss
                scaler.scale(loss / grad_accumulation).backward()
                accumulated += 1
                loss_sum += float(loss.detach().cpu())
                if accumulated == grad_accumulation or batch_index == len(loader):
                    scaler.unscale_(optimizer)
                    if accumulated != grad_accumulation:
                        for parameter in parameters:
                            if parameter.grad is not None:
                                parameter.grad.mul_(
                                    grad_accumulation / accumulated
                                )
                    torch.nn.utils.clip_grad_norm_(
                        parameters, float(training["max_grad_norm"])
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    accumulated = 0

            original = predict(validation_sequences)
            swapped_back = swap_probability_columns(
                predict(swapped_validation_sequences)
            )
            metrics = evaluate_experiment_probabilities(
                y_validation, original, swapped_back
            )
            metrics["train_loss"] = loss_sum / len(loader)
            metrics["learning_rate"] = float(scheduler.get_last_lr()[0])
            run.log_metrics(metrics, namespace="selection_validation", step=epoch)
            history.append(
                {"epoch": epoch, "optimizer_steps": global_step, **metrics}
            )
            final_metrics = metrics

        assert original is not None
        assert swapped_back is not None
        assert final_metrics is not None
        averaged = normalize_probabilities(0.5 * (original + swapped_back))
        train_seconds = float(time.perf_counter() - training_started)
        peak_gpu_mb = float(torch.cuda.max_memory_allocated() / (1024**2))

        checkpoint_path = save_checkpoint(run)
        prediction_path = run.artifact_path("predictions/fold7_predictions.csv")
        prediction_frame = pd.DataFrame(
            {
                "id": validation_part["id"].to_numpy(),
                "y_true": y_validation,
                **{
                    f"original_{name}": original[:, index]
                    for index, name in enumerate(TARGET_COLUMNS)
                },
                **{
                    f"swapped_back_{name}": swapped_back[:, index]
                    for index, name in enumerate(TARGET_COLUMNS)
                },
                **{
                    name: averaged[:, index]
                    for index, name in enumerate(TARGET_COLUMNS)
                },
            }
        )
        prediction_frame.to_csv(prediction_path, index=False)

        study = {
            "experiment_id": config["experiment_id"],
            "run_id": run.run_id,
            "seed": seed,
            "candidate": candidate,
            "fixed_epochs": epochs,
            "scheduler_horizon_epochs": scheduler_horizon,
            "final_metrics": final_metrics,
            "history": history,
            "protocol": {
                "training_folds": list(roles.training_folds),
                "selection_fold": roles.validation_fold,
                "calibration_fold_unopened": roles.calibration_fold,
                "final_holdout_fold_unopened": roles.final_holdout_fold,
                "data_hashes": verified_hashes,
                "gpu": gpu_name,
                "compute_dtype": str(compute_dtype),
                "optimizer_steps": global_step,
                "train_runtime_seconds": train_seconds,
                "peak_gpu_memory_mb": peak_gpu_mb,
            },
        }
        study_path = run.artifact_path("study.json")
        study_path.write_text(
            json.dumps(study, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        manifest = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "run_id": run.run_id,
            "seed": seed,
            "git_commit": run.git["commit"],
            "model": config["model"],
            "class_order": list(TARGET_COLUMNS),
            "protocol": {
                "training_folds": list(roles.training_folds),
                "selection_fold": roles.validation_fold,
                "epochs": epochs,
                "scheduler_horizon_epochs": scheduler_horizon,
                "swap_averaged_inference": True,
            },
            "artifacts": {
                "adapter": {
                    "path": "models/qlora_adapter.zip",
                    "sha256": sha256(checkpoint_path),
                    "bytes": checkpoint_path.stat().st_size,
                },
                "fold7_predictions": {
                    "path": "predictions/fold7_predictions.csv",
                    "sha256": sha256(prediction_path),
                    "bytes": prediction_path.stat().st_size,
                    "rows": len(prediction_frame),
                },
            },
        }
        manifest_path = run.artifact_path("model_manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

        run.log_artifact("models/qlora_adapter.zip", checkpoint_path)
        run.log_artifact("predictions/fold7_predictions.csv", prediction_path)
        run.log_artifact("study.json", study_path)
        run.log_artifact("model_manifest.json", manifest_path)
        run.log_metrics(
            {
                **{
                    name: value
                    for name, value in final_metrics.items()
                    if name not in {"train_loss", "learning_rate"}
                },
                "train_runtime_seconds": train_seconds,
                "peak_gpu_memory_mb": peak_gpu_mb,
            },
            namespace="validation",
        )

    validation = validate_run_directory(run.result_dir, project_root=root)
    if not validation.valid:
        raise RuntimeError("Invalid run: " + "; ".join(validation.errors))
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "experiment_id": config["experiment_id"],
                "seed": seed,
                "epochs": epochs,
                "log_loss": final_metrics["log_loss"],
                "result_dir": str(run.result_dir),
                "artifact_dir": str(run.artifact_dir),
            },
            ensure_ascii=False,
        )
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
