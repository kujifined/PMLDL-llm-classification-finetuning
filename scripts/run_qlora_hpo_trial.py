#!/usr/bin/env python3
"""Run one pre-registered, epoch-aware QLoRA selection trial on one GPU.

The launcher starts this script once per candidate.  It intentionally does not
query Kaggle, fold 8, or fold 9: all decisions use the frozen selection fold 7.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    if str(root / "src") not in os.sys.path:
        os.sys.path.insert(0, str(root / "src"))

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup, set_seed

    from pmldl_llm.config import validate_fold_roles
    from pmldl_llm.data import (decode_turns, load_checksum_manifest, load_competition_data,
        swap_probability_columns, swap_target_indices, target_indices, verify_competition_data_dir)
    from pmldl_llm.evaluation import evaluate_experiment_probabilities, normalize_probabilities
    from pmldl_llm.experiment import ExperimentRun
    from pmldl_llm.run_validation import validate_run_directory
    from pmldl_llm.split import load_frozen_folds
    from pmldl_llm.truncation import balanced_head_tail_truncate

    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training = config["training"]
    candidates = {candidate["id"]: candidate for candidate in training["hpo_trials"]}
    if args.trial_id not in candidates:
        raise ValueError(f"Unknown trial {args.trial_id!r}; expected {sorted(candidates)}")
    candidate = candidates[args.trial_id]
    selection = training["selection"]
    seed = int(config["seed"])
    smoke = bool(config["smoke_test"])
    max_length = 128 if smoke else int(training["max_length"])
    effective_batch_size = int(training["effective_batch_size"])
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA HPO requires CUDA.")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    if "H100" not in gpu_name.upper():
        raise RuntimeError(f"Expected H100, got {gpu_name!r}")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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

    data_candidates = [Path(os.environ[name]) for name in ("PMLDL_DATA_DIR",) if os.environ.get(name)]
    data_candidates += [root / "data" / "llm-classification-finetuning", Path("/kaggle/input/llm-classification-finetuning")]
    data_dir = next((path for path in data_candidates if (path / "train.csv").is_file()), None)
    if data_dir is None:
        raise FileNotFoundError("Competition data not found: " + ", ".join(map(str, data_candidates)))
    hashes = load_checksum_manifest(root / "data" / "checksums.sha256")
    verified_hashes = verify_competition_data_dir(data_dir, hashes)
    train_frame, _ = load_competition_data(data_dir)
    split_path = root / "configs" / "split.json"
    roles = validate_fold_roles(json.loads(split_path.read_text(encoding="utf-8")))
    folds = load_frozen_folds(train_frame, root / "data" / "splits" / "folds.csv", split_path,
        n_splits=roles.n_splits, metadata_path=root / "data" / "splits" / "metadata.json", dataset_hashes=hashes)
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
        train_indices, validation_indices = balanced_indices(train_mask, 24), balanced_indices(validation_mask, 12)
    else:
        train_indices, validation_indices = np.flatnonzero(train_mask), np.flatnonzero(validation_mask)
    train_part = train_frame.iloc[train_indices].reset_index(drop=True)
    validation_part = train_frame.iloc[validation_indices].reset_index(drop=True)
    y_train, y_validation = target_indices(train_part), target_indices(validation_part)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], revision=config["model"]["revision"], use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    cls_id = tokenizer.cls_token_id or tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id or tokenizer.eos_token_id
    if cls_id is None or sep_id is None or tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer needs CLS/BOS, SEP/EOS and PAD tokens.")

    def render(value: object) -> str:
        turns: list[str] = []
        for index, turn in enumerate(decode_turns(value)):
            text = "null_response" if turn is None else (turn if isinstance(turn, str) else json.dumps(turn, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            turns.append(f"<turn_{index}> {text}")
        return "\n<turn_boundary>\n".join(turns)

    def encode(frame: pd.DataFrame, swaps: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
        labels = target_indices(frame).copy()
        sequences: list[list[int]] = []
        for index, row in enumerate(frame.itertuples(index=False)):
            a, b = (row.response_b, row.response_a) if swaps[index] else (row.response_a, row.response_b)
            prompt_ids = tokenizer.encode(render(row.prompt), add_special_tokens=False)
            a_ids = tokenizer.encode(render(a), add_special_tokens=False)
            b_ids = tokenizer.encode(render(b), add_special_tokens=False)
            prompt_ids, a_ids, b_ids, _ = balanced_head_tail_truncate(prompt_ids, a_ids, b_ids, max_length=max_length, special_tokens=4, budget_weights=(1, 2, 2))
            sequences.append([cls_id, *prompt_ids, sep_id, *a_ids, sep_id, *b_ids, sep_id])
        labels[swaps] = swap_target_indices(labels[swaps])
        return sequences, labels

    class PreferenceDataset(Dataset):
        def __init__(self, sequences: list[list[int]], labels: np.ndarray | None = None) -> None:
            self.sequences, self.labels = sequences, labels
        def __len__(self) -> int: return len(self.sequences)
        def __getitem__(self, index: int) -> dict[str, object]:
            item: dict[str, object] = {"input_ids": self.sequences[index]}
            if self.labels is not None: item["labels"] = int(self.labels[index])
            return item

    def collate(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        longest = min(max_length, int(math.ceil(max(len(item["input_ids"]) for item in batch) / 8) * 8))
        input_ids = torch.full((len(batch), longest), int(tokenizer.pad_token_id), dtype=torch.long)
        attention = torch.zeros((len(batch), longest), dtype=torch.long)
        labels: list[int] = []
        for index, item in enumerate(batch):
            values = item["input_ids"]
            input_ids[index, :len(values)] = torch.tensor(values, dtype=torch.long)
            attention[index, :len(values)] = 1
            if "labels" in item: labels.append(int(item["labels"]))
        output = {"input_ids": input_ids, "attention_mask": attention}
        if labels: output["labels"] = torch.tensor(labels, dtype=torch.long)
        return output

    rng = np.random.default_rng(seed)
    train_sequences, encoded_y_train = encode(train_part, rng.random(len(train_part)) < 0.5)
    validation_sequences, _ = encode(validation_part, np.zeros(len(validation_part), dtype=bool))
    swapped_validation_sequences, _ = encode(validation_part, np.ones(len(validation_part), dtype=bool))

    load_kwargs: dict[str, object] = {"revision": config["model"]["revision"], "num_labels": 3, "ignore_mismatched_sizes": True, "low_cpu_mem_usage": True,
        "quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype, llm_int8_skip_modules=["pooler", "classifier"]), "device_map": {"": 0}}
    model = AutoModelForSequenceClassification.from_pretrained(config["model"]["name"], **load_kwargs)
    model.config.pad_token_id = int(tokenizer.pad_token_id)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=int(candidate["r"]), lora_alpha=int(candidate["alpha"]), lora_dropout=float(candidate["dropout"]), target_modules=["query_proj", "value_proj"], modules_to_save=["pooler", "classifier"], bias="none"))

    def cast_quantized_input(module: object, hook_args: tuple[object, ...]) -> tuple[object, ...] | None:
        if hook_args and torch.is_tensor(hook_args[0]) and hook_args[0].dtype != compute_dtype:
            return (hook_args[0].to(compute_dtype), *hook_args[1:])
        return None
    for module in model.modules():
        if type(module).__name__ == "Linear4bit": module.register_forward_pre_hook(cast_quantized_input)

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    batches_per_epoch = math.ceil(len(train_sequences) / micro_batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / grad_accumulation)
    max_epochs = 1 if smoke else int(selection["max_epochs"])
    optimizer = torch.optim.AdamW(parameters, lr=float(candidate["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=math.ceil(updates_per_epoch * max_epochs * float(training["warmup_ratio"])), num_training_steps=updates_per_epoch * max_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=compute_dtype == torch.float16)

    def predict(sequences: list[list[int]]) -> np.ndarray:
        loader = DataLoader(PreferenceDataset(sequences), batch_size=64, shuffle=False, collate_fn=collate, pin_memory=True)
        result: list[np.ndarray] = []
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=compute_dtype): logits = model(**batch).logits
                result.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
        return normalize_probabilities(np.concatenate(result))

    def checkpoint(run: ExperimentRun) -> Path:
        base = run.artifact_path("models/best_checkpoint")
        with tempfile.TemporaryDirectory(dir=run.artifact_dir, prefix="best-") as temporary:
            directory = Path(temporary) / "checkpoint"
            model.save_pretrained(directory, safe_serialization=True)
            tokenizer.save_pretrained(directory)
            return Path(shutil.make_archive(str(base), "zip", root_dir=directory))

    with ExperimentRun(config_path, project_root=root) as run:
        run.log_metrics({"rank": candidate["r"], "dropout": candidate["dropout"], "learning_rate": candidate["learning_rate"]}, namespace="trial_parameters")
        best_metrics: dict[str, float] | None = None
        best_epoch, non_improving, global_step = 0, 0, 0
        history: list[dict[str, object]] = []
        started = time.perf_counter()
        for epoch in range(1, max_epochs + 1):
            reset_seed(epoch)
            loader = DataLoader(PreferenceDataset(train_sequences, encoded_y_train), batch_size=micro_batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed + epoch), collate_fn=collate, pin_memory=True)
            model.train(); optimizer.zero_grad(set_to_none=True); accumulated = 0; loss_sum = 0.0
            for batch_index, batch in enumerate(loader, start=1):
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=compute_dtype): loss = model(**batch).loss
                scaler.scale(loss / grad_accumulation).backward(); accumulated += 1; loss_sum += float(loss.detach().cpu())
                if accumulated == grad_accumulation or batch_index == len(loader):
                    scaler.unscale_(optimizer)
                    if accumulated != grad_accumulation:
                        for parameter in parameters:
                            if parameter.grad is not None: parameter.grad.mul_(grad_accumulation / accumulated)
                    torch.nn.utils.clip_grad_norm_(parameters, float(training["max_grad_norm"]))
                    scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                    global_step += 1; accumulated = 0
            original = predict(validation_sequences)
            swapped_back = swap_probability_columns(predict(swapped_validation_sequences))
            metrics = evaluate_experiment_probabilities(y_validation, original, swapped_back)
            metrics["train_loss"] = loss_sum / len(loader)
            metrics["learning_rate"] = float(scheduler.get_last_lr()[0])
            run.log_metrics(metrics, namespace="selection_validation", step=epoch)
            entry = {"epoch": epoch, "optimizer_steps": global_step, **metrics}
            history.append(entry)
            if best_metrics is None or float(metrics["log_loss"]) < float(best_metrics["log_loss"]) - float(selection["min_improvement"]):
                best_metrics, best_epoch, non_improving = dict(metrics), epoch, 0
                checkpoint_path = checkpoint(run)
            else:
                non_improving += 1
                if non_improving >= int(selection["early_stopping_patience_epochs"]): break
        assert best_metrics is not None
        payload = {"trial_id": args.trial_id, "candidate": candidate, "best_epoch": best_epoch, "best_metrics": best_metrics, "history": history,
            "protocol": {"selection_fold": roles.validation_fold, "calibration_fold_unopened": roles.calibration_fold, "final_holdout_fold_unopened": roles.final_holdout_fold, "data_hashes": verified_hashes, "gpu": gpu_name, "elapsed_seconds": time.perf_counter() - started}}
        study_path = run.artifact_path("study.json")
        study_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        run.log_artifact("study.json", study_path)
        run.log_artifact("models/best_checkpoint.zip", checkpoint_path)
        run.log_metrics(best_metrics, namespace="validation")
    validation = validate_run_directory(run.result_dir, project_root=root)
    if not validation.valid: raise RuntimeError("Invalid run: " + "; ".join(validation.errors))
    print(json.dumps({"run_id": run.run_id, "trial_id": args.trial_id, "best_epoch": best_epoch, "log_loss": best_metrics["log_loss"]}, ensure_ascii=False))
    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
