"""E4: QLoRA preference training with an A/B consistency objective.

The control is the three-epoch E2 QLoRA arm.  E4 keeps its model, split,
seed, preprocessing, optimiser, and random A/B augmentation fixed.  The only
training change is an extra Jensen-Shannon penalty between a primary example
and the same example with responses A and B exchanged.
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import random
import shutil
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import validate_fold_roles
from .data import (
    decode_turns,
    load_checksum_manifest,
    load_competition_data,
    swap_probability_columns,
    swap_target_indices,
    target_indices,
    verify_competition_data_dir,
)
from .evaluation import normalize_probabilities
from .notebook import ExperimentOutput, NotebookExperimentSetup
from .split import load_frozen_folds
from .truncation import balanced_head_tail_truncate


E2_EXPERIMENT_ID = "E20260905212645934620"
DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"
DEFAULT_MODEL_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"


def mean_js_divergence(
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return mean Jensen-Shannon divergence for two probability matrices."""
    first = normalize_probabilities(probabilities_a)
    second = normalize_probabilities(probabilities_b)
    if first.shape != second.shape:
        raise ValueError("Jensen-Shannon inputs must have the same shape.")
    mixture = 0.5 * (first + second)
    first_term = first * (
        np.log(np.clip(first, epsilon, None))
        - np.log(np.clip(mixture, epsilon, None))
    )
    second_term = second * (
        np.log(np.clip(second, epsilon, None))
        - np.log(np.clip(mixture, epsilon, None))
    )
    return float(0.5 * np.mean(np.sum(first_term + second_term, axis=1)))


def _render_turns(value: Any) -> str:
    rendered: list[str] = []
    for index, turn in enumerate(decode_turns(value)):
        if turn is None:
            content = "null_response"
        elif isinstance(turn, str):
            content = turn
        else:
            content = json.dumps(
                turn, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        rendered.append(f"<turn_{index}> {content}")
    return "\n<turn_boundary>\n".join(rendered)


def _balanced_smoke_indices(
    mask: np.ndarray,
    targets: np.ndarray,
    *,
    per_class: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_index in range(3):
        candidates = np.flatnonzero(mask & (targets == class_index)).copy()
        rng.shuffle(candidates)
        selected.extend(candidates[:per_class].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _find_data_dir(root: Path) -> Path:
    candidates: list[Path] = []
    configured = os.environ.get("PMLDL_DATA_DIR")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            root / "data" / "llm-classification-finetuning",
            Path("/kaggle/input/llm-classification-finetuning"),
        ]
    )
    for candidate in candidates:
        required = ("train.csv", "test.csv", "sample_submission.csv")
        if all((candidate / name).is_file() for name in required):
            return candidate.resolve()
    checked = ", ".join(path.as_posix() for path in candidates)
    raise FileNotFoundError(f"Competition data is unavailable. Checked: {checked}")


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def run_e4_consistency_experiment(
    run: Any,
    setup: NotebookExperimentSetup,
    project_root: str | Path,
) -> ExperimentOutput:
    """Train and evaluate the E4 QLoRA consistency ablation."""
    try:
        import torch
        import torch.nn.functional as functional
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
    except ImportError as exc:
        raise RuntimeError(
            "E4 requires the packages pinned in requirements-transformer.lock."
        ) from exc

    root = Path(project_root).resolve()
    training = dict(setup.training)
    epochs = int(training.get("epochs", 3))
    effective_batch_size = int(training.get("effective_batch_size", 32))
    max_length = int(training.get("max_length", 512))
    consistency_lambda = float(training.get("consistency_lambda", 0.1))
    learning_rate = float(training.get("learning_rate", 2e-4))
    max_projected_runtime_seconds = float(
        training.get("max_projected_runtime_seconds", 27000)
    )
    seed = int(setup.seed)
    smoke = bool(setup.smoke_test)
    if epochs < 1 or effective_batch_size < 1 or max_length < 16:
        raise ValueError("Invalid E4 training dimensions.")
    if not 0.0 < consistency_lambda <= 1.0:
        raise ValueError("consistency_lambda must be in (0, 1].")
    if not torch.cuda.is_available():
        raise RuntimeError("E4 QLoRA requires a CUDA GPU.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")

    data_dir = _find_data_dir(root)
    manifest_path = root / "data" / "checksums.sha256"
    expected_hashes = load_checksum_manifest(manifest_path)
    verified_hashes = verify_competition_data_dir(data_dir, expected_hashes)
    train_frame, _test_frame = load_competition_data(data_dir)
    split_config_path = root / "configs" / "split.json"
    split_config = json.loads(split_config_path.read_text(encoding="utf-8"))
    roles = validate_fold_roles(split_config)
    folds = load_frozen_folds(
        train_frame,
        root / "data" / "splits" / "folds.csv",
        split_config_path,
        n_splits=roles.n_splits,
        metadata_path=root / "data" / "splits" / "metadata.json",
        dataset_hashes=expected_hashes,
    )
    fold_values = folds["fold"].to_numpy()
    all_targets = target_indices(train_frame)
    train_mask = np.isin(fold_values, list(roles.training_folds))
    validation_mask = fold_values == roles.validation_fold
    if (train_mask & validation_mask).any():
        raise RuntimeError("Training and validation folds overlap.")
    if smoke:
        train_indices = _balanced_smoke_indices(
            train_mask, all_targets, per_class=24, seed=seed
        )
        validation_indices = _balanced_smoke_indices(
            validation_mask, all_targets, per_class=12, seed=seed
        )
    else:
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)
    train_part = train_frame.iloc[train_indices].reset_index(drop=True)
    validation_part = train_frame.iloc[validation_indices].reset_index(drop=True)
    y_validation = target_indices(validation_part)

    model_name = setup.model_name or DEFAULT_MODEL_NAME
    model_revision = setup.model_revision or DEFAULT_MODEL_REVISION
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    cls_token_id = tokenizer.cls_token_id
    if cls_token_id is None:
        cls_token_id = tokenizer.bos_token_id
    sep_token_id = tokenizer.sep_token_id
    if sep_token_id is None:
        sep_token_id = tokenizer.eos_token_id
    if cls_token_id is None or sep_token_id is None or tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer must define CLS/BOS, SEP/EOS and PAD tokens.")

    max_length_in_use = 128 if smoke else max_length

    def encode_rows(
        frame: pd.DataFrame,
        swap_flags: Iterable[bool],
    ) -> tuple[list[list[int]], np.ndarray]:
        flags = np.asarray(list(swap_flags), dtype=bool)
        if len(flags) != len(frame):
            raise ValueError("swap_flags length does not match the frame.")
        labels = target_indices(frame).copy()
        sequences: list[list[int]] = []
        for row_index, row in enumerate(frame.itertuples(index=False)):
            response_a = row.response_b if flags[row_index] else row.response_a
            response_b = row.response_a if flags[row_index] else row.response_b
            prompt_ids = tokenizer.encode(
                _render_turns(row.prompt), add_special_tokens=False
            )
            response_a_ids = tokenizer.encode(
                _render_turns(response_a), add_special_tokens=False
            )
            response_b_ids = tokenizer.encode(
                _render_turns(response_b), add_special_tokens=False
            )
            prompt_ids, response_a_ids, response_b_ids, _stats = (
                balanced_head_tail_truncate(
                    prompt_ids,
                    response_a_ids,
                    response_b_ids,
                    max_length=max_length_in_use,
                    special_tokens=4,
                    budget_weights=(1, 2, 2),
                )
            )
            sequences.append(
                [
                    int(cls_token_id),
                    *prompt_ids,
                    int(sep_token_id),
                    *response_a_ids,
                    int(sep_token_id),
                    *response_b_ids,
                    int(sep_token_id),
                ]
            )
        labels[flags] = swap_target_indices(labels[flags])
        return sequences, labels

    swap_rng = np.random.default_rng(seed)
    primary_swap_flags = swap_rng.random(len(train_part)) < 0.5
    primary_sequences, primary_targets = encode_rows(
        train_part, primary_swap_flags
    )
    counterpart_sequences, counterpart_targets = encode_rows(
        train_part, ~primary_swap_flags
    )
    expected_counterpart_targets = swap_target_indices(primary_targets)
    if not np.array_equal(counterpart_targets, expected_counterpart_targets):
        raise RuntimeError("Paired A/B labels are inconsistent.")
    validation_sequences, _ = encode_rows(
        validation_part, np.zeros(len(validation_part), dtype=bool)
    )
    swapped_validation_sequences, _ = encode_rows(
        validation_part, np.ones(len(validation_part), dtype=bool)
    )

    class PairDataset(Dataset):
        def __len__(self) -> int:
            return len(primary_sequences)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {
                "primary_ids": primary_sequences[index],
                "counterpart_ids": counterpart_sequences[index],
                "label": int(primary_targets[index]),
            }

    class SequenceDataset(Dataset):
        def __init__(self, sequences: list[list[int]]) -> None:
            self.sequences = sequences

        def __len__(self) -> int:
            return len(self.sequences)

        def __getitem__(self, index: int) -> list[int]:
            return self.sequences[index]

    def pad_sequences(sequences: list[list[int]]) -> tuple[Any, Any]:
        longest = max(len(sequence) for sequence in sequences)
        longest = min(max_length_in_use, int(math.ceil(longest / 8.0) * 8))
        input_ids = torch.full(
            (len(sequences), longest),
            int(tokenizer.pad_token_id),
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(sequences), longest), dtype=torch.long)
        for row_index, sequence in enumerate(sequences):
            input_ids[row_index, : len(sequence)] = torch.tensor(
                sequence, dtype=torch.long
            )
            attention_mask[row_index, : len(sequence)] = 1
        return input_ids, attention_mask

    def pair_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        primary = [item["primary_ids"] for item in batch]
        counterpart = [item["counterpart_ids"] for item in batch]
        input_ids, attention_mask = pad_sequences(primary + counterpart)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
            "pair_count": len(batch),
        }

    def sequence_collate(batch: list[list[int]]) -> dict[str, Any]:
        input_ids, attention_mask = pad_sequences(batch)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    gpu_name = torch.cuda.get_device_name(0)
    h100_fast_path = "H100" in gpu_name.upper()
    micro_batch_size = 16 if h100_fast_path else 2
    if effective_batch_size % micro_batch_size:
        raise ValueError("effective_batch_size must be divisible by micro_batch_size.")
    gradient_accumulation_steps = effective_batch_size // micro_batch_size
    evaluation_batch_size = 32 if h100_fast_path else 8
    device = torch.device("cuda:0")
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
        llm_int8_skip_modules=["pooler", "classifier"],
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=model_revision,
        num_labels=3,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True,
        quantization_config=quantization,
        device_map={"": 0},
    )
    model.config.pad_token_id = int(tokenizer.pad_token_id)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora = dict(training.get("lora", {}))
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=list(
                lora.get("target_modules", ["query_proj", "value_proj"])
            ),
            modules_to_save=list(
                lora.get("modules_to_save", ["pooler", "classifier"])
            ),
            bias="none",
        ),
    )

    def cast_quantized_input(_module: Any, arguments: tuple[Any, ...]) -> Any:
        if (
            arguments
            and torch.is_tensor(arguments[0])
            and arguments[0].dtype != compute_dtype
        ):
            return (arguments[0].to(compute_dtype),) + arguments[1:]
        return None

    for submodule in model.modules():
        if type(submodule).__name__ == "Linear4bit":
            submodule.register_forward_pre_hook(cast_quantized_input)

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        PairDataset(),
        batch_size=micro_batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        collate_fn=pair_collate,
        pin_memory=True,
    )
    microbatch_goal = 3 if smoke else len(train_loader) * epochs
    optimizer_steps = math.ceil(microbatch_goal / gradient_accumulation_steps)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(
            optimizer_steps * float(training.get("warmup_ratio", 0.06))
        ),
        num_training_steps=optimizer_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=compute_dtype == torch.float16)
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    model.train()
    curve: list[dict[str, float | int]] = []
    processed_microbatches = 0
    optimizer_step = 0
    accumulated = 0
    group_target = min(gradient_accumulation_steps, microbatch_goal)
    accumulated_ce = 0.0
    accumulated_js = 0.0
    training_started = time.perf_counter()
    projected_runtime_seconds: float | None = None

    while processed_microbatches < microbatch_goal:
        for batch in train_loader:
            if processed_microbatches >= microbatch_goal:
                break
            if accumulated == 0:
                remaining = microbatch_goal - processed_microbatches
                group_target = min(gradient_accumulation_steps, remaining)
            pair_count = int(batch.pop("pair_count"))
            labels = batch.pop("labels").to(device, non_blocking=True)
            inputs = {
                key: value.to(device, non_blocking=True) for key, value in batch.items()
            }
            autocast_context = torch.autocast(
                device_type="cuda", dtype=compute_dtype
            )
            with autocast_context:
                combined_logits = model(**inputs).logits
                primary_logits = combined_logits[:pair_count]
                counterpart_logits = combined_logits[pair_count:]
                ce_loss = functional.cross_entropy(primary_logits, labels)
            counterpart_back_logits = counterpart_logits[:, [1, 0, 2]].float()
            log_primary = functional.log_softmax(primary_logits.float(), dim=-1)
            log_counterpart = functional.log_softmax(
                counterpart_back_logits, dim=-1
            )
            primary_probabilities = log_primary.exp()
            counterpart_probabilities = log_counterpart.exp()
            mixture = 0.5 * (primary_probabilities + counterpart_probabilities)
            log_mixture = torch.log(mixture.clamp_min(1e-12))
            js_loss = 0.5 * (
                (
                    primary_probabilities * (log_primary - log_mixture)
                ).sum(dim=-1)
                + (
                    counterpart_probabilities * (log_counterpart - log_mixture)
                ).sum(dim=-1)
            ).mean()
            loss = ce_loss.float() + consistency_lambda * js_loss
            scaler.scale(loss / group_target).backward()
            accumulated += 1
            processed_microbatches += 1
            accumulated_ce += float(ce_loss.detach().cpu())
            accumulated_js += float(js_loss.detach().cpu())
            if accumulated == group_target:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training.get("max_grad_norm", 1.0))
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                mean_ce = accumulated_ce / group_target
                mean_js = accumulated_js / group_target
                point = {
                    "step": optimizer_step,
                    "cross_entropy": mean_ce,
                    "consistency_js": mean_js,
                    "loss": mean_ce + consistency_lambda * mean_js,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
                curve.append(point)
                if optimizer_step == 20 and not smoke:
                    elapsed = time.perf_counter() - training_started
                    projected_runtime_seconds = elapsed / optimizer_step * optimizer_steps
                    if projected_runtime_seconds > max_projected_runtime_seconds:
                        raise RuntimeError(
                            "Projected E4 training runtime exceeds the configured safety "
                            f"limit: {projected_runtime_seconds:.0f}s > "
                            f"{max_projected_runtime_seconds:.0f}s."
                        )
                if optimizer_step == 1 or optimizer_step % 20 == 0:
                    run.log_metrics(
                        {
                            "loss": float(point["loss"]),
                            "cross_entropy": mean_ce,
                            "consistency_js": mean_js,
                            "learning_rate": float(point["learning_rate"]),
                        },
                        namespace="train",
                        step=optimizer_step,
                    )
                accumulated = 0
                accumulated_ce = 0.0
                accumulated_js = 0.0
        if len(train_loader) == 0:
            raise RuntimeError("Training dataloader is empty.")

    train_runtime_seconds = time.perf_counter() - training_started
    if optimizer_step != optimizer_steps:
        raise RuntimeError(
            f"Expected {optimizer_steps} optimizer steps, got {optimizer_step}."
        )

    def predict(sequences: list[list[int]]) -> np.ndarray:
        loader = DataLoader(
            SequenceDataset(sequences),
            batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=sequence_collate,
            pin_memory=True,
        )
        batches: list[np.ndarray] = []
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                inputs = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch.items()
                }
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=compute_dtype)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with autocast_context:
                    logits = model(**inputs).logits
                batches.append(
                    functional.softmax(logits.float(), dim=-1).cpu().numpy()
                )
        if not batches:
            raise RuntimeError("Prediction dataloader is empty.")
        return normalize_probabilities(np.concatenate(batches, axis=0))

    original_probabilities = predict(validation_sequences)
    swapped_probabilities = predict(swapped_validation_sequences)
    swapped_back_probabilities = swap_probability_columns(swapped_probabilities)
    averaged_probabilities = normalize_probabilities(
        0.5 * (original_probabilities + swapped_back_probabilities)
    )
    validation_js = mean_js_divergence(
        original_probabilities, swapped_back_probabilities
    )
    peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated() / (1024**2))

    predictions_path = run.artifact_path("predictions/validation_predictions.csv")
    predictions = pd.DataFrame(
        {
            "id": validation_part["id"].astype(str),
            "target": y_validation,
            "original_winner_model_a": original_probabilities[:, 0],
            "original_winner_model_b": original_probabilities[:, 1],
            "original_winner_tie": original_probabilities[:, 2],
            "swapped_back_winner_model_a": swapped_back_probabilities[:, 0],
            "swapped_back_winner_model_b": swapped_back_probabilities[:, 1],
            "swapped_back_winner_tie": swapped_back_probabilities[:, 2],
            "winner_model_a": averaged_probabilities[:, 0],
            "winner_model_b": averaged_probabilities[:, 1],
            "winner_tie": averaged_probabilities[:, 2],
        }
    )
    predictions.to_csv(predictions_path, index=False)
    curve_path = _write_json(run.artifact_path("curves/training.json"), curve)
    study_path = _write_json(
        run.artifact_path("study.json"),
        {
            "experiment_id": setup.experiment_id,
            "parent_experiment_id": setup.parent_experiment_id,
            "control": {
                "experiment_id": E2_EXPERIMENT_ID,
                "qlora_log_loss": 1.03428689372673,
                "qlora_swap_error_l1": 0.16203328866204142,
            },
            "changed_factor": {
                "name": "ab_consistency_loss",
                "lambda": consistency_lambda,
                "cross_entropy_view": "same deterministic random swap as E2",
                "consistency_partner": "opposite A/B order",
                "divergence": "Jensen-Shannon",
            },
            "protocol": {
                "seed": seed,
                "training_folds": list(roles.training_folds),
                "validation_fold": roles.validation_fold,
                "calibration_fold_unopened": roles.calibration_fold,
                "final_holdout_fold_unopened": roles.final_holdout_fold,
                "train_rows": len(train_part),
                "validation_rows": len(validation_part),
                "epochs": epochs,
                "max_length": max_length_in_use,
                "effective_batch_size": effective_batch_size,
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "optimizer_steps": optimizer_steps,
                "learning_rate": learning_rate,
            },
            "diagnostics": {
                "validation_consistency_js": validation_js,
                "train_runtime_seconds": train_runtime_seconds,
                "projected_runtime_seconds": projected_runtime_seconds,
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
            },
        },
    )
    environment_path = _write_json(
        run.artifact_path("environment.json"),
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
            "bitsandbytes": __import__("bitsandbytes").__version__,
            "cuda": torch.version.cuda,
            "gpu_name": gpu_name,
            "compute_dtype": str(compute_dtype),
            "verified_data_hashes": verified_hashes,
            "model": {"name": model_name, "revision": model_revision},
        },
    )

    checkpoint_base = run.artifact_path("models/e4_consistency_qlora")
    with tempfile.TemporaryDirectory(prefix="e4-checkpoint-", dir=run.artifact_dir) as temporary:
        checkpoint_directory = Path(temporary) / "checkpoint"
        model.save_pretrained(checkpoint_directory, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint_directory)
        checkpoint_path = Path(
            shutil.make_archive(
                str(checkpoint_base), "zip", root_dir=checkpoint_directory
            )
        )

    run.log_metrics(
        {
            "consistency_lambda": consistency_lambda,
            "validation_consistency_js": validation_js,
            "train_runtime_seconds": train_runtime_seconds,
            "projected_runtime_seconds": projected_runtime_seconds or 0.0,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "trainable_parameters": float(trainable_parameters),
            "total_parameters": float(total_parameters),
        },
        namespace="e4_diagnostics",
    )
    artifacts = {
        "predictions/validation_predictions.csv": predictions_path,
        "curves/training.json": curve_path,
        "study.json": study_path,
        "environment.json": environment_path,
        "models/e4_consistency_qlora.zip": checkpoint_path,
    }
    output = ExperimentOutput.from_predictions(
        y_true=y_validation,
        original_probabilities=original_probabilities,
        swapped_back_probabilities=swapped_back_probabilities,
        artifacts=artifacts,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return output
