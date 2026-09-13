"""Sprint 2 Gemma-2 9B LoRA/QLoRA comparison.

The experiment repeats the frozen E2 data and evaluation protocol while
replacing DeBERTa with Gemma-2-9B-IT.  FP16 LoRA and 4-bit NF4 QLoRA share
the same adapter, optimiser, augmentation, truncation, and epoch schedule.
Selection-fold metrics are recorded after epochs 1, 2, and 3.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import tempfile
import time
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
from .evaluation import evaluate_experiment_probabilities, normalize_probabilities
from .notebook import ExperimentOutput, NotebookExperimentSetup
from .split import load_frozen_folds
from .truncation import balanced_head_tail_truncate


E2_EXPERIMENT_ID = "E20260905212645934620"
DEFAULT_MODEL_NAME = "google/gemma-2-9b-it"
DEFAULT_MODEL_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
CLASS_ORDER = ("winner_model_a", "winner_model_b", "winner_tie")


class ProjectedRuntimeLimit(RuntimeError):
    """Raised when a full arm cannot finish inside the configured budget."""


def full_finetune_adamw_memory_lower_bound_gib(
    parameter_count: int,
    *,
    bytes_per_parameter: int = 16,
) -> float:
    """Return an activation-free AdamW memory lower bound.

    The default counts FP16 weights and gradients plus FP32 master weights and
    two FP32 Adam moments.  Activations and allocator overhead are deliberately
    excluded, so passing this gate is necessary but not sufficient.
    """
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive.")
    if bytes_per_parameter <= 0:
        raise ValueError("bytes_per_parameter must be positive.")
    return float(parameter_count * bytes_per_parameter / (1024**3))


def full_finetune_preflight(
    *,
    parameter_count: int,
    total_gpu_memory_gib: float,
    bytes_per_parameter: int = 16,
    usable_fraction: float = 0.9,
) -> dict[str, float | str]:
    """Describe whether full fine-tuning clears a conservative memory gate."""
    if total_gpu_memory_gib <= 0:
        raise ValueError("total_gpu_memory_gib must be positive.")
    if not 0.0 < usable_fraction <= 1.0:
        raise ValueError("usable_fraction must be in (0, 1].")
    required = full_finetune_adamw_memory_lower_bound_gib(
        parameter_count,
        bytes_per_parameter=bytes_per_parameter,
    )
    usable = float(total_gpu_memory_gib * usable_fraction)
    return {
        "status": "allowed" if required <= usable else "skipped_resource_limit",
        "parameter_count_estimate": float(parameter_count),
        "bytes_per_parameter_lower_bound": float(bytes_per_parameter),
        "required_memory_lower_bound_gib": required,
        "total_gpu_memory_gib": float(total_gpu_memory_gib),
        "usable_gpu_memory_gib": usable,
    }


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
    selected: list[int] = []
    rng = np.random.default_rng(seed)
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
        if all(
            (candidate / filename).is_file()
            for filename in ("train.csv", "test.csv", "sample_submission.csv")
        ):
            return candidate.resolve()
    checked = ", ".join(path.as_posix() for path in candidates)
    raise FileNotFoundError(f"Competition data is unavailable. Checked: {checked}")


def _resolve_model_source(
    training: dict[str, Any],
    model_name: str,
    model_revision: str,
) -> tuple[str, str | None, str]:
    candidates: list[Path] = []
    configured = os.environ.get("PMLDL_MODEL_DIR")
    if configured:
        candidates.append(Path(configured))
    source_config = dict(training.get("model_source", {}))
    candidates.extend(
        Path(value) for value in source_config.get("kaggle_path_candidates", [])
    )
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return str(candidate.resolve()), None, "attached_local_model"
    return model_name, model_revision, "huggingface_hub"


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_gemma2_experiment(
    run: Any,
    setup: NotebookExperimentSetup,
    project_root: str | Path,
) -> ExperimentOutput:
    """Run the tracked Gemma-2 9B LoRA/QLoRA comparison."""
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
            "Gemma-2 training requires requirements-transformer.lock."
        ) from exc

    root = Path(project_root).resolve()
    training = dict(setup.training)
    seed = int(setup.seed)
    smoke = bool(setup.smoke_test)
    max_epochs = int(training.get("max_epochs", 3))
    epoch_checkpoints = tuple(
        int(value) for value in training.get("epoch_checkpoints", [1, 2, 3])
    )
    if epoch_checkpoints != tuple(range(1, max_epochs + 1)):
        raise ValueError("epoch_checkpoints must cover every epoch from 1 to max_epochs.")
    effective_batch_size = int(training.get("effective_batch_size", 32))
    smoke_effective_batch_size = int(training.get("smoke_effective_batch_size", 4))
    max_length = int(training.get("max_length", 512))
    smoke_max_length = int(training.get("smoke_max_length", 128))
    learning_rate = float(training.get("learning_rate", 2e-4))
    weight_decay = float(training.get("weight_decay", 0.01))
    warmup_ratio = float(training.get("warmup_ratio", 0.06))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    runtime_probe_steps = int(training.get("runtime_probe_optimizer_steps", 10))
    max_arm_runtime_seconds = float(
        training.get("max_projected_arm_runtime_seconds", 39600)
    )
    smoke_optimizer_steps = int(training.get("smoke_optimizer_steps", 2))
    if max_epochs != 3 or epoch_checkpoints != (1, 2, 3):
        raise ValueError("Sprint 2 requires metrics after epochs 1, 2, and 3.")
    if effective_batch_size < 1 or max_length < 32:
        raise ValueError("Invalid Gemma training dimensions.")
    if not torch.cuda.is_available():
        raise RuntimeError("Gemma-2 9B PEFT training requires a CUDA GPU.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")

    data_dir = _find_data_dir(root)
    expected_hashes = load_checksum_manifest(root / "data" / "checksums.sha256")
    verified_hashes = verify_competition_data_dir(data_dir, expected_hashes)
    train_frame, _ = load_competition_data(data_dir)
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
        raise RuntimeError("Training and selection folds overlap.")
    if smoke:
        train_indices = _balanced_smoke_indices(
            train_mask, all_targets, per_class=6, seed=seed
        )
        validation_indices = _balanced_smoke_indices(
            validation_mask, all_targets, per_class=3, seed=seed
        )
    else:
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)
    train_part = train_frame.iloc[train_indices].reset_index(drop=True)
    validation_part = train_frame.iloc[validation_indices].reset_index(drop=True)
    y_train = target_indices(train_part)
    y_validation = target_indices(validation_part)
    required_features = {"prompt", "response_a", "response_b"}
    if not required_features.issubset(train_part.columns):
        raise RuntimeError("Required text fields are absent.")
    if {"model_a", "model_b"}.intersection(required_features):
        raise RuntimeError("Model identity columns are forbidden as features.")

    model_name = setup.model_name or DEFAULT_MODEL_NAME
    model_revision = setup.model_revision or DEFAULT_MODEL_REVISION
    model_source, source_revision, model_source_kind = _resolve_model_source(
        training, model_name, model_revision
    )
    tokenizer_kwargs: dict[str, Any] = {"use_fast": True}
    if source_revision:
        tokenizer_kwargs["revision"] = source_revision
    tokenizer = AutoTokenizer.from_pretrained(model_source, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    bos_token_id = tokenizer.bos_token_id
    separator_token_id = tokenizer.eos_token_id
    if (
        bos_token_id is None
        or separator_token_id is None
        or tokenizer.pad_token_id is None
    ):
        raise RuntimeError("Gemma tokenizer must define BOS, EOS, and PAD tokens.")

    max_length_in_use = smoke_max_length if smoke else max_length

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
            prompt_ids, response_a_ids, response_b_ids, _ = (
                balanced_head_tail_truncate(
                    prompt_ids,
                    response_a_ids,
                    response_b_ids,
                    max_length=max_length_in_use,
                    special_tokens=4,
                    budget_weights=(1, 2, 2),
                )
            )
            sequence = [
                int(bos_token_id),
                *prompt_ids,
                int(separator_token_id),
                *response_a_ids,
                int(separator_token_id),
                *response_b_ids,
                int(separator_token_id),
            ]
            if len(sequence) > max_length_in_use:
                raise RuntimeError("Encoded sequence exceeds max_length.")
            sequences.append(sequence)
        labels[flags] = swap_target_indices(labels[flags])
        return sequences, labels

    swap_probability = float(
        dict(training.get("preprocessing", {})).get(
            "random_ab_swap_probability", 0.5
        )
    )
    if swap_probability != 0.5:
        raise ValueError("The frozen E2 protocol requires 50% random A/B swap.")
    swap_rng = np.random.default_rng(seed)
    training_swap_flags = swap_rng.random(len(train_part)) < swap_probability
    train_sequences, swapped_y_train = encode_rows(train_part, training_swap_flags)
    validation_sequences, _ = encode_rows(
        validation_part, np.zeros(len(validation_part), dtype=bool)
    )
    swapped_validation_sequences, _ = encode_rows(
        validation_part, np.ones(len(validation_part), dtype=bool)
    )
    if not np.array_equal(
        swapped_y_train[training_swap_flags],
        swap_target_indices(y_train[training_swap_flags]),
    ):
        raise RuntimeError("A/B swap labels are inconsistent.")

    class PreferenceDataset(Dataset):
        def __init__(self, sequences: list[list[int]], labels: np.ndarray | None = None):
            self.sequences = sequences
            self.labels = labels

        def __len__(self) -> int:
            return len(self.sequences)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item: dict[str, Any] = {"input_ids": self.sequences[index]}
            if self.labels is not None:
                item["labels"] = int(self.labels[index])
            return item

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        longest = max(len(item["input_ids"]) for item in batch)
        longest = min(max_length_in_use, int(math.ceil(longest / 8.0) * 8))
        input_ids = torch.full(
            (len(batch), longest),
            int(tokenizer.pad_token_id),
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(batch), longest), dtype=torch.long)
        labels: list[int] = []
        for row_index, item in enumerate(batch):
            sequence = item["input_ids"]
            input_ids[row_index, : len(sequence)] = torch.tensor(
                sequence, dtype=torch.long
            )
            attention_mask[row_index, : len(sequence)] = 1
            if "labels" in item:
                labels.append(int(item["labels"]))
        result: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if labels:
            result["labels"] = torch.tensor(labels, dtype=torch.long)
        return result

    gpu_names = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    gpu_total_memory_gib = [
        float(torch.cuda.get_device_properties(index).total_memory / (1024**3))
        for index in range(torch.cuda.device_count())
    ]
    total_gpu_memory_gib = float(sum(gpu_total_memory_gib))
    full_config = dict(training.get("full_finetune_preflight", {}))
    parameter_count_estimate = int(
        full_config.get("parameter_count_estimate", 9_000_000_000)
    )
    full_preflight = full_finetune_preflight(
        parameter_count=parameter_count_estimate,
        total_gpu_memory_gib=total_gpu_memory_gib,
        bytes_per_parameter=int(full_config.get("bytes_per_parameter", 16)),
        usable_fraction=float(full_config.get("usable_fraction", 0.9)),
    )
    if full_preflight["status"] == "allowed":
        full_preflight["status"] = "not_requested_protocol_scope"

    bf16_supported = bool(torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
    arms_config = dict(training.get("arms", {}))
    arm_order = ("lora", "qlora")
    arm_results: dict[str, Any] = {}
    arm_probabilities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    artifacts: dict[str, Path] = {}

    def reset_arm_seed() -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        set_seed(seed)

    def build_model(arm_name: str) -> tuple[Any, Any, int, int]:
        reset_arm_seed()
        arm = dict(arms_config[arm_name])
        quantized = arm_name == "qlora"
        load_kwargs: dict[str, Any] = {
            "num_labels": 3,
            "ignore_mismatched_sizes": True,
            "low_cpu_mem_usage": True,
        }
        if source_revision:
            load_kwargs["revision"] = source_revision
        if quantized:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
                llm_int8_skip_modules=["score"],
            )
            load_kwargs["device_map"] = {"": 0}
        else:
            single_gpu_usable = gpu_total_memory_gib[0] * 0.9
            fp16_weight_gib = parameter_count_estimate * 2 / (1024**3)
            if single_gpu_usable < fp16_weight_gib and torch.cuda.device_count() < 2:
                raise torch.cuda.OutOfMemoryError(
                    "FP16 LoRA base weights do not fit on one available GPU."
                )
            load_kwargs["torch_dtype"] = compute_dtype
            load_kwargs["device_map"] = "auto"
            memory_fraction = float(arm.get("max_gpu_memory_fraction", 0.72))
            if not 0.0 < memory_fraction < 1.0:
                raise ValueError("max_gpu_memory_fraction must be in (0, 1).")
            load_kwargs["max_memory"] = {
                index: f"{max(1, int(memory * memory_fraction))}GiB"
                for index, memory in enumerate(gpu_total_memory_gib)
            }
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source, **load_kwargs
        )
        model.config.pad_token_id = int(tokenizer.pad_token_id)
        model.config.use_cache = False
        if quantized:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        else:
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=int(arm["r"]),
                lora_alpha=int(arm["alpha"]),
                lora_dropout=float(arm["dropout"]),
                target_modules=list(arm["target_modules"]),
                modules_to_save=list(arm.get("modules_to_save", ["score"])),
                bias="none",
            ),
        )
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in model.parameters())
        input_device = model.get_input_embeddings().weight.device
        return model, input_device, int(trainable), int(total)

    def predict(
        model: Any,
        input_device: Any,
        sequences: list[list[int]],
        batch_size: int,
    ) -> np.ndarray:
        loader = DataLoader(
            PreferenceDataset(sequences),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
            pin_memory=True,
        )
        batches: list[np.ndarray] = []
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                inputs = {
                    key: value.to(input_device, non_blocking=True)
                    for key, value in batch.items()
                }
                with torch.autocast(device_type="cuda", dtype=compute_dtype):
                    logits = model(**inputs).logits
                batches.append(
                    functional.softmax(logits.float(), dim=-1).cpu().numpy()
                )
        if not batches:
            raise RuntimeError("Prediction dataloader is empty.")
        return normalize_probabilities(np.concatenate(batches, axis=0))

    for arm_name in arm_order:
        model: Any | None = None
        optimizer: Any | None = None
        scheduler: Any | None = None
        scaler: Any | None = None
        train_loader: Any | None = None
        batch: Any | None = None
        inputs: Any | None = None
        logits: Any | None = None
        loss: Any | None = None
        labels: Any | None = None
        arm = dict(arms_config[arm_name])
        micro_batch_size = int(arm.get("micro_batch_size", 1))
        evaluation_batch_size = int(arm.get("evaluation_batch_size", 2))
        arm_effective_batch = (
            smoke_effective_batch_size if smoke else effective_batch_size
        )
        if arm_effective_batch % micro_batch_size:
            raise ValueError(
                f"Effective batch size is not divisible for {arm_name}."
            )
        gradient_accumulation_steps = arm_effective_batch // micro_batch_size
        try:
            torch.cuda.reset_peak_memory_stats()
            arm_started = time.perf_counter()
            model, input_device, trainable_parameters, total_parameters = build_model(
                arm_name
            )
            optimizer = torch.optim.AdamW(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            batches_per_epoch = math.ceil(len(train_sequences) / micro_batch_size)
            if smoke:
                total_optimizer_steps = smoke_optimizer_steps
            else:
                total_optimizer_steps = (
                    math.ceil(batches_per_epoch / gradient_accumulation_steps)
                    * max_epochs
                )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=math.ceil(total_optimizer_steps * warmup_ratio),
                num_training_steps=total_optimizer_steps,
            )
            scaler = torch.amp.GradScaler(
                "cuda", enabled=compute_dtype == torch.float16
            )
            optimizer.zero_grad(set_to_none=True)
            training_started = time.perf_counter()
            optimizer_step = 0
            train_curve: list[dict[str, float | int]] = []
            epoch_results: list[dict[str, Any]] = []
            best_log_loss = math.inf
            best_epoch = 0
            best_original: np.ndarray | None = None
            best_swapped_back: np.ndarray | None = None
            best_checkpoint_root = Path(
                tempfile.mkdtemp(prefix=f"{arm_name}-best-", dir=run.artifact_dir)
            )
            epochs_to_run = (1,) if smoke else epoch_checkpoints
            for epoch in epochs_to_run:
                model.train()
                generator = torch.Generator().manual_seed(seed + epoch - 1)
                train_loader = DataLoader(
                    PreferenceDataset(train_sequences, swapped_y_train),
                    batch_size=micro_batch_size,
                    shuffle=True,
                    generator=generator,
                    num_workers=0,
                    collate_fn=collate,
                    pin_memory=True,
                )
                accumulated = 0
                loss_sum = 0.0
                for batch in train_loader:
                    if smoke and optimizer_step >= smoke_optimizer_steps:
                        break
                    labels = batch.pop("labels")
                    inputs = {
                        key: value.to(input_device, non_blocking=True)
                        for key, value in batch.items()
                    }
                    with torch.autocast(device_type="cuda", dtype=compute_dtype):
                        logits = model(**inputs).logits
                        loss = functional.cross_entropy(
                            logits.float(), labels.to(logits.device)
                        )
                    scaler.scale(loss / gradient_accumulation_steps).backward()
                    accumulated += 1
                    loss_sum += float(loss.detach().cpu())
                    is_epoch_end = accumulated == gradient_accumulation_steps
                    if not is_epoch_end:
                        continue
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                    mean_loss = loss_sum / accumulated
                    point = {
                        "epoch": epoch,
                        "step": optimizer_step,
                        "loss": mean_loss,
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                    }
                    train_curve.append(point)
                    if optimizer_step == 1 or optimizer_step % 10 == 0:
                        run.log_metrics(
                            {
                                "loss": mean_loss,
                                "learning_rate": point["learning_rate"],
                            },
                            namespace=f"{arm_name}_train",
                            step=optimizer_step,
                        )
                    if (
                        not smoke
                        and optimizer_step == runtime_probe_steps
                        and optimizer_step < total_optimizer_steps
                    ):
                        elapsed = time.perf_counter() - training_started
                        projected = elapsed / optimizer_step * total_optimizer_steps
                        if projected > max_arm_runtime_seconds:
                            raise ProjectedRuntimeLimit(
                                f"Projected {arm_name} runtime {projected:.0f}s exceeds "
                                f"the {max_arm_runtime_seconds:.0f}s arm limit."
                            )
                    accumulated = 0
                    loss_sum = 0.0
                if accumulated and (not smoke or optimizer_step < smoke_optimizer_steps):
                    correction = gradient_accumulation_steps / accumulated
                    scaler.unscale_(optimizer)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1
                    train_curve.append(
                        {
                            "epoch": epoch,
                            "step": optimizer_step,
                            "loss": loss_sum / accumulated,
                            "learning_rate": float(scheduler.get_last_lr()[0]),
                        }
                    )

                original = predict(
                    model,
                    input_device,
                    validation_sequences,
                    evaluation_batch_size,
                )
                swapped = predict(
                    model,
                    input_device,
                    swapped_validation_sequences,
                    evaluation_batch_size,
                )
                swapped_back = swap_probability_columns(swapped)
                metrics = evaluate_experiment_probabilities(
                    y_validation, original, swapped_back
                )
                epoch_result = {
                    "epoch": epoch,
                    "optimizer_steps": optimizer_step,
                    "metrics": metrics,
                }
                epoch_results.append(epoch_result)
                run.log_metrics(
                    metrics,
                    namespace=f"{arm_name}_epoch_{epoch}_validation",
                    step=epoch,
                )
                if float(metrics["log_loss"]) < best_log_loss:
                    best_log_loss = float(metrics["log_loss"])
                    best_epoch = epoch
                    best_original = original.copy()
                    best_swapped_back = swapped_back.copy()
                    checkpoint_dir = best_checkpoint_root / "checkpoint"
                    if checkpoint_dir.exists():
                        shutil.rmtree(checkpoint_dir)
                    model.save_pretrained(checkpoint_dir, safe_serialization=True)
                    tokenizer.save_pretrained(checkpoint_dir)

            if best_original is None or best_swapped_back is None:
                raise RuntimeError(f"{arm_name} produced no epoch evaluation.")
            averaged = normalize_probabilities(
                0.5 * (best_original + best_swapped_back)
            )
            prediction_path = run.artifact_path(
                f"predictions/{arm_name}_fold7_predictions.csv"
            )
            pd.DataFrame(
                {
                    "id": validation_part["id"].astype(str),
                    "winner_model_a": averaged[:, 0],
                    "winner_model_b": averaged[:, 1],
                    "winner_tie": averaged[:, 2],
                }
            ).to_csv(prediction_path, index=False)
            checkpoint_base = run.artifact_path(f"models/{arm_name}_best_adapter")
            checkpoint_path = Path(
                shutil.make_archive(
                    str(checkpoint_base),
                    "zip",
                    root_dir=best_checkpoint_root / "checkpoint",
                )
            )
            shutil.rmtree(best_checkpoint_root)
            curve_path = _write_json(
                run.artifact_path(f"curves/{arm_name}.json"), train_curve
            )
            arm_runtime = time.perf_counter() - arm_started
            peak_memory = float(torch.cuda.max_memory_allocated() / (1024**2))
            arm_results[arm_name] = {
                "status": "completed",
                "quantization": arm.get("quantization", "none"),
                "best_epoch": best_epoch,
                "best_log_loss": best_log_loss,
                "epoch_results": epoch_results,
                "optimizer_steps": optimizer_step,
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
                "trainable_fraction": float(
                    trainable_parameters / max(total_parameters, 1)
                ),
                "runtime_seconds": arm_runtime,
                "peak_gpu_memory_mb": peak_memory,
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            }
            arm_probabilities[arm_name] = (best_original, best_swapped_back)
            artifacts[f"predictions/{arm_name}_fold7_predictions.csv"] = (
                prediction_path
            )
            artifacts[f"models/{arm_name}_best_adapter.zip"] = checkpoint_path
            artifacts[f"curves/{arm_name}.json"] = curve_path
            run.log_metrics(
                {
                    "best_epoch": float(best_epoch),
                    "best_log_loss": best_log_loss,
                    "runtime_seconds": arm_runtime,
                    "peak_gpu_memory_mb": peak_memory,
                    "trainable_parameters": float(trainable_parameters),
                    "total_parameters": float(total_parameters),
                },
                namespace=f"{arm_name}_summary",
            )
        except (torch.cuda.OutOfMemoryError, ProjectedRuntimeLimit) as exc:
            status = (
                "skipped_resource_limit"
                if isinstance(exc, torch.cuda.OutOfMemoryError)
                else "skipped_runtime_limit"
            )
            arm_results[arm_name] = {
                "status": status,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            run.log_metric("resource_limited", 1.0, namespace=f"{arm_name}_summary")
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        finally:
            model = None
            optimizer = None
            scheduler = None
            scaler = None
            train_loader = None
            batch = None
            inputs = None
            logits = None
            loss = None
            labels = None
            gc.collect()
            torch.cuda.empty_cache()

    completed_arms = [
        name for name in arm_order if arm_results.get(name, {}).get("status") == "completed"
    ]
    if not completed_arms:
        raise RuntimeError("No Gemma LoRA/QLoRA arm completed successfully.")
    best_arm = min(
        completed_arms, key=lambda name: float(arm_results[name]["best_log_loss"])
    )
    best_original, best_swapped_back = arm_probabilities[best_arm]
    best_averaged = normalize_probabilities(
        0.5 * (best_original + best_swapped_back)
    )
    best_prediction_path = run.artifact_path("fold7_predictions.csv")
    pd.DataFrame(
        {
            "id": validation_part["id"].astype(str),
            "winner_model_a": best_averaged[:, 0],
            "winner_model_b": best_averaged[:, 1],
            "winner_tie": best_averaged[:, 2],
        }
    ).to_csv(best_prediction_path, index=False)
    artifacts["fold7_predictions.csv"] = best_prediction_path

    best_checkpoint = artifacts[f"models/{best_arm}_best_adapter.zip"]
    manifest_payload = {
        "schema_version": 1,
        "run_id": run.run_id,
        "experiment_id": setup.experiment_id,
        "seed": seed,
        "class_order": list(CLASS_ORDER),
        "selected_arm": best_arm,
        "selected_epoch": int(arm_results[best_arm]["best_epoch"]),
        "model": {
            "name": model_name,
            "revision": model_revision,
            "load_source_kind": model_source_kind,
            "load_source": model_source,
        },
        "files": {
            "adapter": {
                "path": best_checkpoint.relative_to(run.artifact_dir).as_posix(),
                "sha256": _sha256(best_checkpoint),
            },
            "fold7_predictions": {
                "path": best_prediction_path.relative_to(
                    run.artifact_dir
                ).as_posix(),
                "sha256": _sha256(best_prediction_path),
            },
        },
    }
    manifest_path = _write_json(
        run.artifact_path("model_manifest.json"), manifest_payload
    )
    artifacts["model_manifest.json"] = manifest_path

    study_path = _write_json(
        run.artifact_path("study.json"),
        {
            "experiment_id": setup.experiment_id,
            "parent_experiment_id": setup.parent_experiment_id,
            "control": dict(training.get("control", {})),
            "protocol": {
                "seed": seed,
                "training_folds": list(roles.training_folds),
                "selection_fold": roles.validation_fold,
                "calibration_fold_unopened": roles.calibration_fold,
                "final_holdout_fold_unopened": roles.final_holdout_fold,
                "train_rows": len(train_part),
                "selection_rows": len(validation_part),
                "max_length": max_length_in_use,
                "effective_batch_size": (
                    smoke_effective_batch_size if smoke else effective_batch_size
                ),
                "epoch_checkpoints": list((1,) if smoke else epoch_checkpoints),
                "random_ab_swap_probability": swap_probability,
                "swap_averaged_inference": True,
                "model_identity_features_used": False,
            },
            "full_finetune_preflight": full_preflight,
            "arms": arm_results,
            "selected_arm": best_arm,
        },
    )
    artifacts["study.json"] = study_path
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
            "gpu_names": gpu_names,
            "gpu_total_memory_gib": gpu_total_memory_gib,
            "compute_dtype": str(compute_dtype),
            "verified_data_hashes": verified_hashes,
            "model": {"name": model_name, "revision": model_revision},
            "model_source_kind": model_source_kind,
        },
    )
    artifacts["environment.json"] = environment_path

    return ExperimentOutput.from_predictions(
        y_true=y_validation,
        original_probabilities=best_original,
        swapped_back_probabilities=best_swapped_back,
        artifacts=artifacts,
    )
