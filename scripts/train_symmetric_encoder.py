
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pmldl_llm.config import validate_fold_roles
from pmldl_llm.constants import (
    DEFAULT_CHECKSUM_MANIFEST,
    DEFAULT_DATA_DIR,
    DEFAULT_FOLD_PATH,
    DEFAULT_SPLIT_CONFIG,
    TARGET_COLUMNS,
)
from pmldl_llm.data import (
    load_competition_data,
    swap_probability_columns,
    swap_target_indices,
    swapped_frame,
    target_indices,
    verify_competition_data_dir,
    verify_checksum_manifest,
)
from pmldl_llm.evaluation import evaluate_probabilities, normalize_probabilities
from pmldl_llm.provenance import build_provenance
from pmldl_llm.split import load_frozen_folds
from pmldl_llm.symmetric_model import SymmetricPreferenceModel
from pmldl_llm.text import flatten_conversation
from pmldl_llm.truncation import balanced_head_tail_truncate


def encode_side(
    tokenizer,
    prompt_text: str,
    response_text: str,
    max_length: int,
    special_tokens: int,
    prompt_weight: int,
    response_weight: int,
) -> tuple[list[int], list[int]]:

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    truncated_prompt, truncated_response, _empty, _stats = balanced_head_tail_truncate(
        prompt_ids,
        response_ids,
        [],
        max_length=max_length,
        special_tokens=special_tokens,
        budget_weights=(prompt_weight, response_weight, response_weight),
    )
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    input_ids = [cls_id] + truncated_prompt + [sep_id] + truncated_response + [sep_id]
    attention_mask = [1] * len(input_ids)
    pad_length = max_length - len(input_ids)
    if pad_length > 0:
        pad_id = tokenizer.pad_token_id
        input_ids = input_ids + [pad_id] * pad_length
        attention_mask = attention_mask + [0] * pad_length
    return input_ids[:max_length], attention_mask[:max_length]


class PreferencePairDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer,
        targets: np.ndarray | None,
        config: dict,
    ) -> None:
        self.prompts = [flatten_conversation(value) for value in frame["prompt"]]
        self.responses_a = [
            flatten_conversation(value) for value in frame["response_a"]
        ]
        self.responses_b = [
            flatten_conversation(value) for value in frame["response_b"]
        ]
        self.ids = frame["id"].to_numpy()
        self.targets = targets
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict:
        cfg = self.config
        input_ids_a, mask_a = encode_side(
            self.tokenizer,
            self.prompts[index],
            self.responses_a[index],
            cfg["max_length"],
            cfg["special_tokens"],
            cfg["prompt_weight"],
            cfg["response_weight"],
        )
        input_ids_b, mask_b = encode_side(
            self.tokenizer,
            self.prompts[index],
            self.responses_b[index],
            cfg["max_length"],
            cfg["special_tokens"],
            cfg["prompt_weight"],
            cfg["response_weight"],
        )
        item = {
            "input_ids_a": torch.tensor(input_ids_a, dtype=torch.long),
            "attention_mask_a": torch.tensor(mask_a, dtype=torch.long),
            "input_ids_b": torch.tensor(input_ids_b, dtype=torch.long),
            "attention_mask_b": torch.tensor(mask_b, dtype=torch.long),
        }
        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[index], dtype=torch.long)
        return item


@torch.no_grad()
def predict_probabilities(
    model: SymmetricPreferenceModel,
    frame: pd.DataFrame,
    tokenizer,
    config: dict,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    model.eval()

    def run(data_frame: pd.DataFrame) -> np.ndarray:
        dataset = PreferencePairDataset(data_frame, tokenizer, None, config)
        loader = DataLoader(dataset, batch_size=config["eval_batch_size"])
        all_probs = []
        for batch in loader:
            logits = model(
                batch["input_ids_a"].to(device),
                batch["attention_mask_a"].to(device),
                batch["input_ids_b"].to(device),
                batch["attention_mask_b"].to(device),
            )
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return normalize_probabilities(np.concatenate(all_probs, axis=0))

    original = run(frame)
    swapped_back = swap_probability_columns(run(swapped_frame(frame)))
    raw_symmetry_error = float(np.abs(original - swapped_back).sum(axis=1).mean())
    return original, raw_symmetry_error


def train_one_epoch(
    model: SymmetricPreferenceModel,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        logits = model(
            batch["input_ids_a"].to(device),
            batch["attention_mask_a"].to(device),
            batch["input_ids_b"].to(device),
            batch["attention_mask_b"].to(device),
        )
        loss = loss_fn(logits, batch["target"].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def main() -> None:
    started_at = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "symmetric_encoder.json",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLD_PATH)
    parser.add_argument(
        "--checksum-manifest", type=Path, default=DEFAULT_CHECKSUM_MANIFEST
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "symmetric_encoder",
    )
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=0,
        help="If > 0, subsample this many training/validation rows for a fast smoke test.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.manual_seed(config["seed"])

    split_config = json.loads(args.split_config.read_text(encoding="utf-8"))
    roles = validate_fold_roles(split_config)
    dataset_hashes = verify_checksum_manifest(args.checksum_manifest)
    verify_competition_data_dir(args.data_dir, dataset_hashes)
    train, _test = load_competition_data(args.data_dir)
    folds = load_frozen_folds(
        train,
        args.folds,
        args.split_config,
        n_splits=roles.n_splits,
        dataset_hashes=dataset_hashes,
    )
    fold_values = folds["fold"].to_numpy()
    is_training = np.isin(fold_values, list(roles.training_folds))
    is_validation = fold_values == roles.validation_fold

    y = target_indices(train)
    training_frame = train.loc[is_training].reset_index(drop=True)
    validation_frame = train.loc[is_validation].reset_index(drop=True)
    y_train = y[is_training]
    y_validation = y[is_validation]

    if args.smoke_rows > 0:
        training_frame = training_frame.iloc[: args.smoke_rows].reset_index(drop=True)
        y_train = y_train[: args.smoke_rows]
        validation_frame = validation_frame.iloc[: args.smoke_rows].reset_index(
            drop=True
        )
        y_validation = y_validation[: args.smoke_rows]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = SymmetricPreferenceModel(config["model_name"]).to(device)

    augmented_frame = pd.concat(
        [training_frame, swapped_frame(training_frame)], ignore_index=True
    )
    augmented_targets = np.concatenate([y_train, swap_target_indices(y_train)])

    train_dataset = PreferencePairDataset(
        augmented_frame, tokenizer, augmented_targets, config
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": config["learning_rate"]},
            {
                "params": list(model.preference_head.parameters())
                + list(model.tie_head.parameters()),
                "lr": config["head_learning_rate"],
            },
        ],
        weight_decay=config["weight_decay"],
    )
    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    epoch_losses = []
    for epoch in range(config["epochs"]):
        epoch_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, config["max_grad_norm"]
        )
        epoch_losses.append(epoch_loss)
        print(f"epoch {epoch + 1}/{config['epochs']}: train_loss={epoch_loss:.4f}")

    validation_probability, raw_symmetry_error = predict_probabilities(
        model, validation_frame, tokenizer, config, device
    )

    metrics = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
        },
        "protocol": {
            "fold_assignment": "data/splits/folds.csv",
            "n_splits": roles.n_splits,
            "training_folds": list(roles.training_folds),
            "validation_fold": roles.validation_fold,
            "model_seed": config["seed"],
            "train_rows": int(len(training_frame)),
            "validation_rows": int(len(validation_frame)),
            "a_b_swap_augmentation": True,
            "architecture": "swap-equivariant siamese encoder (linear preference head, no post-hoc averaging)",
        },
        "symmetric_encoder": evaluate_probabilities(
            y_validation, validation_probability
        ),
        "raw_symmetry_l1": raw_symmetry_error,
        "epoch_losses": epoch_losses,
        "config": config,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": validation_frame["id"].to_numpy(),
            "target": y_validation,
            TARGET_COLUMNS[0]: validation_probability[:, 0],
            TARGET_COLUMNS[1]: validation_probability[:, 1],
            TARGET_COLUMNS[2]: validation_probability[:, 2],
        }
    ).to_csv(args.output_dir / "validation_predictions.csv", index=False)

    torch.save(model.state_dict(), args.output_dir / "model_state_dict.pt")

    metrics["provenance"] = build_provenance(
        PROJECT_ROOT,
        __file__,
        [args.config, args.split_config],
        args.folds,
        dataset_hashes,
    )
    metrics["runtime_seconds"] = float(time.perf_counter() - started_at)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()