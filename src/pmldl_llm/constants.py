from pathlib import Path

TARGET_COLUMNS = (
    "winner_model_a",
    "winner_model_b",
    "winner_tie",
)
TARGET_NAMES = ("model_a", "model_b", "tie")
TEXT_COLUMNS = ("prompt", "response_a", "response_b")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "llm-classification-finetuning"
DEFAULT_CHECKSUM_MANIFEST = PROJECT_ROOT / "data" / "checksums.sha256"
DEFAULT_SPLIT_CONFIG = PROJECT_ROOT / "configs" / "split.json"
DEFAULT_FOLD_PATH = PROJECT_ROOT / "data" / "splits" / "folds.csv"
DEFAULT_FOLD_METADATA_PATH = PROJECT_ROOT / "data" / "splits" / "metadata.json"
