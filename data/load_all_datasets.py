# data/load_all_datasets.py
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path to allow importing 'data' module
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pickle

from data.load_data import Repertoire, load_dataset

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Paths to the dataset directories
# We assume the script is run from airr_ml_project_template/
TRAIN_ROOT = Path("data/dataset_new/train_datasets/train_datasets")
TEST_ROOT = Path("data/dataset_new/test_datasets/test_datasets")

# Keys are internal dataset names; values are directory names
TRAIN_DATASETS: Dict[str, str] = {
    "ds1": "train_dataset_1",
    "ds2": "train_dataset_2",
    "ds3": "train_dataset_3",
    "ds4": "train_dataset_4",
    "ds5": "train_dataset_5",
    "ds6": "train_dataset_6",
    "ds7": "train_dataset_7",
    "ds8": "train_dataset_8",
}

TEST_DATASETS: Dict[str, str] = {
    "ds1": "test_dataset_1",
    "ds2": "test_dataset_2",
    "ds3": "test_dataset_3",
    "ds4": "test_dataset_4",
    "ds5": "test_dataset_5",
    "ds6": "test_dataset_6",
    "ds7_1": "test_dataset_7_1",
    "ds7_2": "test_dataset_7_2",
    "ds8_1": "test_dataset_8_1",
    "ds8_2": "test_dataset_8_2",
    "ds8_3": "test_dataset_8_3",
}


def save_repertoires_pickle(reps: List[Repertoire], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(reps, f)
    print(f"[load_all_datasets] Saved {len(reps)} repertoires to {path}")


def load_repertoires_pickle(path: Path) -> List[Repertoire]:
    with path.open("rb") as f:
        reps = pickle.load(f)
    return reps


def build_train_pickles():
    for ds_name, dirname in TRAIN_DATASETS.items():
        raw_path = TRAIN_ROOT / dirname
        out_path = PROCESSED_DIR / f"{ds_name}_train.pkl"
        if out_path.exists():
            print(f"[build_train_pickles] {out_path} exists, skipping.")
            continue
        # Load using the new directory-aware function
        reps = load_dataset(raw_path, dataset_name=ds_name)
        save_repertoires_pickle(reps, out_path)


def build_test_pickles():
    for ds_name, dirname in TEST_DATASETS.items():
        raw_path = TEST_ROOT / dirname
        out_path = PROCESSED_DIR / f"{ds_name}_test.pkl"
        if out_path.exists():
            print(f"[build_test_pickles] {out_path} exists, skipping.")
            continue
        # Test usually has no labels; label_col=None is default behavior if metadata missing
        reps = load_dataset(raw_path, dataset_name=ds_name)
        save_repertoires_pickle(reps, out_path)


def main():
    build_train_pickles()
    build_test_pickles()


if __name__ == "__main__":
    main()
