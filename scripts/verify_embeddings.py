import sys
import logging
from pathlib import Path
import random
import torch
import pickle
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data.load_all_datasets import TRAIN_DATASETS, TEST_DATASETS, PROCESSED_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("verify_embeddings.log")
    ]
)
logger = logging.getLogger("Verify")

EMBEDDINGS_DIR = PROJECT_ROOT / "data/embeddings"

def check_dataset(ds_name: str, split: str = "train"):
    """
    Checks integrity for a single dataset.
    1. Loads the repertoire list from data/processed/{ds_name}_{split}.pkl
    2. Checks if data/embeddings/{ds_name}/{rep_id}.pt exists.
    3. Randomly consistency checks a few files.
    """
    pkl_name = f"{ds_name}_{split}.pkl"
    pkl_path = PROCESSED_DIR / pkl_name
    
    if not pkl_path.exists():
        logger.error(f"❌ Pickle not found: {pkl_path}")
        logger.error(f"   Run 'python scripts/run_full_pipeline.py' to generate pickles first (it will stop after build_datasets if you interrupt it).")
        return {"total": 0, "missing": 0, "corrupt": 0}

    try:
        with open(pkl_path, "rb") as f:
            reps = pickle.load(f)
    except Exception as e:
        logger.error(f"❌ Failed to load pickle {pkl_path}: {e}")
        return {"total": 0, "missing": 0, "corrupt": 0}

    logger.info(f"🧐 Checking {ds_name} ({split}): {len(reps)} repertoires")
    
    missing_count = 0
    corrupt_count = 0
    checked_count = 0
    
    # Map ds7_1 -> ds7, ds8_1 -> ds8
    embed_ds_folder = ds_name.split("_")[0]
    
    # Structure: data/embeddings/{ds_name}/{split}/{rep_id}.npy
    dataset_embed_dir = EMBEDDINGS_DIR / embed_ds_folder / split
    
    if not dataset_embed_dir.exists():
        logger.warning(f"   ⚠️ Folder {dataset_embed_dir} does not exist. Assuming all missing.")
        # Fallback check: maybe it's just {ds_name}/{rep_id}.npy?
        if (EMBEDDINGS_DIR / embed_ds_folder / f"{reps[0].rep_id}.npy").exists():
             logger.info("   ℹ️ Found flattened structure (no split folder).")
             dataset_embed_dir = EMBEDDINGS_DIR / embed_ds_folder
        else:
             return {"total": len(reps), "missing": len(reps), "corrupt": 0}

    for rep in tqdm(reps, desc=f"Scanning {ds_name}"):
        rep_id = rep.rep_id
        # sanitize rep_id if needed, though usually it matches filename
        file_path = dataset_embed_dir / f"{rep_id}.npy"
        
        if not file_path.exists():
            missing_count += 1
            continue
            
        # Random Integrity Check (1% chance)
        if random.random() < 0.01:
            try:
                # Just try to load headers/content
                _ = torch.load(file_path, map_location="cpu")
            except Exception as e:
                logger.error(f"   ❌ Corrupt file found: {file_path} ({e})")
                corrupt_count += 1
        
        checked_count += 1

    if missing_count == 0 and corrupt_count == 0:
        logger.info(f"   ✅ {ds_name}: OK ({len(reps)} checked)")
    else:
        logger.warning(f"   ⚠️ {ds_name}: {missing_count} missing, {corrupt_count} corrupt")
        
    return {"total": len(reps), "missing": missing_count, "corrupt": corrupt_count}

def main():
    logger.info("========================================")
    logger.info("   Embedding Verification Tool          ")
    logger.info("========================================")
    logger.info(f"Embeddings Root: {EMBEDDINGS_DIR}")
    
    if not EMBEDDINGS_DIR.exists():
        logger.error(f"CRITICAL: Embeddings directory {EMBEDDINGS_DIR} does not exist.")
        sys.exit(1)

    total_stats = {"total": 0, "missing": 0, "corrupt": 0}
    
    # Check TRain
    for ds_name in TRAIN_DATASETS.keys():
        stats = check_dataset(ds_name, split="train")
        for k in total_stats: total_stats[k] += stats[k]
        
    # Check Test
    for ds_name in TEST_DATASETS.keys():
        stats = check_dataset(ds_name, split="test")
        for k in total_stats: total_stats[k] += stats[k]
        
    logger.info("\n========================================")
    logger.info("   FINAL SUMMARY")
    logger.info("========================================")
    logger.info(f"Total Repertoires Expected: {total_stats['total']}")
    logger.info(f"Total Missing files:        {total_stats['missing']}")
    logger.info(f"Total Corrupt (Sampled):    {total_stats['corrupt']}")
    
    if total_stats["missing"] == 0 and total_stats["corrupt"] == 0:
        logger.info("\n✅ INTEGRITY VERIFIED. You can likely reuse these embeddings.")
    else:
        logger.info("\n⚠️ ISSUES FOUND. See verify_embeddings.log for details.")

if __name__ == "__main__":
    main()
