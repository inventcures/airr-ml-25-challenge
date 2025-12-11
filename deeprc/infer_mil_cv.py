import sys
import os
from pathlib import Path

# Add project root to path to allow importing 'data' module
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

from data.load_all_datasets import load_repertoires_pickle, PROCESSED_DIR, TEST_DATASETS
from deeprc.dataset import DeepRCDataset, collate_mil
from deeprc.mil_model import AttentionMIL

# Directories
MODELS_CV_DIR = Path("models/deeprc_cv")
OUTPUTS_CV_DIR = Path("outputs/deeprc_cv_preds")
EMBEDDINGS_DIR = Path("data/embeddings")

OUTPUTS_CV_DIR.mkdir(parents=True, exist_ok=True)

def infer_dataset_cv(dataset_name: str, split: str, model_ds_name: str, n_folds: int, device: torch.device, batch_size: int = 8):
    logging.info(f"Inferring {dataset_name} ({split}) using models from {model_ds_name} (Ensemble of {n_folds} folds)...")

    # Output path
    out_csv = OUTPUTS_CV_DIR / f"{dataset_name}_{split}_deeprc_preds.csv"
    if out_csv.exists():
        logging.info(f"  ✅ Output already exists: {out_csv}. Skipping.")
        return

    # Checkpoint path for partial inference
    checkpoint_path = OUTPUTS_CV_DIR / f"{dataset_name}_{split}_infer_checkpoint.pth"

    # Load Repertoires
    pkl_path = PROCESSED_DIR / f"{dataset_name}_{split}.pkl"
    if not pkl_path.exists():
        logging.error(f"  Pickle not found: {pkl_path}")
        return
        
    reps = load_repertoires_pickle(pkl_path)
    
    # Load Embeddings
    emb_ds_lookup = dataset_name.split("_")[0] # ds7_1 -> ds7
    emb_dir = EMBEDDINGS_DIR / emb_ds_lookup
    
    if not emb_dir.exists():
        if (EMBEDDINGS_DIR / dataset_name).exists():
            emb_dir = EMBEDDINGS_DIR / dataset_name
        else:
            logging.error(f"  Embeddings dir not found: {emb_dir}")
            return
            
    # Create Dataset
    ds = DeepRCDataset(reps, emb_dir, max_seqs=10000)
    if len(ds) == 0:
        logging.error("  No embeddings found.")
        return
    
    # Check for inference checkpoint
    final_preds = []
    start_idx = 0
    if checkpoint_path.exists():
        logging.info(f"  Found checkpoint {checkpoint_path}. Resuming inference...")
        try:
            ckpt = torch.load(checkpoint_path)
            final_preds = ckpt.get('final_preds', [])
            start_idx = len(final_preds)
            pct = (start_idx / len(ds)) * 100
            logging.info(f"  Resuming from index {start_idx}/{len(ds)} ({pct:.1f}% complete).")
        except Exception as e:
            logging.error(f"  Failed to load checkpoint: {e}. Starting from scratch.")
            final_preds = []
            start_idx = 0
            
    if start_idx >= len(ds):
        logging.info("  Inference already completed in checkpoint. Proceeding to save CSV.")
    else:
        # Create loader for remaining data
        # DeepRCDataset is indexable, so we can use Subset
        subset_indices = range(start_idx, len(ds))
        subset = Subset(ds, subset_indices)
        
        loader = DataLoader(
            subset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_mil, 
            num_workers=4,
            pin_memory=True
        )
        
        # Load all models
        models = []
        full_model_dir = MODELS_CV_DIR / model_ds_name
        if not full_model_dir.exists():
            logging.error(f"  Model directory not found: {full_model_dir}")
            return

        for fold in range(n_folds):
            fold_path = full_model_dir / f"fold{fold}_model.pth"
            if not fold_path.exists():
                logging.error(f"  Model for fold {fold} not found at {fold_path}")
                return
                
            m = AttentionMIL(input_dim=480, hidden_dim=128).to(device)
            m.load_state_dict(torch.load(fold_path, map_location=device))
            m.eval()
            models.append(m)
            
        logging.info(f"  Loaded {len(models)} models.")
        
        # Inference Loop
        batches_processed = 0
        save_interval = 10 
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Inferring {dataset_name}", dynamic_ncols=True)
            for bags, labels in pbar:
                bags = [b.to(device) for b in bags]
                
                # Run all models
                batch_probs_list = []
                for m in models:
                    logits, _ = m(bags)
                    probs = torch.sigmoid(logits).cpu() 
                    batch_probs_list.append(probs)
                
                # Average predictions
                stacked = torch.stack(batch_probs_list)
                mean_probs = stacked.mean(dim=0).flatten().tolist()
                
                final_preds.extend(mean_probs)
                batches_processed += 1
                
                # Periodic Checkpoint
                if batches_processed % save_interval == 0:
                    torch.save({'final_preds': final_preds}, checkpoint_path)
                    # logging.debug(f"  💾 Checkpoint saved at {batches_processed} batches.") 
                    
            # Final save of checkpoint after loop
            torch.save({'final_preds': final_preds}, checkpoint_path)
            logging.info(f"  💾 Checkpoint saved. Inference finished for {dataset_name}.")

    # Map back to Repertoires
    results = []
    # Note: ds.valid_indices maps 0..len(ds) to index in 'reps'
    # We have predictions for 0..len(ds)
    valid_reps = [reps[i] for i in ds.valid_indices]
    
    if len(final_preds) != len(valid_reps):
        logging.error(f"  Mismatch: {len(final_preds)} preds vs {len(valid_reps)} valid reps.")
        return

    for i, r in enumerate(valid_reps):
        results.append({
            "repertoire_id": r.rep_id,
            "label": r.label if r.label is not None else -1, 
            "p_deeprc": final_preds[i]
        })
        
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    logging.info(f"  Saved averaged preds to {out_csv}")
    
    # Cleanup checkpoint
    if checkpoint_path.exists():
        os.remove(checkpoint_path)
        logging.info("  Removed checkpoint.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--datasets", type=str, nargs="+", help="Specific test datasets to infer (e.g. ds1 ds2). If empty, infers ALL test datasets.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference (default: 8)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/deeprc_infer_cv.log")
        ]
    )
    
    datasets_to_run = args.datasets if args.datasets else list(TEST_DATASETS.keys())
    
    for test_ds in datasets_to_run:
        train_ds = test_ds.split("_")[0] # ds1 -> ds1, ds7_1 -> ds7
        
        infer_dataset_cv(test_ds, "test", train_ds, args.folds, device, batch_size=args.batch_size)

if __name__ == "__main__":
    main()