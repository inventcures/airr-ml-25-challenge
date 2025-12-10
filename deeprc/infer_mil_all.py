import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import sys

from data.load_all_datasets import load_repertoires_pickle, PROCESSED_DIR, TRAIN_DATASETS, TEST_DATASETS
from deeprc.dataset import DeepRCDataset, collate_mil
from deeprc.mil_model import AttentionMIL

MODELS_DIR = Path("models/deeprc")
PREDS_DIR = Path("outputs/deeprc_preds")
PREDS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR = Path("data/embeddings")

def infer_dataset(dataset_name: str, split: str, model: nn.Module, device: torch.device):
    print(f"  Inferring {dataset_name} ({split})...")
    
    out_csv = PREDS_DIR / f"{dataset_name}_{split}_deeprc_preds.csv"
    if out_csv.exists():
        print(f"    ✅ Output already exists: {out_csv}. Skipping.")
        return
    
    pkl_path = PROCESSED_DIR / f"{dataset_name}_{split}.pkl"
    if not pkl_path.exists():
        print(f"    Pickle not found: {pkl_path}")
        return
        
    reps = load_repertoires_pickle(pkl_path)
    
    # Map ds7_1 -> ds7 for embeddings
    emb_ds_name = dataset_name.split("_")[0]
    emb_dir = EMBEDDINGS_DIR / emb_ds_name
    
    if not emb_dir.exists():
        # Fallback
        if (EMBEDDINGS_DIR / dataset_name).exists():
            emb_dir = EMBEDDINGS_DIR / dataset_name
        else:
            print(f"    Embeddings dir not found: {emb_dir}")
            return
    
    # Create Dataset
    # We might want to use all repertoires, even if embeddings are missing (return 0.5?)
    # DeepRCDataset filters missing embeddings.
    # We should handle missing ones in the output dataframe.
    
    ds = DeepRCDataset(reps, emb_dir, max_seqs=10000)
    if len(ds) == 0:
        print("    No embeddings found.")
        return
        
    loader = DataLoader(
        ds, 
        batch_size=8, 
        shuffle=False, 
        collate_fn=collate_mil,
        num_workers=4,
        pin_memory=True
    )
    
    model.eval()
    all_preds = []
    
    # We need to map back to rep_ids.
    # The dataset filters, so we need to know which indices were kept.
    valid_indices = ds.valid_indices
    valid_reps = [reps[i] for i in valid_indices]
    
    with torch.no_grad():
        for bags, labels in tqdm(loader, desc="Inferring"):
            bags = [b.to(device) for b in bags]
            
            logits, _ = model(bags)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(probs)
            
    # Create DataFrame
    results = []
    pred_idx = 0
    
    # Map back to original repertoires
    # We iterate original reps. If in valid_indices, we take next pred. Else NaN or 0.5.
    
    # Actually, simpler: create DF for valid ones, then merge?
    # Or just output valid ones.
    
    for i, r in enumerate(valid_reps):
        p = all_preds[i]
        results.append({
            "repertoire_id": r.rep_id,
            "label": r.label,
            "p_deeprc": p
        })
        
    df = pd.DataFrame(results)
    out_csv = PREDS_DIR / f"{dataset_name}_{split}_deeprc_preds.csv"
    df.to_csv(out_csv, index=False)
    print(f"    Saved preds to {out_csv}")

def infer_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/deeprc_inference.log")
        ]
    )
    
    logging.info(f"[DeepRC Inference] Using device: {device}")
    
    # We need to know which model to use for which dataset.
    # Assuming one model per training dataset (ds1_deeprc_model.pth for ds1, etc.)
    # For test datasets, we need to know which model to apply.
    # Usually we apply the model trained on the corresponding training set?
    # Or is it a single model for all?
    # The challenge has multiple datasets.
    # Let's assume ds1_test uses ds1_deeprc_model.
    
    # Train datasets
    for ds_name in TRAIN_DATASETS.keys():
        model_path = MODELS_DIR / f"{ds_name}_deeprc_model.pth"
        if not model_path.exists():
            print(f"Model not found for {ds_name}: {model_path}")
            continue
            
        print(f"Loading model for {ds_name}...")
        model = AttentionMIL(input_dim=480, hidden_dim=128).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Infer on Train (for meta-ensemble CV? No, we need CV preds for that)
        # We can't use the model trained on all data for meta-ensemble training.
        # We need CV predictions.
        # But for now, let's just infer on Train to see performance (overfitted).
        # infer_dataset(ds_name, "train", model, device)
        
        # Infer on Test
        # Check if there is a corresponding test set
        if ds_name in TEST_DATASETS:
             infer_dataset(ds_name, "test", model, device)
        
        # Also handle sub-datasets like ds7_1, ds7_2?
        # We need a mapping.
        # For now, let's just loop TEST_DATASETS and try to find a matching model.
        # Heuristic: if test ds name starts with train ds name.
        
    # Better loop: Iterate all TEST datasets and find appropriate model.
    for test_ds in TEST_DATASETS.keys():
        # Find matching train ds
        # ds1 -> ds1
        # ds7_1 -> ds7?
        train_ds = test_ds.split("_")[0] # ds1 -> ds1, ds7 -> ds7
        if "ds" not in train_ds:
             # maybe ds7_1 -> ds7
             # split by _ and take first part?
             pass
             
        # Actually, let's look at the keys.
        # ds1, ds2... ds7_1, ds7_2.
        # Train keys: ds1..ds8.
        # So ds7_1 should use ds7 model.
        
        base_ds = test_ds.split("_")[0] # ds7
        if len(test_ds.split("_")) > 1 and test_ds.split("_")[1].isdigit():
             # ds7_1
             base_ds = test_ds.rsplit("_", 1)[0] # ds7_1 -> ds7? No.
             # ds7_1 -> ds7
             pass
        
        # Regex or simple check
        # Try exact match first
        if test_ds in TRAIN_DATASETS:
            model_ds = test_ds
        else:
            # Try stripping suffix
            # ds7_1 -> ds7
            if "_" in test_ds:
                model_ds = test_ds.split("_")[0]
            else:
                model_ds = test_ds
        
        model_path = MODELS_DIR / f"{model_ds}_deeprc_model.pth"
        if not model_path.exists():
            print(f"  No model found for {test_ds} (expected {model_ds})")
            continue
            
        print(f"Inferring {test_ds} using model from {model_ds}...")
        model = AttentionMIL(input_dim=480, hidden_dim=128).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        infer_dataset(test_ds, "test", model, device)

if __name__ == "__main__":
    infer_all()
