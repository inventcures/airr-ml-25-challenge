import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import copy
import logging
from tqdm import tqdm
import sys
import os

# Adjust python path if needed
from data.load_all_datasets import load_repertoires_pickle, PROCESSED_DIR
from deeprc.dataset import DeepRCDataset, collate_mil
from deeprc.mil_model import AttentionMIL

# Directories
MODELS_CV_DIR = Path("models/deeprc_cv")
OUTPUTS_CV_DIR = Path("outputs/deeprc_cv_preds")
EMBEDDINGS_DIR = Path("data/embeddings")

MODELS_CV_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_CV_DIR.mkdir(parents=True, exist_ok=True)

def save_checkpoint(path, state):
    torch.save(state, path)

def load_checkpoint(path, device):
    return torch.load(path, map_location=device)

def train_cv(args):
    dataset_name = args.dataset
    n_folds = args.folds
    
    # Setup dataset-specific model dir
    model_ds_dir = MODELS_CV_DIR / dataset_name
    model_ds_dir.mkdir(parents=True, exist_ok=True)

    Path("logs").mkdir(exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"logs/deeprc_train_cv_{dataset_name}.log")
        ]
    )
    
    logging.info(f"[DeepRC CV] Training on {dataset_name} with {n_folds} folds...")
    
    # Load repertoires
    pkl_path = PROCESSED_DIR / f"{dataset_name}_train.pkl"
    if not pkl_path.exists():
        logging.error(f"Pickle not found: {pkl_path}")
        return
        
    reps = load_repertoires_pickle(pkl_path)
    
    # Filter labeled
    labeled_reps = [r for r in reps if r.label is not None]
    if not labeled_reps:
        logging.error("No labeled data.")
        return
        
    # Check embeddings dir
    emb_ds_name = dataset_name.split("_")[0]
    emb_dir = EMBEDDINGS_DIR / emb_ds_name
    
    if not emb_dir.exists():
        if (EMBEDDINGS_DIR / dataset_name).exists():
            emb_dir = EMBEDDINGS_DIR / dataset_name
        else:
            logging.error(f"Embeddings dir not found: {emb_dir}")
            return
        
    # Create Full Dataset
    full_dataset = DeepRCDataset(labeled_reps, emb_dir, max_seqs=args.max_seqs)
    if len(full_dataset) == 0:
        logging.error("No valid embeddings found.")
        return
    
    # Get labels and original repertoire IDs for StratifiedKFold
    dataset_labels = []
    dataset_rep_ids = []
    
    for i in range(len(full_dataset)):
        real_idx = full_dataset.valid_indices[i]
        rep = full_dataset.repertoires[real_idx]
        dataset_labels.append(rep.label)
        dataset_rep_ids.append(rep.rep_id)
        
    dataset_labels = np.array(dataset_labels)
    dataset_indices = np.arange(len(full_dataset))
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(skf.split(dataset_indices, dataset_labels))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Checkpoint
    checkpoint_path = model_ds_dir / "checkpoint.pth"
    start_fold = 0
    start_epoch = 0
    oof_results = []
    best_auc = 0.0
    best_model_wts = None
    
    # If resuming
    resumed = False
    if checkpoint_path.exists():
        logging.info(f"Found checkpoint at {checkpoint_path}. Resuming...")
        ckpt = load_checkpoint(checkpoint_path, device)
        start_fold = ckpt['fold']
        start_epoch = ckpt['epoch']
        oof_results = ckpt['oof_results']
        # Note: We only need model/optimizer state if we are resuming MID-fold.
        # If start_epoch == 0, it means we finished the previous fold or are just starting.
        resumed = True
        
    if start_fold >= n_folds:
        logging.info("Training already completed for all folds.")
        return

    for fold in range(start_fold, n_folds):
        train_idx, val_idx = splits[fold]
        
        logging.info(f"--- Fold {fold} ---")
        
        train_sub = Subset(full_dataset, train_idx)
        val_sub = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(
            train_sub, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_mil,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_sub, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate_mil,
            num_workers=4,
            pin_memory=True
        )
        
        # Init Model
        model = AttentionMIL(input_dim=480, hidden_dim=128).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.BCEWithLogitsLoss()
        
        # If we are resuming this exact fold
        if resumed and fold == start_fold:
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'best_auc' in ckpt:
                best_auc = ckpt['best_auc']
            if 'best_model_wts' in ckpt:
                best_model_wts = ckpt['best_model_wts']
            
            # Reset resumed flag after handling the first fold
            resumed = False
        else:
            # Fresh start for this fold
            best_auc = 0.0
            best_model_wts = copy.deepcopy(model.state_dict())
            start_epoch = 0 # Ensure we start from 0 if not resuming this fold
        
        # Training Loop
        for epoch in range(start_epoch, args.epochs):
            model.train()
            running_loss = 0.0
            
            # Using tqdm for progress
            pbar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{args.epochs}", leave=False, dynamic_ncols=True)
            for bags, labels in pbar:
                bags = [b.to(device) for b in bags]
                labels = labels.to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                logits, _ = model(bags)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * len(bags)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            epoch_loss = running_loss / len(train_sub)
            
            # Validation
            model.eval()
            val_labels_epoch = []
            val_preds_epoch = []
            
            with torch.no_grad():
                for bags, labels in val_loader:
                    bags = [b.to(device) for b in bags]
                    labels = labels.to(device).unsqueeze(1)
                    
                    logits, _ = model(bags)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    
                    val_labels_epoch.extend(labels.cpu().numpy())
                    val_preds_epoch.extend(probs)
            
            try:
                val_auc = roc_auc_score(val_labels_epoch, val_preds_epoch)
            except:
                val_auc = 0.5
            
            logging.info(f"  Fold {fold} Epoch {epoch+1} - Loss: {epoch_loss:.4f} - Val AUC: {val_auc:.4f}")
                
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_wts = copy.deepcopy(model.state_dict())
                logging.info(f"    New best AUC for fold {fold}: {best_auc:.4f}")

            # Save Checkpoint after every epoch
            # We save the NEXT epoch index so we start there
            save_checkpoint(checkpoint_path, {
                'fold': fold,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'best_model_wts': best_model_wts,
                'oof_results': oof_results
            })
            
        logging.info(f"Fold {fold} Finished. Best Val AUC: {best_auc:.4f}")
        
        # Save best model for this fold
        fold_model_path = model_ds_dir / f"fold{fold}_model.pth"
        torch.save(best_model_wts, fold_model_path)
        logging.info(f"Saved best model for fold {fold} to {fold_model_path}")
        
        # Load best weights for OOF inference
        model.load_state_dict(best_model_wts)
        model.eval()
        
        with torch.no_grad():
            current_ptr = 0
            for bags, labels in val_loader:
                bags = [b.to(device) for b in bags]
                logits, _ = model(bags)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                
                batch_size = len(bags)
                batch_indices = val_idx[current_ptr : current_ptr + batch_size]
                current_ptr += batch_size
                
                for i, prob in enumerate(probs):
                    original_idx = batch_indices[i]
                    rep_id = dataset_rep_ids[original_idx]
                    true_label = dataset_labels[original_idx]
                    
                    oof_results.append({
                        "repertoire_id": rep_id,
                        "label": true_label,
                        "p_deeprc": prob,
                        "fold": fold
                    })
        
        # Update Checkpoint for NEXT fold
        # We clear model state from checkpoint to force init for next fold
        save_checkpoint(checkpoint_path, {
            'fold': fold + 1,
            'epoch': 0,
            'oof_results': oof_results
            # No model_state_dict means next load will init fresh
        })

    # Save OOF predictions
    oof_df = pd.DataFrame(oof_results)
    out_csv = OUTPUTS_CV_DIR / f"{dataset_name}_oof.csv"
    oof_df.to_csv(out_csv, index=False)
    logging.info(f"Saved OOF predictions to {out_csv}")
    
    # Cleanup checkpoint
    if checkpoint_path.exists():
        os.remove(checkpoint_path)
        logging.info("Training complete. Removed checkpoint.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. ds1)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seqs", type=int, default=10000)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    train_cv(args)