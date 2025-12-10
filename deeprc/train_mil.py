import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import copy

from data.load_all_datasets import load_repertoires_pickle, PROCESSED_DIR
from deeprc.dataset import DeepRCDataset, collate_mil
from deeprc.mil_model import AttentionMIL

MODELS_DIR = Path("models/deeprc")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR = Path("data/embeddings")

def train(args):
    dataset_name = args.dataset
    print(f"[DeepRC] Training on {dataset_name}...")
    
    # Load repertoires
    pkl_path = PROCESSED_DIR / f"{dataset_name}_train.pkl"
    if not pkl_path.exists():
        print(f"Pickle not found: {pkl_path}")
        return
        
    reps = load_repertoires_pickle(pkl_path)
    
    # Filter labeled
    labeled_reps = [r for r in reps if r.label is not None]
    if not labeled_reps:
        print("No labeled data.")
        return
        
    # Check embeddings dir
    # Map ds7_1 -> ds7, ds8_2 -> ds8, etc.
    emb_ds_name = dataset_name.split("_")[0]
    emb_dir = EMBEDDINGS_DIR / emb_ds_name
    
    if not emb_dir.exists():
        # Fallback: maybe it IS in ds7_1?
        if (EMBEDDINGS_DIR / dataset_name).exists():
            emb_dir = EMBEDDINGS_DIR / dataset_name
        else:
            print(f"Embeddings dir not found: {emb_dir} (checked {dataset_name} too)")
            return
        
    # Create Dataset
    full_dataset = DeepRCDataset(labeled_reps, emb_dir, max_seqs=args.max_seqs)
    if len(full_dataset) == 0:
        print("No valid embeddings found.")
        return
        
    # Split Train/Val
    # -------------------------------------------------------------------------
    # EDUCATIONAL NOTE: Train/Validation Split
    # -------------------------------------------------------------------------
    # We split our data into two parts:
    # 1. Training Set (80%): Used to update the model's weights.
    # 2. Validation Set (20%): Used to evaluate the model's performance on unseen data.
    #
    # Why? If we only looked at training accuracy, we wouldn't know if the model
    # is memorizing the data (overfitting) or actually learning patterns.
    # If Val Loss goes UP while Train Loss goes DOWN, we are overfitting!
    # -------------------------------------------------------------------------
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
    
    # collate_fn=collate_mil is crucial here!
    # It handles "bags" of sequences that have different sizes (e.g., one rep has 100 seqs, another 500).
    # Standard PyTorch default_collate assumes all items are the same size.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mil)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mil)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AttentionMIL(input_dim=480, hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Checkpoint path
    checkpoint_path = MODELS_DIR / f"{dataset_name}_checkpoint.pth"
    start_epoch = 0
    best_auc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # Resume from checkpoint if exists
    if checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_auc = checkpoint.get('best_auc', 0.0)
        best_model_wts = checkpoint.get('best_model_wts', copy.deepcopy(model.state_dict()))
        print(f"Resuming from epoch {start_epoch}")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        
        for bags, labels in train_loader:
            bags = [b.to(device) for b in bags]
            labels = labels.to(device).unsqueeze(1) # (B, 1)
            
            optimizer.zero_grad()
            
            logits, _ = model(bags)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(bags)
            
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probs)
            
        epoch_loss = running_loss / len(train_ds)
        try:
            train_auc = roc_auc_score(all_labels, all_preds)
        except:
            train_auc = 0.5
            
        # Validation
        model.eval()
        val_labels = []
        val_preds = []
        val_loss = 0.0
        
        with torch.no_grad():
            for bags, labels in val_loader:
                bags = [b.to(device) for b in bags]
                labels = labels.to(device).unsqueeze(1)
                
                logits, _ = model(bags)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(bags)
                
                probs = torch.sigmoid(logits).cpu().numpy()
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(probs)
                
        val_loss = val_loss / len(val_ds)
        try:
            val_auc = roc_auc_score(val_labels, val_preds)
        except:
            val_auc = 0.5
            
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {epoch_loss:.4f} AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_auc': best_auc,
            'best_model_wts': best_model_wts
        }, checkpoint_path)
            
    print(f"Best Val AUC: {best_auc:.4f}")
    
    # Save best model
    model.load_state_dict(best_model_wts)
    out_path = MODELS_DIR / f"{dataset_name}_deeprc_model.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. ds1)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seqs", type=int, default=10000)
    args = parser.parse_args()
    train(args)
