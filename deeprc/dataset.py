import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
import random

from data.load_data import Repertoire

class DeepRCDataset(Dataset):
    def __init__(
        self, 
        repertoires: List[Repertoire], 
        embeddings_dir: Path, 
        max_seqs: int = 10000, 
        sample_mode: str = "random" # random or top_k (if we had scores, but we don't yet)
    ):
        """
        Args:
            repertoires: List of Repertoire objects.
            embeddings_dir: Directory containing .npy embedding files.
            max_seqs: Maximum number of sequences to sample per repertoire.
        """
        self.repertoires = repertoires
        self.embeddings_dir = embeddings_dir
        self.max_seqs = max_seqs
        self.sample_mode = sample_mode
        
        # Filter out repertoires that don't have embeddings?
        # Or just error/return empty during getitem?
        # Better to filter upfront to avoid runtime errors during training.
        self.valid_indices = []
        self.file_map = {} # rep_id -> full_path
        
        # Pre-scan directory to find all .npy files
        # This handles train/test subfolders and flat structures
        print(f"[DeepRCDataset] Scanning {self.embeddings_dir}...")
        if self.embeddings_dir.exists():
            for p in self.embeddings_dir.rglob("*.npy"):
                # stem is filename without extension (rep_id)
                self.file_map[p.stem] = p
        
        for i, r in enumerate(self.repertoires):
            if str(r.rep_id) in self.file_map:
                self.valid_indices.append(i)
            else:
                # Debug: print first few missing
                if len(self.valid_indices) < 5:
                    print(f"  Warning: Missing embedding for {r.rep_id}")
        
        print(f"[DeepRCDataset] Found embeddings for {len(self.valid_indices)} / {len(self.repertoires)} repertoires.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        rep = self.repertoires[real_idx]
        
        # Load embeddings
        emb_path = self.file_map[str(rep.rep_id)]
        try:
            embeddings = np.load(emb_path)
            # Ensure 2D
            if embeddings.ndim == 1:
                if len(embeddings) == 0:
                    # Empty repertoire
                    embeddings = np.zeros((1, 1280), dtype=np.float32) # Dummy
                else:
                    embeddings = embeddings.reshape(1, -1)
        except Exception:
            # Corrupt file? Return dummy
            embeddings = np.zeros((1, 1280), dtype=np.float32)

        n_seqs = embeddings.shape[0]
        
        # Subsample if needed
        if n_seqs > self.max_seqs:
            indices = np.random.choice(n_seqs, self.max_seqs, replace=False)
            embeddings = embeddings[indices]
        
        # Convert to tensor
        embeddings_tensor = torch.from_numpy(embeddings).float()
        
        # Label
        label = rep.label if rep.label is not None else -1
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return embeddings_tensor, label_tensor

def collate_mil(batch):
    """
    Collate function for MIL.
    Since bags have different sizes, we cannot stack them directly into a tensor of (B, N, D).
    We can either:
    1. Pad to max size in batch (wasteful).
    2. Return a list of tensors.
    3. Return a packed sequence or similar.
    
    DeepRC usually handles a list of bags.
    """
    bags, labels = zip(*batch)
    labels = torch.stack(labels)
    return list(bags), labels
