import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class AttentionMIL(nn.Module):
    def __init__(
        self, 
        input_dim: int = 1280, 
        hidden_dim: int = 128, 
        n_classes: int = 1, 
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature extractor (optional, to reduce dim or transform embeddings)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism
        # a = softmax(w^T tanh(V h))
        self.attention_V = nn.Linear(hidden_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, bags: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            bags: List of tensors, each (N_i, D)
        Returns:
            logits: (B, n_classes)
            attention_weights: List of (N_i, 1)
        """
        batch_logits = []
        batch_attention = []
        
        for bag in bags:
            # bag: (N, D)
            if bag.size(0) == 0:
                # Handle empty bag
                # Should not happen if dataset filters, but just in case
                device = self.attention_w.weight.device
                batch_logits.append(torch.zeros(1, device=device)) # Logit 0 -> Prob 0.5
                batch_attention.append(torch.zeros(0, 1, device=device))
                continue
                
            # 1. Feature extraction
            h = self.feature_extractor(bag) # (N, H)
            
            # 2. Attention
            # A = w^T tanh(V h)
            a_raw = self.attention_w(torch.tanh(self.attention_V(h))) # (N, 1)
            a = F.softmax(a_raw, dim=0) # (N, 1)
            
            # 3. Aggregation
            z = torch.sum(a * h, dim=0) # (H,)
            
            # 4. Classification
            logits = self.classifier(z) # (n_classes,)
            
            batch_logits.append(logits)
            batch_attention.append(a)
            
        return torch.stack(batch_logits), batch_attention
