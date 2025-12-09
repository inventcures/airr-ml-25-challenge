import numpy as np
import pandas as pd
from typing import List, Dict
import torch

from malid.esm_seq_model import ESMSequenceClassifier

def score_sequences(
    sequences: List[str], 
    embeddings: np.ndarray, 
    model: ESMSequenceClassifier
) -> List[float]:
    """
    Score sequences based on their likelihood of being disease-associated.
    Using the trained ESM sequence classifier.
    """
    if len(sequences) == 0:
        return []
        
    if len(embeddings) != len(sequences):
        raise ValueError("Mismatch between sequences and embeddings length")
        
    # Predict probability of class 1
    probs = model.predict_proba_sequences(embeddings)[:, 1]
    return probs.tolist()

def rank_sequences(
    sequences: List[str], 
    embeddings: np.ndarray, 
    model: ESMSequenceClassifier,
    top_k: int = 100
) -> pd.DataFrame:
    """
    Rank sequences and return top k.
    """
    scores = score_sequences(sequences, embeddings, model)
    
    df = pd.DataFrame({
        "sequence": sequences,
        "score": scores
    })
    
    # Sort descending
    df = df.sort_values("score", ascending=False)
    
    return df.head(top_k)
