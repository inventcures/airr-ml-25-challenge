import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from data.load_all_datasets import load_repertoires_pickle, PROCESSED_DIR, TEST_DATASETS
from malid.esm_seq_model import ESMSequenceClassifier
from scripts.rank_sequences_task2_core import rank_sequences

MODELS_DIR = Path("models/esm_seq")
EMBEDDINGS_DIR = Path("data/embeddings")
OUTPUT_DIR = Path("outputs/task2_ranking")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def rank_sequences_task2_all():
    # Task 2 is about ranking sequences in specific test datasets.
    # We need to know which datasets are for Task 2.
    # Usually it's all test datasets or specific ones.
    # We will generate rankings for ALL test datasets.
    
    for test_ds in TEST_DATASETS.keys():
        print(f"\n[Task 2] Processing {test_ds}...")
        
        # Determine corresponding model (from train ds)
        # Heuristic: ds1 -> ds1
        train_ds = test_ds.split("_")[0]
        # Handle ds7_1 -> ds7
        if train_ds not in MODELS_DIR.glob(f"{train_ds}_esm_seq_model.joblib"):
             # Try stripping suffix
             if "_" in test_ds:
                 train_ds = test_ds.split("_")[0]
        
        model_path = MODELS_DIR / f"{train_ds}_esm_seq_model.joblib"
        if not model_path.exists():
            print(f"  Model not found for {test_ds} (expected {train_ds}). Skipping.")
            continue
            
        print(f"  Loading model from {model_path}...")
        model = ESMSequenceClassifier.load(model_path)
        
        # Load repertoires
        pkl_path = PROCESSED_DIR / f"{test_ds}_test.pkl"
        if not pkl_path.exists():
            print(f"  Pickle not found: {pkl_path}")
            continue
            
        reps = load_repertoires_pickle(pkl_path)
        
        # Load embeddings
        emb_dir = EMBEDDINGS_DIR / test_ds
        if not emb_dir.exists():
            print(f"  Embeddings dir not found: {emb_dir}")
            continue
            
        all_rankings = []
        
        for r in reps:
            emb_path = emb_dir / f"{r.rep_id}.npy"
            if not emb_path.exists():
                continue
                
            try:
                emb = np.load(emb_path)
                if emb.ndim == 1:
                    if len(emb) == 0:
                        continue
                    emb = emb.reshape(1, -1)
            except:
                continue
                
            # Filter valid sequences (length > 0)
            valid_seqs = [s for s in r.junction_aa if len(s) > 0]
            if len(valid_seqs) != len(emb):
                # Mismatch? Maybe some sequences failed embedding?
                # Or maybe empty strings were skipped?
                # embed_esm650m.py filters valid_seqs.
                # So they should match.
                if len(valid_seqs) == 0:
                    continue
                # Truncate to min length just in case
                min_len = min(len(valid_seqs), len(emb))
                valid_seqs = valid_seqs[:min_len]
                emb = emb[:min_len]
            
            # Rank
            df_rank = rank_sequences(valid_seqs, emb, model, top_k=100)
            df_rank["repertoire_id"] = r.rep_id
            
            # Add V/J calls
            # We need to map back from sequence to V/J.
            # Since valid_seqs are unique (usually), we can map.
            # But wait, r.junction_aa might have duplicates?
            # embed_esm650m.py uses r.junction_aa directly.
            # If there are duplicates, embeddings might be computed for each?
            # Let's assume 1-to-1 mapping with r.junction_aa indices.
            
            # We need to find the index of the sequence in r.junction_aa to get v_call/j_call.
            # This is slow if we search.
            # Better: construct a lookup or pass indices.
            
            # Let's just create a DF of the repertoire first
            rep_df = pd.DataFrame({
                "sequence": r.junction_aa,
                "v_call": r.v_call,
                "j_call": r.j_call
            })
            # Drop duplicates if any, or keep first?
            # If we drop duplicates, we lose 1-to-1 with embeddings if embeddings were computed on full list.
            # embed_esm650m.py: "valid_seqs = [s for s in r.junction_aa if len(s) > 0]"
            # It preserves order.
            
            # Filter rep_df to valid seqs
            rep_df = rep_df[rep_df["sequence"].str.len() > 0]
            
            # Now merge with df_rank
            # df_rank has "sequence" and "score"
            # We merge on "sequence".
            # Note: if a sequence appears multiple times with different V/J, we might get duplicates.
            # We just need one V/J pair.
            
            merged = df_rank.merge(rep_df, on="sequence", how="left")
            # Drop duplicates (same sequence, same score, same V/J)
            merged = merged.drop_duplicates(subset=["sequence"])
            
            all_rankings.append(merged)
            
        if all_rankings:
            final_df = pd.concat(all_rankings)
            # Format: repertoire_id, sequence, score, v_call, j_call
            final_df = final_df[["repertoire_id", "sequence", "score", "v_call", "j_call"]]
            out_csv = OUTPUT_DIR / f"{test_ds}_ranking.csv"
            final_df.to_csv(out_csv, index=False)
            print(f"  Saved rankings to {out_csv}")
        else:
            print("  No rankings generated.")

if __name__ == "__main__":
    rank_sequences_task2_all()
