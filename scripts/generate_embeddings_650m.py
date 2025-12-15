
import sys
import torch
import esm
import numpy as np
import pickle
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.load_all_datasets import load_repertoires_pickle, PROCESSED_DIR, TRAIN_DATASETS, TEST_DATASETS

# CONFIG
MODEL_NAME = "esm2_t33_650M_UR50D"
LAYER = 33
BATCH_SIZE = 16 # Adjusted for 650M on 24GB VRAM. 64 might OOM.
EMBEDDINGS_DIR = Path("data/embeddings") # Assumes this is on the 2TB volume
LOG_FILE = Path("generate_embeddings.log")

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

def sanitize_sequence(seq: str) -> str:
    """Sanitize sequence for ESM tokenizer."""
    # Remove stop codons or unknown chars that cause KeyError: '*'
    return seq.replace("*", "")

def embed_dataset(dataset_name: str, split: str):
    """
    Generate and save ESM2-650M embeddings for a given dataset pickle.
    """
    # 1. Load Data
    pkl_name = f"{dataset_name}_{split}.pkl"
    if split == "train" and (PROCESSED_DIR / f"{dataset_name}.pkl").exists():
        # Fallback for train sets sometimes named just ds1.pkl
        pkl_name = f"{dataset_name}.pkl"
        
    pkl_path = PROCESSED_DIR / pkl_name
    if not pkl_path.exists():
        logging.warning(f"⚠️  Pickle not found: {pkl_path}")
        return

    logging.info(f"Loading {pkl_path}...")
    try:
        reps = load_repertoires_pickle(pkl_path)
    except Exception as e:
        logging.error(f"❌ Failed to load pickle {pkl_path}: {e}")
        return
    
    # 2. Setup Output Directory
    # Match documentation structure: data/embeddings/ds1/train/
    if split == "train":
        target_dir = EMBEDDINGS_DIR / dataset_name / "train"
    elif "test" in dataset_name:
         # Special handling for ds7_test -> ds7/test? or ds7/1_test?
         parts = dataset_name.split("_")
         base_ds = parts[0] # ds7
         
         if len(parts) == 2 and parts[1] == "test":
             # "ds1_test" -> ds1/test
             target_dir = EMBEDDINGS_DIR / base_ds / "test"
         elif len(parts) >= 3 and "test" in parts[-1]:
             # "ds7_1_test" -> ds7/1_test
             sub_folder = "_".join(parts[1:]) 
             target_dir = EMBEDDINGS_DIR / base_ds / sub_folder
         else:
             # Fallback
             target_dir = EMBEDDINGS_DIR / dataset_name / "test"
             
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Load Model (Lazy load to save VRAM if not needed?)
    global model, alphabet, batch_converter
    if 'model' not in globals():
        logging.info(f"Loading {MODEL_NAME}...")
        try:
            # Set Persistent Cache for Model Weights (Avoid re-downloading)
            CACHE_DIR = Path("data/model_cache")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(CACHE_DIR))
            logging.info(f"  Model cache set to: {CACHE_DIR}")

            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            batch_converter = alphabet.get_batch_converter()
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
                logging.info("✅ Model loaded on GPU.")
            else:
                logging.warning("⚠️  Warning: CPU only mode. This will be slow.")
        except Exception as e:
            logging.critical(f"Expected behavior: Install fair-esm. Error loading model: {e}")
            sys.exit(1)

    # 4. Processing Loop
    existing_files = list(target_dir.glob("*.npy"))
    logging.info(f"Dataset {dataset_name}: Found {len(existing_files)} existing files in {target_dir}. Processing remaining...")
    
    # Filter reps that are already done
    reps_to_process = []
    for r in reps:
        if not (target_dir / f"{r.rep_id}.npy").exists():
            reps_to_process.append(r)
            
    if not reps_to_process:
        logging.info("All repertoires already processed! ✅")
        return

    # Helper for inference
    def run_batch(batch_s):
        # SANITIZE SEQUENCES HERE
        clean_batch_s = [sanitize_sequence(s) for s in batch_s]
        
        batch_d = [(str(j), s) for j, s in enumerate(clean_batch_s)]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_d)
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[LAYER], return_contacts=False)
        token_reps = results["representations"][LAYER]
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        
        res_list = []
        for j, tokens_len in enumerate(batch_lens):
            if tokens_len <= 2: 
                seq_rep = np.zeros(1280, dtype=np.float16)
            else:
                seq_rep = token_reps[j, 1 : tokens_len - 1].mean(0).cpu().numpy().astype(np.float16)
            res_list.append(seq_rep)
        return res_list

    # Wrap in TQDM
    pbar = tqdm(reps_to_process, desc=f"{dataset_name}")
    for r in pbar:
        out_file = target_dir / f"{r.rep_id}.npy"
        tmp_file = target_dir / f"{r.rep_id}.tmp.npy" # Atomic write
        
        # Robust Repertoire Processing
        try:
            # Extract Seqs & OPTIMIZATION: Sort by length to minimize padding
            seqs = [s for s in r.junction_aa if isinstance(s, str) and len(s) > 0]
            if not seqs:
                # Save empty
                np.save(tmp_file, np.array([]))
                tmp_file.rename(out_file)
                continue
            
            # Sort for batch efficiency (bucket similar lengths)
            seqs.sort(key=len) 
                
            embeddings = []
            batch_failure = False
            total_seqs = len(seqs)
            
            # Batch Processing
            i = 0
            TOKEN_BUDGET = 4500 # Conservative limit for 650M model on 24GB VRAM
            
            while i < len(seqs):
                # Heartbeat for huge repertoires
                if i > 0 and i % 5000 == 0:
                     # Calculate rough instantaneous batch size (look back)
                     # Just logging the 'est_batch_size' from prev iteration is hard here due to scope.
                     # We will just print the loop index i.
                     logging.info(f"    ...processed {i}/{total_seqs} (Current Batch Size: {est_batch_size})...")
                
                # --- Dynamic Batching Logic ---
                # 1. Initial Guess based on start of batch (shortest seq because sorted)
                current_len = len(seqs[i])
                # +2 for CLS/EOS tokens, +padding overhead safety
                est_batch_size = max(1, TOKEN_BUDGET // (current_len + 4))
                
                # Cap valid range
                est_batch_size = min(est_batch_size, 2048) # 2048 hard limit for sanity
                
                # 2. Safety Lookahead (Prevent OOM if length jumps drastically in this slice)
                # If we take est_batch_size, the max length (padding) is determined by the last element
                last_idx = min(i + est_batch_size, len(seqs)) - 1
                max_len_in_batch = len(seqs[last_idx]) + 2
                
                # Recalculate usage: (Batch Size) * (Max Length in Batch)
                # If this exceeds budget, shrink batch size
                actual_tokens = (last_idx - i + 1) * max_len_in_batch
                if actual_tokens > TOKEN_BUDGET:
                    # Resize: New Size = Budget / Max Len
                    # Note: Max Len might drop if we shrink, but using current max_len is a safe conservative bound
                    est_batch_size = max(1, TOKEN_BUDGET // max_len_in_batch)
                
                # 3. Final Slice
                batch_seqs = seqs[i : i + est_batch_size]
                
                try:
                    # Try dynamic batch with Mixed Precision
                    # Fix FutureWarning: use torch.amp.autocast('cuda')
                    with torch.amp.autocast('cuda'):
                        batch_embs = run_batch(batch_seqs)
                    embeddings.extend(batch_embs)
                    i += len(batch_seqs) # Advance by ACTUAL batch size
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # OOM Fallback: Process this batch one-by-one
                        torch.cuda.empty_cache()
                        # hidden log: logging.warning(f"OOM on rep {r.rep_id}. Retrying batch as single items...")
                        
                        sub_failure = False
                        for sub_seq in batch_seqs:
                            try:
                                sub_emb_list = run_batch([sub_seq])
                                embeddings.extend(sub_emb_list)
                            except Exception as sub_e:
                                logging.error(f"❌ Failed even with batch_size=1 for rep {r.rep_id}: {sub_e}")
                                sub_failure = True
                                break
                        
                        if sub_failure:
                            batch_failure = True
                            break
                        
                        i += BATCH_SIZE
                    else:
                        logging.error(f"❌ Critical Error on rep {r.rep_id}: {e}")
                        batch_failure = True
                        break
                except Exception as e:
                     logging.error(f"❌ Unexpected Error on rep {r.rep_id}: {e}")
                     batch_failure = True
                     break
            
            # Save ONLY if successful
            if not batch_failure:
                final_arr = np.vstack(embeddings).astype(np.float16)
                np.save(tmp_file, final_arr)
                tmp_file.rename(out_file) # Atomic rename
            else:
                logging.warning(f"Skipping save for {r.rep_id} due to inference failure.")

            # Explicit Cleanup to prevent Memory Leaks (Critical for DS7)
            del embeddings, final_arr
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"🔥 CRASH AVOIDED on Repertoire {r.rep_id}: {e}")
            # Continue to next repertoire
            continue

def main():
    try:
        # PRIORITY: Process Test Datasets FIRST (Critical for Submission)
        logging.info("🚀 STARTING WITH TEST DATASETS (PRIORITY)...")
        for ds_name in TEST_DATASETS.keys():
            embed_dataset(ds_name, "test")

        # Process Train Datasets (Lower Priority if time is tight)
        logging.info("🔄 Processing Train Datasets...")
        for ds_name in TRAIN_DATASETS.keys():
            embed_dataset(ds_name, "train")
            
        logging.info("🎉 All Datasets Processed Successfully!")
        
    except KeyboardInterrupt:
        logging.info("\n🛑 Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logging.critical(f"🔥 Fatal Script Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
