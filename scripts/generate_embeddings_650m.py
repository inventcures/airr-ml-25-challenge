
import sys
import torch
import esm
import numpy as np
import pickle
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
        print(f"⚠️  Pickle not found: {pkl_path}")
        return

    print(f"\nLoading {pkl_path}...")
    reps = load_repertoires_pickle(pkl_path)
    
    # 2. Setup Output Directory
    # Match documentation structure: data/embeddings/ds1/train/
    if split == "train":
        target_dir = EMBEDDINGS_DIR / dataset_name / "train"
    elif "test" in dataset_name:
         # Special handling for ds7_test -> ds7/test? or ds7/1_test?
         # The keys in TEST_DATASETS are like "ds1_test", "ds7_1_test", "ds7_2_test"
         # We need to parse this carefully to reconstruct the folder path.
         
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
    # No, we need it.
    global model, alphabet, batch_converter
    if 'model' not in globals():
        print(f"Loading {MODEL_NAME}...")
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            print("✅ Model loaded on GPU.")
        else:
            print("⚠️  Warning: CPU only mode. This will be slow.")

    # 4. Processing Loop
    # 4. Processing Loop
    existing_files = list(target_dir.glob("*.npy"))
    print(f"Dataset {dataset_name}: Found {len(existing_files)} existing files in {target_dir}. Processing remaining...")
    
    # Filter reps that are already done
    reps_to_process = []
    for r in reps:
        if not (target_dir / f"{r.rep_id}.npy").exists():
            reps_to_process.append(r)
            
    if not reps_to_process:
        print("All repertoires already processed! ✅")
        return

    for r in tqdm(reps_to_process, desc=f"{dataset_name}"):
        out_file = target_dir / f"{r.rep_id}.npy"
        tmp_file = target_dir / f"{r.rep_id}.tmp.npy" # Atomic write
        
        # Extract Seqs
        seqs = [s for s in r.junction_aa if isinstance(s, str) and len(s) > 0]
        if not seqs:
            try:
                np.save(tmp_file, np.array([]))
                tmp_file.rename(out_file)
            except Exception as e:
                print(f"Failed to save empty rep {r.rep_id}: {e}")
            continue
            
        # Helper for inference
        def run_batch(batch_s):
            batch_d = [(str(j), s) for j, s in enumerate(batch_s)]
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

        embeddings = []
        batch_failure = False
        
        # Batch Processing
        i = 0
        while i < len(seqs):
            batch_seqs = seqs[i : i + BATCH_SIZE]
            try:
                # Try standard batch
                batch_embs = run_batch(batch_seqs)
                embeddings.extend(batch_embs)
                i += BATCH_SIZE
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # OOM Fallback: Process this batch one-by-one
                    torch.cuda.empty_cache()
                    # print(f"⚠️ OOM on rep {r.rep_id} idx {i}. Retrying with batch_size=1...")
                    
                    sub_failure = False
                    for sub_seq in batch_seqs:
                        try:
                            sub_emb_list = run_batch([sub_seq])
                            embeddings.extend(sub_emb_list)
                        except Exception as sub_e:
                            print(f"❌ Failed even with batch_size=1 for rep {r.rep_id}: {sub_e}")
                            sub_failure = True
                            break
                    
                    if sub_failure:
                        batch_failure = True
                        break
                    
                    i += BATCH_SIZE
                else:
                    print(f"❌ Critical Error on rep {r.rep_id}: {e}")
                    batch_failure = True
                    break
        
        # Save ONLY if successful
        if not batch_failure:
            try:
                final_arr = np.vstack(embeddings).astype(np.float16)
                np.save(tmp_file, final_arr)
                tmp_file.rename(out_file) # Atomic rename
            except Exception as e:
                print(f"Failed to write file for {r.rep_id}: {e}")
                if tmp_file.exists(): tmp_file.unlink()
        else:
            print(f"Skipping save for {r.rep_id} due to inference failure.")

def main():
    # Process Train Datasets
    for ds_name in TRAIN_DATASETS.keys():
        embed_dataset(ds_name, "train")
        
    # Process Test Datasets
    for ds_name in TEST_DATASETS.keys():
        # Usually test sets are named "ds1_test" in the dict keys
        # The split argument is mostly for file suffix checking
        embed_dataset(ds_name, "test")

if __name__ == "__main__":
    main()
