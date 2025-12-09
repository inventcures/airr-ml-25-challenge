import modal
from pathlib import Path
import pickle
import numpy as np
import time
import sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

app = modal.App("verify-counts")

# Mount volumes
vol_35m = modal.Volume.from_name("airr-ml-25-data-35m")
vol_data = modal.Volume.from_name("airr-ml-25-data")

volumes = {
    "/data_35m": vol_35m,
    "/data": vol_data
}

# Image with local code for pickle loading
image = (
    modal.Image.debian_slim()
    .pip_install("numpy", "pandas", "tabulate", "tqdm")
    .add_local_dir("data", remote_path="/root/data")
)

def check_file(args):
    fpath, rep_id = args
    try:
        # mmap_mode='r' reads only the header to get shape, very fast
        data = np.load(fpath, mmap_mode='r')
        return rep_id, data.shape[0], None
    except Exception as e:
        return rep_id, 0, str(e)

@app.function(image=image, volumes=volumes, timeout=3600)
def verify():
    import sys
    sys.path.append("/root")
    from data.load_data import Repertoire # Ensure class is available
    
    print("🚀 Starting Sanity Check (Parallelized)...")
    
    # Expected Data (from User)
    expected_data = [
        # Train
        {"dataset": "ds1", "split": "train", "reps": 400, "seqs": 10000000},
        {"dataset": "ds2", "split": "train", "reps": 400, "seqs": 10000000},
        {"dataset": "ds3", "split": "train", "reps": 400, "seqs": 10000000},
        {"dataset": "ds4", "split": "train", "reps": 400, "seqs": 10000000},
        {"dataset": "ds5", "split": "train", "reps": 400, "seqs": 10000000},
        {"dataset": "ds6", "split": "train", "reps": 400, "seqs": 10000000},
        {"dataset": "ds7", "split": "train", "reps": 302, "seqs": 93928788},
        {"dataset": "ds8", "split": "train", "reps": 908, "seqs": 93636523},
        # Test
        {"dataset": "ds1", "split": "test", "reps": 400, "seqs": 10000000},
        {"dataset": "ds2", "split": "test", "reps": 400, "seqs": 10000000},
        {"dataset": "ds3", "split": "test", "reps": 400, "seqs": 10000000},
        {"dataset": "ds4", "split": "test", "reps": 400, "seqs": 10000000},
        {"dataset": "ds5", "split": "test", "reps": 400, "seqs": 10000000},
        {"dataset": "ds6", "split": "test", "reps": 400, "seqs": 10000000},
        {"dataset": "ds7", "split": "1_test", "reps": 76, "seqs": 22877643},
        {"dataset": "ds7", "split": "2_test", "reps": 100, "seqs": 16827870},
        {"dataset": "ds8", "split": "1_test", "reps": 390, "seqs": 40366159},
        {"dataset": "ds8", "split": "2_test", "reps": 857, "seqs": 39721528},
        {"dataset": "ds8", "split": "3_test", "reps": 390, "seqs": 40366159},
    ]
    
    results = []
    
    base_dir = Path("/data_35m/embeddings_35m")
    
    for item in expected_data:
        ds = item["dataset"]
        split = item["split"]
        exp_reps = item["reps"]
        exp_seqs = item["seqs"]
        
        print(f"\n📂 Checking {ds} {split}...")
        
        # 1. Load Pickle to get IDs
        pkl_name = f"{ds}_{split}.pkl"
        pkl_path = Path(f"/data/processed/{pkl_name}")
        if not pkl_path.exists():
            pkl_path = Path(f"/data/data/{pkl_name}")
            
        if not pkl_path.exists():
            print(f"  ❌ Pickle missing: {pkl_name}")
            results.append({
                "Dataset": ds, "Split": split, 
                "Status": "Pickle Missing", 
                "Reps (Found/Exp)": f"0/{exp_reps}",
                "Seqs (Found/Exp)": f"0/{exp_seqs}"
            })
            continue
            
        try:
            with open(pkl_path, "rb") as f:
                reps = pickle.load(f)
        except Exception as e:
            print(f"  ❌ Pickle load error: {e}")
            continue
            
        target_ids = [r.rep_id for r in reps]
        
        # 2. Determine Folder Path
        folder_ds = ds
        folder_split = "test" if "test" in split else "train"
        search_dir = base_dir / folder_ds / folder_split
        
        # 3. Check Files
        print(f"  Listing files in {search_dir}...")
        if search_dir.exists():
            existing_files = set(p.name for p in search_dir.glob("*.npy"))
        else:
            existing_files = set()
            
        tasks = []
        missing_ids = []
        
        for rep_id in target_ids:
            fname = f"{rep_id}.npy"
            if fname in existing_files:
                tasks.append((search_dir / fname, rep_id))
            else:
                missing_ids.append(rep_id)
                
        found_reps = 0
        found_seqs = 0
        
        if tasks:
            print(f"  Verifying {len(tasks)} files (Parallel)...")
            # Use ThreadPoolExecutor for parallel I/O
            with ThreadPoolExecutor(max_workers=32) as executor:
                # tqdm for progress bar
                results_iter = list(tqdm(executor.map(check_file, tasks), total=len(tasks), unit="rep", ncols=80))
                
            for rep_id, seq_count, error in results_iter:
                if error:
                    print(f"    ❌ Error reading {rep_id}: {error}")
                else:
                    found_reps += 1
                    found_seqs += seq_count
        else:
            print("  ⚠️  No matching files found.")
        
        # Status
        if found_reps == exp_reps and found_seqs == exp_seqs:
            status = "✅ OK"
        elif found_reps == exp_reps:
            status = "⚠️ Seqs Mismatch"
        else:
            status = "❌ Reps Missing"
            
        print(f"  Result: {status} | Reps: {found_reps}/{exp_reps} | Seqs: {found_seqs}/{exp_seqs}")
        if missing_ids:
            print(f"  Missing IDs (first 5): {missing_ids[:5]}")
            
        results.append({
            "Dataset": ds, "Split": split, 
            "Status": status, 
            "Reps (Found/Exp)": f"{found_reps}/{exp_reps}",
            "Seqs (Found/Exp)": f"{found_seqs}/{exp_seqs}"
        })
        
    # Print Table
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("SANITY CHECK RESULTS")
    print("="*80)
    print(df.to_markdown(index=False))
    print("="*80)

@app.local_entrypoint()
def main():
    verify.remote()
