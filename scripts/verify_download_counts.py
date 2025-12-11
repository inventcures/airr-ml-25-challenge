
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import sys

# Expected Data Configuration
EXPECTED_DATA = [
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

# Paths - Assuming standard project structure
BASE_DIR = Path(".") 
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Add project root to path for data.load_data import if needed
sys.path.append(str(BASE_DIR))

# Mock Repertoire class if import fails (so we don't crash if environment is minimal)
try:
    from data.load_data import Repertoire
except ImportError:
    class Repertoire:
        def __init__(self, rep_id):
            self.rep_id = rep_id

def check_file_local(args):
    """
    Worker function to check a single .npy file.
    Returns: (rep_id, seq_count, error_message)
    """
    fpath, rep_id = args
    try:
        if not fpath.exists():
            return rep_id, 0, "File not found"
        # mmap_mode='r' reads only the header to get shape, very fast
        data = np.load(fpath, mmap_mode='r')
        return rep_id, data.shape[0], None
    except Exception as e:
        return rep_id, 0, str(e)

def verify_local():
    print("🚀 Starting Local Sanity Check of Embeddings...")
    print(f"   Embeddings Root: {EMBEDDINGS_DIR.resolve()}")
    
    results = []
    
    for item in EXPECTED_DATA:
        ds = item["dataset"]
        split = item["split"]
        exp_reps = item["reps"]
        exp_seqs = item["seqs"]
        
        print(f"\n📂 Checking {ds} {split}...")
        
        # 1. Load Pickle to get Expected IDs
        #    Logic: Try loading from processed/ first.
        pkl_name = f"{ds}_{split}.pkl"
        
        # Determine strict pickle path
        # Note: In load_all_datasets.py, ds7_1_test -> ds7_1_test.pkl (usually)
        # But 'split' here is "1_test", so the name is "ds7_1_test.pkl"?
        # Let's handle the underscore naming convention carefully.
        
        # logic from expected_data keys to filename:
        # ds1, test -> ds1_test.pkl
        # ds7, 1_test -> ds7_1_test.pkl
        
        if "_" in split and split[0].isdigit():
             formatted_pkl_name = f"{ds}_{split}.pkl"
        else:
             formatted_pkl_name = f"{ds}_{split}.pkl"
             
        pkl_path = PROCESSED_DIR / formatted_pkl_name
        
        if not pkl_path.exists():
            print(f"  ⚠️  Pickle not found at {pkl_path}, counting files only.")
            # Fallback: Just look at the directory count if pickle is missing
            target_ids = []
            use_pickle = False
        else:
            try:
                with open(pkl_path, "rb") as f:
                    reps = pickle.load(f)
                target_ids = [r.rep_id for r in reps]
                use_pickle = True
            except Exception as e:
                print(f"  ❌ Pickle load error: {e}")
                target_ids = []
                use_pickle = False

        # 2. Determine Local Folder Path
        # Structure: data/embeddings/ds1/train or data/embeddings/ds7/test/1_test
        # Wait, the structure on disk from sync_data_robust.py download is:
        # ds1/train, ds1/test.
        # ds7/train.
        # ds7/test/1_test ? Or just ds7/1_test?
        
        # Re-reading sync_data_robust.py behavior:
        # For ds7_1_test, it tries to download to ...
        # Actually sync_data_robust iterates ["ds7", "1_test"]
        # It downloads to EMBEDDINGS_LOCAL_DIR / ds / split
        # So for ds7, 1_test -> data/embeddings/ds7/1_test
        # Verify this against user's tree command output? 
        # User output showed ds7/test exists, but user said "spl test folders for ds7 & ds8 not downloaded".
        # Then manual modal commands were run targeting data/embeddings/ds7/test.
        # So likely: data/embeddings/ds7/test/1_test
        
        folder_ds = ds
        
        # Standardize path based on typical structure
        if "test" in split and split != "test":
            # e.g. 1_test
            # It might be in ds7/test/1_test OR ds7/1_test.
            # Let's probe.
            path_a = EMBEDDINGS_DIR / ds / "test" / split
            path_b = EMBEDDINGS_DIR / ds / split
            
            if path_a.exists():
                search_dir = path_a
            elif path_b.exists():
                search_dir = path_b
            else:
                # Default to path_a for error reporting
                search_dir = path_a
        else:
            search_dir = EMBEDDINGS_DIR / ds / split

        # 3. List Local Files
        # print(f"  Looking in {search_dir}...")
        if search_dir.exists():
            existing_files = set(p.name for p in search_dir.glob("*.npy"))
        else:
            existing_files = set()
            print(f"  ❌ Directory not found: {search_dir}")

        # 4. Prepare Tasks
        tasks = []
        missing_ids = []
        
        if use_pickle:
            # Check for specific IDs
            for rep_id in target_ids:
                fname = f"{rep_id}.npy"
                if fname in existing_files:
                    tasks.append((search_dir / fname, rep_id))
                else:
                    missing_ids.append(rep_id)
        else:
            # Check all found files
            for fname in existing_files:
                rep_id = fname.replace(".npy", "")
                tasks.append((search_dir / fname, rep_id))
        
        # 5. Run Checks
        found_reps = 0
        found_seqs = 0
        
        if tasks:
            print(f"  Verifying {len(tasks)} files...")
            with ThreadPoolExecutor(max_workers=32) as executor:
                # Using tqdm
                results_iter = list(tqdm(executor.map(check_file_local, tasks), total=len(tasks), unit="rep", ncols=80, leave=False))
                
            for rep_id, seq_count, error in results_iter:
                if error:
                    print(f"    ❌ Error reading {rep_id}: {error}")
                else:
                    found_reps += 1
                    found_seqs += seq_count
        else:
            print("  ⚠️  No files to check.")

        # 6. Validate
        if use_pickle:
             expected_count = len(target_ids) # Should match exp_reps usually
        else:
             expected_count = exp_reps

        # Check Rep Count
        if found_reps == expected_count:
            status = "✅ OK"
        elif found_reps > 0:
            status = f"⚠️ Partial {found_reps}/{expected_count}"
        else:
            status = "❌ Missing"
            
        # Check Seq Count (Tolerance? Exact match expected generally)
        seq_status = "✅" if found_seqs == exp_seqs else f"❌ {found_seqs} vs {exp_seqs}"
        
        print(f"  Result: {status} | Reps: {found_reps}/{expected_count} | Seqs: {found_seqs}/{exp_seqs}")
        
        results.append({
            "Dataset": ds, 
            "Split": split, 
            "Status": status, 
            "SeqStatus": "OK" if found_seqs == exp_seqs else "Mismatch",
            "Found Reps": found_reps, 
            "Exp Reps": expected_count,
            "Found Seqs": found_seqs,
            "Exp Seqs": exp_seqs
        })
        
    # Summary Table
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("LOCAL SANITY CHECK RESULTS")
    print("="*80)
    # Simple formatting if tabulate isn't installed
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    verify_local()
