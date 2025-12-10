import subprocess
import sys
import time
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

PICKLES = [
    "ds1_train.pkl", "ds1_test.pkl",
    "ds2_train.pkl", "ds2_test.pkl",
    "ds3_train.pkl", "ds3_test.pkl",
    "ds4_train.pkl", "ds4_test.pkl",
    "ds5_train.pkl", "ds5_test.pkl",
    "ds6_train.pkl", "ds6_test.pkl",
    "ds7_train.pkl", "ds7_1_test.pkl", "ds7_2_test.pkl",
    "ds8_train.pkl", "ds8_1_test.pkl", "ds8_2_test.pkl", "ds8_3_test.pkl"
]

PICKLE_REMOTE_PATHS = ["data/processed", "data", "processed"]
PICKLE_LOCAL_DIR = Path("data/processed")

EMBEDDINGS_LOCAL_DIR = Path("data/embeddings")
EMBEDDINGS_EXPECTED = [
    {"dataset": "ds1", "split": "train", "reps": 400},
    {"dataset": "ds2", "split": "train", "reps": 400},
    {"dataset": "ds3", "split": "train", "reps": 400},
    {"dataset": "ds4", "split": "train", "reps": 400},
    {"dataset": "ds5", "split": "train", "reps": 400},
    {"dataset": "ds6", "split": "train", "reps": 400},
    {"dataset": "ds7", "split": "train", "reps": 302},
    {"dataset": "ds8", "split": "train", "reps": 908},
    {"dataset": "ds1", "split": "test", "reps": 400},
    {"dataset": "ds2", "split": "test", "reps": 400},
    {"dataset": "ds3", "split": "test", "reps": 400},
    {"dataset": "ds4", "split": "test", "reps": 400},
    {"dataset": "ds5", "split": "test", "reps": 400},
    {"dataset": "ds6", "split": "test", "reps": 400},
    {"dataset": "ds7", "split": "1_test", "reps": 76},
    {"dataset": "ds7", "split": "2_test", "reps": 100},
    {"dataset": "ds8", "split": "1_test", "reps": 390},
    {"dataset": "ds8", "split": "2_test", "reps": 857},
    {"dataset": "ds8", "split": "3_test", "reps": 390},
]

# =============================================================================
# Helper Functions
# =============================================================================

def run_modal_command(args, retries=3):
    """Run a modal CLI command with retries."""
    cmd = ["uv", "run", "modal"] + args
    for attempt in range(retries):
        try:
            # Check=True raises CalledProcessError on non-zero exit code
            # Capture output so we can control what the user sees
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return False
    return False

# =============================================================================
# Part 1: Download Pickles
# =============================================================================

def download_pickles():
    print("\n🥒 [1/3] Checking Processed Data (Pickles)...")
    PICKLE_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # We use tqdm to show progress through the list of pickle files
    for pkl_name in tqdm(PICKLES, desc="Pickles", unit="file"):
        local_path = PICKLE_LOCAL_DIR / pkl_name
        if local_path.exists():
            continue  # Resume: Skip existing

        success = False
        # Try multiple possible remote paths (folder structure resilience)
        for remote_base in PICKLE_REMOTE_PATHS:
            remote_path = f"{remote_base}/{pkl_name}"
            # modal volume get airr-ml-25-data <remote> <local>
            if run_modal_command(["volume", "get", "--force", "airr-ml-25-data", remote_path, str(local_path)]):
                success = True
                break
        
        if not success:
            tqdm.write(f"❌ Failed to download {pkl_name}")

# =============================================================================
# Part 2: Download Embeddings
# =============================================================================

def count_local_files(ds, split):
    target_dir = EMBEDDINGS_LOCAL_DIR / ds / split
    if not target_dir.exists():
        return 0
    return len(list(target_dir.glob("*.npy")))

def download_embeddings():
    print("\n🧠 [2/3] Checking Embeddings (Smart Sync)...")
    
    # Iterate through all dataset splits
    for item in tqdm(EMBEDDINGS_EXPECTED, desc="Datasets"):
        ds = item["dataset"]
        split = item["split"]
        expected = item["reps"]
        
        current = count_local_files(ds, split)
        
        if current >= expected:
            continue # Resume: Fully complete
            
        # Determine Strategy
        missing_count = expected - current
        remote_dir = f"embeddings_35m/{ds}/{split}"
        local_dir = EMBEDDINGS_LOCAL_DIR / ds / split
        
        # If > 50% missing or empty, download folder (Faster batch)
        # If < 50% missing, repair individually
        if current == 0 or missing_count > (expected * 0.5):
            tqdm.write(f"  ⬇️  Downloading folder {ds}/{split} (bulk)...")
            
            # Ensure clean start for bulk download
            if local_dir.is_file():
                local_dir.unlink()
            local_dir.mkdir(parents=True, exist_ok=True)
            
            success = run_modal_command(
                ["volume", "get", "--force", "airr-ml-25-data-35m", remote_dir, str(local_dir)], 
                retries=5
            )
            if not success:
                tqdm.write(f"  ❌ Failed to download folder {ds}/{split}")
        else:
            tqdm.write(f"  🔧 Repairing {ds}/{split} ({missing_count} missing)...")
            # Identify missing indices
            missing_indices = []
            for i in range(expected):
                if not (local_dir / f"{i}.npy").exists():
                    missing_indices.append(i)
            
            # Download missing files individually with sub-progress bar
            for idx in tqdm(missing_indices, desc=f"  Repairing {ds}", leave=False, unit="file"):
                remote_file = f"{remote_dir}/{idx}.npy"
                local_file = f"{local_dir}/{idx}.npy"
                
                if not run_modal_command(["volume", "get", "--force", "airr-ml-25-data-35m", remote_file, local_file]):
                    tqdm.write(f"  ❌ Failed to download {idx}.npy")

# =============================================================================
# Part 3: Enforce Structure
# =============================================================================

def enforce_structure():
    print("\n🛡️  [3/3] Enforcing Clean Structure...")
    if not EMBEDDINGS_LOCAL_DIR.exists():
        return

    # Scan for loose .npy files in data/embeddings/dsX/ (not in split subfolders)
    # The structure must be data/embeddings/dsX/train/*.npy
    # If we find data/embeddings/dsX/*.npy, that is wrong.
    
    count_deleted = 0
    for ds_dir in EMBEDDINGS_LOCAL_DIR.iterdir():
        if not ds_dir.is_dir() or not ds_dir.name.startswith("ds"):
            continue
            
        loose_files = list(ds_dir.glob("*.npy"))
        if loose_files:
            for f in loose_files:
                f.unlink()
            count_deleted += len(loose_files)
            
    if count_deleted > 0:
        print(f"  ✅ Cleaned up {count_deleted} loose files.")
    else:
        print("  ✅ Structure is clean.")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Ensure uv is installed/available or just assume environment is set
    try:
        download_pickles()
        download_embeddings()
        enforce_structure()
        print("\n✨ Sync Complete! All systems go.")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted.")
        sys.exit(1)
