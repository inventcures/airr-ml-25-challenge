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
        except subprocess.CalledProcessError as e:
            error_output = e.stderr.decode().strip()
            if "AuthenticationError" in error_output or "Missing credentials" in error_output:
                print("\n❌ Modal Authentication Error: Please run 'modal token set ...' first.", file=sys.stderr)
                sys.exit(1)
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return False
    return False

# =============================================================================
# Authentication Check
# =============================================================================
def check_modal_authentication():
    print("🔑 Checking Modal authentication...")
    # Attempt a simple modal command that requires authentication
    # to confirm credentials are set. Using 'modal volume list' as it's safe.
    try:
        subprocess.run(["uv", "run", "modal", "volume", "list"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("  ✅ Authenticated with Modal.")
    except subprocess.CalledProcessError:
        print("\n❌ Modal Authentication Required!")
        print("Please run the following command with your credentials:")
        print("  modal token set --token-id <YOUR_TOKEN_ID> --token-secret <YOUR_TOKEN_SECRET>")
        print("You can find these at: https://modal.com/settings/tokens")
        sys.exit(1)

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
            
            # Fix: Download to PARENT directory because modal volume get creates the leaf directory
            # if we target '.../train', it creates '.../train/train' if '.../train' exists.
            # Targeting '.../ds1' (parent) will create '.../ds1/train' correctly.
            
            # However, we must be careful. if we download 'embeddings_35m/ds1/train' to 'data/embeddings/ds1', 
            # it will create 'data/embeddings/ds1/train'.
            
            success = run_modal_command(
                ["volume", "get", "--force", "airr-ml-25-data-35m", remote_dir, str(local_dir.parent)], 
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
# Part 3: Enforce Structure & Fix Nesting
# =============================================================================

def fix_nested_structure():
    """
    Detects and fixes 'Russian doll' nested directories like:
    data/embeddings/ds1/train/train/*.npy
    Moves them to:
    data/embeddings/ds1/train/*.npy
    """
    print("\n🧹 [3/4] Checking for nested 'Russian Doll' directories...")
    if not EMBEDDINGS_LOCAL_DIR.exists():
        return

    # iterate ds1, ds2...
    for ds_dir in EMBEDDINGS_LOCAL_DIR.iterdir():
        if not ds_dir.is_dir(): continue
        
        # iterate train, test...
        for split_dir in ds_dir.iterdir():
            if not split_dir.is_dir(): continue
            
            # Check for nested same-name dir: e.g. ds1/train/train
            nested_same = split_dir / split_dir.name
            
            if nested_same.exists() and nested_same.is_dir():
                print(f"  ⚠️  Found nested dir: {nested_same}")
                
                # Move files up
                moved_count = 0
                for f in nested_same.iterdir():
                    if f.is_file():
                        dest = split_dir / f.name
                        if not dest.exists():
                            f.rename(dest)
                            moved_count += 1
                        else:
                            # If dest exists (maybe partial dup), just delete source? 
                            # Or overwrite? Let's overwrite to be safe we have latest? 
                            # Actually if they are identical, doesn't matter.
                            f.replace(dest) 
                            moved_count += 1
                
                # Attempt to remove the empty nested dir
                try:
                    nested_same.rmdir()
                    print(f"  ✅ Fixed {nested_same} (Moved {moved_count} files)")
                except OSError:
                    print(f"  ❌ Could not remove {nested_same} (not empty?)")

def enforce_structure():
    print("\n🛡️  [4/4] Enforcing Clean Structure (Loose files)...")
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
    check_modal_authentication() # Verify authentication before starting downloads
    try:
        # download_pickles()
        # download_embeddings()
        fix_nested_structure()
        # enforce_structure()
        print("\n✨ Fix Complete! Nested directories repaired.")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted.")
        sys.exit(1)
