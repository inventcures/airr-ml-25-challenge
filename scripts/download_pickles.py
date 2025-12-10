import subprocess
from pathlib import Path
import sys

# Define pickles to download
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

REMOTE_PATHS_TO_TRY = [
    "data/processed",  # Most likely
    "data",            # Possible
    "processed"        # Possible
]

LOCAL_DIR = Path("data/processed")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

def download_pickle(filename):
    local_path = LOCAL_DIR / filename
    if local_path.exists():
        print(f"✅ {filename} already exists.")
        return True
        
    print(f"⬇️  Downloading {filename}...")
    
    for remote_base in REMOTE_PATHS_TO_TRY:
        remote_path = f"{remote_base}/{filename}"
        cmd = [
            "uv", "run", "modal", "volume", "get", 
            "--force",
            "airr-ml-25-data", 
            remote_path, 
            str(local_path)
        ]
        
        try:
            # Capture output to avoid clutter
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"  ✅ Found in {remote_base}")
            return True
        except subprocess.CalledProcessError:
            continue
            
    print(f"  ❌ Failed to find {filename} in any known location.")
    return False

def main():
    print("🥒 Starting Pickle Download...")
    success_count = 0
    for pkl in PICKLES:
        if download_pickle(pkl):
            success_count += 1
            
    print(f"\nSummary: {success_count}/{len(PICKLES)} pickles downloaded.")
    if success_count < len(PICKLES):
        sys.exit(1)

if __name__ == "__main__":
    main()
