import subprocess
import time
from pathlib import Path
import sys

# Expected counts (from verify_counts.py)
EXPECTED_COUNTS = [
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

BASE_LOCAL_DIR = Path("data/embeddings")

def count_local_files(ds, split):
    # Handle folder structure mapping
    # ds1/train -> data/embeddings/ds1/train
    # ds7/1_test -> data/embeddings/ds7/1_test
    
    # Logic from verify_counts.py:
    # folder_ds = ds
    # folder_split = "test" if "test" in split else "train"
    # BUT verify_counts logic was for the OLD structure or generalized.
    # Let's check embed_cloud.py output path:
    # out_dir = Path(f"/data_35m/embeddings_35m/{dataset_name}/{split}")
    # So it is EXACTLY {ds}/{split}
    
    target_dir = BASE_LOCAL_DIR / ds / split
    if not target_dir.exists():
        return 0
    return len(list(target_dir.glob("*.npy")))

def download_split(ds, split):
    remote_path = f"embeddings_35m/{ds}/{split}"
    local_path = f"data/embeddings/{ds}/{split}"
    
    print(f"⬇️  Downloading {ds}/{split}...")
    cmd = [
        "uv", "run", "modal", "volume", "get", 
        "--force",
        "airr-ml-25-data-35m", 
        remote_path, 
        local_path
    ]
    
    # Retry loop
    max_retries = 5
    for i in range(max_retries):
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError:
            print(f"⚠️  Download failed for {ds}/{split}. Retrying ({i+1}/{max_retries})...")
            time.sleep(5)
            
    print(f"❌ Failed to download {ds}/{split} after {max_retries} attempts.")
    return False

def main():
    print("🚀 Starting Smart Download...")
    
    for item in EXPECTED_COUNTS:
        ds = item["dataset"]
        split = item["split"]
        expected = item["reps"]
        
        current = count_local_files(ds, split)
        
        if current >= expected:
            print(f"✅ {ds}/{split}: Complete ({current}/{expected})")
        else:
            print(f"⏳ {ds}/{split}: Incomplete ({current}/{expected})")
            download_split(ds, split)
            
    print("\n🎉 All downloads checked!")

if __name__ == "__main__":
    main()
