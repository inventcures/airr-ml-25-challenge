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

def download_file(ds, split, index):
    remote_path = f"embeddings_35m/{ds}/{split}/{index}.npy"
    local_path = f"data/embeddings/{ds}/{split}/{index}.npy"
    
    cmd = [
        "uv", "run", "modal", "volume", "get", 
        "--force",
        "airr-ml-25-data-35m", 
        remote_path, 
        local_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False

def download_split(ds, split, current_count, expected_count):
    remote_dir_path = f"embeddings_35m/{ds}/{split}"
    local_dir_path = f"data/embeddings/{ds}/{split}"
    
    # Strategy:
    # 1. If 0 files, download entire directory (Fastest)
    # 2. If > 50% missing, download entire directory (Likely faster than 100s of individual calls)
    # 3. If < 50% missing, download individually (Saves bandwidth/time on existing files)
    
    missing_indices = []
    if current_count > 0:
        for i in range(expected_count):
            if not (Path(local_dir_path) / f"{i}.npy").exists():
                missing_indices.append(i)
    
    is_partial_repair = current_count > 0 and len(missing_indices) < (expected_count * 0.5)
    
    if is_partial_repair:
        print(f"🔧 Repairing {ds}/{split}: Downloading {len(missing_indices)} missing files...", flush=True)
        success_count = 0
        for idx in missing_indices:
            print(f"  ⬇️  Downloading {idx}.npy ({success_count+1}/{len(missing_indices)})...", end="\r", flush=True)
            if download_file(ds, split, idx):
                success_count += 1
            else:
                print(f"\n  ❌ Failed to download {idx}.npy", flush=True)
        
        if success_count == len(missing_indices):
            print(f"\n  ✅ Repair complete.", flush=True)
            return True
        else:
            print(f"\n  ⚠️  Repair incomplete. {len(missing_indices) - success_count} files failed.", flush=True)
            return False
            
    else:
        # Full Directory Download
        print(f"⬇️  Downloading folder {ds}/{split}...", flush=True)
        cmd = [
            "uv", "run", "modal", "volume", "get", 
            "--force",
            "airr-ml-25-data-35m", 
            remote_dir_path, 
            local_dir_path
        ]
        
        # Retry loop
        max_retries = 5
        for i in range(max_retries):
            try:
                # Capture output to silence the generic "✓ Finished" messages
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"  ✅ Downloaded successfully.", flush=True)
                return True
            except subprocess.CalledProcessError as e:
                print(f"  ⚠️  Attempt {i+1} failed. Error: {e.stderr.decode().strip()}", flush=True)
                print(f"  Retrying in 5 seconds...", flush=True)
                time.sleep(5)
                
        print(f"❌ Failed to download {ds}/{split} after {max_retries} attempts.", flush=True)
        return False

def main():
    print("🚀 Starting Smart Download...", flush=True)
    
    for item in EXPECTED_COUNTS:
        ds = item["dataset"]
        split = item["split"]
        expected = item["reps"]
        
        current = count_local_files(ds, split)
        timestamp = time.strftime("%H:%M:%S")
        
        if current >= expected:
            print(f"[{timestamp}] ✅ {ds}/{split}: Complete ({current}/{expected})", flush=True)
        else:
            print(f"[{timestamp}] ⏳ {ds}/{split}: Incomplete ({current}/{expected}) - Starting Download...", flush=True)
            download_split(ds, split, current, expected)
            
    print("\n🎉 All downloads checked!", flush=True)

if __name__ == "__main__":
    main()
