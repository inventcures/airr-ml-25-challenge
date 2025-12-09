import modal
from pathlib import Path
import pickle
import shutil
import time

app = modal.App("reorganize-volume")

# Mount both volumes
vol_35m = modal.Volume.from_name("airr-ml-25-data-35m")
vol_data = modal.Volume.from_name("airr-ml-25-data")

volumes = {
    "/data_35m": vol_35m,
    "/data": vol_data
}

# Define image with local code mount
image = (
    modal.Image.debian_slim()
    .pip_install("numpy", "pandas")
    .add_local_dir("data", remote_path="/root/data")
)

@app.function(image=image, volumes=volumes, timeout=3600)
def reorganize():
    import sys
    # Add root to path so pickle can find 'data.load_data'
    sys.path.append("/root")
    
    print("🚀 Starting robust reorganization...")
    start_time = time.time()
    
    # Define datasets and splits
    datasets = {
        "ds1": ["train", "test"],
        "ds2": ["train", "test"],
        "ds3": ["train", "test"],
        "ds4": ["train", "test"],
        "ds5": ["train", "test"],
        "ds6": ["train", "test"],
        "ds7": ["train", "1_test", "2_test"],
        "ds8": ["train", "1_test", "2_test", "3_test"]
    }
    
    base_dir = Path("/data_35m/embeddings_35m")
    summary = []
    
    for ds_name, splits in datasets.items():
        ds_dir = base_dir / ds_name
        if not ds_dir.exists():
            print(f"⚠️  Skipping {ds_name}: Directory not found at {ds_dir}")
            summary.append(f"{ds_name}: SKIPPED (Dir not found)")
            continue
            
        print(f"\n📂 Processing {ds_name}...")
        ds_stats = []
        
        for split in splits:
            # 1. Load the pickle to get the IDs for this split
            if "_" in split and "test" in split: # e.g. 1_test
                pkl_name = f"{ds_name}_{split}.pkl"
            else:
                pkl_name = f"{ds_name}_{split}.pkl"
                
            pkl_path = Path(f"/data/processed/{pkl_name}")
            if not pkl_path.exists():
                pkl_path = Path(f"/data/data/{pkl_name}")
            
            if not pkl_path.exists():
                print(f"  ❌ Error: Pickle not found for {ds_name} {split}")
                ds_stats.append(f"{split}: Pickle Missing")
                continue
                
            try:
                with open(pkl_path, "rb") as f:
                    reps = pickle.load(f)
            except Exception as e:
                print(f"  ❌ Error loading pickle {pkl_path}: {e}")
                ds_stats.append(f"{split}: Pickle Load Error")
                continue
            
            target_ids = set(r.rep_id for r in reps)
            print(f"  ℹ️  {split}: Found {len(target_ids)} target IDs in pickle")
            
            # 2. Create the split directory
            split_dir = ds_dir / split
            split_dir.mkdir(exist_ok=True)
            
            # 3. Move matching files
            moved_count = 0
            already_in_place = 0
            missing_files = 0
            
            for rep_id in target_ids:
                # Check if already in destination
                dst_file = split_dir / f"{rep_id}.npy"
                if dst_file.exists():
                    already_in_place += 1
                    continue

                # Check source (root of ds folder)
                src_file = ds_dir / f"{rep_id}.npy"
                
                if src_file.exists():
                    try:
                        shutil.move(str(src_file), str(dst_file))
                        moved_count += 1
                    except Exception as e:
                        print(f"    ❌ Failed to move {src_file.name}: {e}")
                else:
                    missing_files += 1
            
            print(f"  ✅ {split}: Moved {moved_count}, Already Correct {already_in_place}, Missing {missing_files}")
            ds_stats.append(f"{split}: +{moved_count} (Total: {moved_count + already_in_place})")
            
        summary.append(f"{ds_name}: {', '.join(ds_stats)}")
            
    vol_35m.commit()
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*50)
    print(f"🏁 Reorganization Complete in {duration:.2f}s")
    print("="*50)
    for line in summary:
        print(line)
    print("="*50)

@app.local_entrypoint()
def main():
    reorganize.remote()
