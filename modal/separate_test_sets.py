import modal
from pathlib import Path
import pickle
import shutil
import sys

app = modal.App("separate-test-sets")

# Mount volumes
vol_35m = modal.Volume.from_name("airr-ml-25-data-35m")
vol_data = modal.Volume.from_name("airr-ml-25-data")

volumes = {
    "/data_35m": vol_35m,
    "/data": vol_data
}

image = (
    modal.Image.debian_slim()
    .pip_install("numpy", "pandas")
    .add_local_dir("data", remote_path="/root/data")
)

@app.function(image=image, volumes=volumes, timeout=3600)
def separate():
    import sys
    sys.path.append("/root")
    from data.load_data import Repertoire
    
    print("🚀 Starting Separation of Test Sets...")
    
    # Define what to separate
    tasks = [
        {"ds": "ds7", "splits": ["1_test", "2_test"]},
        {"ds": "ds8", "splits": ["1_test", "2_test", "3_test"]}
    ]
    
    base_dir = Path("/data_35m/embeddings_35m")
    
    for task in tasks:
        ds_name = task["ds"]
        splits = task["splits"]
        
        # The current mixed folder
        mixed_dir = base_dir / ds_name / "test"
        if not mixed_dir.exists():
            print(f"⚠️  {mixed_dir} does not exist. Skipping {ds_name}.")
            continue
            
        print(f"\n📂 Processing {ds_name} (Splitting {mixed_dir})...")
        
        for split in splits:
            # 1. Load Pickle to get IDs
            pkl_name = f"{ds_name}_{split}.pkl"
            pkl_path = Path(f"/data/processed/{pkl_name}")
            if not pkl_path.exists():
                pkl_path = Path(f"/data/data/{pkl_name}")
                
            if not pkl_path.exists():
                print(f"  ❌ Pickle missing: {pkl_name}")
                continue
                
            print(f"  Loading {pkl_name}...")
            with open(pkl_path, "rb") as f:
                reps = pickle.load(f)
            target_ids = set(r.rep_id for r in reps)
            
            # 2. Create Subfolder
            # e.g. ds7/test/1_test
            target_dir = mixed_dir / split
            target_dir.mkdir(exist_ok=True)
            
            # 3. Move Files
            moved = 0
            already_there = 0
            missing = 0
            
            for rep_id in target_ids:
                fname = f"{rep_id}.npy"
                
                # It might be in mixed_dir OR already in target_dir
                src = mixed_dir / fname
                dst = target_dir / fname
                
                if dst.exists():
                    already_there += 1
                elif src.exists():
                    shutil.move(str(src), str(dst))
                    moved += 1
                else:
                    # Check if it's in ANY other subfolder (maybe we ran this partially?)
                    # But for now just count as missing from root
                    missing += 1
                    
            print(f"  ✅ {split}: Moved {moved} | Already in subfolder {already_there} | Missing {missing}")
            
    vol_35m.commit()
    print("\n🏁 Separation Complete!")

@app.local_entrypoint()
def main():
    separate.remote()
