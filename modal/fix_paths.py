import modal
from pathlib import Path
import shutil
import time

app = modal.App("fix-paths")

vol_35m = modal.Volume.from_name("airr-ml-25-data-35m")
volumes = {"/data_35m": vol_35m}

@app.function(volumes=volumes, timeout=3600)
def fix_paths():
    print("🚀 Starting path fix...")
    base_dir = Path("/data_35m/embeddings_35m")
    
    # List of datasets that might have files in the wrong place
    # These are the ones we run with --dataset-name "ds7_1" etc.
    targets = ["ds7_1", "ds7_2", "ds8_1", "ds8_2", "ds8_3"]
    
    for ds_name in targets:
        ds_dir = base_dir / ds_name
        if not ds_dir.exists():
            print(f"Skipping {ds_name} (not found)")
            continue
            
        print(f"Checking {ds_name}...")
        
        # We want to move files from ds_dir/*.npy to ds_dir/test/*.npy
        test_dir = ds_dir / "test"
        test_dir.mkdir(exist_ok=True)
        
        moved = 0
        for f in ds_dir.glob("*.npy"):
            if f.is_file():
                dst = test_dir / f.name
                if not dst.exists():
                    shutil.move(str(f), str(dst))
                    moved += 1
                else:
                    # If it already exists in test, just delete the duplicate in root?
                    # Or keep it safe. Let's just leave it if dst exists.
                    pass
                    
        print(f"  Moved {moved} files to {ds_name}/test")
        
    vol_35m.commit()
    print("✅ Fix complete!")

@app.local_entrypoint()
def main():
    fix_paths.remote()
