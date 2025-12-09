import modal
from pathlib import Path
import shutil
import time

app = modal.App("consolidate-test-sets")

vol_35m = modal.Volume.from_name("airr-ml-25-data-35m")
volumes = {"/data_35m": vol_35m}

@app.function(volumes=volumes, timeout=3600)
def consolidate():
    print("🚀 Starting consolidation...")
    base_dir = Path("/data_35m/embeddings_35m")
    
    # Mapping: Source Folder -> Destination Folder
    # We want to move contents of ds7_1/test -> ds7/test
    moves = {
        "ds7": ["ds7_1", "ds7_2"],
        "ds8": ["ds8_1", "ds8_2", "ds8_3"]
    }
    
    for dest_ds, sources in moves.items():
        dest_dir = base_dir / dest_ds / "test"
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📂 Consolidating into {dest_dir}...")
        
        total_moved = 0
        
        for src_ds in sources:
            src_dir = base_dir / src_ds / "test"
            if not src_dir.exists():
                print(f"  ⚠️  Source not found: {src_dir}")
                continue
                
            print(f"  Processing {src_ds}...")
            files = list(src_dir.glob("*.npy"))
            print(f"    Found {len(files)} files")
            
            for f in files:
                dest_file = dest_dir / f.name
                if not dest_file.exists():
                    shutil.move(str(f), str(dest_file))
                    total_moved += 1
                else:
                    # Duplicate?
                    pass
            
            # Cleanup empty source dir
            try:
                src_dir.rmdir() # Only works if empty
                (base_dir / src_ds).rmdir() # Try to remove parent if empty
                print(f"    Cleaned up {src_ds}")
            except:
                pass
                
        print(f"✅ Moved {total_moved} files to {dest_ds}/test")
        
    vol_35m.commit()
    print("\n🏁 Consolidation Complete!")

@app.local_entrypoint()
def main():
    consolidate.remote()
