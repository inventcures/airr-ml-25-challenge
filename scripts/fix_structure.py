import shutil
from pathlib import Path
import os

BASE_DIR = Path("data/embeddings")

def fix_structure():
    print("🔧 Starting Directory Fixer...")
    
    # Find all .npy files
    # We want to move them to data/embeddings/{ds}/{split}/filename.npy
    
    files = list(BASE_DIR.rglob("*.npy"))
    print(f"Found {len(files)} .npy files.")
    
    moved_count = 0
    
    for file_path in files:
        # file_path might be: data/embeddings/embeddings_35m/ds1/train/0.npy
        # or: data/embeddings/ds1/train/train/0.npy
        
        parts = file_path.parts
        
        # We look for "dsX" in the path to anchor ourselves
        ds = None
        split = None
        
        for part in parts:
            if part.startswith("ds") and part[2:].isdigit():
                ds = part
            if part in ["train", "test", "1_test", "2_test", "3_test"]:
                split = part
                
        if ds and split:
            # Construct correct path
            target_dir = BASE_DIR / ds / split
            target_file = target_dir / file_path.name
            
            # If it's already in the right place, skip
            if file_path.parent == target_dir:
                continue
                
            # Move it
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
            except FileExistsError:
                # This happens if a parent component is a file
                # Check if target_dir itself is a file
                if target_dir.is_file():
                    print(f"⚠️  Conflict: {target_dir} is a file but should be a directory.")
                    backup_name = target_dir.with_name(target_dir.name + "_CONFLICT_BACKUP")
                    print(f"   Renaming to {backup_name}")
                    target_dir.rename(backup_name)
                    target_dir.mkdir(parents=True, exist_ok=True)
                else:
                    # Check parents
                    for parent in target_dir.parents:
                        if parent == BASE_DIR:
                            break
                        if parent.is_file():
                            print(f"⚠️  Conflict: Parent {parent} is a file.")
                            backup_name = parent.with_name(parent.name + "_CONFLICT_BACKUP")
                            print(f"   Renaming to {backup_name}")
                            parent.rename(backup_name)
                            # Retry mkdir
                            target_dir.mkdir(parents=True, exist_ok=True)
                            break
            
            shutil.move(str(file_path), str(target_file))
            moved_count += 1
            
    print(f"✅ Moved {moved_count} files to correct locations.")
    
    # Cleanup empty directories
    for root, dirs, files in os.walk(BASE_DIR, topdown=False):
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except OSError:
                pass # Directory not empty
                
    print("🧹 Cleaned up empty directories.")

if __name__ == "__main__":
    fix_structure()
