import os
from pathlib import Path

def diagnose():
    target = Path("data/embeddings/ds1")
    print(f"🕵️‍♀️ Diagnosing {target}...")
    
    if not target.exists():
        print("❌ Target directory does not exist.")
        return

    for root, dirs, files in os.walk(target):
        level = root.replace(str(target), "").count(os.sep)
        indent = " " * 4 * (level)
        print(f"{indent}📂 {os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        
        # Print first few files
        for f in files[:3]:
            print(f"{subindent}📄 {f}")
        if len(files) > 3:
            print(f"{subindent}... ({len(files)} files total)")

if __name__ == "__main__":
    diagnose()
