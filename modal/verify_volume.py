import modal
from pathlib import Path

app = modal.App("verify-volume")
vol = modal.Volume.from_name("airr-ml-25-data")

@app.function(volumes={"/data": vol})
def verify_content():
    print("Verifying volume content...")
    
    base_dir = Path("/data")
    data_dir = base_dir / "data"
    embed_dir = base_dir / "embeddings"
    
    print(f"Searching for datasets in {base_dir}...")
    for path in base_dir.rglob("*"):
        if "train_datasets" in str(path) or "test_datasets" in str(path) or path.suffix == ".csv":
            print(f"FOUND: {path}")

@app.local_entrypoint()
def main():
    verify_content.remote()
