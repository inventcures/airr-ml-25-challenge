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
    
    datasets = [f"ds{i}" for i in range(1, 9)]
    splits = ["train", "test"]
    
    print(f"{'Dataset':<10} {'Split':<10} {'Pickle':<10} {'Embeddings':<15}")
    print("-" * 50)
    
    for ds in datasets:
        for split in ["train"]: # Focus on train for now
            # Check pickle
            pkl_name = f"{ds}_{split}.pkl"
            # Check multiple locations for pickle
            pkl_found = False
            for p in [base_dir / pkl_name, data_dir / pkl_name, base_dir / "processed" / pkl_name]:
                if p.exists():
                    pkl_found = True
                    break
            
            # Check embeddings
            ds_embed_dir = embed_dir / ds
            n_embeds = 0
            if ds_embed_dir.exists():
                n_embeds = len(list(ds_embed_dir.glob("*.npy")))
            
            print(f"{ds:<10} {split:<10} {'✅' if pkl_found else '❌':<10} {n_embeds:<15}")

@app.local_entrypoint()
def main():
    verify_content.remote()
