import modal
from pathlib import Path
import pickle
import numpy as np
from typing import List, Dict, Any

def download_model():
    import esm
    # This downloads the model to the image cache
    esm.pretrained.esm2_t12_35M_UR50D()

# Define the image
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "fair-esm", "numpy", "pandas", "tqdm")
    .run_function(download_model)
    .add_local_dir(
        "data", 
        remote_path="/root/data",
        ignore=["embeddings", "processed", "__pycache__", "*.pkl", "*.npy"]
    ) # Mount local code directly to image, excluding data
)

app = modal.App("airr-ml-25-esm-cloud")

# Volume configuration
# We use a separate volume for 35M embeddings to avoid mixing with 650M data
vol = modal.Volume.from_name("airr-ml-25-data-35m")
# We also need read access to the original data volume
data_vol = modal.Volume.from_name("airr-ml-25-data")

volumes = {
    "/data_35m": vol,
    "/data": data_vol
}

@app.function(
    image=image,
    gpu="A10G", # Cost-effective
    timeout=86400, # 24 hours
    max_containers=20, # High throughput
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0),
    volumes=volumes
)
def embed_sequences(sequences: List[str], batch_size: int = 64) -> List[np.ndarray]:
    import torch
    import esm
    from tqdm import tqdm

    # Load ESM-2 model
    # Load ESM-2 35M model
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(sequences), batch_size), desc="Batches"):
        batch_seqs = sequences[i:i + batch_size]
        # Sanitize sequences: replace '*' (stop codon) with 'X' (unknown)
        # ESM tokenizer crashes on '*'
        batch_seqs = [s.replace('*', 'X') for s in batch_seqs]
        
        batch_data = [(str(j), seq) for j, seq in enumerate(batch_seqs)]
        
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12], return_contacts=False)
        
        token_representations = results["representations"][12]
        
        for j, tokens_len in enumerate(batch_lens):
            # Mean pooling over sequence (1 : tokens_len-1)
            seq_rep = token_representations[j, 1 : tokens_len - 1].mean(0)
            embeddings.append(seq_rep.cpu().numpy())
            
    return embeddings

# Cloud Orchestrator
@app.function(
    image=image,
    volumes=volumes,
    timeout=86400 # 24 hours for orchestration of large datasets (ds7/ds8)
)
def orchestrate_dataset(dataset_name: str, split: str):
    import sys
    from tqdm import tqdm
    
    # Paths in the cloud volume
    # We assume data was uploaded to /data directly
    # Try multiple paths just in case
    possible_paths = [
        Path(f"/data/data/{dataset_name}_{split}.pkl"), # Found via debug
        Path(f"/data/{dataset_name}_{split}.pkl"),
        Path(f"/data/processed/{dataset_name}_{split}.pkl")
    ]
    
    pkl_path = None
    for p in possible_paths:
        if p.exists():
            pkl_path = p
            break
            
    if pkl_path is None:
        print(f"Error: Pickle not found for {dataset_name}_{split}")
        print(f"Checked: {[str(p) for p in possible_paths]}")
        print("Did you upload the data? 'modal volume put airr-ml-25-data data/processed /data'")
        return
        
    out_dir = Path(f"/data_35m/embeddings_35m/{dataset_name}/{split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {pkl_path}...")
    if not pkl_path.exists():
        print(f"Error: Pickle not found at {pkl_path}")
        print("Did you upload the data? 'modal volume put airr-ml-25-data data/processed /data'")
        return

    with open(pkl_path, "rb") as f:
        reps = pickle.load(f)
        
    print(f"Loaded {len(reps)} repertoires.")
    
    # Prepare inputs
    inputs = []
    output_paths = []
    
    for r in reps:
        out_file = out_dir / f"{r.rep_id}.npy"
        if out_file.exists():
            continue
            
        valid_seqs = [s for s in r.junction_aa if len(s) > 0]
        if not valid_seqs:
            np.save(out_file, np.array([]))
            continue
            
        inputs.append(valid_seqs)
        output_paths.append(out_file)
        
    if not inputs:
        print("All repertoires already processed!")
        return

    print(f"Submitting {len(inputs)} repertoires to workers...")
    
    # Parallel map
    results = embed_sequences.map(inputs, return_exceptions=True)
    
    completed = 0
    for result, out_file in tqdm(zip(results, output_paths), total=len(inputs), desc=f"Processing {dataset_name}"):
        if isinstance(result, Exception):
            print(f"Error processing {out_file.name}: {result}")
        else:
            np.save(out_file, np.array(result))
            completed += 1
            
    # Force sync to ensure data is persisted to volume
    vol.commit()
    print(f"Finished! Processed {completed}/{len(inputs)} repertoires.")

@app.local_entrypoint()
def main(dataset_name: str = "ds1", split: str = "train"):
    print(f"Triggering cloud orchestration for {dataset_name}...")
    orchestrate_dataset.remote(dataset_name, split)
