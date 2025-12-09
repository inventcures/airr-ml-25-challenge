import modal
from pathlib import Path
import pickle
import numpy as np
from typing import List, Dict, Any

# Define the image
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "fair-esm", "numpy", "pandas", "tqdm")
)

app = modal.App("airr-ml-25-esm")

@app.function(
    image=image,
    gpu="A10G", # or "H100" if available and needed
    timeout=3600, # 1 hour
    max_containers=10,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0), # Resilience
    volumes={
        "/data": modal.Volume.from_name("airr-ml-25-data"),
        "/root/.cache/torch": modal.Volume.from_name("airr-ml-25-data") # Cache model weights
    }
)
def embed_sequences(sequences: List[str], batch_size: int = 64) -> List[np.ndarray]:
    import torch
    import esm

    # Load ESM-2 model
    # esm2_t33_650M_UR50D is a good balance
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    embeddings = []
    
    from tqdm import tqdm
    # Process in batches
    for i in tqdm(range(0, len(sequences), batch_size), desc="Batches"):
        batch_seqs = sequences[i:i + batch_size]
        # Format for ESM: list of (id, seq)
        batch_data = [(str(j), seq) for j, seq in enumerate(batch_seqs)]
        
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        
        token_representations = results["representations"][33]
        
        # Extract sequence embeddings (mean over tokens, or CLS?)
        # Usually mean over tokens (excluding padding and start/end tokens) is good for seq level.
        # Or CLS token.
        # Let's use mean pooling for now as it's robust.
        
        for j, tokens_len in enumerate(batch_lens):
            # tokens_len includes start/end tokens?
            # batch_tokens includes <cls> and <eos>
            # We want to average over the sequence itself (1 to len-1)
            # But batch_lens counts all non-padding.
            # So actual seq is 1 : tokens_len-1
            
            # Note: ESM-2 adds <cls> at 0 and <eos> at end.
            # So valid tokens are 1..tokens_len-2?
            # Let's just take 1 : tokens_len-1 (excluding eos)
            
            seq_rep = token_representations[j, 1 : tokens_len - 1].mean(0)
            embeddings.append(seq_rep.cpu().numpy())
            
    return embeddings

# Local entrypoint to trigger the job
@app.local_entrypoint()
def main(dataset_name: str = "ds1", split: str = "train"):
    # This script assumes it's run locally and triggers remote execution
    # It reads local pickles, sends sequences to Modal, and saves results locally (or to Volume)
    
    # Ideally, we should upload the data to Modal Volume first, then run processing there.
    # But for simplicity, we can send lists of sequences.
    
    from data.load_all_datasets import load_repertoires_pickle, PROCESSED_DIR
    
    pkl_path = PROCESSED_DIR / f"{dataset_name}_{split}.pkl"
    if not pkl_path.exists():
        print(f"Pickle not found: {pkl_path}")
        return

    reps = load_repertoires_pickle(pkl_path)
    print(f"Loaded {len(reps)} repertoires from {dataset_name}")

    # For each repertoire, check if embeddings exist
    out_dir = Path(f"data/embeddings/{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare list of inputs for parallel processing
    inputs = []
    output_paths = []
    
    from tqdm import tqdm
    
    print(f"Preparing inputs for {len(reps)} repertoires...")
    for r in reps:
        out_file = out_dir / f"{r.rep_id}.npy"
        if out_file.exists():
            continue
            
        # Filter valid sequences (length > 0)
        valid_seqs = [s for s in r.junction_aa if len(s) > 0]
        if not valid_seqs:
            # Save empty immediately
            np.save(out_file, np.array([]))
            continue
            
        inputs.append(valid_seqs)
        output_paths.append(out_file)
        
    if not inputs:
        print("All repertoires already processed!")
        return

    print(f"Submitting {len(inputs)} repertoires to Modal (parallel)...")
    
    # Use .map() for parallel execution
    # return_exceptions=True allows some to fail without crashing all
    results = embed_sequences.map(inputs, return_exceptions=True)
    
    # Iterate results as they finish
    completed = 0
    for result, out_file in tqdm(zip(results, output_paths), total=len(inputs), desc=f"Processing {dataset_name}"):
        if isinstance(result, Exception):
            print(f"Error processing {out_file.name}: {result}")
        else:
            np.save(out_file, np.array(result))
            completed += 1
            
    print(f"Finished! Processed {completed}/{len(inputs)} repertoires.")

