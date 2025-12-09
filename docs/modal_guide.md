# Modal.com Setup & Usage Guide for AIRR-ML-25

This project uses [Modal](https://modal.com/) to parallelize the generation of ESM-2 embeddings for millions of sequences. Modal allows us to spin up hundreds of containers with GPUs/CPUs on demand, significantly speeding up this compute-intensive step.

## 1. Prerequisites

1.  **Create a Modal Account**: Go to [modal.com](https://modal.com/) and sign up.
2.  **Install Modal Client**:
    ```bash
    pip install modal
    # or if using uv
    uv pip install modal
    ```
3.  **Authenticate**:
    ```bash
    python3 -m modal setup
    ```
    This will open a browser window to authenticate your terminal with your Modal account.

## 2. The Embedding Script: `modal/embed_esm650m.py`

### What it does
1.  **Defines a Container Image**: It builds a Docker-like image with `torch`, `fair-esm`, `numpy`, and `pandas` installed.
2.  **Defines a Remote Function**: The `embed_sequences` function is decorated with `@app.function`. When called, this function runs **in the cloud** on Modal's infrastructure (using A10G GPUs by default).
3.  **Local Entrypoint**: The `main` function runs locally on your machine. It:
    -   Loads the processed pickle files (e.g., `data/processed/ds1_train.pkl`).
    -   Extracts CDR3 sequences.
    -   Checks if embeddings already exist in `data/embeddings/`.
    -   Sends batches of sequences to the remote `embed_sequences` function.
    -   Saves the returned embeddings as `.npy` files locally.

### Key Configuration
-   **Model**: Uses `esm2_t33_650M_UR50D` (650M parameters).
-   **GPU**: Requests `A10G` GPUs. You can change this to `T4` (cheaper) or `H100` (faster) in the script.
-   **Volume**: Uses a Modal Volume `airr-ml-25-data` to cache models or data if configured (optional for this script as we save locally).

## 3. How to Run

To generate embeddings for a specific dataset (e.g., `ds1` training set):

```bash
modal run modal/embed_esm650m.py --dataset-name ds1 --split train
```

To run for all datasets, you can write a simple shell loop or python script, or just run them sequentially:

```bash
# Example loop
for ds in ds1 ds2 ds3 ds4; do
    modal run modal/embed_esm650m.py --dataset-name $ds --split train
done
```

## 4. Monitoring

-   **Dashboard**: You can monitor running apps, logs, and costs at [modal.com/dashboard](https://modal.com/dashboard).
-   **Logs**: The terminal will show progress bars and logs from the remote workers.

## 5. Troubleshooting

-   **Timeout**: If a batch takes too long, increase `timeout` in the `@app.function` decorator.
-   **Out of Memory**: Decrease `batch_size` in the script.
-   **Authentication Error**: Run `python3 -m modal setup` again.
