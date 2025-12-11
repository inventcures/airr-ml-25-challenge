# Cloud Setup (Modal & RunPod)

This pipeline leverages a hybrid cloud architecture to handle large-scale AIRR data efficiently.

## Architecture

| Component | Platform | Why? |
| :--- | :--- | :--- |
| **Data Storage** | **Modal Volumes** | High-throughput, shared between jobs. |
| **ESM Embeddings** | **Modal** | Serverless scaling for massive parallel inference. |
| **DeepRC Training** | **RunPod** | Persistent GPU instances for long-running LSTM/Attention training. |
| **Meta-Ensemble** | **Local** | Lightweight aggregation and submission building. |

---

## ☁️ Setting up Modal

**Modal** is used for hosting the ~35GB dataset and generating ESM2 embeddings.

1.  **Sign Up**: Create an account at [modal.com](https://modal.com).
2.  **Install CLI**:
    ```bash
    pip install modal
    ```
3.  **Authenticate**:
    ```bash
    modal token new
    ```
4.  **Verify Volume**:
    Check that you can access the shared volume:
    ```bash
    modal volume list
    ```
    You should see `airr-ml-25-data-35m`.

---

## ⚡ Setting up RunPod

**RunPod** is used for training the DeepRC MIL models (5-Fold CV).

1.  **Rent a Pod**:
    -   Go to [runpod.io](https://runpod.io).
    -   Select a GPU instance (e.g., RTX 3090 or 4090).
    -   Image: Use PyTorch 2.0+ template.
2.  **SSH Access**:
    -   Set up SSH keys in your RunPod settings.
    -   Connect: `ssh root@<IP> -p <PORT>`
3.  **Environment Sync**:
    -   Clone your repo on the pod.
    -   **Important**: Run `scripts/sync_data_robust.py` on the pod to download data from Modal to the pod's local NVMe storage.

```bash
# On RunPod
python scripts/sync_data_robust.py
```
