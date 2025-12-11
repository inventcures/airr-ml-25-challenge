# ☁️ RunPod Master Guide

This guide is your cockpit for managing the heavy GPU training on RunPod.

---

## ✅ Pre-Flight Checklist
Before you spend money renting a GPU, ensure you have:
1.  [ ] **RunPod Account**: With at least $10 credit.
2.  [ ] **SSH Key**: Added to your RunPod settings (see below if not).
3.  [ ] **Modal Token**: Open [Modal Settings](https://modal.com/settings/tokens) and keep it ready.
4.  [ ] **Embeddings Ready**: Your Modal volume `airr-ml-25-data-35m` should be full (~900GB).

---

## 🚀 Step 1: Launch the Pod

1.  Go to **RunPod Console** -> **Pods** -> **Deploy**.
2.  Choose **Secure Cloud** (recommended for stability).
3.  **GPU**: Select **RTX 3090** or **RTX 4090**.
4.  **Template**: Search for and select `RunPod PyTorch 2.1`.
5.  **Customize Deployment** (Click the "Edit" button):
    *   **Container Disk**: `50 GB` (Enough for code).
    *   **Volume Disk**: `1000 GB` (CRITICAL! You need 1TB for data).
6.  Click **Deploy**.

---

## 🔗 Step 2: Connect via SSH

1.  Wait for the pod to show **"Running"**.
2.  Click **Connect** -> **SSH Command**.
3.  Copy the command (e.g., `ssh root@194.x.x.x -p 12345`).
4.  Paste it into your **Laptop Terminal**.

*Trouble connecting?*
*   If it asks for a password, your SSH key isn't set up correctly.
*   Check `~/.ssh/id_rsa.pub` on your laptop and add it to RunPod Settings -> SSH Keys.

---

## 📦 Step 3: The Great Data Transfer (Cloud-to-Cloud)

We will pull data directly from Modal to RunPod. This is fast because it uses data center internet speeds.

**Run these commands inside your RunPod SSH terminal:**

### 1. Setup Environment (The Fast Way ⚡️)
```bash
cd /workspace

# Install uv (it's much faster than pip!)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install Modal using uv
uv pip install modal --system
```

### 2. Authenticate with Modal
Paste your token command from the Modal dashboard:
```bash
modal token set --token-id ... --token-secret ...
```

### 3. Clone Your Code
```bash
git clone https://github.com/your-username/adaptive-immune-profiling-challenge-2025.git
mv adaptive-immune-profiling-challenge-2025 airr_ml_project_template
cd airr_ml_project_template
```

### 4. Download Data (The Big One)
```bash
# Create directories
mkdir -p data/processed data/embeddings

# 1. Download Labels (Small)
# Note: The pickles are inside the 'data' folder on the volume
uv run modal volume get airr-ml-25-data data/ data/processed/

# 2. Download Embeddings (Huge - 900GB)
# This will take 1-2 hours. Do not close your terminal!
uv run modal volume get airr-ml-25-data-35m embeddings_35m/ data/embeddings/
```

### 5. Verify Data
Run this to make sure everything arrived safely:
```bash
ls -R data/embeddings/ds7/test | head
# You should see subfolders like '1_test', '2_test'
```

---

## 🧠 Step 4: Train DeepRC (CV)

Now the fun part. We train the AI using Cross-Validation for robustness.

### 1. Install Dependencies
```bash
# Use uv for lightning-fast installs
uv pip install torch numpy pandas scikit-learn tqdm fair-esm --system
```

### 2. Run Training Loop
```bash
for i in {1..8}; do
    echo "Training ds$i (CV)..."
    python deeprc/train_mil_cv.py --dataset "ds$i" --folds 5 --epochs 20 --batch-size 4
done
```
*   **Time**: ~2-3 hours per dataset (for 5 folds).
*   **Output**: Models saved to `models/deeprc_cv/`.
*   **Resumable**: If stopped, just run it again! It picks up where it left off.

### 3. Run Inference
After training, run inference to generate predictions for the meta-ensemble:
```bash
python deeprc/infer_mil_cv.py --folds 5
```
*   **What it does**: Uses the trained ensembles to predict on both train (for meta-ensemble) and test sets.
*   **Output**: Prediction CSVs in `outputs/deeprc_cv_preds/`.

### 4. Run Downstream Models (Stats, ESM, Clustering, Task 2)
Since we are on RunPod with access to the embeddings, we must run the other models now.

```bash
# 1. Stats Model (Fast)
uv run python malid/train_stats_all.py

# 2. ESM Sequence Model (Heavy, resumes automatically)
uv run python malid/train_esm_seq_all.py

# 3. Clustering Model
uv run python malid/run_clustering_all.py

# 4. Task 2 Sequence Ranking
uv run python scripts/rank_sequences_task2_all.py
```

### 5. Push Results to GitHub
Save your models and predictions to Git so you can access them on your laptop:
```bash
# Add ALL predictions and models
git add -f models/deeprc_cv/
git add -f outputs/deeprc_cv_preds/
git add -f outputs/stats_preds/
git add -f outputs/esm_seq_preds/
git add -f outputs/cluster_preds/
git add -f outputs/task2_ranking/

git commit -m "Add valid RunPod results"
git push origin main
```

---

## ⬇️ Step 5: Bring Models Home

Once training finishes, we need to save the brains (models) to your laptop.

**Open a NEW terminal on your Laptop:**
```bash
# Replace IP and PORT with your Pod's details
scp -P PORT -r root@IP:/workspace/airr_ml_project_template/models/deeprc_cv .
scp -P PORT -r root@IP:/workspace/airr_ml_project_template/outputs/deeprc_cv_preds .
```

### 5. Detached Mode (Resilience) 🛡️
To keep training running even if your laptop sleeps or disconnects, use `tmux`:

1.  **Start a Session**:
    ```bash
    # Install tmux if missing (takes 10s)
    apt-get update && apt-get install -y tmux
    
    tmux new -s training
    ```
2.  **Run Training**:
    ```bash
    # Run the CV training loop
    for i in {1..8}; do python deeprc/train_mil_cv.py --dataset "ds$i" --folds 5; done
    ```
3.  **Detach**: Press `Ctrl+B`, then `D`. You can now close the terminal.
4.  **Re-attach**:
    ```bash
    tmux attach -t training
    ```

### 6. Monitoring 📱
RunPod doesn't have a mobile app, but you can monitor via SSH or Web:

*   **Web Dashboard**: Check GPU utilization on [runpod.io](https://runpod.io).
*   **Mobile SSH**: Use an app like **Termius** (iOS/Android) to SSH in and run `tail -f logs/deeprc_train_cv_*.log`.

### 7. Resuming Interrupted Runs 🔄
The scripts are **auto-resumable**:
*   **Checkpoints**: Saved every epoch inside `models/deeprc_cv/`.
*   **Resume**: Just run the python training loop again. It will detect the checkpoint and resume from the exact fold and epoch!

---

## 🛑 Step 6: Terminate (Save Money!)

1.  Go to RunPod Dashboard.
2.  Find your pod.
3.  Click the **Trash Can Icon (Terminate)**.
4.  Confirm.

**Warning**: If you just "Stop" it, you still pay for the 1TB storage! Terminate it completely when done.
