# 🚀 AIRR-ML-25: The Complete Beginner's Guide

Welcome to the **Adaptive Immune Profiling Challenge 2025**!
This guide is your roadmap from "zero" to "leaderboard submission". We will use **Modal** for heavy data processing and **RunPod** for training deep learning models.

---

## 🗺️ The Roadmap (The "Big Picture")

We are building a pipeline that moves data through three stages:

1.  **Phase 1: Preparation (Local)** 💻
    *   Goal: Download raw data and format it.
    *   Time: ~10 mins.
2.  **Phase 2: The Factory (Modal)** ☁️
    *   Goal: Convert biological sequences into "Embeddings" (numbers the AI understands).
    *   Tool: Modal (A10G GPUs).
    *   Time: ~2-4 hours.
3.  **Phase 3: The Brain (RunPod)** 🧠
    *   Goal: Train the **DeepRC** model to classify diseases.
    *   Tool: RunPod (RTX 3090/4090).
    *   Time: ~4-8 hours.
4.  **Phase 4: The Finish Line (Local)** 🏁
    *   Goal: Combine everything into a final submission.
    *   Time: ~20 mins.

---

## 🟢 Phase 1: Preparation (Local)

First, we need to get the raw data ready.

### Step 1: Run the Data Loader
Open your terminal and run:
```bash
uv run python data/load_all_datasets.py
```
*   **What it does**: Reads the raw folders (`ds1`, `ds2`...) and saves them as neat `.pkl` files in `data/processed/`.
*   **Success Check**: Look in `data/processed/`. You should see files like `ds1_train.pkl`, `ds1_test.pkl`, etc.

---

## 🔵 Phase 2: The Factory (Modal)

Now we need to generate **Embeddings**. This is computationally heavy, so we use Modal.

### Step 1: Upload Data to Modal
```bash
# Create a cloud volume
uv run modal volume create airr-ml-25-data

# Upload your processed data
uv run modal volume put airr-ml-25-data data/processed /data
```

### Step 2: Run the Embedding Generators
This runs the **ESM-2 35M** model on thousands of sequences in parallel.

```bash
# 1. Train Datasets (ds1-ds8)
for i in {1..8}; do
    uv run modal run --detach modal/embed_cloud.py --dataset-name "ds$i" --split train
done

# 2. Test Datasets (ds1-ds6)
for i in {1..6}; do
    uv run modal run --detach modal/embed_cloud.py --dataset-name "ds$i" --split test
done

# 3. Special Test Datasets (ds7 & ds8)
# These have multiple parts, but our script handles them!
uv run modal run --detach modal/embed_cloud.py --dataset-name "ds7" --split test
uv run modal run --detach modal/embed_cloud.py --dataset-name "ds8" --split test
```

### Step 3: Organize & Verify
Once the jobs finish (check your Modal dashboard), run these cleanup scripts:
```bash
# Consolidate scattered test files
uv run modal run modal/consolidate_test_sets.py

# Separate them into clean subfolders
uv run modal run modal/separate_test_sets.py

# (Optional) Verify counts
uv run modal run modal/verify_counts.py
```
*   **Success Check**: You now have a Modal Volume `airr-ml-25-data-35m` containing ~900GB of embeddings.

---

## 🟣 Phase 3: The Brain (RunPod)

Now we move to **RunPod** to train the DeepRC model. This model learns to diagnose diseases from the embeddings.

### Step 1: Rent a GPU Pod
1.  Go to **RunPod.io**.
2.  Deploy a **Secure Cloud** pod with **RTX 3090** or **4090**.
3.  **CRITICAL**: Set **Volume Disk** to **1000 GB** (1 TB). You need this space!
4.  Connect via SSH (see `docs/runpod_guide.md` for details).

### Step 2: Transfer Data (Cloud-to-Cloud) ☁️
**Do NOT download to your laptop.** inside your RunPod terminal:
```bash
# Install Modal
pip install modal

# Login (copy command from Modal Settings)
modal token set ...

# Download Embeddings (Fast!)
mkdir -p data/embeddings
uv run modal volume get airr-ml-25-data-35m embeddings_35m/ data/embeddings/
```

### Step 3: Train DeepRC (Cross-Validation)
Inside your RunPod terminal:
```bash
# Run the CV training loop for all datasets
for i in {1..8}; do
    echo "Training ds$i (CV)..."
    python deeprc/train_mil_cv.py --dataset "ds$i" --folds 5 --epochs 20 --batch-size 4
done
```
*   **What it does**: Trains 5 models per dataset (Cross-Validation) and saves them to `models/deeprc_cv/`. It is auto-resumable!

### Step 4: Run DeepRC Inference (Still on RunPod!)
After training completes, we need to generate predictions using the trained models.
**Why on RunPod?** The inference script needs access to the 900GB of embeddings. It's much faster to run it here than to download all that data.

```bash
# Generate predictions for all datasets (train + test)
python deeprc/infer_mil_cv.py --folds 5
```
*   **What it does**: Loads each ensemble of 5 models and predicts on both train and test sets.
*   **Output**: Prediction CSVs saved to `outputs/deeprc_cv_preds/`.

### Step 5: Run Downstream Models (Still on RunPod!)
While we are on RunPod (where the data is), we **must** run the other models that need embeddings.

```bash
# 1. Stats Model (Fast)
uv run python malid/train_stats_all.py

# 2. ESM Sequence Model (Needs Embeddings)
uv run python malid/train_esm_seq_all.py

# 3. Clustering Model (Choose ONE)
# Option A: Standard (CPU FAISS) - Stable
uv run python malid/run_clustering_all.py

# Option B: Voyager (Flash Fast CPU) - RECOMMENDED 🚀
# pip install voyager
uv run python malid/run_clustering_voyager.py

# Option C: LanceDB (GPU Accelerated)
# pip install lancedb
uv run python malid/run_clustering_lancedb.py

# 4. Task 2 Sequence Ranking (Needs Embeddings)
uv run python scripts/rank_sequences_task2_all.py
```
**Why?** Scripts 2, 3, and 4 require reading the 900GB embedding files. This is instant on RunPod but would take days to download to your laptop.

### Step 6: Push Results to GitHub
Now we save our work (models + predictions) to GitHub so we can access them on our laptop.
```bash
# Add models and predictions
git add -f models/deeprc_cv/
git add -f outputs/deeprc_cv_preds/
git add -f outputs/esm_seq_preds/
git add -f outputs/cluster_preds/
git add -f outputs/stats_preds/
git add -f outputs/task2_ranking/

# Commit and push
git commit -m "Add all trained models and predictions"
git push origin main
```

### Step 7: Pull Results to Laptop
Back on your **Laptop**:
```bash
git pull
```

---

## 🏁 Phase 4: The Finish Line (Local)

You have the predictions from all models (DeepRC, Stats, Clustering, ESM) on your laptop now. Let's combine them!

### Step 1: Run the Meta-Ensemble
This script combines the CSVs from DeepRC, Stats, and Clustering/ESM to make the final decision.
```bash
uv run python malid/train_meta_and_predict.py
```

### Step 2: Submit!
The script generates `outputs/submission/submission.csv`.
Upload this file to the competition platform.

**Congratulations! You have successfully built and run a state-of-the-art immune profiling pipeline!** 🥂
