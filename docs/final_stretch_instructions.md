# 🏁 The Final Stretch: Execution Checklist

This is your master checklist for the final phase of the challenge. Follow these steps in order.

## 🟢 Part 1: RunPod Execution (The Heavy Lifting)

**Prerequisite**: Ensure you have pulled the latest code on RunPod.
```bash
git pull
```

### 1. Train & Infer DeepRC
*(If you haven't already)*
```bash
# Train
for i in {1..8}; do python deeprc/train_mil_cv.py --dataset "ds$i" --folds 5; done

# Infer (Generates outputs/deeprc_cv_preds/)
python deeprc/infer_mil_cv.py --folds 5
```

### 2. Run Downstream Models
These scripts require the 900GB embeddings (available only on RunPod).

```bash
# A. Stats Model (Generates outputs/stats_preds/)
uv run python malid/train_stats_all.py

# B. ESM Sequence Model (Generates outputs/esm_seq_preds/)
uv run python malid/train_esm_seq_all.py

# C. Clustering Model (Choose ONE)
# FAISS (Standard)
uv run python malid/run_clustering_all.py

# OR Voyager (Fastest - Recommended)
# pip install voyager
uv run python malid/run_clustering_voyager.py

# OR LanceDB (GPU Accelerated)
# pip install lancedb
uv run python malid/run_clustering_lancedb.py

# D. Task 2 Ranking (Generates outputs/task2_ranking/)
uv run python scripts/rank_sequences_task2_all.py
```

---

## 🔵 Part 2: Sync Results (Cloud -> Git)

Push all generated results to GitHub so your laptop can see them.

```bash
# Force add folders (ignoring .gitignore rules for outputs)
git add -f models/deeprc_cv/
git add -f outputs/deeprc_cv_preds/
git add -f outputs/stats_preds/
git add -f outputs/esm_seq_preds/
git add -f outputs/cluster_preds/
git add -f outputs/task2_ranking/

# Commit & Push
git commit -m "Add final RunPod results"
git push origin main
```

---

## 🟣 Part 3: Local Assembly (Laptop)

Switch to your **Local Machine**.

### 1. Pull Results
```bash
git pull origin main
```

### 2. Verify Files
Ensure you have files in `outputs/` corresponding to all datasets.

### 3. Run Meta-Ensemble
Aggregates all predictions into specific "Part" files.
```bash
uv run python malid/train_meta_and_predict.py
```

### 4. Build Final Submission
Combines parts into the final CSV.
```bash
uv run python scripts/build_submission.py
```

---

## ✅ Part 4: Final Submission

1.  Locate `submission.csv` in your project root (or `outputs/submission/`).
2.  Upload to the challenge portal.
3.  Celebrate! 🥂
