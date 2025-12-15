# 🚀 650M Embeddings Workflow

This guide details how to switch the pipeline to use the larger **ESM2-650M** embeddings.

## ✅ Time Savers (Skipped Steps)
You do **NOT** need to re-run the following, as they are independent of embeddings:
- `python malid/train_stats_all.py` (Statistical baseline uses metadata only).
- **Keep** `outputs/stats_preds/*.csv`.

## 🛑 Prerequisite: Cleanup Old Models
Since we reuse file names (to compatibility with the meta-learner), you **MUST** clear old artifacts to ensure you are training on new data.

Run on Pod:
```bash
# 1. Clear Old ESM Models (35M)
rm -rf models/esm_seq/*.joblib
rm -rf models/esm_seq_ensemble/*.joblib

# 2. Clear Old ESM Predictions
rm -rf outputs/esm_seq_preds/*.csv

# 3. Clear Old Task 2 Rankings
rm -rf outputs/task2_ranking/*.csv
rm -rf outputs/task2_ranking/*.pkl

# 4. Clear Meta-Learner (So it retrains on new inputs)
rm -rf models/meta/*.joblib
rm -rf outputs/submission/*
```

---

## 🏃 STEP 1: Train ESM Models (650M) ~2 Hours

The `-m 650` flag tells the script to look in `data/embeddings` (where 650M embeddings are generated), instead of `data/embeddings/35m`.

```bash
# Train on 650M embeddings
python malid/train_esm_seq_all.py -m 650
```
*Output: Generates new `outputs/esm_seq_preds/*_esm_preds.csv`.*

---

## 🧠 STEP 2: Retrain Meta-Learner (~5 Mins)

The meta-learner automatically picks up the new predictions from `outputs/esm_seq_preds/`.

```bash
# ⚠️ Ensure you ran the cleanup step above!
python malid/train_meta_and_predict.py
```
*Output: Generates meta-model & `outputs/submission/submission.csv`.*

---

## 🏆 STEP 3: Rank Sequences (Task 2) (~3 Hours)

We force re-ranking using the 650M embeddings.

```bash
# Rank using 650M embeddings
python scripts/rank_sequences_task2_all.py -m 650 --force
```
*Output: Generates `outputs/task2_ranking/*_ranking.csv`.*

---

## 📦 STEP 4: Build Submission (~1 Min)

Combine everything into the final Kaggle file.

```bash
python scripts/build_submission.py
```
*Output: `submission_[date].csv` ready for upload.*
