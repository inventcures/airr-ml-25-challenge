
# 🚀 650M Embeddings Workflow (Namespaced)

This guide details how to switch the pipeline to use the larger **ESM2-650M** embeddings.

## ✨ Safe & Isolated
All scripts now accept `-m 650`. This isolates all outputs to `*_650m` directories.
**You do NOT need to delete old models.** Your 35M baseline is safe! 🛡️

## ✅ Time Savers (Skipped Steps)
You do **NOT** need to re-run the following:
- `python malid/train_stats_all.py` (Independent of ESM).

---

## 🏃 STEP 1: Train ESM Models (650M) ~2 Hours

```bash
# Uses: data/embeddings
# Writes to: models/esm_seq_650m/, outputs/esm_seq_preds_650m/
python malid/train_esm_seq_all.py -m 650
```

---

## 🧠 STEP 2: Retrain Meta-Learner (~5 Mins)

```bash
# Reads from: outputs/esm_seq_preds_650m/
# Writes to: models/meta_650m/, outputs/submission_650m/
python malid/train_meta_and_predict.py -m 650
```

---

## 🏆 STEP 3: Rank Sequences (Task 2) (~3 Hours)

```bash
# Writes to: outputs/task2_ranking_650m/
python scripts/rank_sequences_task2_all.py -m 650 --force
```

---

## 📦 STEP 4: Build Submission (~1 Min)

```bash
# Combines: outputs/submission_650m/ & outputs/task2_ranking_650m/
# Writes to: submission_650m.csv
python scripts/build_submission.py -m 650
```

**Final Output:** `submission_650m.csv` 🚀
