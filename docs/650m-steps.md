
# 🚀 650M Embeddings Workflow (Namespaced)

This guide details how to switch the pipeline to use the larger **ESM2-650M** embeddings.

## ✨ Safe & Isolated
All scripts now accept `-m 650`. This isolates all outputs to `*_650m` directories.
**You do NOT need to delete old models.** Your 35M baseline is safe! 🛡️

# 🚨 EMERGENCY PLAN: "Test First" Strategy (Updated Dec 16 03:00 IST)

**Context:**
- Deadline: Dec 17, 12:00 PM IST.
- Remaining Work: ~28 Hours of Embedding Generation (Test Sets).
- Strategy: We prioritize generating **TEST** set embeddings first. If we run out of time, we skip Training set embeddings for DS7/8 and use models trained on DS1-6.

## 📉 Timeline & Milestones

1.  **NOW (Dec 16 03:00 AM)**: `generate_embeddings_650m.py` is running.
    -   It is processing **TEST** datasets first (DS1..8).
    -   Expected Speed: ~11.5s/repo (Dynamic Batching Active).

2.  **TOMORROW MORNING (Dec 17 08:00 AM)**: Decision Point.
    -   **Scenario A:** Script finished DS8 Test. -> SUCCESS. Proceed to Submission.
    -   **Scenario B:** Script is still on DS8 Test. -> **KILL IT**.
    -   *Note:* We need ~3-4 hours to run the Ranking & Submission script. So at T-4 hours (08:00 AM IST), we must STOP embedding generation and just submit what we have (or whatever completed folds we have).

## 🪜 Step-by-Step Instructions (Post-Embeddings)

Once `generate_embeddings_650m.py` finishes (or you stop it at 08:00 AM IST):

### 1. Train Models (Skip if DS7/8 Train missing)
If you have DS7/8 **Training** embeddings (unlikely):
```bash
python malid/train_esm_seq_all.py -m 650
```
*If you DO NOT have DS7/8 Training embeddings (likely), skip this step. We will use the models trained on DS1-6.*

### 2. Rank Sequences (Task 2)
This generates the ranking for DS7/8.
```bash
python scripts/rank_sequences_task2_all.py -m 650
```

### 3. Build Submission
```bash
python scripts/build_submission.py -m 650
```
-   Output: `outputs/submission_650m/submission.csv`

### 4. Upload to Kaggle
-   Go to Kaggle Competition.
-   Upload `submission.csv`.

**Final Output:** `submission_650m.csv` 🚀
