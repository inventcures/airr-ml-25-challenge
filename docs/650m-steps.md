# 🚀 650M Embeddings Workflow (Namespaced)

This guide details how to switch the pipeline to use the larger **ESM2-650M** embeddings.

## ✨ Safe & Isolated
All scripts now accept `-m 650`. This isolates all outputs to `*_650m` directories.
**You do NOT need to delete old models.** Your 35M baseline is safe! 🛡️

## 🕷️ The "Spider-Man" Strategy (5-Pod Parallelization)

To beat the clock, we have split the work across 5 Pods for maximum throughput.

### Pod 1: "The Captain" (DS7 Test + DS1-6)
*This pod prioritizes the critical DS7 Test set, then handles the smaller datasets.*
**Phase 1 (DS7 Test Only):**
```bash
python scripts/generate_embeddings_650m.py --include ds7 --only-split test
```
**Phase 2 (DS1-6 Clean Sweep - After Phase 1):**
```bash
python scripts/generate_embeddings_650m.py --include ds1 ds2 ds3 ds4 ds5 ds6
```

### Pod 2: "The Heavy Lifter" (DS8 Test Parts 1 & 2)
*This pod crunches the bulk of DS8 Test.*
```bash
python scripts/generate_embeddings_650m.py --include ds8_1 ds8_2
```

### Pod 3: "The Closer" (DS8 Test Part 3)
*This pod finishes the DS8 Test set.*
```bash
python scripts/generate_embeddings_650m.py --include ds8_3
```

### Pod 4: "Train Engine A" (DS8 Train)
*Focuses purely on the massive DS8 Training set (94M sequences).*
**Quick Start:**
```bash
curl -LsSf https://raw.githubusercontent.com/inventcures/airr-ml-25-challenge/main/start_pod4.sh | bash
```
**Manual Command:**
```bash
python scripts/generate_embeddings_650m.py --include ds8 --only-split train
```

### Pod 5: "Train Engine B" (DS7 Train)
*Focuses purely on the massive DS7 Training set (94M sequences).*
**Quick Start:**
```bash
curl -LsSf https://raw.githubusercontent.com/inventcures/airr-ml-25-challenge/main/start_pod5.sh | bash
```
**Manual Command:**
```bash
python scripts/generate_embeddings_650m.py --include ds7 --only-split train
```

### 🌪️ How to Merge (Tomorrow)
When all pods finish:
1.  On Pods 2-5, zip up their results:
    ```bash
    cd data/embeddings
    zip -r ds_embeddings_partX.zip .
    ```
2.  Download these zips.
3.  Upload/SCP them to **Pod 1**.
4.  Unzip them on Pod 1 to merge into `data/embeddings/`.

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
