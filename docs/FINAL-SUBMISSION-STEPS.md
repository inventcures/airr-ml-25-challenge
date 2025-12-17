# 🏁 FINAL SUBMISSION STEPS (The "Endgame" Guide)

**Objective:** Merge all data to **Pod 1** and generate the submission before the deadline (16 Hours Remaining).

---

## 🌍 Network Reference

| Pod Role | IP Address | Port | Data to Transfer |
| :--- | :--- | :--- | :--- |
| **POD 1 (Master)** | **`203.57.40.123`** | `10052` | *Destination for all data* |
| **POD 2** | `103.196.86.192` | `18332` | DS8 Test (Parts 1 & 2) |
| **POD 3** | `103.196.86.130` | `14457` | DS8 Train (Shard 1/2) |
| **POD 4** | `213.173.98.86` | `18444` | DS8 Train (Shard 0/2) |
| **POD 5** | `213.173.98.90` | `12987` | DS7 Train (Shard 1/2) |

---

## 1️⃣ STEP 1: Pack & Send (On Satellite Pods)

*Run these commands on the respective pods efficiently once they finish.*

### 🛠️ ON POD 5 (DS7 Train - Shard 1/2)
```bash
# 1. Zip the results
cd /workspace/airr-ml-25-challenge/data/embeddings
zip -r ds7_train_shard1.zip ds7/train

# 2. Send to Pod 1
scp -P 10052 ds7_train_shard1.zip root@203.57.40.123:/workspace/airr-ml-25-challenge/data/embeddings/
```

### 🛠️ ON POD 4 (DS8 Train - Shard 0/2)
```bash
# 1. Zip
cd /workspace/airr-ml-25-challenge/data/embeddings
zip -r ds8_train_shard0.zip ds8/train

# 2. Send to Pod 1
scp -P 10052 ds8_train_shard0.zip root@203.57.40.123:/workspace/airr-ml-25-challenge/data/embeddings/
```

### 🛠️ ON POD 3 (DS8 Train - Shard 1/2 + DS8 Test Part 3)
```bash
# 1. Zip (Includes both Train Shard and Test Part 3 if present)
cd /workspace/airr-ml-25-challenge/data/embeddings
zip -r ds8_pod3_results.zip ds8

# 2. Send to Pod 1
scp -P 10052 ds8_pod3_results.zip root@203.57.40.123:/workspace/airr-ml-25-challenge/data/embeddings/
```

### 🛠️ ON POD 2 (DS8 Test Parts 1 & 2)
```bash
# 1. Zip
cd /workspace/airr-ml-25-challenge/data/embeddings
zip -r ds8_test_pod2.zip ds8

# 2. Send to Pod 1
scp -P 10052 ds8_test_pod2.zip root@203.57.40.123:/workspace/airr-ml-25-challenge/data/embeddings/
```

---

## 2️⃣ STEP 2: Merge & Unzip (On Pod 1)

**A. Transfer (Pull Method - Run on Pod 1)**
*Alternative to pushing from remote pods. Run these in parallel tabs for speed.*

**Pod 2 (DS8 Test):**
```bash
scp -P 18332 root@103.196.86.192:/workspace/airr-ml-25-challenge/data/embeddings/pod2_results_ds8_test_1_2.zip .
```
**Pod 3 (DS8 Train/Test):**
```bash
scp -P 14457 root@103.196.86.130:/workspace/airr-ml-25-challenge/data/embeddings/pod3_results.zip .
```
**Pod 4 (DS8 Train Shard 0):**
```bash
scp -P 18444 root@213.173.98.86:/workspace/airr-ml-25-challenge/data/embeddings/pod4_results_ds8_train_shard0.zip .
```
**Pod 5 (DS7 Train Shard 1):**
```bash
scp -P 12987 root@213.173.98.90:/workspace/airr-ml-25-challenge/data/embeddings/pod5_results_ds7_train_shard1.zip .
```

**B. Merge & Fix (Scripted)**
*The merge logic is now encapsulated in a safe, logging-enabled script.*

Run this single command:
```bash
./scripts/merge_650m_embeddings.sh
```

*This script will:*
1.  Unzip all 4 files.
2.  Fix the directory structure (merging split folders).
3.  Clean up empty directories.
4.  Run the verification scan automatically.
5.  Save a detailed log to `logs/merge_650m_embeddings.log`.

echo "✅ Merge & Repair Complete!"
```

---

## 3️⃣ STEP 3: Execution Pipeline (On Pod 1)

*These scripts are fast compared to embeddings. You can run them sequentially.*

### A. Task 2 Ranking (Crowd Labeling)
*   **Input:** DS7/DS8 Embeddings (Train & Test).
*   **Output:** Ranked submission file.
*   **Est. Time:** 20-40 Minutes.

```bash
cd /workspace/airr-ml-25-challenge
uv run scripts/rank_sequences_task2_all.py -m 650
```

### B. Task 1 Ensemble Prediction (Binding)
*   **Input:** DS1-6 & DS8 Embeddings.
*   **Output:** Predictions for Task 1.
*   **Est. Time:** 15-30 Minutes.

```bash
uv run scripts/generate_ensemble_preds_task1.py -m 650
```

### C. Build Final Submission
*   **Input:** Results from A & B.
*   **Output:** `submission.zip`.
*   **Est. Time:** < 1 Minute.

```bash
uv run scripts/build_submission.py -m 650
```

---
### Step 4: Download & Submit
1.  Navigate to `outputs/submissions_650m/`
2.  Find the folder with the latest timestamp (e.g., `submission_20251216...`).
3.  Download `submission.csv` within it.
4.  **Zip it** (Warning: Portal creates zip automatically? No, usually you upload CSV or Zip. Zip is safer).
    ```bash
    cd outputs/submissions_650m/submission_YYYY...
    zip -r submission_FINAL.zip submission.csv
    ```
5.  Upload `submission_FINAL.zip` to the Challenge Portal!

---
# C. Pipeline & Emergency Fallback
*What if some embeddings are missing?*
**Don't Panic.** I have patched the scripts to handle "Partial Data" gracefully:
1.  **Task 1 (Ensemble):** Defaults to `0.5` probability if embedding is missing. (Already Safe)
2.  **Task 2 (Ranking):** **NEW!** Now falls back to selecting the first valid sequence if embedding is missing.

**Scenario C: Verification Fails (RED LIGHT) but Deadline is Imminent**
If `scripts/merge_650m_embeddings.sh` fails because of verification:
1.  **Run Pipeline ANYWAY.** The scripts will now auto-fill missing data.
2.  Go to **Step 3** above and execute normally.
3.  Submit whatever you have. A partial score > No score.
