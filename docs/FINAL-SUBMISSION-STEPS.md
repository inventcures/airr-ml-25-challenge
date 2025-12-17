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

**B. Unzip & Fix Structure**
*`unzip -o` is perfectly safe. It merges directories and overwrites identical files.*

```bash
cd /workspace/airr-ml-25-challenge/data/embeddings

# 1. Unzip Everything
unzip -o pod2_results_ds8_test_1_2.zip
unzip -o pod3_results.zip
unzip -o pod4_results_ds8_train_shard0.zip
unzip -o pod5_results_ds7_train_shard1.zip

# 2. 🚨 CRITICAL FIX: Move Flat Folders to Nested Structure 🚨
# Downstream scripts expect: data/embeddings/ds8/test/3_test
# Pods likely created:       data/embeddings/ds8_3/test

# Fix DS8 Test (Handle Pod 2's ds8_1/ds8_2 and Pod 3's ds8_3)
mkdir -p ds8/test
if [ -d "ds8_1/test" ]; then
    mkdir -p ds8/test/1_test
    # Merge contents of ds8_1/test into ds8/test/1_test
    rsync -a --remove-source-files ds8_1/test/ ds8/test/1_test/
fi
if [ -d "ds8_2/test" ]; then
    mkdir -p ds8/test/2_test
    # Merge contents of ds8_2/test into ds8/test/2_test
    rsync -a --remove-source-files ds8_2/test/ ds8/test/2_test/
fi
if [ -d "ds8_3/test" ]; then
    mkdir -p ds8/test/3_test
    # Merge contents of ds8_3/test into ds8/test/3_test
    rsync -a --remove-source-files ds8_3/test/ ds8/test/3_test/
fi

# Fix DS8 Train (Pod 3 and 4 outputted standard 'ds8/train', so unzip worked correctly. No manual merge needed.)

# Fix DS7
mkdir -p ds7/test
if [ -d "ds7_1/test" ]; then
    mkdir -p ds7/test/1_test
    rsync -a --remove-source-files ds7_1/test/ ds7/test/1_test/
fi
if [ -d "ds7_2/test" ]; then
    mkdir -p ds7/test/2_test
    rsync -a --remove-source-files ds7_2/test/ ds7/test/2_test/
fi

# 3. Cleanup Empty Shells
rmdir ds7_1 ds7_2 ds8_1 ds8_2 ds8_3 2>/dev/null || true
rm -rf ds8_3 # Force remove if rsync left empty dirs

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

## 4️⃣ STEP 4: Download & Submit

**📍 Run on your Local Machine (Laptop):**

```bash
scp -P 10052 root@203.57.40.123:/workspace/airr-ml-25-challenge/submission.zip ./submission_FINAL.zip
```

**Upload `submission_FINAL.zip` to the Challenge Portal!** 🚀
