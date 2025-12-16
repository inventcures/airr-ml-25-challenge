# 🕷️ The "Spider-Man" Strategy: Multi-Pod Execution Guide

**Objective:** Triple our speed by splitting the workload across 3 RunPod instances.
**Deadline:** Dec 17, 12:00 PM IST.

---

## 🏗️ Phase 1: Setup The Pods

1.  **Pod 1 (The Captain):** Your current running pod.
2.  **Pod 2 (The Heavy Lifter):** Rent a NEW GPU Pod (Template: RunPod PyTorch 2.0+).
3.  **Pod 3 (The Closer):** Rent a NEW GPU Pod (Template: RunPod PyTorch 2.0+).

*Tip: Use the same GPU type (e.g., A40, A6000, or A100) if possible for consistent speed, but even an A5000 is better than nothing.*

---

## 📦 Phase 2: Data Transfer (The "Care Package")

The new pods are empty. They need the **Input Data (Pickles)** to know what sequences to process.
Instead of downloading the whole dataset again (slow), we will copy just the pickles from Pod 1.

### Step 2.A: On Pod 1 (Create the Zips)
Run these commands in the Terminal of Pod 1:

```bash
# 1. Update code
git pull

# 2. Go to data directory
cd data/processed

# 3. Zip data for Pod 2 (DS8 Parts 1 & 2 Test)
zip pod2_input.zip ds8_1_test.pkl ds8_2_test.pkl

# 4. Zip data for Pod 3 (DS8 Part 3 Test)
zip pod3_input.zip ds8_3_test.pkl

# (Optional) Zip DS8 Train if you plan to try it (Warning: huge)
# zip ds8_train_input.zip ds8_train.pkl
```

### Step 2.B: Download to your Computer
1.  In Pod 1's JupyterLab File Browser, find `data/processed/pod2_input.zip` and `pod3_input.zip`.
2.  Right-click -> **Download**.

### Step 2.C: Upload to New Pods
1.  Open **Pod 2** JupyterLab.
2.  Drag-and-drop `pod2_input.zip` into the file browser.
3.  Open **Pod 3** JupyterLab.
4.  Drag-and-drop `pod3_input.zip` into the file browser.

---

## 🚀 Phase 3: Launch Execution

Run these commands on the respective Pods.

### 🎮 Pod 1: "The Captain" (DS1-DS7)

*Mission: Finish DS7 Test. Then aim for DS7 Train.*

```bash
# 1. Update Code
git pull

# 2. Run (Prioritizes Test)
python scripts/generate_embeddings_650m.py --include ds1 ds2 ds3 ds4 ds5 ds6 ds7
```
*(Note: Since you already did DS1-6, it will skip them instantly and focus on DS7).*

### 🏋️ Pod 2: "The Heavy Lifter" (DS8 Parts 1 & 2)

*Mission: Crush the first 2/3rds of DS8 Test.*

```bash
# 1. Clone Repo
git clone https://github.com/WIT-AIRR-Challenge/adaptive-immune-profiling-challenge-2025.git airr
cd airr/airr_ml_project_template

# 2. Install dependencies (Fast)
pip install torch esm pandas numpy tqdm

# 3. Setup Data
mkdir -p data/processed
mkdir -p data/embeddings
# Move the uploaded zip here
mv ~/pod2_input.zip data/processed/
cd data/processed
unzip pod2_input.zip
cd ../..

# 4. Run!
python scripts/generate_embeddings_650m.py --include ds8_1 ds8_2
```

### ⚾ Pod 3: "The Closer" (DS8 Part 3)

*Mission: Finish the last chunk of DS8 Test.*

```bash
# 1. Clone & Install (Same as Pod 2)
git clone https://github.com/WIT-AIRR-Challenge/adaptive-immune-profiling-challenge-2025.git airr
cd airr/airr_ml_project_template
pip install torch esm pandas numpy tqdm

# 2. Setup Data
mkdir -p data/processed
mkdir -p data/embeddings
# Move the uploaded zip here
mv ~/pod3_input.zip data/processed/
cd data/processed
unzip pod3_input.zip
cd ../..

# 3. Run!
python scripts/generate_embeddings_650m.py --include ds8_3
```

---

## 🌪️ Phase 4: The Merge (Tomorrow Morning)

**When:** Dec 17, 08:00 AM IST (or whenever Pods 2/3 finish).

### Step 4.A: Pack Results (On Pod 2 & 3)
On Pod 2:
```bash
cd data/embeddings
zip -r pod2_results.zip ds8
```
*(Download `pod2_results.zip`)*

On Pod 3:
```bash
cd data/embeddings
zip -r pod3_results.zip ds8
```
*(Download `pod3_results.zip`)*

### Step 4.B: Merge on Pod 1
1.  Upload `pod2_results.zip` and `pod3_results.zip` to **Pod 1**.
2.  Unzip them:
    ```bash
    mv pod2_results.zip data/embeddings/
    mv pod3_results.zip data/embeddings/
    cd data/embeddings
    unzip -o pod2_results.zip  # -o means overwrite/merge
    unzip -o pod3_results.zip
    ```

### Step 4.C: Final Check
Ensure `data/embeddings/ds8/1_test` (etc) exists on Pod 1.

Then run the final submission script on Pod 1! 🏁
