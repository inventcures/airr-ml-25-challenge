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

### 1. Setup Environment
```bash
cd /workspace
pip install modal
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
uv run modal volume get airr-ml-25-data processed/ data/processed/

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

## 🧠 Step 4: Train DeepRC

Now the fun part. We train the AI.

### 1. Install Dependencies
```bash
pip install torch numpy pandas scikit-learn tqdm fair-esm
```

### 2. Run Training Loop
Copy-paste this entire block:
```bash
for i in {1..8}; do
    echo "========================================"
    echo "Training DeepRC on ds$i..."
    echo "========================================"
    python deeprc/train_mil.py --dataset "ds$i" --epochs 20 --batch-size 4
done
```
*   **Time**: ~30-60 mins per dataset.
*   **Output**: Models saved to `models/deeprc/`.

---

## ⬇️ Step 5: Bring Models Home

Once training finishes, we need to save the brains (models) to your laptop.

**Open a NEW terminal on your Laptop:**
```bash
# Replace IP and PORT with your Pod's details
scp -P PORT -r root@IP:/workspace/airr_ml_project_template/models/deeprc models/
```

---

## 🛑 Step 6: Terminate (Save Money!)

1.  Go to RunPod Dashboard.
2.  Find your pod.
3.  Click the **Trash Can Icon (Terminate)**.
4.  Confirm.

**Warning**: If you just "Stop" it, you still pay for the 1TB storage! Terminate it completely when done.
