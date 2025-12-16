#!/bin/bash

# 1. Install UV (Fast Python Manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. Install System Deps
apt update && apt install -y tree tmux zip unzip git

# 3. Install Python Libs (System-wide for simplicity in Pods)
uv pip install numpy pandas scikit-learn joblib tqdm faiss-cpu fair-esm python-igraph torch --system

# 4. Setup Repo
cd /workspace
if [ -d "airr-ml-25-challenge" ]; then
    echo "Repo exists, pulling latest..."
    cd airr-ml-25-challenge
    git pull
else
    git clone https://github.com/inventcures/airr-ml-25-challenge
    cd airr-ml-25-challenge
fi

# 5. WARNING: DATA CHECK
# Use checks if data/processed exists. If not, you might need to SCP it or run setup.
if [ ! -d "data/processed" ]; then
    echo "⚠️  WARNING: data/processed not found! You may need to transfer pickles or run setup."
fi

# 6. RUN POD 4 (DS8 TRAIN)
# --include ds8 : Only process user 8 datasets
# --only-split train : Only process the training split (94M sequences)
echo "🚀 Starting POD 4 (DS8 Train)..."
uv run python scripts/generate_embeddings_650m.py --include ds8 --only-split train
