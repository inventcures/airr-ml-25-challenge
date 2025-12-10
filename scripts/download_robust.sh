#!/bin/bash

# Robust Download Script for RunPod
# Usage: nohup ./scripts/download_robust.sh > download.log 2>&1 &

echo "Starting robust download..."

# 1. Download Pickles
echo "Downloading Pickles..."
uv run python scripts/download_pickles.py
echo "Pickles downloaded."

# 2. Download Embeddings (Smart Mode)
echo "Starting Smart Download (skips existing files)..."

# Run the smart python script
# It handles retries and skipping internally
# -u: Unbuffered output so you see logs immediately
python3 -u scripts/download_smart.py

echo "SUCCESS: Smart Download finished!"
