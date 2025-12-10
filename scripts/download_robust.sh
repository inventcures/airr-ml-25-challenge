#!/bin/bash

# Robust Download Script for RunPod
# Usage: nohup ./scripts/download_robust.sh > download.log 2>&1 &

echo "Starting robust download..."

# 1. Download Pickles (Skipped - already downloaded)
# echo "Downloading Pickles..."
# until uv run modal volume get airr-ml-25-data data/ data/processed/; do
#     echo "Pickle download failed. Retrying in 5 seconds..."
#     sleep 5
# done
# echo "Pickles downloaded."

# 2. Download Embeddings (Smart Mode)
echo "Starting Smart Download (skips existing files)..."

# Run the smart python script
# It handles retries and skipping internally
python3 scripts/download_smart.py

echo "SUCCESS: Smart Download finished!"
