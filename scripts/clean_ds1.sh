#!/bin/bash

echo "🧹 Cleaning up local ds1 directory..."
# Remove the entire ds1 directory to clear loose files and bad structure
rm -rf data/embeddings/ds1

echo "✨ ds1 wiped clean."

echo "🚀 Restarting robust download to fetch ds1 correctly..."
# The robust script calls download_smart.py, which will see ds1 is missing and download it from the correct remote subfolders
nohup ./scripts/download_robust.sh > download.log 2>&1 &

echo "✅ Download restarted. Monitor with: tail -f download.log"
