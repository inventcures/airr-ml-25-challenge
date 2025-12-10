#!/bin/bash

# Robust Training Script
# Usage: nohup ./scripts/train_robust.sh > training.log 2>&1 &

echo "🚀 Starting Robust Training Loop..."

mkdir -p models/deeprc

for i in {1..8}; do
    DATASET="ds$i"
    MODEL_FILE="models/deeprc/${DATASET}_deeprc_model.pth"
    
    if [ -f "$MODEL_FILE" ]; then
        echo "✅ Model for $DATASET already exists. Skipping."
        continue
    fi
    
    echo "----------------------------------------------------------------"
    echo "🧠 Training $DATASET..."
    echo "----------------------------------------------------------------"
    
    # Run training
    # If it crashes, the loop will stop (unless we add '|| true')
    # But since we have checkpointing in python, we can just re-run this script to resume!
    uv run python deeprc/train_mil.py --dataset "$DATASET" --epochs 20 --batch-size 4
    
    if [ $? -eq 0 ]; then
        echo "✅ Finished $DATASET"
    else
        echo "❌ Failed $DATASET"
        exit 1
    fi
done

echo "🎉 All training jobs completed!"
