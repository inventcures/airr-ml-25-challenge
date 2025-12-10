#!/bin/bash

echo "📊 DeepRC Training Monitor"
echo "Press Ctrl+C to exit."
echo "--------------------------------"

# Check if training is running
if pgrep -f "train_mil.py" > /dev/null; then
    echo "🟢 Training is ACTIVE."
else
    echo "🔴 Training is NOT running."
fi

echo "--------------------------------"
echo "Latest logs (tailing logs/deeprc_train_*.log):"
echo "Waiting for activity..."

# Tail all deeprc logs, showing new lines as they appear
tail -f logs/deeprc_train_*.log 2>/dev/null
