#!/bin/bash

# ============================================
# RunPod → Backblaze B2 Transfer Script
# ============================================

# ----- CONFIGURE THESE -----
B2_KEY_ID="0e4a7fc50333"
B2_APP_KEY="0045f7221afdbceba6961f7880a73ac8b928359055"
B2_BUCKET="airr-data-processed"
SOURCE_DIR="/workspace/airr_ml_project_template/data/processed"
# ---------------------------

# Optional: subfolder in bucket (leave empty for root)
B2_PATH=""

# Create rclone config
mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf << EOF
[b2]
type = b2
account = ${B2_KEY_ID}
key = ${B2_APP_KEY}
EOF

# Build destination path
if [ -z "$B2_PATH" ]; then
    DEST="b2:${B2_BUCKET}"
else
    DEST="b2:${B2_BUCKET}/${B2_PATH}"
fi

echo "============================================"
echo "Starting transfer: ${SOURCE_DIR} → ${DEST}"
echo "============================================"

# Run transfer with optimized settings
rclone sync "$SOURCE_DIR" "$DEST" \
    --transfers 16 \
    --checkers 8 \
    --b2-chunk-size 64M \
    --buffer-size 16M \
    --fast-list \
    --progress \
    --stats 30s \
    --log-file=transfer_$(date +%Y%m%d_%H%M%S).log \
    --log-level INFO

echo "============================================"
echo "Transfer complete!"
echo "Log saved to transfer_*.log"
echo "============================================"