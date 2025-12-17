#!/bin/bash
set -e # Exit on error
set -u # Exit on undefined variables

# --- CONFIG ---
LOG_FILE="logs/merge_650m_embeddings.log"
VERIFY_SCRIPT="scripts/verify_embeddings_650m.py"
DATA_DIR="/workspace/airr-ml-25-challenge/data/embeddings"
mkdir -p logs

# Logging Setup: Redirect stdout/stderr to console AND log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "🚀 STARTING 650M EMBEDDING MERGE: $(date)"
echo "========================================================"
echo "Working Directory: $DATA_DIR"

cd "$DATA_DIR" || { echo "❌ Failed to cd to $DATA_DIR"; exit 1; }

# --- FUNCTION: ROBUST MERGE ---
robust_merge() {
    local src="$1"
    local dest="$2"
    if [ -d "$src" ]; then
        echo "  🔄 Merging $src -> $dest..."
        mkdir -p "$dest"
        # rsync -a --remove-source-files merges contents and deletes source files
        rsync -a --remove-source-files "$src" "$dest"
        # Clean up empty source dir
        rmdir "$src" 2>/dev/null || true
        echo "     ✅ Done."
    else
        echo "     ℹ️ Skip: $src not found (ok)"
    fi
}

echo ""
echo "--- 1️⃣ UNZIPPING FILES ---"

unzip_fast() {
    local zipfile="$1"
    if [ -f "$zipfile" ]; then
        echo "  📦 Unzipping $zipfile..."
        unzip -o -q "$zipfile"
        echo "     ✅ Unzipped."
    else
         echo "  ⚠️ Warning: $zipfile not found. Skipping."
    fi
}

unzip_fast "pod2_results_ds8_test_1_2.zip"
unzip_fast "pod3_results.zip"
unzip_fast "pod4_results_ds8_train_shard0.zip"
unzip_fast "pod5_results_ds7_train_shard1.zip"

echo ""
echo "--- 2️⃣ FIXING DIRECTORY STRUCTURE ---"

# Fix DS8 Test (Handle Pod 2's ds8_1/ds8_2 and Pod 3's ds8_3)
echo "🔍 Checking DS8 Test Splits..."
mkdir -p ds8/test

# DS8_1 -> 1_test
robust_merge "ds8_1/test/" "ds8/test/1_test/"

# DS8_2 -> 2_test
robust_merge "ds8_2/test/" "ds8/test/2_test/"

# DS8_3 -> 3_test
robust_merge "ds8_3/test/" "ds8/test/3_test/"

# Fix DS7 Test
echo "🔍 Checking DS7 Test Splits..."
mkdir -p ds7/test

# DS7_1 -> 1_test
robust_merge "ds7_1/test/" "ds7/test/1_test/"

# DS7_2 -> 2_test
robust_merge "ds7_2/test/" "ds7/test/2_test/"

# Cleanup
echo ""
echo "--- 3️⃣ CLEANUP ---"
echo "  🧹 Removing empty shells..."
rmdir ds7_1 ds7_2 ds8_1 ds8_2 ds8_3 2>/dev/null || true
rm -rf ds8_3 # Force remove if rsync left empty dirs (safety)

echo ""
echo "========================================================"
echo "✅ MERGE COMPLETE"
echo "========================================================"
echo ""
echo "--- 4️⃣ RUNNING VERIFICATION SCRIPT ---"
echo "Running: uv run $VERIFY_SCRIPT"

# Go back to project root to run python script
cd /workspace/airr-ml-25-challenge

if [ -f "$VERIFY_SCRIPT" ]; then
    uv run "$VERIFY_SCRIPT"
else
     echo "❌ Verification script $VERIFY_SCRIPT not found!"
fi

echo ""
echo "📜 LOG SAVED TO: $LOG_FILE"
