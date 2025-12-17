#!/bin/bash
set -u # Exit on undefined variables

# --- CONFIGURATION ---
LOG_FILE="logs/merge_650m_embeddings.log"
VERIFY_SCRIPT="scripts/verify_embeddings_650m.py"
DATA_DIR="/workspace/airr-ml-25-challenge/data/embeddings"

# --- TRAP SIGNALS ---
trap 'echo ""; echo "❌ SCRIPT INTERRUPTED BY USER."; exit 1' SIGINT
trap 'echo "❌ ERROR OCCURRED AT LINE $LINENO"; exit 1' ERR

# --- SETUP LOGGING ---
mkdir -p logs
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "🛡️  HARDENED MERGE SCRIPT STARTING: $(date)"
echo "========================================================"
echo "Working Directory: $DATA_DIR"

# Check Dependencies
command -v rsync >/dev/null 2>&1 || { echo "❌ rsync not found. Run 'apt-get install -y rsync'"; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo "❌ unzip not found. Run 'apt-get install -y unzip'"; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "❌ uv not found. Install uv first."; exit 1; }

# Check Disk Space (Warn if < 10GB free, just in case)
# ... omit complexity, assume handled by user, focus on robust logic

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data dir $DATA_DIR does not exist."
    exit 1
fi

cd "$DATA_DIR" || exit 1

# --- HELPER FUNCTIONS ---

print_progress() {
    echo "⏳ $1..."
}

robust_merge() {
    local src="$1"
    local dest="$2"
    
    if [ -d "$src" ]; then
        echo "  🔄 Merging $src -> $dest"
        mkdir -p "$dest"
        
        # rsync options:
        # -a: archive mode (recursive, preserve perms)
        # --remove-source-files: deletes from src AFTER successful transfer (Safe for resume)
        # --info=progress2: Global progress bar
        # --no-inc-recursive: Better progress estimation
        
        rsync -a --remove-source-files --info=progress2 --no-inc-recursive "$src" "$dest"
        
        # Verify empty before rmdir
        if [ -z "$(ls -A $src)" ]; then
             rmdir "$src"
             echo "     ✅ Cleaned up $src"
        else
             echo "     ⚠️  Warning: $src not empty after merge. Hidden files?"
        fi
    else
        echo "     ℹ️  Skip: $src not found (Already merged?)"
    fi
}

unzip_robust() {
    local zipfile="$1"
    if [ -f "$zipfile" ]; then
        echo "--------------------------------------------------------"
        echo "📦 Unzipping $zipfile"
        echo "--------------------------------------------------------"
        # -o: Overwrite existing (Standard for resume/fix)
        unzip -o "$zipfile" | awk 'BEGIN {ORS="."} {if(NR%1000==0) print "."}' 
        echo ""
        echo "✅ Unzip Complete."
    else
        echo "⚠️  File not found: $zipfile (Skipping)"
    fi
}

# --- STEP 1: UNZIP ---
echo ""
echo ">>> STEP 1: UNZIPPING (Overwrite Mode - Safe to Resume)"
unzip_robust "pod2_results_ds8_test_1_2.zip"
unzip_robust "pod3_results.zip"
unzip_robust "pod4_results_ds8_train_shard0.zip"
unzip_robust "pod5_results_ds7_train_shard1.zip"

# --- STEP 2: MERGE LOGIC ---
echo ""
echo ">>> STEP 2: FIXING DIRECTORY STRUCTURE"

# DS8 Test Merges
echo "--- Fixing DS8 Test ---"
robust_merge "ds8_1/test/" "ds8/test/1_test/"
robust_merge "ds8_2/test/" "ds8/test/2_test/"
robust_merge "ds8_3/test/" "ds8/test/3_test/"

# DS7 Test Merges
echo "--- Fixing DS7 Test ---"
robust_merge "ds7_1/test/" "ds7/test/1_test/"
robust_merge "ds7_2/test/" "ds7/test/2_test/"

# --- STEP 3: FINAL CLEANUP ---
echo ""
echo ">>> STEP 3: FINAL CLEANUP"
rmdir ds7_1 ds7_2 ds8_1 ds8_2 ds8_3 2>/dev/null || true
# Force remove ds8_3 only if empty
if [ -d "ds8_3" ]; then
    rmdir ds8_3 2>/dev/null || echo "Info: ds8_3 not removed (not empty?)"
fi

# --- STEP 4: VERIFICATION ---
echo ""
echo ">>> STEP 4: VERIFICATION"
echo "Running sanity check..."
cd /workspace/airr-ml-25-challenge || exit 1
uv run "$VERIFY_SCRIPT"

echo ""
echo "========================================================"
echo "✅ SUCCESS: All Operations Complete."
echo "📜 Log saved to: $LOG_FILE"
echo "========================================================"
