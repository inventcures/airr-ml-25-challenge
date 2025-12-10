#!/bin/bash
set -e

# Usage: bash scripts/restore_data_from_modal.sh

echo "📦 Installing Modal (if needed)..."
if ! command -v modal &> /dev/null; then
    pip install modal
fi

echo "🔑 Authentication Check"
# Try to list files to check auth
if ! modal volume list > /dev/null 2>&1; then
    echo "❌ You are not authenticated with Modal."
    echo "Please run the following command now (copy from https://modal.com/settings/tokens):"
    echo "  modal token set --token-id <YOUR_TOKEN_ID> --token-secret <YOUR_TOKEN_SECRET>"
    exit 1
fi

echo "📂 Creating local directories..."
mkdir -p data/processed
mkdir -p data/embeddings

# 1. Download Processed Pickles (Metadata/Sequences)
echo "⬇️  Downloading Processed Data (Pickles)..."
# Matches: uv run modal volume get airr-ml-25-data data/ data/processed/
modal volume get airr-ml-25-data data/ data/processed/ --force

# 2. Download Embeddings (The big volume)
echo "⬇️  Downloading Embeddings (This is large, please wait)..."
# Matches: uv run modal volume get airr-ml-25-data-35m embeddings_35m/ data/embeddings/
modal volume get airr-ml-25-data-35m embeddings_35m/ data/embeddings/ --force

echo "✅ Restoration Complete!"
echo "   Processed files: $(ls data/processed | wc -l)"
echo "   Embedding folders: $(ls data/embeddings | wc -l)"
