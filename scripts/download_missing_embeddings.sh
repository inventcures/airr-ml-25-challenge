#!/bin/bash
set -e

# Usage: bash scripts/download_missing_embeddings.sh

echo "📦 Installing Modal (if needed)..."
# Check if modal is installed in the current uv environment
if ! uv run python -c "import modal" &> /dev/null; then
    uv pip install modal
fi

echo "🔑 Check Auth..."
if ! uv run modal volume list > /dev/null 2>&1; then
    echo "❌ You are not authenticated with Modal."
    echo "   Run: uv run modal token set --token-id ... --token-secret ..."
    exit 1
fi

echo "⬇️  Downloading missing chunks..."

# Ensure parent dirs exist
mkdir -p data/embeddings/ds7/test
mkdir -p data/embeddings/ds8/test

# DS7
echo "   Fetching ds7/test/1_test..."
uv run modal volume get airr-ml-25-data-35m embeddings_35m/ds7/test/1_test data/embeddings/ds7/test/1_test --force

echo "   Fetching ds7/test/2_test..."
uv run modal volume get airr-ml-25-data-35m embeddings_35m/ds7/test/2_test data/embeddings/ds7/test/2_test --force

# DS8
echo "   Fetching ds8/test/1_test..."
uv run modal volume get airr-ml-25-data-35m embeddings_35m/ds8/test/1_test data/embeddings/ds8/test/1_test --force

echo "   Fetching ds8/test/2_test..."
uv run modal volume get airr-ml-25-data-35m embeddings_35m/ds8/test/2_test data/embeddings/ds8/test/2_test --force

echo "   Fetching ds8/test/3_test..."
uv run modal volume get airr-ml-25-data-35m embeddings_35m/ds8/test/3_test data/embeddings/ds8/test/3_test --force

echo "✅ Download Complete."
