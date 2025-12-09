#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.

echo "========== STEP 0: Preprocess data =========="
python -m data.load_all_datasets

echo "========== STEP 1: Train stats models (Mal-ID Model 1) =========="
python -m malid.train_stats_all

echo "========== STEP 2: Generate ESM2-650M embeddings with Modal =========="
echo ">> This step runs remotely via Modal. Make sure your Modal account is set up."
modal run modal/embed_esm650m.py || echo 'Modal step failed or skipped; ensure embeddings exist under ./embeddings/embeddings before continuing.'

echo "========== STEP 3: Train ESM sequence classifiers (Mal-ID Model 3) =========="
python -m malid.train_esm_seq_all

echo "========== STEP 4: Run clustering + cluster model (Mal-ID Model 2) =========="
python -m malid.run_clustering_all

echo "========== STEP 5: Train DeepRC MIL models on RunPod =========="
echo ">> This step is heavy and typically run on RunPod or another GPU box."
echo ">> Example:"
echo "   python -m deeprc.train_mil --dataset ds1 --epochs 20 --batch-size 4 --lr 1e-4"

echo "========== STEP 6: Run DeepRC MIL inference for train (p_deeprc) =========="
python -m deeprc.infer_mil_all

echo "========== STEP 7: Train meta-ensemble and predict Task 1 for test =========="
python -m malid.train_meta_and_predict

echo "========== STEP 8: Rank sequences for Task 2 (top 50k per training dataset) =========="
python -m scripts.rank_sequences_task2_all

echo "========== STEP 9: Validate all components (optional but recommended) =========="
python -m scripts.validate_components || echo '>> Validation reported issues but pipeline continues.'

echo "========== STEP 10: Build final Kaggle submission =========="
python -m scripts.build_submission

echo "========== DONE =========="
echo "Submission CSV is at: submission.csv"
