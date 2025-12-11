
# DeepRC Cross-Validation & Submission Workflow

This workflow describes the step-by-step process to run 5-Fold Cross-Validation for DeepRC, aggregate the results, and generate the final submission. The workload is split between **RunPod (GPU)** for training/inference and your **Local Machine (CPU)** for data assembly and submission.

---

## **Part 1: Running on RunPod (GPU)**

Since DeepRC training is computationally intensive, these steps must be run on your RunPod instance where the data is located.

### 1. Run 5-Fold CV Training (Sequentially for all Datasets)
This trains 5 models per dataset (one per fold) and generates the Out-Of-Fold (OOF) predictions essential for the meta-ensemble.

```bash
# Run this inside the 'airr_ml_project_template' directory on RunPod
for i in {1..8}; do
    echo "Starting training for ds${i}..."
    uv run deeprc/train_mil_cv.py --dataset "ds${i}" --epochs 20 --folds 5 --batch-size 4
done
```
**Output:** OOF predictions saved to `outputs/deeprc_cv_preds/{dataset}_oof.csv`.

### 2. Run CV Inference on Test Data
This uses the trained models to generate predictions for the test set. It uses an ensemble of the 5 models (Soft Voting).

```bash
for i in {1..8}; do
    echo "Running Inference for ds${i}..."
    uv run deeprc/infer_mil_cv.py --dataset "ds${i}" --folds 5 --batch-size 4
done
```
**Output:** Test predictions saved to `outputs/deeprc_cv_preds/{dataset}_test_deeprc_preds.csv`.

### 3. Run Downstream Models (ESM, Stats, Task 2)
Since we are on RunPod with access to the embeddings, we must run the other models now.

```bash
# 1. Stats Model
uv run python malid/train_stats_all.py

# 2. ESM Sequence Model
uv run python malid/train_esm_seq_all.py

# 3. Task 2 Sequence Ranking
uv run python scripts/rank_sequences_task2_all.py
```

### 4. Sync Results via Git (Easy Option)
Push the results to your git repository so you can easily pull them locally.

```bash
# Add the new predictions
# Add the new predictions
git add outputs/deeprc_cv_preds/*.csv
git add outputs/cluster_preds/*.csv
git add outputs/esm_seq_preds/*.csv
git add outputs/stats_preds/*.csv
git add outputs/task2_ranking/*.csv

# Commit and Push
git commit -m "Add DeepRC CV predictions from RunPod"
git push
```
*(Note: Ensure git is configured on RunPod with your credentials).*

---

## **Part 2: Running on Local Machine (CPU)**

These steps are lightweight and should be run on your local laptop to generate the final submission file.

### 4. Pull Results
Get the predictions you just pushed from RunPod.

```bash
# Pull changes from the repo
git pull
```

### 5. Train Meta-Ensemble & Predict
This script now points to the new `outputs/deeprc_cv_preds` directory. It trains the meta-learner using the OOF predictions and generates the final probabilities for the test set.

```bash
python -m malid.train_meta_and_predict
```

### 6. Rank Sequences (Task 2)
Generate the sequence rankings (top 50k sequences) for the second challenge task.

```bash
python -m scripts.rank_sequences_task2_all
```

### 7. Build Submission
Aggregates everything into the final `submission.csv` file.

```bash
python -m scripts.build_submission
```

---

**🎉 Done!** You now have a `submission.csv` ready for upload.
