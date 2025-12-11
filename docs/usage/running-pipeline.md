# Running the Pipeline

This guide details the step-by-step execution of the pipeline. You can run steps individually or use the orchestrator script.

## The Orchestrator

The easiest way to run everything (mostly locally) is:

```bash
./run_all.sh
```

However, steps involving **Heavy GPU** usage (DeepRC) are often run manually on cloud instances.

---

## Step-by-Step Breakdown

### 1. Data Preprocessing
Standardizes raw pickles into a consistent format.
```bash
python -m data.load_all_datasets
```
- **Input**: Raw `data/` directory.
- **Output**: `data/processed/*.pkl`.

### 2. Mal-ID Model 1: Stats
Extracts V/J usage statistics and trains a baseline model.
```bash
python -m malid.train_stats_all
```
- **Output**: `outputs/stats_preds/*.csv`.

### 3. ESM Embeddings (Remote)
Generates embeddings using Modal.
```bash
modal run modal/embed_esm650m.py
```
- **Note**: This runs on the cloud and syncs results back to `data/embeddings/`.

### 4. Mal-ID Model 3: ESM Sequence Classifier
Trains a classifier on the generated ESM embeddings.
```bash
python -m malid.train_esm_seq_all
```
- **Output**: `outputs/esm_seq_preds/*.csv`.

### 5. DeepRC: 5-Fold CV (The Heavy Lifter)
Trains the Multiple Instance Learning model. **Run this on RunPod.**

**Training:**
```bash
for i in {1..8}; do
    uv run deeprc/train_mil_cv.py --dataset "ds${i}" --epochs 20 --folds 5
done
```

**Inference:**
```bash
for i in {1..8}; do
    uv run deeprc/infer_mil_cv.py --dataset "ds${i}" --folds 5
done
```
- **Output**: `outputs/deeprc_cv_preds/*.csv`.
- **Action**: Transfer these files back to your local machine (`git push`/`pull` or `scp`).

### 6. Meta-Ensemble Training
Combines predictions from Stats, ESM, and DeepRC (CV) models.
```bash
python -m malid.train_meta_and_predict
```
- **Output**: `outputs/submission/submission.csv` (Task 1).

### 7. Task 2 Banking
Generates sequence rankings for the top-k sequences.
```bash
python -m scripts.rank_sequences_task2_all
```
- **Output**: `outputs/task2_ranking/*.csv`.

### 8. Build Final Submission
Aggregates Task 1 and Task 2 results into the Kaggle-ready format.
```bash
python -m scripts.build_submission
```
- **Final Output**: `submission.csv`.
