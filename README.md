# AIRR-ML-25 Kaggle Pipeline Template

This repository is a **template project** for the AIRR-ML-25 / Adaptive Immune Profiling Challenge 2025.

It implements, in a modular way, a multi-model ensemble inspired by:

- **Mal-ID**: ensemble of global stats, clustering, and protein language model sequence classifiers.
- **DeepRC-style MIL**: attention-based multiple-instance learning over TCR sequences.
- **Meta-ensemble**: a logistic regression layer on top of model outputs.
- Full support for **Task 1 (repertoire-level classification)** and **Task 2 (sequence ranking)**.


---

## 1. Repository Layout

```text
project/
  data/
    load_data.py              # Load AIRR-format files and group into Repertoire objects
    load_all_datasets.py      # Load all train/test datasets and cache them as pickles

  malid/
    stats_model.py            # Global repertoire stats model (Mal-ID Model 1)
    train_stats_all.py        # Train stats model per dataset
    cluster_model.py          # FAISS + Louvain clustering + cluster enrichment (Mal-ID Model 2)
    run_clustering_all.py     # Run clustering on all train datasets, save models + sequence-cluster map
    esm_seq_model.py          # ESM2 sequence-level classifier + top-k aggregation (Mal-ID Model 3)
    train_esm_seq_all.py      # Train ESM seq classifier per dataset
    meta_ensemble.py          # Build meta-training tables, train global meta LR model, infer Task 1
    train_meta_and_predict.py # One-shot train meta + predict Task 1 for all test datasets

  deeprc/
    dataset.py                # PyTorch Dataset to load repertoire embeddings
    mil_model.py              # Attention-based MIL model (DeepRC-style)
    train_mil.py              # Train MIL per dataset (for use on a GPU box like RunPod)
    infer_mil_all.py          # Inference: compute p_deeprc and optional attention weights

  modal/
    embed_esm650m.py          # Modal app: parallel ESM2-650M embeddings over repertoires

  scripts/
    validate_components.py    # Compute AUROC per dataset per component + meta
    rank_sequences_task2_core.py # Build per-sequence table, combine scores, rank top 50k
    rank_sequences_task2_all.py  # Loop over all training datasets for Task 2
    build_submission.py       # Merge Task 1 + Task 2 into final Kaggle submission CSV

  run_all.sh                  # Orchestration script (uses checkpointing/idempotency)
  README.md                   # This file
```

---

## 2. High-Level Pipeline

Conceptually, the pipeline is:

1. **Data ingestion**
   - Read AIRR-format files (TSV/CSV).
   - Group per `(dataset, repertoire_id)` into `Repertoire` objects.
   - Cache as pickled Python objects in `data/processed/`.

2. **Mal-ID Model 1: Global Stats**
   - Extract repertoire-level features:
     - V gene usage
     - J gene usage
     - CDR3 length distribution
     - Amino acid composition
   - Train **L1-regularized logistic regression** per dataset.
   - Save:
     - Model, vocab
     - Training predictions `p_stats` per repertoire (for meta-ensemble).

3. **ESM2 Embeddings + Sequence Classifier (Mal-ID Model 3)**
   - Use **Modal** with GPU to embed each CDR3 with **ESM2-650M (t33)**.
   - Save `embeddings/embeddings/{dataset}/{rep_id}.npz` with `emb` array `[N, D]`.
   - Build sequence-level dataset:
     - Each embedding inherits the repertoire label (weak supervision).
   - Train a sequence-level classifier (e.g. LogisticRegression) per dataset.
   - For each repertoire:
     - Compute per-sequence probabilities.
     - Aggregate via **top-k mean** to get `p_esm`.

4. **Mal-ID Model 2: Clustering**
   - Subsample up to `max_seqs_per_rep` sequences from each repertoire.
   - Build a FAISS L2 index on all embeddings.
   - Run kNN search to build a graph.
   - Run **Louvain clustering** to get clusters of similar sequences.
   - Compute **cluster enrichment** for case vs control.
   - Select top enriched clusters and build repertoire-level cluster features.
   - Train logistic regression on features, giving `p_clusters`.
   - Save a **sequence-cluster map**: for each sequence, `(cluster_id, rep_id, local_index)`.

5. **DeepRC-style MIL**
   - Use ESM embeddings directly as sequence embeddings.
   - Train an **attention-based MIL model**:
     - Inputs: sequence embeddings `[N, D]`.
     - Attention network outputs weights `[N]`.
     - Repertoire embedding = weighted sum of sequence embeddings.
     - Final classifier: small MLP over repertoire embedding → `p_deeprc`.
   - Run inference to compute:
     - Repertoire-level `p_deeprc`.
     - Optional per-sequence attention weights (for Task 2).

6. **Meta-Ensemble (Task 1)**
   - Build a global table over all train datasets with:
     - `p_stats`
     - `p_clusters`
     - `p_esm`
     - `p_deeprc`
     - Label (0/1)
   - Train a single **logistic regression** meta-model across all datasets.
   - Apply to all test repertoires to produce:
     - `outputs/task1_probs/task1_all_test.csv`:
       - `ID, dataset, label_positive_probability`.

7. **Task 2 (Sequence Ranking)**
   - Build a per-sequence table that combines:
     - ESM sequence probability `p_seq_esm`
     - DeepRC attention `attn`
     - Cluster enrichment `cluster_score`
   - Compute a combined score:
     - `score = w_attn * attn + w_esm * p_seq_esm + w_cluster * cluster_score`
   - Rank sequences per training dataset and take top 50k.
   - Save per-dataset files under `outputs/task2_top/`.

8. **Final Submission**
   - Task 1 block: test repertoires + probabilities, AIRR fields as `-999.0`.
   - Task 2 block: top 50k sequences per train dataset with:
     - `ID, dataset, label_positive_probability (-999.0), junction_aa, v_call, j_call`
   - Concatenate into `submission.csv`.

---

## 3. Prerequisites

### Python & packages

Use Python 3.10+ if possible. Create a virtual environment and install:

```bash
pip install torch fair-esm faiss-cpu igraph python-igraph scipy scikit-learn pandas numpy modal-client joblib
```

(You may need system packages for igraph on some OSes.)

### Modal

- Create a Modal account.
- Install the CLI: `pip install modal-client`
- Run `modal token new` to log in.
- Ensure the ESM embeddings script points to the correct image and GPU type available in your account.

### RunPod (or other GPU provider for DeepRC)

- Spin up a GPU pod with CUDA + PyTorch.
- Mount / sync this repo.
- Run the DeepRC training script there (`deeprc/train_mil.py`).

---

## 4. How to Use This Template

1. **Unzip the project** and `cd` into `project/`.
2. Replace each stub Python file with the full content from your ChatGPT session.
3. Place all raw AIRR files into `data/raw/`, and update file names in `data/load_all_datasets.py` to match:
   - `TRAIN_DATASETS = { "ds1": "train_dataset_1.tsv", ... }`
   - `TEST_DATASETS = { "ds9": "test_dataset_9.tsv", ... }`
4. Run the full pipeline:

   ```bash
   chmod +x run_all.sh
   ./run_all.sh
   ```

   Steps that depend on external services (Modal, RunPod) are clearly marked.

5. After everything completes, the final Kaggle submission file will be:

   ```text
   submission.csv
   ```

---

## 5. Checkpointing & Resilience

Every major script is designed to be **idempotent** and **resume-friendly**:

- If an output file (model, predictions, sequence map, submission) already exists:
  - The script **skips recomputation** for that part.
- Per-dataset loops are wrapped in `try/except`:
  - A failure on one dataset does **not** kill the entire run.
- Task 2 ranking and meta-ensemble training:
  - Check existing outputs before running.
- The orchestration script `run_all.sh`:
  - Can be run multiple times; already-done steps will be mostly skipped.

If a step fails (e.g., network errors with Modal, GPU pre-emption on RunPod), fix the issue and re-run:

```bash
./run_all.sh
```

It will pick up from where artifacts are missing.

---

## 6. Customization Tips

- **Feature engineering**:
  - You can expand stats features (e.g., k-mer distributions, V-J pairing frequencies).
  - Adjust ESM embedding layer, top-k aggregation strategy, or sequence-level classifier.

- **Model hyperparameters**:
  - `ClusterConfig` (kNN k, min_cluster_size, top_clusters).
  - `Task2Config` (weights for attn, ESM, cluster scores).
  - DeepRC embedding dimension, hidden sizes, dropout, max sequences per repertoire.

- **Meta model**:
  - We use logistic regression for interpretability.
  - You can swap for an MLP or gradient boosting model if desired.

---

## 7. Next Steps

1. Paste the full script code into each stub file.
2. Test each component on a **single dataset** first:
   - `python -m malid.train_stats_all`
   - `python -m malid.train_esm_seq_all`
   - `python -m malid.run_clustering_all`
   - `python -m deeprc.train_mil --dataset ds1 ...`
3. Use `python -m scripts.validate_components` to see AUROC per component.
4. Iterate on:
   - hyperparameters,
   - data preprocessing,
   - ensemble weights / meta model,
   - Task 2 scoring weights.

This template gives you a **solid, competition-grade starting point** you can extend and tune for the AIRR-ML-25 challenge.
