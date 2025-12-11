# Architecture Overview

The AIRR ML Pipeline uses a **Meta-Ensembling** strategy, often called "Stacking". Instead of relying on a single model, we train multiple distinct models that capture different biological signals and combine their outputs.

## The Three Pillars

### 1. Global Statistics (Small & Fast)
-   **Concept**: Immune repertoires have structural biases (e.g., V-gene usage distribution).
-   **Model**: Simple Logistic Regression or Random Forest on V/J gene counts.
-   **Strength**: Very fast, captures high-level population shifts.

### 2. Sequence Embeddings (Local & Semantic)
-   **Concept**: Specific amino acid patterns (motifs) drive binding.
-   **Model**: **ESM-2 (650M)**. We classify individual sequences and aggregate their scores.
-   **Strength**: Captures biochemical properties and evolutionary context.
-   **Detail**: See [ESM Architecture](esm.md).

### 3. Multiple Instance Learning (Holistic & Deep)
-   **Concept**: A repertoire is a "bag" of sequences. Only *some* sequences (witnesses) indicate disease.
-   **Model**: **DeepRC** with Attention-based pooling.
-   **Strength**: End-to-end learning on the full repertoire without manual aggregation.
-   **Detail**: See [DeepRC Architecture](deeprc.md).

## The Meta-Learner

The outputs of pillars 1, 2, and 3 (probabilities) become the **inputs** for the Meta-Learner.

-   **Algorithm**: Logistic Regression.
-   **Training Data**: Generated via **Out-Of-Fold (OOF)** predictions on the training set.
-   **Why?**: The meta-learner learns to trust the model that is most confident for a given sample, calibrating the final probability.
