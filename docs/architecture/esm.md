# ESM (Evolutionary Scale Modeling)

We utilize **ESM-2 (650M parameters)**, a state-of-the-art protein language model developed by Meta AI.

## Why Transformers?
Unlike simple 1D CNNs, Transformers capture **global context** and **long-range dependencies** within a sequence. They understand that an amino acid at position 1 might interact with position 15 due to protein folding properties.

## Pipeline Usage

### 1. Generating Embeddings
We run the 650M model on every unique CDR3 sequence in the dataset.
-   **Layer**: Last hidden layer.
-   **Token Strategy**: Mean pooling of all tokens (CLS token is alternative).
-   **Output**: A dense vector (size 1280) representing the sequence semantics.

### 2. Sequence Classifier (Mal-ID Model 3)
We train a lightweight classifier (Random Forest / MLP) on top of these 1280-dim vectors.
-   **Input**: `(Batch, 1280)`
-   **Label**: Use the repertoire label (weakly supervised) â€“ we assume all sequences in a positive bag are "positive" initially, then refine. (Note: This is a simplification; in practice, we aggregate scores).

### 3. Repertoire Score
To get a score for the patient, we aggregate the predictions of their top sequences.
-   **Max/Mean Pooling**: We take the max probability or the mean of the top-k highest scoring sequences.
