# DeepRC (Multiple Instance Learning)

**DeepRC** is a specialized architecture for immune repertoire classification. It treats the problem as a Multiple Instance Learning (MIL) task.

## The MIL Concept
-   **Bag**: The Patient (Repertoire).
-   **Instance**: The Sequence (TCR/BCR CDR3).
-   **Label**: Disease Status (Binary).
-   **Assumption**: A "Positive" bag contains at least one "Positive" instance (disease-associated sequence). A "Negative" bag contains only negative instances.

## Model Components

### 1. Sequence Encoder (1D CNN)
Each amino acid sequence is one-hot encoded (or embedded) and passed through 1D Convolutional layers.
-   **Input**: `(Batch, 21, Length)`
-   **Output**: Feature vector per sequence `(Batch, Hidden_Dim)`.

### 2. Attention Mechanism (The "Pooling" Layer)
This is the core of DeepRC. Instead of simply averaging all sequence vectors (which dilutes the signal of the rare disease sequences), the network learns to assign **Attention Weights**.

$$ A = \text{softmax}(w^T \cdot \tanh(V \cdot H^T)) $$

-   **High Weight**: The model "pays attention" to this sequence. It thinks this sequence is relevant to the label.
-   **Low Weight**: The model ignores this sequence (background noise).

### 3. Classification Head
The weighted sum of sequence vectors forms the **Repertoire Representation**. This is fed into a dense layer to produce the final probability.

## 5-Fold Cross-Validation
To prevent overfitting and generate robust OOF predictions for the meta-ensemble, we use:
-   **Splitting**: Stratified K-Fold (maintains disease ratio).
-   **Ensembling**: At test time, we average the logits of the 5 models.
