# AIRR ML Pipeline Documentation

Welcome to the **AIRR ML Pipeline** documentation! This project implements a cutting-edge Machine Learning pipeline for satisfying the **Adaptive Immune Profiling Challenge 2025**. It leverages Multiple Instance Learning (DeepRC) and Large Language Models (ESM-2) to classify immune repertoires.

## 🚀 Quick Links

??? tip "Quick Start"
    Run the full pipeline with a single command:
    ```bash
    ./run_all.sh
    ```
    See [Running the Pipeline](usage/running-pipeline.md) for details.

<div class="grid cards" markdown>

-   :material-rocket-launch: __Getting Started__
    -   [Installation](getting-started/installation.md)
    -   [Cloud Setup (Modal/RunPod)](getting-started/cloud-setup.md)

-   :material-console: __User Guide__
    -   [Pipeline Workflow](usage/running-pipeline.md)
    -   [CLI Reference](usage/cli-reference.md)

-   :material-graph: __Architecture__
    -   [DeepRC (MIL) Details](architecture/deeprc.md)
    -   [ESM Sequence Embeddings](architecture/esm.md)

-   :material-hand-shake: __Community__
    -   [Contribute](contributing.md)

</div>

## 🏗 System Architecture

The pipeline consists of three main stages: Data Processing, Feature Extraction (DeepRC & ESM), and Meta-Ensembling.

```mermaid
graph TD
    Data[Raw Data (Pickles)] --> Processed[Processed Data]
    
    subgraph "Feature Extraction"
        Processed --> DeepRC[DeepRC (MIL Model)]
        Processed --> ESM[ESM-2 (Seq Embeddings)]
        Processed --> Stats[Statistical Features]
    end
    
    subgraph "Level 1 Predictions"
        DeepRC --> P_DeepRC[Probabilities (DeepRC)]
        ESM --> P_ESM[Probabilities (ESM)]
        Stats --> P_Stats[Probabilities (Stats)]
    end
    
    subgraph "Meta Ensemble"
        P_DeepRC & P_ESM & P_Stats --> Meta[Logistic Regression Stacking]
        Meta --> Final[Final Submission]
    end
```

## 🎯 Task Objectives

1.  **Task 1 (Repertoire Classification)**: Predict the disease status (0/1) for a patient based on their entire VDJ repertoire.
2.  **Task 2 (Sequence Ranking)**: Identify the specific associated sequences that are driving the positive classification.
