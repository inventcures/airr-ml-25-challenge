# Installation

This guide covers setting up your local environment to run the AIRR ML Pipeline.

## Prerequisites

-   **Operating System**: macOS or Linux (Windows via WSL2).
-   **Python**: 3.9+.
-   **Package Manager**: `uv` (Recommended) or `pip`.
-   **Hardware**:
    -   **Heavy Lifting**: NVIDIA GPU (24GB+ VRAM recommended) for DeepRC/ESM training.
    -   **Local**: Mac (Apple Silicon) or standard CPU for Meta-Ensemble and Submission building.

## 1. Clone the Repository

```bash
git clone https://github.com/inventcures/airr-ml-25-challenge.git
cd airr_ml_project_template
```

## 2. Setup Virtual Environment

We recommend using `uv` for ultra-fast dependency management.

=== "Using uv (Recommended)"

    Process is automatic if you use `uv run`. To explicitly sync:
    ```bash
    # Install uv if needed
    pip install uv
    
    # Create venv and install dependencies
    uv sync
    ```

=== "Using pip"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## 3. Environment Variables

Create a `.env` file (if needed by specific extensions like Modal) or ensure your shell exports necessary keys.

```bash
# Example for Modal
export MODAL_TOKEN_ID="your_token_id"
export MODAL_TOKEN_SECRET="your_token_secret"
```

## 4. Verify Installation

Run a quick syntax check on key scripts:

```bash
uv run python -c "import torch; print(f'Torch: {torch.__version__}')"
```
