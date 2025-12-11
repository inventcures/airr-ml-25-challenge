# Contributing

We welcome contributions! Please follow these guidelines to ensure smooth collaboration.

## Development Setup

1.  **Clone & Install**:
    ```bash
    git clone https://github.com/inventcures/airr-ml-25-challenge.git
    uv sync
    ```
2.  **Pre-commit Hooks**:
    We use pre-commit to ensure code quality (formatting, linting).
    ```bash
    pip install pre-commit
    pre-commit install
    ```

## Testing
Always run the validation suite before pushing.

```bash
python scripts/validate_components.py
```
This checks that all model files and output CSVs are structurally correct.

## Documentation
If you change code, update the docs!
1.  **Serve locally**:
    ```bash
    pip install mkdocs-material
    mkdocs serve
    ```
2.  Open `http://127.0.0.1:8000` to preview changes.

## Pull Requests
-   Create a feature branch: `git checkout -b feature/my-new-feature`.
-   Commit changes: `git commit -m "Add feature X"`.
-   Push: `git push origin feature/my-new-feature`.
-   Open a PR on GitHub.
