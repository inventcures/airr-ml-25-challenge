# CLI Reference

Technical reference for the core scripts.

## `stats/sync_data_robust.py`

Downloads and synchronizes data from Modal volumes.

```bash
python scripts/sync_data_robust.py [OPTIONS]
```

| Option | Description |
| :--- | :--- |
| `None` | (No arguments) Runs the robust sync + nest-fix logic. |

---

## `deeprc/train_mil_cv.py`

Trains the DeepRC MIL model using 5-Fold Stratified Cross-Validation.

```bash
python -m deeprc.train_mil_cv --dataset DS1 [OPTIONS]
```

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | `str` | **Required** | Dataset ID (e.g., `ds1`, `ds7`). |
| `--folds` | `int` | `5` | Number of CV folds. |
| `--epochs` | `int` | `20` | Training epochs per fold. |
| `--batch-size` | `int` | `4` | Bag batch size. |
| `--lr` | `float` | `1e-4` | Learning rate. |
| `--max-seqs` | `int` | `10000` | Max sequences per repertoire (subsampling). |

---

## `deeprc/infer_mil_cv.py`

Runs inference using the ensemble of 5 trained models per dataset.

```bash
python -m deeprc.infer_mil_cv [OPTIONS]
```

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--datasets` | `list` | `All` | Specific datasets to infer (e.g. `ds1 ds2`). |
| `--folds` | `int` | `5` | Number of models to ensemble. |

---

## `malid/train_meta_and_predict.py`

Trains the logistic regression meta-ensemble and generating final probabilities.

**Usage**: `python -m malid.train_meta_and_predict` (No arguments)

**Inputs**:
- `outputs/deeprc_cv_preds/*_oof.csv` (for training)
- `outputs/deeprc_cv_preds/*_preds.csv` (for test)
- `outputs/stats_preds/*.csv`
- `outputs/esm_seq_preds/*.csv`
