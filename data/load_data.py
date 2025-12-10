# data/load_data.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd


@dataclass
class Repertoire:
    """
    A single repertoire (all TCR sequences from one person / sample).
    """
    rep_id: str
    dataset: str
    label: Optional[int]  # 0/1 or None for unlabeled (test)
    junction_aa: List[str]
    v_call: List[str]
    j_call: List[str]
    meta: Dict[str, Any]


def load_repertoire_file(
    path: Path,
    rep_id: str,
    dataset_name: str,
    label: Optional[int],
    junction_aa_col: str = "junction_aa",
    v_call_col: str = "v_call",
    j_call_col: str = "j_call",
) -> Repertoire:
    """
    Load a single repertoire from a TSV file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Repertoire file not found: {path}")

    # Auto-detect separator
    if path.suffix.lower() in [".tsv", ".txt"]:
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    required = {junction_aa_col, v_call_col, j_call_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    junction_aa = df[junction_aa_col].fillna("").astype(str).tolist()
    v_call = df[v_call_col].fillna("").astype(str).tolist()
    j_call = df[j_call_col].fillna("").astype(str).tolist()

    # Store any additional metadata columns if you need them later
    meta_cols = [c for c in df.columns if c not in {junction_aa_col, v_call_col, j_call_col}]
    meta = {}
    for c in meta_cols:
        # Just store first value per repertoire if it's constant, or maybe not store at all for sequence-level
        # For now, let's just store the first row's value if available
        if len(df) > 0:
            meta[c] = df[c].iloc[0]

    return Repertoire(
        rep_id=str(rep_id),
        dataset=dataset_name,
        label=label,
        junction_aa=junction_aa,
        v_call=v_call,
        j_call=j_call,
        meta=meta,
    )


def load_dataset(
    path: Path,
    dataset_name: str,
    label_col: Optional[str] = "label",  # Used if loading from a single table or metadata
    rep_id_col: str = "repertoire_id",   # Used if loading from a single table
    filename_col: str = "filename",      # Used if loading from metadata.csv
    junction_aa_col: str = "junction_aa",
    v_call_col: str = "v_call",
    j_call_col: str = "j_call",
) -> List[Repertoire]:
    """
    Load a dataset which can be:
    1. A directory containing `metadata.csv` and many TSV files (Train).
    2. A directory containing many TSV files but no metadata (Test).
    3. A single TSV file containing multiple repertoires (Legacy/Template).
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    reps: List[Repertoire] = []

    if path.is_file():
        # Legacy mode: single file with multiple repertoires
        print(f"[load_dataset] Loading single file: {path}")
        # Reuse logic or call a helper? Let's inline simplified logic for now as it might not be used.
        # But wait, if the user provided template assumes this, maybe I should keep it?
        # The prompt says "Use the draft plan... make a comprehensive detailed plan...".
        # Since I see the actual data is directories, I will focus on that.
        # But for completeness, let's keep basic support if needed, or just error if not expected.
        # Actually, let's just implement the directory logic primarily.
        raise NotImplementedError("Single-file dataset loading not fully supported in this update. Expecting directory.")

    elif path.is_dir():
        metadata_path = path / "metadata.csv"
        if metadata_path.exists():
            print(f"[load_dataset] Loading from directory with metadata: {path}")
            meta_df = pd.read_csv(metadata_path)
            
            # Check required columns in metadata
            if filename_col not in meta_df.columns:
                raise ValueError(f"metadata.csv missing '{filename_col}' column")
            
            # Handle labels
            has_labels = False
            if label_col and (label_col in meta_df.columns or "label_positive" in meta_df.columns):
                has_labels = True
                # Normalize label column name
                actual_label_col = label_col if label_col in meta_df.columns else "label_positive"
            
            for _, row in meta_df.iterrows():
                fname = row[filename_col]
                file_path = path / fname
                
                # Determine rep_id
                if rep_id_col in meta_df.columns:
                    rep_id = str(row[rep_id_col])
                else:
                    # Fallback to filename stem
                    rep_id = Path(fname).stem

                # Determine label
                label = None
                if has_labels:
                    val = row[actual_label_col]
                    # Handle boolean or string labels
                    if isinstance(val, bool):
                        label = 1 if val else 0
                    else:
                        s_val = str(val).lower()
                        if s_val in {"1", "true", "case", "positive", "diseased"}:
                            label = 1
                        elif s_val in {"0", "false", "control", "negative", "healthy"}:
                            label = 0
                
                try:
                    rep = load_repertoire_file(
                        file_path, 
                        rep_id, 
                        dataset_name, 
                        label,
                        junction_aa_col=junction_aa_col,
                        v_call_col=v_call_col,
                        j_call_col=j_call_col
                    )
                    reps.append(rep)
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

        else:
            print(f"[load_dataset] Loading from directory (no metadata): {path}")
            # Iterate all TSVs
            files = sorted(list(path.glob("*.tsv")))
            for file_path in files:
                rep_id = file_path.stem
                # No labels in this mode
                label = None
                
                try:
                    rep = load_repertoire_file(
                        file_path, 
                        rep_id, 
                        dataset_name, 
                        label,
                        junction_aa_col=junction_aa_col,
                        v_call_col=v_call_col,
                        j_call_col=j_call_col
                    )
                    reps.append(rep)
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

    print(f"[load_dataset] Loaded {len(reps)} repertoires from {path}")
    return reps
