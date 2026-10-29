"""Dataset ingestion for labeled CSVs and simple CSIC-style text files.

Supported CSV columns:
  request,label
or
  method,url,body,label

Supported text files:
  normal.txt / normalTraffic*.txt -> label 0
  attack.txt / anomalous*.txt / attack*.txt -> label 1

Text requests are split on blank lines. This is a convenience loader, not a
claim that every release of CSIC 2010 has the same layout. Inspect the source
files and verify labels before research use.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUEST_CANDIDATES = ("request", "raw_request", "payload", "text", "http_request")
LABEL_CANDIDATES = ("label", "class", "target", "is_attack")

def _label_to_int(value):
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    s = str(value).strip().lower()
    if s in {"0", "normal", "benign", "legitimate", "valid", "false"}:
        return 0
    if s in {"1", "attack", "anomaly", "malicious", "invalid", "true"}:
        return 1
    raise ValueError(f"Unknown label {value!r}; map labels explicitly to 0=normal, 1=attack.")

def _request_from_row(row):
    for c in REQUEST_CANDIDATES:
        if c in row.index and pd.notna(row[c]):
            return str(row[c])
    parts = []
    for c in ("method", "url", "query", "headers", "body"):
        if c in row.index and pd.notna(row[c]):
            parts.append(f"{c.upper()}: {row[c]}")
    if parts:
        return "\n".join(parts)
    raise ValueError("CSV needs a request/text/payload column or method/url/body columns.")

def load_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    lower = {str(c).lower(): c for c in raw.columns}
    label_col = next((lower[c] for c in LABEL_CANDIDATES if c in lower), None)
    if label_col is None:
        raise ValueError(f"{path}: no label column found. Expected one of {LABEL_CANDIDATES}.")
    records = []
    for _, row in raw.iterrows():
        records.append({"request": _request_from_row(row), "label": _label_to_int(row[label_col])})
    return pd.DataFrame(records)

def _split_requests(text: str):
    chunks = []
    current = []
    for line in text.splitlines():
        if line.strip():
            current.append(line.rstrip())
        elif current:
            chunks.append("\n".join(current))
            current = []
    if current:
        chunks.append("\n".join(current))
    return [x for x in chunks if x.strip()]

def load_text_directory(root: Path) -> pd.DataFrame:
    records = []
    for p in root.rglob("*.txt"):
        name = p.name.lower()
        if any(k in name for k in ("normal", "benign", "legitimate")):
            label = 0
        elif any(k in name for k in ("attack", "anomalous", "malicious", "invalid")):
            label = 1
        else:
            continue
        content = p.read_text(encoding="latin-1", errors="replace")
        for req in _split_requests(content):
            records.append({"request": req, "label": label, "source_file": p.name})
    if not records:
        raise FileNotFoundError(
            f"No recognized labeled .txt files under {root}. Provide a CSV with request,label "
            "or place normal/attack text files in data/raw after verifying their labels."
        )
    return pd.DataFrame(records)

def load_dataset(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {p.resolve()}")
    if p.is_file() and p.suffix.lower() == ".csv":
        df = load_csv(p)
    elif p.is_dir():
        csvs = list(p.glob("*.csv"))
        if csvs:
            # Prefer a single consolidated labeled CSV to avoid accidental file-level label mixing.
            if len(csvs) > 1:
                raise ValueError(f"Multiple CSVs found in {p}. Combine them with explicit labels or pass one CSV.")
            df = load_csv(csvs[0])
        else:
            df = load_text_directory(p)
    else:
        raise ValueError("Pass a labeled CSV or a directory containing supported labeled files.")
    df = df.dropna(subset=["request", "label"]).copy()
    df["request"] = df["request"].astype(str).str.slice(0, 20000)
    df["label"] = df["label"].astype(int)
    df = df[df["request"].str.strip().ne("")]
    if set(df["label"].unique()) != {0, 1}:
        raise ValueError("Dataset must contain both labels: 0=normal and 1=attack.")
    return df.reset_index(drop=True)
