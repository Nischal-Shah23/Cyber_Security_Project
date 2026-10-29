import pandas as pd

source = "data/raw/csic2010/csic_database.csv"
destination = "data/processed/csic2010_ready.csv"

df = pd.read_csv(source)

labels = df["Unnamed: 0"].map({
    "Normal": 0,
    "Anomalous": 1
})

if labels.isna().any():
    raise ValueError("Unexpected labels found in the source dataset.")

def build_request(row):
    method = str(row["Method"]) if pd.notna(row["Method"]) else "GET"
    url = str(row["URL"]) if pd.notna(row["URL"]) else "/"
    parts = [f"{method} {url} HTTP/1.1"]

    for col in row.index:
        if col in {"Unnamed: 0", "classification", "Method", "URL"}:
            continue
        if pd.notna(row[col]) and str(row[col]).strip():
            if col.lower() == "content":
                parts.append(f"Body: {row[col]}")
            else:
                parts.append(f"{col}: {row[col]}")

    return "\n".join(parts)

result = pd.DataFrame({
    "request": df.apply(build_request, axis=1),
    "label": labels.astype(int)
})

result.to_csv(destination, index=False)

print("Saved:", destination)
print("Rows:", len(result))
print("Label counts:")
print(result["label"].value_counts().sort_index())
print("Columns:", result.columns.tolist())
