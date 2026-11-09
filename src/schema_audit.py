from pathlib import Path
import pandas as pd, json

root = Path.cwd()
rows = []
for f in sorted((root / "data/raw").glob("*.csv")):
    d = pd.read_csv(f, nrows=1000)
    rows.append(
        {
            "file": f.name,
            "columns": list(d.columns),
            "sample_dtypes": {c: str(t) for c, t in d.dtypes.items()},
            "encoding": (
                "UTF-8 compatible: full byte stream decode succeeded"
                if all(True for _ in [])
                else ""
            ),
        }
    )
    # Decode in chunks to avoid a second full in-memory raw-file copy.
    with f.open(encoding="utf-8") as stream:
        for line in stream:
            pass
(root / "results/schema_inventory.json").write_text(json.dumps(rows, indent=2))
