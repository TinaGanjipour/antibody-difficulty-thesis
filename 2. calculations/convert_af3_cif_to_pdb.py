import pandas as pd
from pathlib import Path

IN = Path("pairs_af3.csv")
OUT = Path("pairs_af3_pdb.csv")

df = pd.read_csv(IN)

if "pred_pdb" not in df.columns:
    raise SystemExit("pairs_af3.csv missing pred_pdb column")

df["pred_pdb"] = (
    df["pred_pdb"]
    .str.replace("af3_output/", "af3_output_pdb/", regex=False)
    .str.replace(".cif", ".pdb", regex=False)
)

df.to_csv(OUT, index=False)
print(f"Wrote: {OUT}")