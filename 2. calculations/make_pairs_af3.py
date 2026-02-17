"""
this script joins (merges):
1. af3_summary.csv (AF3 outputs per PDB folder)
+
2. your sample sheet for VH/VL chains
+
3. ref_map.csv for the truth structure path
"""
import pandas as pd
from pathlib import Path
AF3_SUMMARY = Path("af3_summary.csv")
SAMPLESHEET = Path("pairs_qc_pass_ultrashort_pruned.csv")
REF_MAP = Path("ref_map.csv")
AF3_TARGETS = Path("af3_export/af3_targets.csv")
OUT = Path("pairs_af3.csv")

def norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def main():
    af3 = pd.read_csv(AF3_SUMMARY)
    ss = pd.read_csv(SAMPLESHEET)
    rm = pd.read_csv(REF_MAP)
    tgt = pd.read_csv(AF3_TARGETS)

    # normalizing ids
    af3["af3_id"] = norm(af3["pdb_id"]) # AF3 folder id
    af3["base_pdb_id"] = af3["af3_id"].str.split("_").str[0]

    ss["pdb_id_norm"] = norm(ss["pdb_id"])
    rm["id_norm"] = norm(rm["id"])

    tgt["label_norm"] = norm(tgt["label"]) # IGAB label like 7SL5_AB
    tgt["base_pdb_id"] = norm(tgt["pdb_id"]) # base PDB ID like 7SL5
    tgt = tgt[["label_norm", "base_pdb_id", "reason"]].drop_duplicates()

    # ensuring one VH/VL pair per PDB ID in samplesheet
    ss_pairs = (
        ss[["pdb_id_norm", "vh_chain_id", "vl_chain_id"]]
        .dropna(subset=["pdb_id_norm", "vh_chain_id", "vl_chain_id"])
        .drop_duplicates(subset=["pdb_id_norm"], keep="first")
        .reset_index(drop=True)
    )

    # merge
    d = af3.merge(rm[["id_norm", "ref_pdb"]],
                  left_on="base_pdb_id", right_on="id_norm", how="left")
    d = d.merge(ss_pairs, left_on="base_pdb_id", right_on="pdb_id_norm", how="left")
    d = d.merge(tgt, on="base_pdb_id", how="left")

    out = pd.DataFrame({
        "id": d["af3_id"],
        "base_pdb_id": d["base_pdb_id"],
        "label_igab": d["label_norm"],
        "reason": d["reason"],
        "method": "AlphaFold3",
        "label": d["best_model"].apply(
            lambda x: f"af3_best_model_{int(x)}" if pd.notna(x) else "af3_best_model_na"
        ),
        "ref_pdb": d["ref_pdb"],
        "pred_pdb": d["model_cif"],
        "vh": d["vh_chain_id"],
        "vl": d["vl_chain_id"],
    })

    out = out.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
    out.to_csv(OUT, index=False)

    print(f"Wrote: {OUT} (rows={len(out)})")
    print("Missing label_igab:", int(out["label_igab"].isna().sum()))
    print("Missing reason:", int(out["reason"].isna().sum()))
    print("Missing ref_pdb:", int(out["ref_pdb"].isna().sum()))
    print("Missing vh:", int(out["vh"].isna().sum()))
    print("Missing vl:", int(out["vl"].isna().sum()))

if __name__ == "__main__":
    main()

"""
Reference:
[1] AlphaFold 3 (deep-learning model for biomolecular complex structure prediction)
Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A., Ronneberger, O., et al. 2024. Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature 630(8016) (2024), 493–500.
https://doi.org/10.1038/s41586-024-07487-w
"""