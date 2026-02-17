from __future__ import annotations
import argparse
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ANARCI_OK = False
try:
    from anarci import anarci as anarci_fn
    ANARCI_OK = True
except Exception:
    try:
        from ANARCI import anarci as anarci_fn
        ANARCI_OK = True
    except Exception:
        ANARCI_OK = False

AA20 = set("ACDEFGHIKLMNPQRSTVWY")

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

def aa_composition(s: str) -> Dict[str, float]:
    s = (s or "").strip()
    n = len(s)
    if n == 0:
        return {f"frac_{a}": 0.0 for a in AA_ORDER}
    c = Counter(s)
    return {f"frac_{a}": float(c.get(a, 0) / n) for a in AA_ORDER}

KD = {
    "A": 1.8,  "C": 2.5,  "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5,  "K": -3.9, "L": 3.8,
    "M": 1.9,  "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2,  "W": -0.9, "Y": -1.3
}
HYDROPHOBIC = set("AILMFWVYC")

def parse_pdb_chain_sequences_ca(pdb_path: str) -> Dict[str, str]:
    AA3_TO_1 = {
        "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G",
        "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
        "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"
    }
    def aa3_to_1(resname: str) -> str:
        return AA3_TO_1.get(resname.strip().upper(), "X")

    chains: Dict[str, Dict[Tuple[int, str], str]] = {}
    seen = set()

    model_idx = 0
    in_model = False
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                model_idx += 1
                in_model = True
                if model_idx > 1:
                    break
                continue
            if line.startswith("ENDMDL") and in_model:
                break
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            alt = line[16]
            if alt not in (" ", "A"):
                continue

            ch = (line[21].strip() or "_")
            resname = line[17:20].strip()
            aa = aa3_to_1(resname)
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = (line[26].strip() or "")
            key = (ch, resseq, icode)
            if key in seen:
                continue
            seen.add(key)

            chains.setdefault(ch, {})[(resseq, icode)] = aa

    out = {}
    for ch, resmap in chains.items():
        seq = "".join(resmap[k] for k in sorted(resmap.keys(), key=lambda t: (t[0], t[1])))
        out[ch] = seq
    return out


def anarci_is_antibody_chain(seq: str) -> Optional[str]:
    if not ANARCI_OK:
        raise RuntimeError("ANARCI required")

    try:
        numbering, _ali, _hits = anarci_fn([("q", seq)], scheme="chothia", allow=["H"])
        if numbering and numbering[0]:
            return "H"
    except Exception:
        pass

    try:
        numbering, _ali, _hits = anarci_fn([("q", seq)], scheme="chothia", allow=["K", "L"])
        if numbering and numbering[0]:
            return "L"
    except Exception:
        pass

    return None

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    ent = 0.0
    for k, v in c.items():
        p = v / n
        ent -= p * math.log(p, 2)
    return ent

def glyco_motif_present(s: str) -> bool:
    for i in range(len(s) - 2):
        if s[i] == "N" and s[i+1] != "P" and s[i+2] in ("S", "T"):
            return True
    return False

def proline_patterns(s: str) -> Tuple[int, int]:
    runs = [len(m.group(0)) for m in re.finditer(r"P+", s)]
    return s.count("P"), (max(runs) if runs else 0)


def cys_patterns(s: str) -> Tuple[bool, bool]:
    has_c = "C" in s
    has_two = s.count("C") >= 2
    return has_c, has_two


def h3_charge(s: str, his_weight: float = 0.1) -> float:
    pos = s.count("K") + s.count("R") + his_weight * s.count("H")
    neg = s.count("D") + s.count("E")
    return float(pos - neg)


def kd_stats(s: str) -> Tuple[float, float]:
    if not s:
        return 0.0, 0.0
    vals = [KD.get(a, 0.0) for a in s]
    mean_kd = float(np.mean(vals)) if vals else 0.0
    frac_h = sum(1 for a in s if a in HYDROPHOBIC) / len(s)
    return mean_kd, float(frac_h)


def frac_set(s: str, chars: str) -> float:
    if not s:
        return 0.0
    return sum(1 for a in s if a in set(chars)) / len(s)


def dataset_kmer_rarity(seqs: List[str], k: int = 3) -> Dict[str, float]:
    km_counts = Counter()
    total = 0
    for s in seqs:
        if not s or len(s) < k:
            continue
        for i in range(len(s) - k + 1):
            km_counts[s[i:i+k]] += 1
            total += 1
    if total == 0:
        return {s: 0.0 for s in seqs}

    km_freq = {km: c / total for km, c in km_counts.items()}

    out = {}
    for s in seqs:
        if not s or len(s) < k:
            out[s] = 0.0
            continue
        vals = []
        for i in range(len(s) - k + 1):
            f = km_freq.get(s[i:i+k], 1e-12)
            vals.append(-math.log10(f))
        out[s] = float(np.mean(vals)) if vals else 0.0
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="RMSD results CSV (e.g., results_backbone_fvnoh3fit.csv)")
    ap.add_argument("--out_csv", default="h3_features.csv")
    ap.add_argument("--plots_dir", default="h3_plots")
    ap.add_argument("--his_weight", type=float, default=0.1)
    ap.add_argument("--rank_by", choices=["h3_rmsd", "rmsd_fv_seqmap"], default="h3_rmsd",
                    help="How to define 'best{N}' and 'worst{N}'")
    args = ap.parse_args()

    if not ANARCI_OK:
        raise SystemExit("ANARCI ERROR")

    df = pd.read_csv(args.in_csv)
    if "h3_seq" not in df.columns:
        raise SystemExit("ERROR: input CSV must contain 'h3_seq'")
    # --- Compatibility columns for downstream plotting scripts ---
    # Provide a standard global-FV column name
    if "rmsd_fv_seqmap" not in df.columns:
        if "rmsd_fv_all_ctx" in df.columns:
            df["rmsd_fv_seqmap"] = pd.to_numeric(df["rmsd_fv_all_ctx"], errors="coerce")
        elif "rmsd_fv_noh3_ctx" in df.columns:
            # fallback, but note: this is "FV excluding H3" not full FV
            df["rmsd_fv_seqmap"] = pd.to_numeric(df["rmsd_fv_noh3_ctx"], errors="coerce")
        else:
            df["rmsd_fv_seqmap"] = np.nan
    
    if "h3_error" not in df.columns:
        df["h3_error"] = ""

    if "h3_rmsd" not in df.columns:
        if "rmsd_h3_ctx" in df.columns:
            df["h3_rmsd"] = pd.to_numeric(df["rmsd_h3_ctx"], errors="coerce")
        elif "rmsd_h3_local" in df.columns:
            df["h3_rmsd"] = pd.to_numeric(df["rmsd_h3_local"], errors="coerce")
        else:
            raise SystemExit(
                "ERROR: no H3 RMSD column found. Expected one of: "
                "'h3_rmsd', 'rmsd_h3_ctx', 'rmsd_h3_local'."
            )

    if "h3_rmsd_ctx" not in df.columns and "rmsd_h3_ctx" in df.columns:
        df["h3_rmsd_ctx"] = pd.to_numeric(df["rmsd_h3_ctx"], errors="coerce")
    if "h3_rmsd_local" not in df.columns and "rmsd_h3_local" in df.columns:
        df["h3_rmsd_local"] = pd.to_numeric(df["rmsd_h3_local"], errors="coerce")

    os.makedirs(args.plots_dir, exist_ok=True)

    bound_state = []
    for _, r in df.iterrows():
        ref_pdb = str(r.get("ref_pdb", "")).strip()
        if not ref_pdb or not os.path.exists(ref_pdb):
            bound_state.append("")
            continue
        try:
            chseqs = parse_pdb_chain_sequences_ca(ref_pdb)
        except Exception:
            bound_state.append("")
            continue

        ab_chains = set()
        partner = False
        for ch, seq in chseqs.items():
            if len(seq) < 30:
                continue
            kind = anarci_is_antibody_chain(seq)
            if kind in ("H", "L"):
                ab_chains.add(ch)
            else:
                partner = True

        bound_state.append("bound" if partner else "unbound")

    df["bound_state"] = bound_state

    df["h3_seq"] = df["h3_seq"].fillna("").astype(str)
    df["h3_len"] = df["h3_seq"].apply(len)
    comp = df["h3_seq"].apply(aa_composition)
    comp_df = pd.DataFrame(list(comp.values), index=df.index)
    comp_df = comp_df.rename(columns={c: f"h3_{c}" for c in comp_df.columns})
    df = pd.concat([df, comp_df], axis=1)

    df["h3_net_charge_pH7"] = df["h3_seq"].apply(lambda s: h3_charge(s, his_weight=args.his_weight))
    df["h3_kd_mean"], df["h3_frac_hydrophobic"] = zip(*df["h3_seq"].apply(kd_stats))

    df["h3_frac_gly_pro"] = df["h3_seq"].apply(lambda s: frac_set(s, "GP"))
    df["h3_frac_aromatic"] = df["h3_seq"].apply(lambda s: frac_set(s, "YFW"))

    df["h3_has_cys"], df["h3_has_cys_pair"] = zip(*df["h3_seq"].apply(cys_patterns))
    df["h3_has_glyco_motif"] = df["h3_seq"].apply(glyco_motif_present)

    df["h3_num_unique"] = df["h3_seq"].apply(lambda s: len(set(s)) if s else 0)
    df["h3_entropy"] = df["h3_seq"].apply(shannon_entropy)

    df["h3_p_count"], df["h3_p_max_run"] = zip(*df["h3_seq"].apply(proline_patterns))

    seqs = df["h3_seq"].tolist()
    rarity_map = dataset_kmer_rarity(seqs, k=3)
    df["h3_kmer3_rarity_dataset"] = df["h3_seq"].map(rarity_map).fillna(0.0)

    df.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")

    rank_col = args.rank_by
    if rank_col not in df.columns:
        raise SystemExit(f"ERROR: rank_by column '{rank_col}' not found in CSV")

    d_ok = df.copy()
    d_ok[rank_col] = pd.to_numeric(d_ok[rank_col], errors="coerce")
    
    if rank_col == "h3_rmsd" and "h3_error" in d_ok.columns:
        h3_err = d_ok["h3_error"].fillna("").astype(str).str.strip()
        d_ok = d_ok[(h3_err == "")]
    N = 30
    d_ok = d_ok[np.isfinite(d_ok[rank_col])]
    if len(d_ok) < N:
        print(f"WARNING: only {len(d_ok)} valid rows for ranking by {rank_col}; using N={len(d_ok)}")
        N = len(d_ok)
    d_rank = d_ok.sort_values(rank_col).drop_duplicates("id", keep="first")
    best = d_rank.head(N)
    worst = d_rank.tail(N)
    
    print("\nBest method counts:")
    print(best["method"].value_counts(dropna=False))
    print("\nWorst method counts:")
    print(worst["method"].value_counts(dropna=False))

    def save_hist(feature: str, fname: str, title: str):
        plt.figure()
        plt.hist(best[feature].dropna().astype(float).values, bins=10, alpha=0.6, label=f"best{N}")
        plt.hist(worst[feature].dropna().astype(float).values, bins=10, alpha=0.6, label=f"worst{N}")
        plt.title(title)
        plt.xlabel(feature)
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.plots_dir, fname), dpi=200)
        plt.close()

    for feat, fn in [
        ("h3_len", "h3_len_best_vs_worst.png"),
        ("h3_net_charge_pH7", "h3_charge_best_vs_worst.png"),
        ("h3_kd_mean", "h3_kd_mean_best_vs_worst.png"),
        ("h3_frac_hydrophobic", "h3_frac_hydrophobic_best_vs_worst.png"),
        ("h3_frac_gly_pro", "h3_frac_glypro_best_vs_worst.png"),
        ("h3_frac_aromatic", "h3_frac_aromatic_best_vs_worst.png"),
        ("h3_entropy", "h3_entropy_best_vs_worst.png"),
        ("h3_kmer3_rarity_dataset", "h3_kmer_rarity_best_vs_worst.png"),
    ]:
        if feat in df.columns:
            save_hist(feat, fn, f"{feat}: best{N} vs worst{N} (rank by {rank_col})")

    plt.figure()
    best_counts = best["bound_state"].value_counts(dropna=False)
    worst_counts = worst["bound_state"].value_counts(dropna=False)
    cats = sorted(set(best_counts.index.tolist()) | set(worst_counts.index.tolist()))
    best_vals = [best_counts.get(c, 0) for c in cats]
    worst_vals = [worst_counts.get(c, 0) for c in cats]

    x = np.arange(len(cats))
    width = 0.4
    plt.bar(x - width/2, best_vals, width, label=f"best{N}")
    plt.bar(x + width/2, worst_vals, width, label=f"worst{N}")
    plt.xticks(x, cats, rotation=0)
    plt.title(f"Bound/unbound in best{N} vs worst{N} (rank by {rank_col})")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.plots_dir, "bound_state_best_vs_worst.png"), dpi=200)
    plt.close()

    cols_to_show = [c for c in ["id", "method", "label", rank_col, "h3_len", "h3_has_glyco_motif", "h3_has_cys", "bound_state"] if c in df.columns]
    best[cols_to_show].to_csv(os.path.join(args.plots_dir, "best_examples.csv"), index=False)
    worst[cols_to_show].to_csv(os.path.join(args.plots_dir, "worst_examples.csv"), index=False)
    print(f"Wrote plots + example CSVs into: {args.plots_dir}")


if __name__ == "__main__":
    main()
"""
References:
[1] Marks & Deane, 2017
Antibody H3 Structure Prediction.
Computational and Structural Biotechnology Journal 15: 222–231.
DOI: 10.1016/j.csbj.2017.01.010

[2] Regep et al., 2017
The H3 loop of antibodies shows unique structural characteristics.
Proteins 85(7): 1311–1318.
DOI: 10.1002/prot.25291

[3] Tsuchiya & Mizuguchi, 2016
The Diversity of H3 Loops Determines the Antigen-Binding Tendencies of Antibody CDR Loops.
Protein Science 25(4): 815–825.
DOI: 10.1002/pro.2874

[4] Shannon, 1948
A Mathematical Theory of Communication.
Bell System Technical Journal 27: 379–423, 623–656.

[5] Cover & Thomas, 2006
Elements of Information Theory (2nd ed.).
Wiley.

[6] Xu et al., 2025
In-Depth Study of Low-Complexity Domains: From Structural Diversity to Disease Mechanisms.
Cells 14(22): 1752.
DOI: 10.3390/cells14221752

[7] Donald et al., 2011
Salt bridges: geometrically specific, designable interactions.
Proteins 79(3): 898–915.
DOI: 10.1002/prot.22927

[8] Li et al., 2015
Rigidity Emerges during Antibody Evolution…
PLoS Computational Biology 11(7): e1004327.
DOI: 10.1371/journal.pcbi.1004327

[9] Sangha et al., 2017
Role of Non-local Interactions between CDR Loops…
Structure 25(12): 1820–1828.e2.
DOI: 10.1016/j.str.2017.10.005

[10] Kyte & Doolittle, 1982
A Simple Method for Displaying the Hydropathic Character of a Protein.
Journal of Molecular Biology 157(1): 105–132.
DOI: 10.1016/0022-2836(82)90515-0

[11] Miao et al., 2004
The Optimal Fraction of Hydrophobic Residues Required to Ensure Protein Collapse.
Journal of Molecular Biology 344(3): 797–811.
DOI: 10.1016/j.jmb.2004.09.061

[12] Jacob et al., 1999
The Role of Proline and Glycine in Determining Backbone Flexibility.
Biophysical Journal 76(3): 1367–1376.
DOI: 10.1016/S0006-3495(99)77298-X

[13] Barlow & Thornton, 1988
Helix geometry in proteins.
Journal of Molecular Biology 201(3): 601–619.
DOI: 10.1016/0022-2836(88)90641-9

[14] Woolfson & Williams, 1990
The Influence of Proline Residues on Alpha-Helical Structure.
FEBS Letters 277(1–2): 185–188.
DOI: 10.1016/0014-5793(90)80839-B

[15] McGaughey et al., 1998
π-Stacking interactions: Alive and well in proteins.
Journal of Biological Chemistry 273(25): 15458–15463.
DOI: 10.1074/jbc.273.25.15458

[16] Infield et al., 2021
Cation-π Interactions and their Functional Roles in Membrane Proteins.
Journal of Molecular Biology 433(17): 167035.
DOI: 10.1016/j.jmb.2021.167035

[17] Peng et al., 2022
Antibody CDR amino acids underlying the functionality of antibody repertoires.
Scientific Reports 12: 12555.
DOI: 10.1038/s41598-022-16841-9

[18] Anishetty et al., 2002
Tripeptide Analysis of Protein Structures.
BMC Structural Biology 2: 9.
DOI: 10.1186/1472-6807-2-9

[19] Vignesh et al., 2024
Ensemble Deep Learning Model for Protein Secondary Structure Prediction Using NLP Metrics and Explainable AI.
Results in Engineering 24: 103435.
DOI: 10.1016/j.rineng.2024.103435
"""
