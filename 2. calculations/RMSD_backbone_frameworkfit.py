
from __future__ import annotations
import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Iterable
import numpy as np
from ANARCI import get_numbered_span, base_num

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", "GLN": "Q", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "HYP": "P", "SEC": "U", "PYL": "O",
}
AA20 = set("ACDEFGHIKLMNPQRSTVWY")
BACKBONE_ATOMS = ("N", "CA", "C", "O")

def aa3_to_1(resname: str) -> str:
    return AA3_TO_1.get(resname.upper(), "X")

@dataclass(frozen=True)
class ResidueRec:
    resseq: int
    icode: str
    aa: str
    atoms: Dict[str, np.ndarray]
    bfac_by_atom: Dict[str, float]

Chain = List[ResidueRec]
NumKey = Tuple[int, str, str]

def _ins_sort_key(ins: str) -> Tuple[int, str]:
    return (0, "") if ins == "" else (1, ins)

def sort_numkeys(keys: Iterable[NumKey]) -> List[NumKey]:
    return sorted(keys, key=lambda k: (k[0], _ins_sort_key(k[1])))


# Reads only CA atoms from the first MODEL (for NMR ensembles or multi-model files it stops after model 1).
# Discards altlocs except blank or "A" (standard practice).
# Kabsch. 1976, Lawrence et al. (2019).
def parse_pdb_backbone_by_chain(path: str) -> Dict[str, Chain]:
    tmp: Dict[str, Dict[Tuple[int, str], Dict[str, object]]] = {}

    in_model = False
    model_idx = 0

    with open(path, "r") as f:
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

            atom = line[12:16].strip()
            if atom not in BACKBONE_ATOMS:
                continue

            alt = line[16]
            if alt not in (" ", "A"):
                continue

            chain_id = (line[21].strip() or "_")
            resname = line[17:20].strip()
            aa = aa3_to_1(resname)

            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = (line[26].strip() or "")

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            try:
                bfac = float(line[60:66])
            except ValueError:
                bfac = float("nan")

            key = (resseq, icode)
            tmp.setdefault(chain_id, {})
            if key not in tmp[chain_id]:
                tmp[chain_id][key] = {"aa": aa, "atoms": {}, "bfac_by_atom": {}}

            tmp[chain_id][key]["atoms"][atom] = np.array([x, y, z], dtype=float)
            tmp[chain_id][key]["bfac_by_atom"][atom] = bfac

    if not tmp:
        raise ValueError(f"No backbone atoms found in {path}")

    out: Dict[str, Chain] = {}
    for ch, residues in tmp.items():
        keys = sorted(residues.keys(), key=lambda k: (k[0], _ins_sort_key(k[1])))
        chain_list: List[ResidueRec] = []
        for (resseq, icode) in keys:
            aa = str(residues[(resseq, icode)]["aa"])
            atoms = dict(residues[(resseq, icode)]["atoms"])
            bfac_by_atom = dict(residues[(resseq, icode)]["bfac_by_atom"])
            chain_list.append(
                ResidueRec(resseq=resseq, icode=icode, aa=aa, atoms=atoms, bfac_by_atom=bfac_by_atom)
            )
        if chain_list:
            out[ch] = chain_list

    if not out:
        raise ValueError(f"No residues parsed in {path}")
    return out

def chain_seq(chain: Chain) -> str:
    return "".join(r.aa for r in chain)

@dataclass(frozen=True)
class NumRec:
    aa: str
    idx: int
    atoms: Dict[str, np.ndarray]
    bfac_by_atom: Dict[str, float]

# ANARCI numbers sequences by aligning to HMMs of germline V domains.
# Below function builds a sequence from the chain’s CA trace.
# Then calls get_numbered_span() to assign Chothia numbering positions
# Cyrus Chothia and Arthur M. Lesk. 1987, James Dunbar and Charlotte M. Deane. 2016.
Produces a map: numbering key → (aa, index, xyz). 
def anarci_numbering_map(chain: Chain, *, allow: List[str], scheme: str = "chothia") -> Dict[NumKey, NumRec]:
    seq = chain_seq(chain)

    clean_to_orig: List[int] = []
    for i, aa in enumerate(seq):
        if aa in AA20:
            clean_to_orig.append(i)

    triples = get_numbered_span(seq, allow=allow, scheme=scheme)
    if not triples:
        return {}

    out: Dict[NumKey, NumRec] = {}
    clean_idx = -1
    for n_label, ins, aa in triples:
        if aa != "-":
            clean_idx += 1
        if aa == "-":
            continue
        if not (0 <= clean_idx < len(clean_to_orig)):
            continue

        orig_idx = clean_to_orig[clean_idx]
        n_int = base_num(n_label)
        ins_s = (ins or "").strip()
        key: NumKey = (n_int, ins_s, str(n_label))

        out[key] = NumRec(
            aa=chain[orig_idx].aa,
            idx=orig_idx,
            atoms=chain[orig_idx].atoms,
            bfac_by_atom=chain[orig_idx].bfac_by_atom,
        )

    return out

@dataclass(frozen=True)
class AlignStats:
    identity: float
    aligned_len: int
    ref_cov: float
    pred_cov: float

def numbering_stats(ref_map: Dict[NumKey, NumRec], pred_map: Dict[NumKey, NumRec]) -> AlignStats:
    if not ref_map or not pred_map:
        return AlignStats(identity=0.0, aligned_len=0, ref_cov=0.0, pred_cov=0.0)

    shared = set(ref_map.keys()) & set(pred_map.keys())
    aligned_len = len(shared)
    if aligned_len == 0:
        return AlignStats(identity=0.0, aligned_len=0, ref_cov=0.0, pred_cov=0.0)

    ident = 0
    denom = 0
    for k in shared:
        a = ref_map[k].aa
        b = pred_map[k].aa
        if a == "X" or b == "X":
            continue
        denom += 1
        if a == b:
            ident += 1

    identity = ident / max(1, denom)
    ref_cov = aligned_len / max(1, len(ref_map))
    pred_cov = aligned_len / max(1, len(pred_map))
    return AlignStats(identity=identity, aligned_len=aligned_len, ref_cov=ref_cov, pred_cov=pred_cov)

def coords_by_numbering_backbone(
    ref_map: Dict[NumKey, NumRec],
    pred_map: Dict[NumKey, NumRec],
    *,
    keep: Optional[Set[NumKey]] = None,
    atoms: Tuple[str, ...] = BACKBONE_ATOMS,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if not ref_map or not pred_map:
        return np.empty((0, 3)), np.empty((0, 3)), 0

    shared = set(ref_map.keys()) & set(pred_map.keys())
    if keep is not None:
        shared &= keep
    if not shared:
        return np.empty((0, 3)), np.empty((0, 3)), 0

    keys = sort_numkeys(shared)
    P: List[np.ndarray] = []
    Q: List[np.ndarray] = []

    for k in keys:
        ra = ref_map[k].atoms
        pa = pred_map[k].atoms
    
        if not all(at in ra for at in atoms):
            continue
        if not all(at in pa for at in atoms):
            continue
    
        for at in atoms:
            P.append(ra[at])
            Q.append(pa[at])

    if not P:
        return np.empty((0, 3)), np.empty((0, 3)), 0
    return np.stack(P), np.stack(Q), len(P)

def mean_bfactor_by_keys(
    num_map: Dict[NumKey, NumRec],
    keys: Set[NumKey],
    *,
    atoms: Tuple[str, ...] = BACKBONE_ATOMS,
) -> float:
    vals: List[float] = []
    for k in keys:
        if k not in num_map:
            continue
        b = num_map[k].bfac_by_atom
        for at in atoms:
            if at in b and np.isfinite(b[at]):
                vals.append(float(b[at]))
    return float(np.mean(vals)) if vals else float("nan")

def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc
    H = Q0.T @ P0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = Pc - R @ Qc
    return R, t

def rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    dif = P - Q
    return math.sqrt(float((dif * dif).sum()) / max(1, len(P)))

def chothia_h3_keys(num_map: Dict[NumKey, NumRec]) -> Set[NumKey]:
    return {k for k in num_map.keys() if 95 <= k[0] <= 102}

def seq_by_keys(num_map: Dict[NumKey, NumRec], keys: Set[NumKey]) -> str:
    ks = sort_numkeys(keys)
    return "".join(num_map[k].aa for k in ks if k in num_map and num_map[k].aa in AA20)

# Below function tries to match to each predicted chain, for each reference chain (VH and VL), by comparing overlap in numbering space. 
# It filters candidates by thresholds (identity, aligned length, coverage).
# Shirai et al., 1999., Jeliazkov et al., 2018.
def choose_chain_mapping_numbered(
    ref_chains: Dict[str, Chain],
    pred_chains: Dict[str, Chain],
    *,
    ref_types: Dict[str, str],
    scheme: str,
    min_identity: float,
    min_ref_cov: float,
    min_pred_cov: float,
    min_aligned_len: int,
):
    ref_num: Dict[str, Dict[NumKey, NumRec]] = {}
    pred_numH: Dict[str, Dict[NumKey, NumRec]] = {}
    pred_numL: Dict[str, Dict[NumKey, NumRec]] = {}

    for rch, chain in ref_chains.items():
        ctype = ref_types.get(rch, "")
        if ctype == "H":
            ref_num[rch] = anarci_numbering_map(chain, allow=["H"], scheme=scheme)
        else:
            ref_num[rch] = anarci_numbering_map(chain, allow=["K", "L"], scheme=scheme)

    for pch, chain in pred_chains.items():
        pred_numH[pch] = anarci_numbering_map(chain, allow=["H"], scheme=scheme)
        pred_numL[pch] = anarci_numbering_map(chain, allow=["K", "L"], scheme=scheme)

    stats: Dict[Tuple[str, str], AlignStats] = {}
    edges: Dict[str, List[str]] = {}
    ref_maps: Dict[Tuple[str, str], Dict[NumKey, NumRec]] = {}
    pred_maps: Dict[Tuple[str, str], Dict[NumKey, NumRec]] = {}

    for rch in ref_chains.keys():
        ctype = ref_types.get(rch, "")
        rmap = ref_num.get(rch, {})
        cands: List[str] = []
        for pch in pred_chains.keys():
            pmap = pred_numH[pch] if ctype == "H" else pred_numL[pch]
            s = numbering_stats(rmap, pmap)
            stats[(rch, pch)] = s
            ref_maps[(rch, pch)] = rmap
            pred_maps[(rch, pch)] = pmap
            if (
                s.identity >= min_identity
                and s.aligned_len >= min_aligned_len
                and s.ref_cov >= min_ref_cov
                and s.pred_cov >= min_pred_cov
            ):
                cands.append(pch)

        if cands:
            cands.sort(key=lambda p: (stats[(rch, p)].identity * stats[(rch, p)].aligned_len), reverse=True)
            edges[rch] = cands

    if not edges:
        return {}, stats, ref_maps, pred_maps

    ref_list = sorted(edges.keys())
    best_map: Dict[str, str] = {}
    best_score = -1.0

    def backtrack(i: int, used: Set[str], cur: Dict[str, str], score: float) -> None:
        nonlocal best_score, best_map
        if i == len(ref_list):
            if cur and score > best_score:
                best_score = score
                best_map = dict(cur)
            return
        rch = ref_list[i]
        for pch in edges.get(rch, []):
            if pch in used:
                continue
            s = stats[(rch, pch)]
            base = s.identity * s.aligned_len
            cur[rch] = pch
            used.add(pch)
            backtrack(i + 1, used, cur, score + base)
            used.remove(pch)
            cur.pop(rch, None)
        backtrack(i + 1, used, cur, score)

    backtrack(0, set(), {}, 0.0)
    return best_map, stats, ref_maps, pred_maps

EMPTY_H3 = {"n_h3_atoms": "", "rmsd_h3_local": "", "rmsd_h3_ctx": "", "h3_seq": ""}

def compute_pair(
    ref_pdb: str,
    pred_pdb: str,
    *,
    ref_vh: str,
    ref_vl: str,
    scheme: str,
    min_identity: float,
    min_ref_cov: float,
    min_pred_cov: float,
    min_aligned_len: int,
):
    def err_out(msg: str, n_used: int = 0, n_aln: int = 0, chain_map_s: str = "", chain_ident_s: str = ""):
        return None, {
            "error": msg,
            "n_chains_used": str(n_used),
            "n_fit_atoms": str(n_aln),
            "chain_map": chain_map_s,
            "chain_identities": chain_ident_s,
            **EMPTY_H3,
            "rmsd_fv_all_ctx": "",
            "rmsd_fv_noh3_ctx": "",
            "pred_conf_h3_bfac_mean": "",
            "pred_conf_fv_bfac_mean": "",
        }

    try:
        ref_chains_all = parse_pdb_backbone_by_chain(ref_pdb)
        pred_chains_all = parse_pdb_backbone_by_chain(pred_pdb)
    except Exception as e:
        return err_out(f"parse error: {e}")

    ref_vh = (ref_vh or "").strip()
    ref_vl = (ref_vl or "").strip()
    if not ref_vh or not ref_vl:
        return err_out("Requires vh and vl columns in pairs.csv (reference chain IDs).")

    if ref_vh not in ref_chains_all or ref_vl not in ref_chains_all:
        return err_out(f"specified ref chains not found in ref_pdb: vh={ref_vh}, vl={ref_vl}")
    if ref_vh == ref_vl:
        return err_out(f"vh==vl ({ref_vh}); single-chain reference not supported")

    if scheme != "chothia":
        return err_out("This script currently defines H3 range only for scheme=chothia.")

    ref_chains = {ref_vh: ref_chains_all[ref_vh], ref_vl: ref_chains_all[ref_vl]}
    ref_types = {ref_vh: "H", ref_vl: "L"}

    chain_map, stats, ref_maps, pred_maps = choose_chain_mapping_numbered(
        ref_chains,
        pred_chains_all,
        ref_types=ref_types,
        scheme=scheme,
        min_identity=min_identity,
        min_ref_cov=min_ref_cov,
        min_pred_cov=min_pred_cov,
        min_aligned_len=min_aligned_len,
    )
    if not chain_map:
        return err_out("no reliable chain map in numbering space (thresholds too strict or ANARCI failed)")

    if ref_vh not in chain_map or ref_vl not in chain_map:
        return err_out(f"incomplete chain map (need both VH and VL). got: {chain_map}", n_used=len(chain_map), n_aln=0)

    chain_map_s = ";".join([f"{rch}->{pch}" for rch, pch in chain_map.items()])
    chain_ident_s = ";".join(
        [
            f"{rch}->{pch}:ident={stats[(rch, pch)].identity:.3f},aln={stats[(rch, pch)].aligned_len},"
            f"refcov={stats[(rch, pch)].ref_cov:.2f},predcov={stats[(rch, pch)].pred_cov:.2f}"
            for (rch, pch) in chain_map.items()
        ]
    )

    p_vh = chain_map[ref_vh]
    p_vl = chain_map[ref_vl]

    refH = ref_maps[(ref_vh, p_vh)]
    predH = pred_maps[(ref_vh, p_vh)]
    refL = ref_maps[(ref_vl, p_vl)]
    predL = pred_maps[(ref_vl, p_vl)]

    if not refH or not predH or not refL or not predL:
        return err_out("ANARCI numbering map empty for at least one chain (ref or pred).", n_used=2, n_aln=0,
                       chain_map_s=chain_map_s, chain_ident_s=chain_ident_s)

    sharedH = set(refH.keys()) & set(predH.keys())
    sharedL = set(refL.keys()) & set(predL.keys())

    h3_keys_ref = chothia_h3_keys(refH)
    h3_keys_pred = chothia_h3_keys(predH)
    h3_keep = (h3_keys_ref & sharedH) & (h3_keys_pred & sharedH)
    h3_seq = seq_by_keys(refH, h3_keep)

    fitH_keep = sharedH - h3_keep
    fitL_keep = sharedL

    PH_fit, QH_fit, nH = coords_by_numbering_backbone(refH, predH, keep=fitH_keep)
    PL_fit, QL_fit, nL = coords_by_numbering_backbone(refL, predL, keep=fitL_keep)

    if (nH + nL) < 60:
        return err_out(
            f"too few backbone atoms in FV-noH3 fit set (VH_noH3_atoms={nH}, VL_atoms={nL})",
            n_used=2,
            n_aln=(nH + nL),
            chain_map_s=chain_map_s,
            chain_ident_s=chain_ident_s,
        )

    Pfit = np.concatenate([PH_fit, PL_fit], axis=0) if nH > 0 else PL_fit
    Qfit = np.concatenate([QH_fit, QL_fit], axis=0) if nH > 0 else QL_fit

    R, t = kabsch(Pfit, Qfit)
    Qfit_ctx = (R @ Qfit.T).T + t
    rmsd_fv_noh3_ctx_val = rmsd(Pfit, Qfit_ctx)

    PH_all, QH_all, _ = coords_by_numbering_backbone(refH, predH, keep=sharedH)
    PL_all, QL_all, _ = coords_by_numbering_backbone(refL, predL, keep=sharedL)
    Pall = np.concatenate([PH_all, PL_all], axis=0)
    Qall = np.concatenate([QH_all, QL_all], axis=0)
    rmsd_fv_all_ctx = rmsd(Pall, (R @ Qall.T).T + t)

    PH3, QH3, nh3_atoms = coords_by_numbering_backbone(refH, predH, keep=h3_keep)
    pred_conf_h3 = mean_bfactor_by_keys(predH, h3_keep)
    pred_conf_fv = mean_bfactor_by_keys(predH, sharedH)
    pred_conf_fv_L = mean_bfactor_by_keys(predL, sharedL)
    pred_conf_fv_both = float(np.nanmean([pred_conf_fv, pred_conf_fv_L]))

    rmsd_h3_ctx = ""
    rmsd_h3_local = ""
    if nh3_atoms >= 12:
        QH3_ctx = (R @ QH3.T).T + t
        rmsd_h3_ctx = f"{rmsd(PH3, QH3_ctx):.6f}"

        Rh3, th3 = kabsch(PH3, QH3)
        QH3_local = (Rh3 @ QH3.T).T + th3
        rmsd_h3_local = f"{rmsd(PH3, QH3_local):.6f}"

    diag = {
        "error": "",
        "n_chains_used": "2",
        "n_fit_atoms": str(len(Pfit)),
        "chain_map": chain_map_s,
        "chain_identities": chain_ident_s,
        "h3_seq": h3_seq,
        "n_h3_atoms": str(nh3_atoms),
        "rmsd_h3_local": rmsd_h3_local,
        "rmsd_h3_ctx": rmsd_h3_ctx,
        "rmsd_fv_all_ctx": f"{rmsd_fv_all_ctx:.6f}",
        "rmsd_fv_noh3_ctx": f"{rmsd_fv_noh3_ctx_val:.6f}",
        "pred_conf_h3_bfac_mean": (f"{pred_conf_h3:.6f}" if np.isfinite(pred_conf_h3) else ""),
        "pred_conf_fv_bfac_mean": (f"{pred_conf_fv_both:.6f}" if np.isfinite(pred_conf_fv_both) else ""),
    }
    return rmsd_fv_noh3_ctx_val, diag

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="pairs.csv with id,ref_pdb,pred_pdb,method,label and REQUIRED vh,vl")
    ap.add_argument("--out", default="results_backbone_fvnoh3fit.csv")
    ap.add_argument("--scheme", default="chothia", choices=["chothia", "kabat", "imgt", "aho"])
    ap.add_argument("--min-identity", type=float, default=0.80)
    ap.add_argument("--min-ref-cov", type=float, default=0.85)
    ap.add_argument("--min-pred-cov", type=float, default=0.85)
    ap.add_argument("--min-aligned-len", type=int, default=80)
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.pairs).open()))

    fieldnames = [
        "id", "method", "label", "ref_pdb", "pred_pdb",
        "rmsd_fv_noh3_ctx",
        "rmsd_fv_all_ctx",
        "rmsd_h3_ctx",
        "rmsd_h3_local",
        "pred_conf_h3_bfac_mean",
        "pred_conf_fv_bfac_mean",
        "h3_seq",
        "n_h3_atoms",
        "n_chains_used",
        "n_fit_atoms",
        "chain_map",
        "chain_identities",
        "error",
    ]

    out_rows: List[Dict[str, str]] = []

    for r in rows:
        rid = (r.get("id") or "").strip()
        ref_pdb = (r.get("ref_pdb") or "").strip()
        pred_pdb = (r.get("pred_pdb") or "").strip()
        method = (r.get("method") or "").strip()
        label = (r.get("label") or "").strip()
        ref_vh = (r.get("vh") or "").strip()
        ref_vl = (r.get("vl") or "").strip()

        row_out = {k: "" for k in fieldnames}
        row_out.update({"id": rid, "method": method, "label": label, "ref_pdb": ref_pdb, "pred_pdb": pred_pdb})

        if not ref_pdb or not Path(ref_pdb).exists():
            row_out["error"] = "ref_pdb not found"
            out_rows.append(row_out)
            continue
        if not pred_pdb or not Path(pred_pdb).exists():
            row_out["error"] = "pred_pdb not found"
            out_rows.append(row_out)
            continue

        val, diag = compute_pair(
            ref_pdb,
            pred_pdb,
            ref_vh=ref_vh,
            ref_vl=ref_vl,
            scheme=args.scheme,
            min_identity=args.min_identity,
            min_ref_cov=args.min_ref_cov,
            min_pred_cov=args.min_pred_cov,
            min_aligned_len=args.min_aligned_len,
        )

        if val is not None:
            row_out["rmsd_fv_noh3_ctx"] = f"{val:.6f}"

        for k, v in diag.items():
            if k in row_out:
                row_out[k] = v

        out_rows.append(row_out)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {args.out} ({len(out_rows)} rows)")

if __name__ == "__main__":
    main()

"""
References:
[1] Brennan Abanades, Wing Ki Wong, Fergus Boyles, and Charlotte M. Deane. 2023. ImmuneBuilder: Deep-Learning models for predicting the structures of immune proteins. Communications Biology 6, 1 (2023), 575. [https://doi.org/10.1038/s42003-023-04927-7](https://doi.org/10.1038/s42003-023-04927-7)

[2] Cadence Design Systems, Inc. [n.d.]. Protein Preparation — OEChem Toolkit Documentation. [https://docs.eyesopen.com/toolkits/cpp/oechemtk/proteinprep.html](https://docs.eyesopen.com/toolkits/cpp/oechemtk/proteinprep.html).

[3] Cyrus Chothia and Arthur M. Lesk. 1987. Canonical structures for the hypervariable regions of immunoglobulins. Journal of Molecular Biology 196, 4 (1987), 901–917. [https://doi.org/10.1016/0022-2836(87)90412-8](https://doi.org/10.1016/0022-2836%2887%2990412-8)

[4] Fabrice PA David and Yum L Yip. 2008. SSMap: a new UniProt-PDB mapping resource for the curation of structural-related information in the UniProt/Swiss-Prot Knowledgebase. BMC Bioinformatics 9 (2008), 391. [https://doi.org/10.1186/1471-2105-9-391](https://doi.org/10.1186/1471-2105-9-391)

[5] James Dunbar, Konrad Krawczyk, Jinwoo Leem, Terry Baker, Angela Fuchs, Guy Georges, Jiahua Shi, and Charlotte M. Deane. 2014. SAbDab: the structural antibody database. Nucleic Acids Research 42, Database issue (2014), D1140–D1146. [https://doi.org/10.1093/nar/gkt1043](https://doi.org/10.1093/nar/gkt1043)

[6] Project Gemmi. 2024. Gemmi: Macromolecular crystallography library. [https://github.com/project-gemmi/gemmi](https://github.com/project-gemmi/gemmi).

[7] Michal Jamroz, Andrzej Kolinski, and Daisuke Kihara. 2016. Ensemble-Based Evaluation for Protein Structure Models. Bioinformatics 32, 12 (2016), i314–i321. [https://doi.org/10.1093/bioinformatics/btw262](https://doi.org/10.1093/bioinformatics/btw262)

[8] Wolfgang Kabsch. 1976. A Solution for the Best Rotation to Relate Two Sets of Vectors. Acta Crystallographica Section A 32, 5 (1976), 922–923. [https://doi.org/10.1107/S0567739476001873](https://doi.org/10.1107/S0567739476001873)

[9] James Lawrence, Javier Bernal, and Christoph Witzgall. 2019. A Purely Algebraic Justification of the Kabsch-Umeyama Algorithm. Journal of Research of the National Institute of Standards and Technology 124 (2019), 1–6. [https://doi.org/10.6028/jres.124.028](https://doi.org/10.6028/jres.124.028)

[10] Pu Liu, Dimitris K. Agrafiotis, and Douglas L. Theobald. 2010. Fast determination of the optimal rotational matrix for macromolecular superpositions. Journal of Computational Chemistry 31, 7 (2010), 1561–1563. [https://doi.org/10.1002/jcc.21439](https://doi.org/10.1002/jcc.21439)

[11] Jeffrey A. Ruffolo, Jeremias Sulam, and Jeffrey J. Gray. 2023. Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies. Nature Communications 14, 1 (2023), 2389. [https://doi.org/10.1038/s41467-023-38063-x](https://doi.org/10.1038/s41467-023-38063-x)

[12] Zirui Zhu, Hossein Ashrafian, Navid Mohammadian Tabrizi, Emily Matas, Louisa Girard, Haowei Ma, and Edouard C. Nice. 2025. Antibody numbering schemes: advances, comparisons and tools for antibody engineering. Protein Engineering, Design and Selection 38 (2025), gzaf005. [https://doi.org/10.1093/protein/gzaf005](https://doi.org/10.1093/protein/gzaf005)

[13]Jeliazkov, J. R., Sljoka, A., Kuroda, D., Tsuchimura, N., Katoh, N., Tsumoto, K., and Gray, J. J. 2018. Repertoire analysis of antibody CDR-H3 loops suggests affinity maturation does not typically result in rigidification. Frontiers in Immunology 9 (2018), 413.
https://doi.org/10.3389/fimmu.2018.00413

[14] Shirai, H., Kidera, A., and Nakamura, H. 1999. H3-rules: identification of CDR-H3 structures in antibodies. FEBS Letters 455(1–2) (1999), 188–197.
https://doi.org/10.1016/S0014-5793(99)00821-2
"""