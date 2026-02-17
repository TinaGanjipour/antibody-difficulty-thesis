"""
ANARCI (Dunbar & Deane, 2015) helpers for antibody variable-domain
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import re
import pandas as pd
from contextlib import redirect_stdout, redirect_stderr
import os
from anarci import run_anarci

# https://www.cryst.bbk.ac.uk/education/AminoAcid/the_twenty.html
# https://www.imgt.org/IMGTeducation/Aide-memoire/_UK/aminoacids/IMGTclasses.html

AA20 = set("ACDEFGHIKLMNPQRSTVWY") # cleans input sequences to the 20 standard amino acids
SCHEME = "chothia" #numbering scheme

# input sequences may contain non-AA characters which are removed
def aa_clean(s: str) -> str:
    return re.sub(rf"[^{''.join(sorted(AA20))}]", "", str(s or "").upper())

def base_num(nlabel: Any) -> Optional[int]:
    m = re.match(r"^(\d+)", str(nlabel))
    return int(m.group(1)) if m else None

def anarci_invoke(pairs, *, scheme: str, output: bool = False, allow=None):
    """
    keeping our pipeline logs clean by minimizing the noise in logs by redirecting stdout/stderr
    Parameters:
        1. pairs (iterable):  (id, sequence) tuples
        2. numbering scheme (str): chothia
        3. allow (None|str|iterable): allowed chain types
            none means allow all of "H","K",and "L" 
            e.g. a string like "HK" is accepted and normalized to a list
    """
    scheme = (scheme or SCHEME).lower()
    valid = {"H", "K", "L"}
    if allow is None:
        allow = ["H", "K", "L"]
    elif isinstance(allow, str):
        allow = [c for c in allow.upper() if c in valid]
        if not allow:
            allow = ["H", "K", "L"]
    else:
        allow = [str(c).upper() for c in allow if str(c).upper() in valid]
        if not allow:
            allow = ["H", "K", "L"]

    fnull = open(os.devnull, "w")
    try:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            kw = dict(scheme=scheme, output=output, allow=allow)
            try:
                out = run_anarci(pairs, **kw, allowed_species=None)
            except TypeError as te:
                if "allowed_species" in str(te):
                    out = run_anarci(pairs, **kw)
                else:
                    raise
    finally:
        fnull.close()

    if not out:
        raise RuntimeError("ANARCI returned no results; check inputs/allow parameter.")
    return out

# triplets: position (number_label, insertion), amino_acid
def _extract_numbering_triples_any(raw: Any) -> List[Tuple[Tuple[Any, Any], str]]:
    def is_pair_list_with_num_ins(x):
        return isinstance(x, list) and len(x) > 0 and all(
            isinstance(t, (list, tuple)) and len(t) == 2 and
            isinstance(t[0], (list, tuple)) and len(t[0]) == 2
            for t in x
        )
    def is_true_triple(t):
        return isinstance(t, (list, tuple)) and len(t) == 3 and isinstance(t[2], str) and t[2] != ""
    def is_triple_list(x):
        return isinstance(x, list) and len(x) > 0 and all(is_true_triple(t) for t in x)

    q = [raw]; seen = set()
    while q:
        o = q.pop(0); oid = id(o)
        if oid in seen:
            continue
        seen.add(oid)
        if is_pair_list_with_num_ins(o):
            return [((ni[0], ni[1]), aa) for (ni, aa) in o]
        if is_triple_list(o):
            return [((t[0], t[1]), t[2]) for t in o]

        if isinstance(o, dict):
            q.extend(v for v in o.values() if not isinstance(v, (str, bytes)))
        elif isinstance(o, (list, tuple)):
            q.extend(v for v in o if not isinstance(v, (str, bytes)))

    return []

def extract_domains(res, fallback_chain_type: Optional[str] = None) -> List[Dict[str, Any]]:
    if not res:
        return []

    assignments = numbering = None
    if isinstance(res, (list, tuple)):
        if len(res) >= 2:
            assignments, numbering = res[0], res[1]
        elif len(res) == 1:
            numbering = res[0]

    chain_type_guess = ""
    try:
        if isinstance(assignments, list) and assignments:
            a0 = assignments[0]
            if isinstance(a0, (list, tuple)) and len(a0) >= 2 and isinstance(a0[1], list) and a0[1]:
                domain_tuple = a0[1][0]
                for item in domain_tuple:
                    if item in ("H", "K", "L"):
                        chain_type_guess = item
                        break
    except Exception:
        pass

    if not chain_type_guess and fallback_chain_type in ("H", "K", "L"):
        chain_type_guess = fallback_chain_type

    triples_norm: List[Tuple[Tuple[Any, Any], str]] = []

    try:
        if isinstance(numbering, list) and numbering:
            n0 = numbering[0]
            if isinstance(n0, (list, tuple)):
                if len(n0) >= 2 and isinstance(n0[1], list):
                    triples_norm = _extract_numbering_triples_any(n0[1])
                elif isinstance(n0, list):
                    triples_norm = _extract_numbering_triples_any(n0)
    except Exception:
        triples_norm = []

    if not triples_norm:
        triples_norm = _extract_numbering_triples_any(res)

    out: List[Dict[str, Any]] = []
    if triples_norm:
        out.append({
            "chain_type": chain_type_guess,
            "triples": triples_norm,
            "details": {},
        })
    return out

def run_and_collect(seq: str, scheme: str = SCHEME, output: bool = True, allow=None) -> List[Dict[str, Any]]:
    s = aa_clean(seq)
    if not s:
        return []
    if allow is None:
        allow = ["H", "K", "L"]
    sid = "seq0"
    res = anarci_invoke([(sid, s)], scheme=scheme, output=output, allow=allow)
    fallback = allow[0] if isinstance(allow, (list, tuple)) and len(allow) == 1 else None
    return extract_domains(res, fallback_chain_type=fallback)

def triples_to_seq(triples: List[Tuple[Tuple[Any, Any], str]]) -> str:
    return "".join(a for (_pos, a) in triples if a and a != "-")

def extract_v_domain_anarci_only(seq: str, allow_types: List[str]) -> Tuple[str, str, Dict]:
    s = aa_clean(seq)
    if not s:
        return "", "", {}
    allow_types = [(str(c).upper()) for c in (allow_types or [])]
    doms = run_and_collect(s, scheme="chothia", output=True, allow=allow_types)
    dom = next((d for d in doms if (d.get("chain_type") in allow_types)), None)
    if dom is None:
        dom = doms[0] if doms else None
    if not dom:
        return "", "", {}
    vseq = triples_to_seq(dom["triples"])
    return vseq, "anarci_direct", dom.get("details", {})

def cdr_slice_from_triples(
    triples: List[Tuple[Tuple[Any, Any], str]], start_num: int, end_num: int
) -> Tuple[object, object, str]:
    """
    Slices a numbered CDR regions from ANARCI position triples ((position (number_label, insertion), amino_acid))
    seq is the extracted amino-acid string
    start_index/end_index are indices into the triples list 
    We included all residues with base number within [start_num, end_num], 
    and insertion residues at the boundaries: (start_num + insertion) and (end_num + insertion).
    Returns start_index, end_index, and sequence_string.
    """
    idx, aa = [], []
    for i, (ni_ii, a) in enumerate(triples):
        if not a or a == "-":
            continue
        n, ins = ni_ii
        nb = base_num(n)
        if nb is None:
            continue
        in_core = (start_num <= nb <= end_num)
        at_end_ins = (nb == end_num and isinstance(ins, str) and len(ins) > 0)
        at_start_ins = (nb == start_num and isinstance(ins, str) and len(ins) > 0)
        if in_core or at_end_ins or at_start_ins:
            idx.append(i)
            aa.append(a)
    if not idx:
        return (pd.NA, pd.NA, "")
    return (min(idx), max(idx), "".join(aa))

# If another similar project provides a canonical CDR-definition utility (e.g., IMGT vs Chothia vs Kabat),
# we can wire it in below because the thesis pipeline can run without it since we fall back to numeric spans:
try:
    from yourmodule.cdrdefs import annotate_regions
except Exception:
    annotate_regions = None


def cdr_labels_from_triples(
    triples: List[Tuple[Tuple[Any, Any], str]],
    chain_type: str,
    numbering_scheme: str = "chothia",
    definition: str = "chothia",
):
    if annotate_regions is None:
        return None
    numbered_sequence = [(pos, aa) for (pos, aa) in triples]
    return annotate_regions(
        numbered_sequence,
        chain_type.replace("K", "L"),
        numbering_scheme=numbering_scheme,
        definition=definition,
    )

def annotate_cdrs_anarci(
    v_seq: str,
    *,
    scheme: str = SCHEME,
) -> Dict[str, Any]:
    """
    Annotate CDR sequences from a variable-domain sequence using ANARCI numbering.
    - This function uses Chothia numeric spans for CDR extraction:
        Heavy: H1=26–32, H2=52–56, H3=95–102
        Light: L1=24–34, L2=50–56, L3=89–97
    (Chothia & Lesk, 1987, Al-Lazikani et al., 1997, Zhu et al., 2024)
    - These spans are a definition choice. Other schemes/definitions (IMGT/Kabat/etc) can shift boundaries and insertion placement.
    Output: a dict with CDR start/end indices into the numbered triple list, the extracted sequences, and the inferred chain type.
    """
    out = {
        "cdr_h1_start": pd.NA, "cdr_h1_end": pd.NA, "h1_seq": "",
        "cdr_h2_start": pd.NA, "cdr_h2_end": pd.NA, "h2_seq": "",
        "cdr_h3_start": pd.NA, "cdr_h3_end": pd.NA, "h3_seq": "", "h3_len": pd.NA,
        "cdr_l1_start": pd.NA, "cdr_l1_end": pd.NA, "l1_seq": "",
        "cdr_l2_start": pd.NA, "cdr_l2_end": pd.NA, "l2_seq": "",
        "cdr_l3_start": pd.NA, "cdr_l3_end": pd.NA, "l3_seq": "",
        "numbering_scheme": scheme.lower(),
        "auto_chain_type": "",
        "full_v_seq": "",
    }

    s = aa_clean(v_seq)
    if not s:
        return out

    domains = run_and_collect(s, scheme=scheme, output=True)
    dom = next((d for d in domains if d.get("chain_type") in ("H", "K", "L", "")), None)
    if not dom:
        return out

    triples = dom["triples"]
    out["auto_chain_type"] = dom.get("chain_type", "")
    out["full_v_seq"] = triples_to_seq(triples)

    is_heavy = (dom.get("chain_type") == "H")
    if is_heavy:
        h1s, h1e, h1 = cdr_slice_from_triples(triples, 26, 32)
        h2s, h2e, h2 = cdr_slice_from_triples(triples, 52, 56)
        h3s, h3e, h3 = cdr_slice_from_triples(triples, 95, 102)
        out.update({
            "cdr_h1_start": h1s, "cdr_h1_end": h1e, "h1_seq": h1,
            "cdr_h2_start": h2s, "cdr_h2_end": h2e, "h2_seq": h2,
            "cdr_h3_start": h3s, "cdr_h3_end": h3e, "h3_seq": h3,
            "h3_len": (len(h3) if h3 else pd.NA),
        })
    else:
        l1s, l1e, l1 = cdr_slice_from_triples(triples, 24, 34)
        l2s, l2e, l2 = cdr_slice_from_triples(triples, 50, 56)
        l3s, l3e, l3 = cdr_slice_from_triples(triples, 89, 97)
        out.update({
            "cdr_l1_start": l1s, "cdr_l1_end": l1e, "l1_seq": l1,
            "cdr_l2_start": l2s, "cdr_l2_end": l2e, "l2_seq": l2,
            "cdr_l3_start": l3s, "cdr_l3_end": l3e, "l3_seq": l3,
        })
    return out

def get_numbered_span(seq: str, allow: List[str], scheme: str = "chothia"):
    s = aa_clean(seq)
    if not s:
        return []
    doms = run_and_collect(s, scheme=scheme, output=True, allow=allow)
    dom = next((d for d in doms if (d.get("chain_type") in allow) or (not d.get("chain_type"))), None)
    if not dom:
        return []
    triples_pair = dom["triples"]
    triples_3 = []
    for (pos, aa) in triples_pair:
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            n_label, ins = pos
        else:
            n_label, ins = pos, ""
        triples_3.append((n_label, ins, aa))
    return triples_3
"""
References:
1. Chothia & Lesk, 1987
2. Al-Lazikani et al., 1997
3. Abanades et al., 2023
4. Ruffolo et al., 2023
5. Dunbar & Deane, 2015
6. Zhu et al., 2024
"""