"""
This script reads all id values from two prediction CSVs (IgFold + ABodyBuilder2),
looks for downloaded truth structures,
and writes ref_map.csv mapping predictions (id) to truth structure (ref_pdb).
"""
import argparse, csv, os
from pathlib import Path

def read_ids_from_pred(csv_path: Path) -> set:
    ids = set()
    with csv_path.open() as f:
        r = csv.DictReader(f)
        if "id" not in (r.fieldnames or []):
            raise ValueError(f"{csv_path} missing 'id' column")
        for row in r:
            pid = (row.get("id") or "").strip()
            if pid:
                ids.add(pid)
    return ids

def load_existing_map(path: Path) -> dict:
    m = {}
    if path.exists():
        with path.open() as f:
            r = csv.DictReader(f)
            if "id" not in (r.fieldnames or []) or "ref_pdb" not in r.fieldnames:
                raise ValueError(f"{path} must have headers: id,ref_pdb")
            for row in r:
                rid = (row.get("id") or "").strip()
                rp = (row.get("ref_pdb") or "").strip()
                if rid:
                    m[rid] = rp
    return m

def best_local_ref(rcsb_dir: Path, pid: str) -> str:
    p_pdb = rcsb_dir / f"{pid.upper()}.pdb"
    p_cif = rcsb_dir / f"{pid.upper()}.cif"
    if p_pdb.exists():
        return str(p_pdb.resolve())
    if p_cif.exists():
        return str(p_cif.resolve())
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--igfold", required=True, help="igfold_predictions.csv")
    ap.add_argument("--ab2", required=True, help="abodybuilder2_predictions.csv")
    ap.add_argument("--rcsb-dir", required=True, help="Folder with downloaded true structures")
    ap.add_argument("--existing", default="ref_map.csv", help="Existing ref_map.csv to merge (if present)")
    ap.add_argument("--out", default="ref_map.csv", help="Output ref_map.csv")
    ap.add_argument("--ids-missing-out", default="ids_missing.txt", help="Write IDs lacking local refs here")
    args = ap.parse_args()

    ig_path = Path(args.igfold).resolve()
    ab_path = Path(args.ab2).resolve()
    rcsb_dir = Path(args.rcsb_dir).resolve()
    out_path = Path(args.out).resolve()
    existing_path = Path(args.existing).resolve()
    ids_missing_path = Path(args.ids_missing_out).resolve() # anything not found locally

    if not ig_path.exists(): raise SystemExit(f"Not found: {ig_path}")
    if not ab_path.exists(): raise SystemExit(f"Not found: {ab_path}")
    if not rcsb_dir.exists(): raise SystemExit(f"RCSB dir not found: {rcsb_dir}")

    all_ids = read_ids_from_pred(ig_path) | read_ids_from_pred(ab_path)
    existing = load_existing_map(existing_path)

    rows = []
    missing = []
    for pid in sorted(all_ids):
        ref_path = existing.get(pid, "")
        if ref_path and Path(ref_path).exists():
            final = ref_path
        else:
            local = best_local_ref(rcsb_dir, pid)
            final = local
        if not final:
            missing.append(pid)
        rows.append({"id": pid, "ref_pdb": final})

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","ref_pdb"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with ids_missing_path.open("w") as f:
        for pid in missing:
            f.write(pid + "\n")

    print(f"Wrote: {out_path}  (total IDs: {len(rows)}, with local refs: {len(rows)-len(missing)}, missing: {len(missing)})")
    print(f"Wrote: {ids_missing_path}  (use it with your downloader to fetch missing truths)")

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
"""