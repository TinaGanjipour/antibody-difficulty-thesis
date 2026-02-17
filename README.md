# Analyzing and Quantifying Antibody Structure Prediction Difficulty

Code and analysis for the MSc Bioinformatics & Systems Biology (joint UvA-VU) minor project thesis:
“Analyzing and Quantifying Antibody Structure Prediction Difficulty”  
Tina Ganjipour (VUNetID: mra160) — February 17, 2026  
Vrije Universiteit Amsterdam / Universiteit van Amsterdam, The Netherlands  
Project supervisor: Katharina Waury • Project investigator: Mitchell Evers • Examiner: Anna Heintz-Buschart


## Summary

Accurate antibody structural modeling is essential for therapeutic design and mechanistic understanding of immune recognition. While modern deep-learning predictors model antibody frameworks well, CDR-H3 remains the hardest region due to its exceptional diversity and conformational variability.

This project:
1. Curates a non-redundant benchmark of 698 antibody structures from SAbDab.
2. After submission of dataset to ABodyBuilder2 and IgFold, evaluates predictions against experimental structures.
3. Uses AlphaFold3 as a tertiary evaluator to contextualize inter-method disagreements (diagnostic, not the main benchmark).
4. Computes sequence-derived CDR-H3 features (length, entropy, composition extremity, motif rarity, etc.)
5. Trains ML models to predict:
   - expected CDR-H3 error (regression), and
   - probability that a target is difficult to predict/hard (calibrated classification)


## Repository content
* `data/`

  * `raw/`

    * `sabdab/` — SAbDab exports
    * `pdb_mmcif/` — experimental structures in mmCIF format
  * `processed/` — processed and curated data products

    * `manifest/` — sample sheets, pair indices, manifests
    * `exports/` — cleaned tables used in analysis (CSV/TSV)
    * `qc/` — quality-control outputs (mmCIF validation, consistency checks)
  * `.ipynb_checkpoints/` — Jupyter notebook artifacts

* `rmsd/` — structural evaluation outputs

  * `rcsb/` — experimental structures converted to PDB format for RMSD evaluation
  * `whole_cohort_outputs/` — RMSD analyses, ECDFs, scatter plots, and violin plots
  * `results_seq.csv` — sequence-level RMSD summaries
  * `predictions.csv` — predictor-specific RMSD tables
  * `h3_features.csv` — final CDR-H3 feature table

* `model/` — difficulty score modeling and analysis

  * `train_difficulty_score*.py` — difficulty score training scripts (multiple versions)
  * `h3_features.csv` — input of difficulty score model
  * `outputs/` — main model outputs (figures, CSVs, JSON summaries)
  * `outputs_engineered/` — engineered-feature models
  * `outputs_ablation*/` — feature ablation studies
  * `outputs_esm_alpha10/` — final ESM-based thesis models and plots
  * `out_diff/`, `out_bound/` — alternative modeling variants
  * `*.joblib` — trained regressors and classifiers

* `notebooks/`

  * `dataset.ipynb` — dataset construction and inspection
  * `ANARCI.py` — antibody numbering and annotation helpers

* `src/` — reusable Python utilities (mapping, RMSD computation, QC helpers)

* `environment.yml` — micromamba environment specification

* `environment.lock.yml` — locked environment for reproducibility

* `.pre-commit-config.yaml` — formatting and hygiene hooks

* `.gitignore`

## Methods overview (pipeline)

### 1) Dataset curation (SAbDab → sample sheet)
- Start from SAbDab metadata export and the non-redundant set.
- Filter to X-ray structures ≤ 3.0 Å resolution.
- Filter deposition dates to reduce overlap with model training data (see thesis for the exact cutoff used).
- Create a sample sheet of (PDB ID, heavy chain ID, light chain ID) triplets and associated metadata using ANARCI (Chothia numbering).

### 2) Experimental structure fetch + validation
- Download mmCIF files from the wwPDB / RCSB distribution.
- Parse/validate structures (Gemmi).
- Resolve altloc handling by selecting primary conformers (blank or “A”).
- Build a cached representation per target (coordinates + sequences).

### 3) Predictor inference
- Run IgFold and ABodyBuilder2
- Runs were executed on an HPC environment (CPU partition).

### 4) Chain mapping + residue correspondence
- Map predicted to experimental VH/VL using ANARCI (Chothia numbering) to ensure immunoglobulin-aware residue matching.
- Accept mappings only if identity/coverage/length thresholds are satisfied (see thesis Methods for exact thresholds).

### 5) Structural evaluation
Compute RMSD using backbone atoms (N, CA, C, O) and rigid-body superposition (Kabsch):
- Fv RMSD (context-aligned) — global measure (sensitive to VH–VL orientation)
- Fv RMSD excluding CDR-H3 — framework quality without H3 penalty
- CDR-H3 context RMSD — framework-aligned loop placement + shape
- CDR-H3 local RMSD — loop-aligned (isolates loop shape independent of placement)

### 6) Feature extraction + difficulty score
- Compute CDR-H3 sequence-derived features (length, entropy, composition, rarity, etc.).
- Train:
  - XGBoost regression (predict expected RMSD)
  - XGBoost classifier (predict hard/not-hard), with probability calibration (Platt scaling via `CalibratedClassifierCV`)
- Interpret with SHAP.


## Key results

- IgFold and ABodyBuilder2 show similar global Fv accuracy (~1.2 Å mean RMSD), and framework-only RMSD improves when excluding CDR-H3.
- Sequence features associated with increased difficulty include: CDR-H3 length, higher entropy, compositional extremity (charge/hydrophobicity), and rare motifs.
- A sequence-only “difficulty score” was trained using XGBoost (classification + regression) and interpreted with SHAP. CDR-H3 length and protein language-model embeddings (ESM2) carried the strongest signal.
