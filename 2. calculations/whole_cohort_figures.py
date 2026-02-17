from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def _as_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def rmsd_bin(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    if x <= 2:
        return "Excellent (≤2Å)"
    if x <= 4:
        return "Good (2–4Å)"
    if x <= 6:
        return "Adequate (4–6Å)"
    return "Poor (>6Å)"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def save_fig(outdir: Path, name: str) -> Path:
    p = outdir / name
    plt.tight_layout()
    plt.savefig(p, dpi=220)
    plt.close()
    return p


def categorize_error(msg: str) -> str:
    m = (msg or "").strip().lower()
    if not m:
        return "ok"
    if "ref_pdb not found" in m:
        return "missing_ref"
    if "pred_pdb not found" in m:
        return "missing_pred"
    if "parse error" in m:
        return "parse_error"
    if "no reliable chain map" in m:
        return "no_chain_map"
    if "expected 2 chains" in m:
        return "one_chain_or_fragment"
    return "other_error"


def summarize_robustness(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["error_cat"] = d["error"].fillna("").map(categorize_error)

    d["rmsd_fv_seqmap"] = _as_float_series(d.get("rmsd_fv_seqmap", pd.Series(dtype=float)))
    d["h3_rmsd"] = _as_float_series(d.get("h3_rmsd", pd.Series(dtype=float)))
    d["n_chains_used"] = pd.to_numeric(d.get("n_chains_used", pd.Series(dtype=float)), errors="coerce")

    d["ok_global"] = (d["error_cat"] == "ok") & (d["n_chains_used"] == 2) & np.isfinite(d["rmsd_fv_seqmap"])
    d["ok_h3"] = (d["h3_error"].fillna("").astype(str).str.strip() == "") & np.isfinite(d["h3_rmsd"])

    rows: List[Dict[str, object]] = []
    for method, g in d.groupby("method", dropna=False):
        row: Dict[str, object] = {"method": method, "n_total": int(len(g))}
        row["n_ok_global"] = int(g["ok_global"].sum())
        row["n_ok_h3"] = int(g["ok_h3"].sum())

        for cat, c in g["error_cat"].value_counts().items():
            if cat == "ok":
                continue
            row[f"n_{cat}"] = int(c)

        h3_err = g["h3_error"].fillna("").astype(str).str.strip()
        row["n_h3_error_nonempty"] = int((h3_err != "").sum())
        rows.append(row)

    return pd.DataFrame(rows).fillna(0).sort_values("method")


def paired_diff(df: pd.DataFrame, value_col: str, ok_mask: pd.Series) -> pd.Series:
    d = df.loc[ok_mask, ["id", "method", value_col]].copy()
    d[value_col] = _as_float_series(d[value_col])
    pv = d.pivot_table(index="id", columns="method", values=value_col, aggfunc="first").dropna()
    if "ABodyBuilder2" not in pv.columns or "IGFold" not in pv.columns:
        return pd.Series(dtype=float)
    return pv["ABodyBuilder2"] - pv["IGFold"]


def unique_by_id(df: pd.DataFrame, mask: pd.Series, *, id_col: str = "id") -> pd.DataFrame:
    d = df.loc[mask].copy()
    d = d.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
    return d


def plot_ecdf_by_method(
    df: pd.DataFrame,
    value_col: str,
    ok_mask: pd.Series,
    title: str,
    xlabel: str,
    outdir: Path,
    fname: str,
) -> None:
    plt.figure()
    for method, g in df.loc[ok_mask].groupby("method"):
        vals = _as_float_series(g[value_col]).to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals) < 5:
            continue
        x, y = ecdf(vals)
        method_s = str(method)
        linestyle = "--" if method_s == "ABodyBuilder2" else "-"
        plt.plot(
            x,
            y,
            linestyle=linestyle,
            linewidth=2.0,
            alpha=0.9,
            label=f"{method_s} (n={len(vals)})",
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("ECDF")
    plt.legend()
    save_fig(outdir, fname)


def plot_violin_by_method(
    df: pd.DataFrame,
    value_col: str,
    ok_mask: pd.Series,
    title: str,
    ylabel: str,
    outdir: Path,
    fname: str,
    *,
    ylim: Tuple[float, float] | None = None,
) -> None:
    d = df.loc[ok_mask, ["method", value_col]].copy()
    d[value_col] = _as_float_series(d[value_col])
    d = d[np.isfinite(d[value_col])]

    methods = sorted(d["method"].unique().tolist())
    data = [d.loc[d["method"] == m, value_col].to_numpy() for m in methods]

    plt.figure()
    plt.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    plt.xticks(np.arange(1, len(methods) + 1), methods, rotation=0)
    plt.title(title)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    save_fig(outdir, fname)


def plot_hist_paired(diff: pd.Series, title: str, xlabel: str, outdir: Path, fname: str, *, bins: int = 40) -> None:
    vals = diff.to_numpy()
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return
    plt.figure()
    plt.hist(vals, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    save_fig(outdir, fname)


def plot_scatter_feature_vs_h3(
    df: pd.DataFrame,
    ok_h3: pd.Series,
    feature_col: str,
    outdir: Path,
    fname_prefix: str,
    *,
    xlabel: str | None = None,
) -> None:
    if feature_col not in df.columns:
        return

    d = df.loc[ok_h3, ["method", feature_col, "h3_rmsd"]].copy()
    d[feature_col] = pd.to_numeric(d[feature_col], errors="coerce")
    d["h3_rmsd"] = _as_float_series(d["h3_rmsd"])
    d = d[np.isfinite(d[feature_col]) & np.isfinite(d["h3_rmsd"])]

    if d.empty:
        return

    for method, g in d.groupby("method"):
        x = g[feature_col].to_numpy(dtype=float)
        y = g["h3_rmsd"].to_numpy(dtype=float)

        plt.figure()
        plt.scatter(x, y, s=12, alpha=0.6)

        if np.all(np.isclose(x, x.astype(int))) and len(np.unique(x.astype(int))) <= 30:
            xs = np.unique(x.astype(int))
            q_rows = []
            for xv in xs:
                mask = x.astype(int) == xv
                if mask.sum() < 10:
                    continue
                yy = y[mask]
                q_rows.append(
                    (
                        float(xv),
                        float(np.quantile(yy, 0.10)),
                        float(np.quantile(yy, 0.50)),
                        float(np.quantile(yy, 0.90)),
                    )
                )
            if q_rows:
                q = pd.DataFrame(q_rows, columns=["x", "q10", "q50", "q90"])
                plt.plot(q["x"], q["q50"])
                plt.plot(q["x"], q["q10"])
                plt.plot(q["x"], q["q90"])
        else:
            bins = np.linspace(np.nanmin(x), np.nanmax(x), 15)
            q_rows = []
            for b0, b1 in zip(bins[:-1], bins[1:]):
                mask = (x >= b0) & (x < b1)
                if mask.sum() < 25:
                    continue
                yy = y[mask]
                q_rows.append(
                    (
                        (b0 + b1) / 2.0,
                        float(np.quantile(yy, 0.10)),
                        float(np.quantile(yy, 0.50)),
                        float(np.quantile(yy, 0.90)),
                    )
                )
            if q_rows:
                q = pd.DataFrame(q_rows, columns=["xmid", "q10", "q50", "q90"])
                plt.plot(q["xmid"], q["q50"])
                plt.plot(q["xmid"], q["q10"])
                plt.plot(q["xmid"], q["q90"])

        plt.title(f"H3 RMSD vs {feature_col} ({method})")
        plt.xlabel(xlabel or feature_col)
        plt.ylabel("H3 RMSD (Å)")
        save_fig(outdir, f"{fname_prefix}_{feature_col}_{method}.png")


def plot_box_by_category(
    df: pd.DataFrame,
    ok_h3: pd.Series,
    category_col: str,
    title: str,
    outdir: Path,
    fname_prefix: str,
    *,
    ordered_categories: List[str] | None = None,
) -> None:
    d = df.loc[ok_h3, ["method", category_col, "h3_rmsd"]].copy()
    d["h3_rmsd"] = _as_float_series(d["h3_rmsd"])
    d = d[np.isfinite(d["h3_rmsd"])]

    for method, g in d.groupby("method"):
        gg = g.copy()
        if ordered_categories is not None:
            gg[category_col] = pd.Categorical(gg[category_col], categories=ordered_categories, ordered=True)
        gg = gg.dropna(subset=[category_col]).sort_values(category_col)

        cats = gg[category_col].astype(str).unique().tolist()
        data = [gg.loc[gg[category_col].astype(str) == c, "h3_rmsd"].to_numpy() for c in cats]
        if len(data) < 2:
            continue

        plt.figure()
        try:
            plt.boxplot(data, tick_labels=cats, showfliers=False)
        except TypeError:
            plt.boxplot(data, labels=cats, showfliers=False)
        plt.title(f"{title} ({method})")
        plt.ylabel("H3 RMSD (Å)")
        plt.xticks(rotation=0)
        save_fig(outdir, f"{fname_prefix}_{method}.png")


def plot_feature_distributions(
    df: pd.DataFrame,
    ok_mask: pd.Series,
    feature_col: str,
    outdir: Path,
    *,
    title_prefix: str = "",
    xlabel: str | None = None,
) -> None:
    if feature_col not in df.columns:
        return

    plot_ecdf_by_method(
        df,
        feature_col,
        ok_mask,
        title=f"{title_prefix}{feature_col}: ECDF by method",
        xlabel=xlabel or feature_col,
        outdir=outdir,
        fname=f"ecdf_feature_{feature_col}.png",
    )
    plot_violin_by_method(
        df,
        feature_col,
        ok_mask,
        title=f"{title_prefix}{feature_col}: distribution by method",
        ylabel=xlabel or feature_col,
        outdir=outdir,
        fname=f"violin_feature_{feature_col}.png",
    )


def summarize_feature_stats(
    df: pd.DataFrame,
    ok_mask: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str],
    outdir: Path,
) -> None:
    d = df.loc[ok_mask].copy()

    miss_rows = []
    for c in numeric_features + categorical_features + ["h3_rmsd"]:
        if c not in d.columns:
            miss_rows.append({"col": c, "n_present": 0, "n_missing": int(len(d)), "frac_missing": 1.0})
            continue
        s = d[c]
        n_missing = int(pd.isna(s).sum())
        miss_rows.append(
            {
                "col": c,
                "n_present": int(len(s) - n_missing),
                "n_missing": n_missing,
                "frac_missing": n_missing / max(1, len(s)),
            }
        )
    (
        pd.DataFrame(miss_rows)
        .sort_values("frac_missing", ascending=False)
        .to_csv(outdir / "feature_missingness_ok_h3.csv", index=False)
    )

    rows = []
    for method, g in d.groupby("method"):
        for col in numeric_features:
            if col not in g.columns:
                continue
            s = pd.to_numeric(g[col], errors="coerce").astype(float)
            s = s[np.isfinite(s)]
            if len(s) == 0:
                continue
            rows.append(
                {
                    "method": method,
                    "feature": col,
                    "n": int(len(s)),
                    "mean": float(np.mean(s)),
                    "std": float(np.std(s, ddof=1)) if len(s) > 1 else np.nan,
                    "median": float(np.median(s)),
                    "q25": float(np.quantile(s, 0.25)),
                    "q75": float(np.quantile(s, 0.75)),
                    "min": float(np.min(s)),
                    "max": float(np.max(s)),
                }
            )
    (
        pd.DataFrame(rows)
        .sort_values(["feature", "method"])
        .to_csv(outdir / "feature_summary_numeric_by_method_ok_h3.csv", index=False)
    )

    cat_rows = []
    for method, g in d.groupby("method"):
        for col in categorical_features:
            if col not in g.columns:
                continue
            vc = g[col].value_counts(dropna=False)
            for k, v in vc.items():
                cat_rows.append({"method": method, "feature": col, "value": k, "n": int(v)})
    (
        pd.DataFrame(cat_rows)
        .sort_values(["feature", "method", "value"])
        .to_csv(outdir / "feature_summary_categorical_by_method_ok_h3.csv", index=False)
    )


def spearman_table(df: pd.DataFrame, ok_h3: pd.Series, features: List[str], outdir: Path) -> None:
    d = df.loc[ok_h3].copy()
    d["h3_rmsd"] = _as_float_series(d["h3_rmsd"])
    d = d[np.isfinite(d["h3_rmsd"])]

    rows = []
    for scope, gg in [("pooled", d)] + [(f"method={m}", g) for m, g in d.groupby("method")]:
        y = gg["h3_rmsd"].to_numpy(dtype=float)
        for col in features:
            if col not in gg.columns:
                continue
            x = pd.to_numeric(gg[col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if int(mask.sum()) < 30:
                continue
            rho, p = stats.spearmanr(x[mask], y[mask])
            rows.append(
                {"scope": scope, "feature": col, "n": int(mask.sum()), "spearman_rho": float(rho), "p_value": float(p)}
            )
    pd.DataFrame(rows).sort_values(["scope", "p_value"]).to_csv(outdir / "spearman_h3_rmsd_vs_features_ok_h3.csv", index=False)


def ols_with_hc3(X: np.ndarray, y: np.ndarray, colnames: List[str]) -> pd.DataFrame:
    n, p = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)

    beta = XtX_inv @ (X.T @ y)
    resid = y - (X @ beta)

    rank = int(np.linalg.matrix_rank(X))
    dof = max(1, n - rank)

    H = X @ XtX_inv @ X.T
    h = np.clip(np.diag(H), 1e-9, 1 - 1e-9)

    u = resid / (1.0 - h)
    S = (X.T * (u * u)) @ X
    cov_hc3 = XtX_inv @ S @ XtX_inv
    se_hc3 = np.sqrt(np.diag(cov_hc3))

    t_hc3 = beta / se_hc3
    p_hc3 = 2.0 * stats.t.sf(np.abs(t_hc3), df=dof)

    ci_lo = beta + stats.t.ppf(0.025, df=dof) * se_hc3
    ci_hi = beta + stats.t.ppf(0.975, df=dof) * se_hc3

    return pd.DataFrame(
        {"term": colnames, "beta": beta, "se_hc3": se_hc3, "t_hc3": t_hc3, "p_hc3": p_hc3, "ci95_lo_hc3": ci_lo, "ci95_hi_hc3": ci_hi}
    )


def build_regression_table(df: pd.DataFrame, ok_h3: pd.Series, outdir: Path) -> pd.DataFrame:
    d = df.loc[ok_h3].copy()
    y = np.log(_as_float_series(d["h3_rmsd"]).to_numpy() + 1e-3)

    h3_len = pd.to_numeric(d["h3_len"], errors="coerce").to_numpy(dtype=float)
    rarity = _as_float_series(d["h3_kmer3_rarity_dataset"]).to_numpy(dtype=float)
    p_count = pd.to_numeric(d["h3_p_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
    has_pp = (pd.to_numeric(d["h3_p_max_run"], errors="coerce").fillna(0).to_numpy(dtype=float) >= 2).astype(float)

    p_ge1 = (p_count >= 1).astype(float)
    p_ge2 = (p_count >= 2).astype(float)

    method_is_ab2 = (d["method"].astype(str) == "ABodyBuilder2").astype(float).to_numpy(dtype=float)
    bound = (d.get("bound_state", pd.Series([""] * len(d))).astype(str) == "bound").astype(float).to_numpy(dtype=float)

    X_cols = [
        ("intercept", np.ones_like(h3_len)),
        ("h3_len", h3_len),
        ("method_is_ab2", method_is_ab2),
        ("bound_is_bound", bound),
        ("p_ge1", p_ge1),
        ("p_ge2", p_ge2),
        ("has_PP", has_pp),
        ("kmer3_rarity", rarity),
    ]
    X = np.column_stack([v for _, v in X_cols])
    colnames = [k for k, _ in X_cols]

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    out = ols_with_hc3(X[mask], y[mask], colnames=colnames)
    out.to_csv(outdir / "ols_log_h3_rmsd_hc3.csv", index=False)
    return out


def plot_ecdf_feature_by_hardness(
    df_targets: pd.DataFrame,
    feature_col: str,
    outdir: Path,
    fname: str,
    *,
    title: str | None = None,
    xlabel: str | None = None,
) -> None:
    if feature_col not in df_targets.columns:
        return
    d = df_targets[[feature_col, "hard_label"]].copy()
    d[feature_col] = pd.to_numeric(d[feature_col], errors="coerce")
    d = d[np.isfinite(d[feature_col])]

    plt.figure()
    for lab, g in d.groupby("hard_label"):
        vals = g[feature_col].to_numpy(dtype=float)
        if len(vals) < 10:
            continue
        x, y = ecdf(vals)
        plt.plot(x, y, linewidth=2.0, alpha=0.9, label=f"{lab} (n={len(vals)})")

    plt.title(title or f"{feature_col}: distribution by difficulty group")
    plt.xlabel(xlabel or feature_col)
    plt.ylabel("ECDF")
    plt.legend()
    save_fig(outdir, fname)


def make_target_level_table(df: pd.DataFrame, ok_h3: pd.Series, *, rmsd_col: str = "h3_rmsd") -> pd.DataFrame:
    d = df.loc[ok_h3, ["id", "method", rmsd_col]].copy()
    d[rmsd_col] = _as_float_series(d[rmsd_col])
    d = d[np.isfinite(d[rmsd_col])]

    # One row per target: median across methods
    tgt = d.groupby("id", as_index=False)[rmsd_col].median().rename(columns={rmsd_col: "h3_rmsd_target"})
    return tgt


def hard_vs_nonhard_stats(df_targets: pd.DataFrame, features: List[str], outdir: Path) -> None:
    rows = []
    for col in features:
        if col not in df_targets.columns:
            continue
        d = df_targets[[col, "hard_label"]].copy()
        d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d[np.isfinite(d[col])]
        a = d.loc[d["hard_label"] == "difficult", col].to_numpy()
        b = d.loc[d["hard_label"] == "not difficult", col].to_numpy()
        if len(a) < 10 or len(b) < 10:
            continue

        # Nonparametric test
        U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {
                "feature": col,
                "n_hard": int(len(a)),
                "n_nonhard": int(len(b)),
                "median_hard": float(np.median(a)),
                "median_nonhard": float(np.median(b)),
                "p_mannwhitney": float(p),
            }
        )
    pd.DataFrame(rows).sort_values("p_mannwhitney").to_csv(outdir / "hard_vs_nonhard_feature_tests.csv", index=False)

def wilcoxon_paired_test(diff: pd.Series, *, alternative: str = "two-sided") -> Dict[str, float]:
    """
    Wilcoxon signed-rank test on paired differences (AB2 - IgFold).
    Returns robust summary + p-value.

    Notes:
    - Drops NaNs/inf and (optionally) handles zeros via zero_method.
    - SciPy wilcoxon ignores zeros under zero_method="wilcox".
    """
    x = pd.to_numeric(diff, errors="coerce").astype(float).to_numpy()
    x = x[np.isfinite(x)]

    out: Dict[str, float] = {
        "n_pairs": float(len(x)),
        "mean_delta": float(np.mean(x)) if len(x) else np.nan,
        "median_delta": float(np.median(x)) if len(x) else np.nan,
        "q25_delta": float(np.quantile(x, 0.25)) if len(x) else np.nan,
        "q75_delta": float(np.quantile(x, 0.75)) if len(x) else np.nan,
        "wilcoxon_stat": np.nan,
        "wilcoxon_p": np.nan,
    }
    if len(x) < 10:
        return out

    # zero_method="wilcox" -> drops zero diffs
    res = stats.wilcoxon(
        x,
        zero_method="wilcox",
        alternative=alternative,
        mode="auto",
    )
    out["wilcoxon_stat"] = float(res.statistic)
    out["wilcoxon_p"] = float(res.pvalue)
    return out


def add_wilcoxon_results(
    out_rows: List[Dict[str, object]],
    *,
    metric: str,
    diff: pd.Series,
    alternative: str = "two-sided",
) -> None:
    r = wilcoxon_paired_test(diff, alternative=alternative)
    out_rows.append(
        {
            "metric": metric,
            "n_pairs": int(r["n_pairs"]) if np.isfinite(r["n_pairs"]) else 0,
            "mean_delta_ab2_minus_igfold": r["mean_delta"],
            "median_delta_ab2_minus_igfold": r["median_delta"],
            "q25_delta": r["q25_delta"],
            "q75_delta": r["q75_delta"],
            "wilcoxon_stat": r["wilcoxon_stat"],
            "wilcoxon_p": r["wilcoxon_p"],
        }
    )

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="h3_features.csv")
    ap.add_argument("--out_dir", default="whole_cohort_outputs")
    ap.add_argument("--hard_thr", type=float, default=3.0, help="Hardness threshold on target-level H3 RMSD (Å)")
    args = ap.parse_args()

    outdir = ensure_dir(Path(args.out_dir))
    df = pd.read_csv(Path(args.in_csv))

    if "error" not in df.columns:
        df["error"] = ""
    if "method" not in df.columns:
        raise SystemExit("ERROR: input CSV must contain 'method'")
    if "id" not in df.columns:
        raise SystemExit("ERROR: input CSV must contain 'id'")

    # Standardize global FV RMSD column name
    if "rmsd_fv_seqmap" not in df.columns:
        if "rmsd_fv_all_ctx" in df.columns:
            df["rmsd_fv_seqmap"] = df["rmsd_fv_all_ctx"]
        else:
            raise SystemExit("ERROR: need 'rmsd_fv_seqmap' or 'rmsd_fv_all_ctx' in input CSV")

    # Standardize H3 RMSD column name
    if "h3_rmsd" not in df.columns:
        if "rmsd_h3_ctx" in df.columns:
            df["h3_rmsd"] = df["rmsd_h3_ctx"]
        elif "rmsd_h3_local" in df.columns:
            df["h3_rmsd"] = df["rmsd_h3_local"]
        else:
            raise SystemExit("ERROR: need 'h3_rmsd' or one of {'rmsd_h3_ctx','rmsd_h3_local'} in input CSV")

    # H3 error column (if absent, treat as OK unless RMSD is NaN)
    if "h3_error" not in df.columns:
        df["h3_error"] = ""

    df["rmsd_fv_seqmap"] = _as_float_series(df["rmsd_fv_seqmap"])
    df["h3_rmsd"] = _as_float_series(df["h3_rmsd"])
    df["n_chains_used"] = pd.to_numeric(df.get("n_chains_used", pd.Series(dtype=float)), errors="coerce")

    df["error_cat"] = df["error"].fillna("").map(categorize_error)
    ok_global = (df["error_cat"] == "ok") & (df["n_chains_used"] == 2) & np.isfinite(df["rmsd_fv_seqmap"])
    ok_h3 = (df["h3_error"].fillna("").astype(str).str.strip() == "") & np.isfinite(df["h3_rmsd"])

    # Target-level table (one row per id) for baseline hard/non-hard comparisons
    tgt = make_target_level_table(df, ok_h3, rmsd_col="h3_rmsd")
    HARD_THR = float(args.hard_thr)
    tgt["hard_label"] = np.where(
        tgt["h3_rmsd_target"] >= HARD_THR,
        "difficult",
        "not difficult",
    )

    # Since sequence-derived features are identical across methods, we take first row per id.
    df_ok_unique = unique_by_id(df, ok_h3, id_col="id")
    df_ok_unique["method"] = "Pooled"
    df_ok_unique = df_ok_unique.merge(tgt[["id", "h3_rmsd_target", "hard_label"]], on="id", how="left")
    ok_unique = pd.Series(True, index=df_ok_unique.index)

    # performance plots
    summarize_robustness(df).to_csv(outdir / "coverage_robustness_by_method.csv", index=False)

    plot_ecdf_by_method(
        df,
        "rmsd_fv_seqmap",
        ok_global,
        title="Global Fv Cα RMSD (orientation-sensitive): ECDF by method",
        xlabel="Global Fv RMSD (Å)",
        outdir=outdir,
        fname="ecdf_global_fv_rmsd.png",
    )
    plot_violin_by_method(
        df,
        "rmsd_fv_seqmap",
        ok_global,
        title="Global Fv Cα RMSD (orientation-sensitive): distribution by method",
        ylabel="Global Fv RMSD (Å)",
        outdir=outdir,
        fname="violin_global_fv_rmsd.png",
    )

    plot_ecdf_by_method(
        df,
        "h3_rmsd",
        ok_h3,
        title="Framework-aligned CDR-H3 RMSD: ECDF by method",
        xlabel="H3 RMSD (Å)",
        outdir=outdir,
        fname="ecdf_h3_rmsd.png",
    )
    plot_violin_by_method(
        df,
        "h3_rmsd",
        ok_h3,
        title="Framework-aligned CDR-H3 RMSD: distribution by method",
        ylabel="H3 RMSD (Å)",
        outdir=outdir,
        fname="violin_h3_rmsd.png",
    )

    diff_global = paired_diff(df, "rmsd_fv_seqmap", ok_global)
    diff_h3 = paired_diff(df, "h3_rmsd", ok_h3)
    # paired Wilcoxon signed-rank test on per-target paired deltas (ABodyBuilder2 − IGFold)
    wilcox_rows: List[Dict[str, object]] = []
    add_wilcoxon_results(wilcox_rows, metric="global_fv_rmsd", diff=diff_global)
    add_wilcoxon_results(wilcox_rows, metric="h3_rmsd", diff=diff_h3)
    
    pd.DataFrame(wilcox_rows).to_csv(outdir / "paired_wilcoxon_ab2_vs_igfold.csv", index=False)
    
    # print summary for logs
    for row in wilcox_rows:
        print(
            f"[{row['metric']}] n={row['n_pairs']}, "
            f"medianΔ={row['median_delta_ab2_minus_igfold']:.4f} Å, "
            f"p={row['wilcoxon_p']:.3g}"
        )

    diff_global.to_csv(outdir / "paired_diff_global_fv_rmsd_ab2_minus_igfold.csv", index=True)
    diff_h3.to_csv(outdir / "paired_diff_h3_rmsd_ab2_minus_igfold.csv", index=True)

    plot_hist_paired(
        diff_global,
        title=f"Paired differences: AB2 − IgFold global Fv RMSD (n={len(diff_global)})",
        xlabel="Δ global Fv RMSD (Å) (negative = AB2 better)",
        outdir=outdir,
        fname="hist_paired_diff_global_fv_rmsd.png",
    )
    plot_hist_paired(
        diff_h3,
        title=f"Paired differences: AB2 − IgFold H3 RMSD (n={len(diff_h3)})",
        xlabel="Δ H3 RMSD (Å) (negative = AB2 better)",
        outdir=outdir,
        fname="hist_paired_diff_h3_rmsd.png",
    )

    # Features
    numeric_features_all = [
        "h3_len",
        "h3_entropy",
        "h3_num_unique",
        "h3_kd_mean",
        "h3_frac_hydrophobic",
        "h3_net_charge_pH7",
        "h3_kmer3_rarity_dataset",
        "h3_frac_gly_pro",
        "h3_frac_aromatic",
        "pred_conf_h3_bfac_mean",
        "pred_conf_fv_bfac_mean",
    ]

    # For hard/non-hard baseline, use target-intrinsic (sequence-derived) features only
    seq_features = [
        "h3_len",
        "h3_entropy",
        "h3_num_unique",
        "h3_kd_mean",
        "h3_frac_hydrophobic",
        "h3_net_charge_pH7",
        "h3_kmer3_rarity_dataset",
        "h3_frac_gly_pro",
        "h3_frac_aromatic",
    ]

    # distributions of all numeric features in pooled unique table
    for col in numeric_features_all:
        if col not in df_ok_unique.columns:
            continue
        plot_feature_distributions(
            df_ok_unique,
            ok_unique,
            col,
            outdir,
            title_prefix="Feature distribution (unique IDs; ok_h3 pooled) — ",
            xlabel=col,
        )

    # hard vs non-hard ECDFs (sequence-only features)
    for col in seq_features:
        if col not in df_ok_unique.columns:
            continue

        plot_ecdf_feature_by_hardness(
            df_ok_unique,
            feature_col=col,
            outdir=outdir,
            fname=f"ecdf_feature_{col}_difficult_vs_notdifficult.png",
            title=f"{col}: difficult vs not difficult (targets; threshold={HARD_THR:.1f} Å)",
            xlabel=col,
        )

        # scatter vs RMSD (method-level)
        plot_scatter_feature_vs_h3(df, ok_h3, col, outdir, "scatter_h3_rmsd_vs")

    if "h3_has_glyco_motif" in df.columns:
        df["glyco_motif"] = pd.to_numeric(df["h3_has_glyco_motif"], errors="coerce").map(
            {1.0: "Motif present", 0.0: "Absent"}
        )
        plot_box_by_category(
            df,
            ok_h3,
            category_col="glyco_motif",
            title="H3 RMSD by N-glycosylation sequon (N-X-S/T, X≠P) in H3",
            outdir=outdir,
            fname_prefix="box_h3_rmsd_by_glyco_motif",
            ordered_categories=["Absent", "Motif present"],
        )

    summarize_feature_stats(
        df,
        ok_h3,
        numeric_features=numeric_features_all,
        categorical_features=["h3_has_glyco_motif"] if "h3_has_glyco_motif" in df.columns else [],
        outdir=outdir,
    )

    spearman_table(df, ok_h3, features=seq_features, outdir=outdir)

    # Regression table
    required_for_ols = {"h3_rmsd", "h3_len", "h3_kmer3_rarity_dataset", "h3_p_count", "h3_p_max_run"}
    if required_for_ols.issubset(set(df.columns)):
        build_regression_table(df, ok_h3, outdir)

    # Winner counts and RMSD bucket counts
    pv = (
        df.loc[ok_global, ["id", "method", "rmsd_fv_seqmap"]]
        .pivot_table(index="id", columns="method", values="rmsd_fv_seqmap", aggfunc="first")
        .dropna()
    )
    if "ABodyBuilder2" in pv.columns and "IGFold" in pv.columns:
        dwin = pv["ABodyBuilder2"] - pv["IGFold"]
        winner = np.where(dwin < -1e-9, "ABodyBuilder2", np.where(dwin > 1e-9, "IGFold", "Tie"))
        (
            pd.Series(winner)
            .value_counts()
            .rename_axis("winner")
            .reset_index(name="n_ids")
            .to_csv(outdir / "win_counts_global_fv_rmsd.csv", index=False)
        )

    dg = df.loc[ok_global, ["method", "rmsd_fv_seqmap"]].copy()
    dg["rmsd_bucket"] = dg["rmsd_fv_seqmap"].apply(rmsd_bin)
    dg.groupby(["method", "rmsd_bucket"]).size().reset_index(name="n").to_csv(
        outdir / "counts_by_global_fv_rmsd_bucket.csv", index=False
    )

    # Hard vs non-hard statistical table (Mann–Whitney) on sequence-only features
    hard_vs_nonhard_stats(df_ok_unique, seq_features, outdir)

    print(df_ok_unique["hard_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()

"""
References:
[1] Brennan Abanades, Wing Ki Wong, Fergus Boyles, and Charlotte M. Deane. 2023.
ImmuneBuilder: Deep-Learning models for predicting the structures of immune proteins.
Communications Biology 6, 1 (2023), 575.
https://doi.org/10.1038/s42003-023-04927-7

[2] Janez Demšar. 2006.
Statistical Comparisons of Classifiers over Multiple Data Sets.
Journal of Machine Learning Research 7 (2006), 1–30.

[3] Rob J. Hyndman and Yanan Fan. 1996.
Sample Quantiles in Statistical Packages.
The American Statistician 50, 4 (1996), 361–365.
https://doi.org/10.2307/2684934

[4] Wolfgang Kabsch. 1976.
A Solution for the Best Rotation to Relate Two Sets of Vectors.
Acta Crystallographica Section A 32, 5 (1976), 922–923.
https://doi.org/10.1107/S0567739476001873

[5] Bernard Rosner, Robert J. Glynn, and Mei-Ling T. Lee. 2006.
The Wilcoxon signed rank test for paired comparisons of clustered data.
Biometrics 62, 1 (2006), 185–192.
https://doi.org/10.1111/j.1541-0420.2005.00389.x

[6] Jeffrey A. Ruffolo, Jeremias Sulam, and Jeffrey J. Gray. 2023.
Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies.
Nature Communications 14, 1 (2023), 2389.
https://doi.org/10.1038/s41467-023-38063-x

[7] Frank Wilcoxon. 1945.
Individual Comparisons by Ranking Methods.
Biometrics Bulletin 1, 6 (1945), 80–83.
https://doi.org/10.2307/3001968
"""