from __future__ import annotations
import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CSV_PATH_DEFAULT = "h3_features.csv"
TARGET_RMSD_COL_DEFAULT = "rmsd_h3_ctx"
OUT_DIR_DEFAULT = "outputs"
RANDOM_STATE = 42
AGREE_TOL_DEFAULT = 1.5
HARD_GE_DEFAULT = 3.0
TEST_SIZE_DEFAULT = 0.15
N_SPLITS_DEFAULT = 5
N_REPEATS_DEFAULT = 5
CALIBRATION_METHOD = "sigmoid"
CALIBRATION_INNER_CV = 3

FEATURE_SET_DEFAULT = "len_comp_physchem"
FEATURE_SET_CHOICES = [
    "len_comp", # length + composition counts (number of unique residues + fractions + amino acid composition + entropy)
    "len_comp_physchem", # en_comp + (physicochemical = charge + hydrophobicity + aromatic + gly/pro + kd mean + hydropathy + aromatic)
    "engineered", # all engineered features (whatever detect_numeric_feature_cols returns)
    "engineered_esm", # main final model
    "kmer_rarity", # k-mer3 (or 3-mer) = a sequence fragment of length 3 amino acids, where k = 1 is single amino acid, k = 2 is dipeptide, and k = 3 is tripeptide (kmer3) (to show a feature we assumed it was correlated with poor cdr-h3 prediction but showed no signal in the model)
]

MAIN_FEATURE_SETS = ["len_comp", "len_comp_physchem", "engineered_esm"]
CONFORMAL_ALPHA_DEFAULT = 0.10 # miscoverage rate for prediction intervals
# features banned for model to see because it would be like giving the answers to the student on a test
BANNED_FEATURE_RE = re.compile(
    r"(rmsd|error|label|target|y_|confidence|conf|plddt|pae|iptm|ptm|score|prob)",
    flags=re.IGNORECASE,
)
# composition
LEN_COMP_RE = re.compile(
    r"(len|length|count|counts|freq|fraction|frac|composition|aa_|residue|unique|entropy)",
    flags=re.IGNORECASE,
) # physicochemical
PHYSCHEM_RE = re.compile(
    r"(charge|net_charge|hydro|hydroph|kyte|kd|hydropathy|aromatic|gly|pro|polar|aliphatic)",
    flags=re.IGNORECASE,
) # kmerrarity
KMER_RE = re.compile(
    r"(kmer|3mer|tri|tripep|ngram|rarity|rare|motif)",
    flags=re.IGNORECASE,
)

def hash_seq(seq: str) -> str:
    return hashlib.md5(seq.encode("utf-8")).hexdigest()

def detect_numeric_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    features starting with h3_ (except h3_seq)
    + excluding anything that looks like label/score leakage
    """
    h3_cols = [c for c in df.columns if c.startswith("h3_")]
    keep = [c for c in h3_cols if c != "h3_seq" and not BANNED_FEATURE_RE.search(c)]
    if not keep:
        raise ValueError("no usable feature columns")
    return keep

def select_feature_subset(all_cols: List[str], feature_set: str) -> List[str]:
    cols = list(all_cols)

    if feature_set in {"engineered", "engineered_esm"}:
        keep = cols
    elif feature_set == "len_comp":
        keep = [c for c in cols if LEN_COMP_RE.search(c) and not PHYSCHEM_RE.search(c)]
    elif feature_set == "len_comp_physchem":
        keep = [c for c in cols if LEN_COMP_RE.search(c) or PHYSCHEM_RE.search(c)]
    elif feature_set == "kmer_rarity":
        keep = [c for c in cols if KMER_RE.search(c)]
    else:
        raise ValueError(f"unknown")

    if not keep:
        raise ValueError(f"produced 0 columns.")
    return keep


def make_preprocessor(feature_cols: List[str], scale: bool) -> ColumnTransformer:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[("num", Pipeline(steps), feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def precision_recall_at_k(
    y_true_01: np.ndarray,
    scores: np.ndarray,
    ks: List[int],
) -> Dict[str, Dict[str, float]]:
    y = np.asarray(y_true_01, dtype=int)
    s = np.asarray(scores, dtype=float)

    n_pos = int(y.sum())
    order = np.argsort(-s)
    out: Dict[str, Dict[str, float]] = {}

    for k in ks:
        kk = int(min(max(k, 1), len(y)))
        topk = order[:kk]
        tp = int(y[topk].sum())
        precision = tp / kk
        recall = 0.0 if n_pos == 0 else tp / n_pos
        out[f"p@{k}"] = {
            "precision": float(precision),
            "recall": float(recall),
            "tp": float(tp),
            "k_eff": float(kk),
            "n_pos": float(n_pos),
        }
    return out

def choose_threshold_on_train(y_true_01: np.ndarray, p1: np.ndarray) -> float:
    """
    picks threshold maximizing F1, using the Precision-Recall curve thresholds
    """
    y = np.asarray(y_true_01, dtype=int)
    p = np.asarray(p1, dtype=float)

    if np.unique(y).size < 2:
        return 0.5

    precision, recall, probability_cutoff_for_classifying_hard = precision_recall_curve(y, p)
    if probability_cutoff_for_classifying_hard.size == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]
    denom = precision + recall
    f1 = np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)

    best_i = int(np.argmax(f1))
    return float(probability_cutoff_for_classifying_hard[best_i])

def plot_pr_curve(y_true_01: np.ndarray, p1: np.ndarray, out_png: Path, title: str) -> None:
    precision, rec, _ = precision_recall_curve(y_true_01, p1)
    ap = average_precision_score(y_true_01, p1) if int(np.sum(y_true_01)) > 0 else float("nan") # ap = average precision (area under Precision-Recall curve)

    plt.figure()
    plt.plot(rec, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} (AP={ap:.3f})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_roc_curve(y_true_01: np.ndarray, p1: np.ndarray, out_png: Path, title: str) -> None:
    if np.unique(y_true_01).size < 2:
        plt.figure()
        plt.title(f"{title} (AUC=nan; one-class)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close()
        return

    fpr, tpr, _ = roc_curve(y_true_01, p1)
    auc = roc_auc_score(y_true_01, p1)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} (AUC={auc:.3f})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_reliability(y_true_01: np.ndarray, p1: np.ndarray, out_png: Path, title: str) -> None:
    if np.unique(y_true_01).size < 2:
        plt.figure()
        plt.title(f"{title} (one-class; not meaningful)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close()
        return

    frac_pos, mean_pred = calibration_curve(y_true_01, p1, n_bins=10, strategy="uniform")

    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_reg_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_png: Path, title: str) -> None:
    plt.figure()
    plt.scatter(y_true, y_pred, s=18)
    plt.xlabel("True RMSD (Å)")
    plt.ylabel("Predicted RMSD (Å)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_reg_error_hist(err: np.ndarray, out_png: Path, title: str) -> None:
    plt.figure()
    plt.hist(err, bins=30)
    plt.xlabel("Prediction error (pred - true) (Å)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def compute_esm_embeddings(
    seqs: List[str],
    model_name: str,
    cache_path: Path,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    mean-pooled ESM2 embeddings sequence-only, from h3_seq
    cache is a CSV created by seq_hash and model_name
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    want = pd.DataFrame({"seq": [str(s) for s in seqs]})
    want["seq_hash"] = want["seq"].map(hash_seq)
    want["model_name"] = model_name

    cached = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
    if cached.empty:
        cached = pd.DataFrame(columns=["seq_hash", "seq", "model_name"])

    cached = cached.drop_duplicates(subset=["seq_hash", "model_name"], keep="last").copy()

    have = set(cached.loc[cached["model_name"] == model_name, "seq_hash"].astype(str).tolist())
    missing = want[~want["seq_hash"].isin(have)]

    if not missing.empty:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval().to(device)

        rows: List[Dict[str, Any]] = []
        with torch.no_grad():
            for seq, sh in zip(missing["seq"].tolist(), missing["seq_hash"].tolist()):
                tokens = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
                tokens = {k: v.to(device) for k, v in tokens.items()}
                out = model(**tokens)

                last = out.last_hidden_state.squeeze(0)  # [L, D]
                core = last[1:-1, :] if last.shape[0] >= 3 else last
                vec = core.mean(dim=0).detach().cpu().numpy().astype(np.float32)

                row: Dict[str, Any] = {"seq_hash": sh, "seq": seq, "model_name": model_name}
                row.update({f"esm_{i}": float(v) for i, v in enumerate(vec)})
                rows.append(row)

        new_df = pd.DataFrame(rows)
        cached = pd.concat([cached, new_df], ignore_index=True)
        cached = cached.drop_duplicates(subset=["seq_hash", "model_name"], keep="last")
        cached.to_csv(cache_path, index=False)

    cached_m = cached[cached["model_name"] == model_name].copy()
    cached_m = cached_m.drop(columns=["seq"], errors="ignore")
    cached_m = cached_m.drop_duplicates(subset=["seq_hash"], keep="last")

    merged = want.merge(cached_m, on=["seq_hash", "model_name"], how="left")

    emb_cols = [c for c in merged.columns if c.startswith("esm_")]
    if merged[emb_cols].isna().any().any():
        raise RuntimeError("missing embeddings")

    return merged[["seq"] + emb_cols].copy()

def conformal_radius(residuals: np.ndarray, alpha: float) -> float:
    """
    it computes a conformal prediction radius (a single number showing typical worst-case absolute error) from the model’s residuals, 
    so we can form a prediction interval: model’s prediction of y±radius
    """
    r = np.asarray(residuals, dtype=float)
    r = np.abs(r[~np.isnan(r)])
    if r.size == 0:
        return float("nan")

    n = r.size
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(np.partition(r, k - 1)[k - 1])

@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame
    y_regression: np.ndarray
    y_classification: np.ndarray
    ids: np.ndarray
    feature_cols: List[str]
    target_rmsd_col: str
    hard_ge: float
    agree_tol: float
    feature_set: str

def build_dataset(
    csv_path: Path,
    target_rmsd_col: str,
    agree_tol: float,
    hard_ge: float,
    feature_set: str,
    use_esm: bool,
    esm_model: str,
    esm_cache_path: Path,
) -> Dataset:
    df = pd.read_csv(csv_path)

    required = {"id", "method", target_rmsd_col, "h3_seq"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"these required columns are missing: {missing}")

    df = df.copy()
    df[target_rmsd_col] = pd.to_numeric(df[target_rmsd_col], errors="coerce")

    # only keeping sequences (pdb ids) that got processed by both methods and the RMSDs are also present
    ids_with_two = df.groupby("id")["method"].nunique()
    ids_with_two = ids_with_two[ids_with_two == 2].index
    df = df[df["id"].isin(ids_with_two)].dropna(subset=[target_rmsd_col]).copy()
    piv = df.pivot_table(index="id", columns="method", values=target_rmsd_col, aggfunc="first").dropna()
    method_cols = list(piv.columns)
    diff = (piv[method_cols[0]] - piv[method_cols[1]]).abs()
    keep_ids = diff[diff <= agree_tol].index.astype(str)
    df2 = df[df["id"].astype(str).isin(keep_ids)].copy()
    agg = df2.groupby("id")[target_rmsd_col].median().reset_index().rename(columns={target_rmsd_col: "y_regression"})
    numeric_cols_all = detect_numeric_feature_cols(df2)
    numeric_cols = select_feature_subset(numeric_cols_all, feature_set)
    feats_num = df2.sort_values(["id", "method"]).groupby("id")[numeric_cols].first().reset_index()
    feats_seq = df2.sort_values(["id", "method"]).groupby("id")["h3_seq"].first().reset_index()
    out = agg.merge(feats_num, on="id", how="left").merge(feats_seq, on="id", how="left")
    out = out.dropna(subset=numeric_cols + ["y_regression", "h3_seq"]).copy()
    X = out[numeric_cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)

    feature_cols = list(numeric_cols)
    if feature_set == "engineered_esm":
        emb = compute_esm_embeddings(
            seqs=out["h3_seq"].astype(str).tolist(),
            model_name=esm_model,
            cache_path=esm_cache_path,
            device="cpu",
        )
        emb_cols = [c for c in emb.columns if c.startswith("esm_")]
        X = pd.concat([X.reset_index(drop=True), emb[emb_cols].reset_index(drop=True)], axis=1) # X = features
        feature_cols += emb_cols

    y_regression = out["y_regression"].to_numpy(dtype=float)
    y_classification = (y_regression >= hard_ge).astype(int)
    ids = out["id"].astype(str).to_numpy()

    return Dataset(
        X=X,
        y_regression=y_regression,
        y_classification=y_classification,
        ids=ids,
        feature_cols=feature_cols,
        target_rmsd_col=target_rmsd_col,
        hard_ge=hard_ge,
        agree_tol=agree_tol,
        feature_set=feature_set,
    )

# regression: y = true RMSD, ŷ = predicted RMSD
def xgb_regressor(feature_cols: List[str]) -> Tuple[str, Any]:
    pre = make_preprocessor(feature_cols, scale=False)
    try:
        from xgboost import XGBRegressor

        reg = XGBRegressor(
            n_estimators=2500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            reg_alpha=0.0,
            min_child_weight=3.0,
            objective="reg:squarederror",
            n_jobs=4,
            random_state=RANDOM_STATE,
        )
        return "XGBReg", Pipeline([("pre", pre), ("reg", reg)])
    except Exception:
        reg = HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.05,
            max_iter=1500,
            l2_regularization=0.5,
            random_state=RANDOM_STATE,
        )
        return "HistGBReg", Pipeline([("pre", pre), ("reg", reg)])

# classification (hard-risk model): model outputs a probability p̂ = P(y=1 | x), then its converted to a class using a threshold t: ŷ =1[p̂ ≥ t]
def xgb_classifier(feature_cols: List[str], scale_pos_weight: float) -> Tuple[str, Any]:
    pre = make_preprocessor(feature_cols, scale=False)
    try:
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=2500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            min_child_weight=3.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=4,
            random_state=RANDOM_STATE,
        )
        return "xgb_classifier", Pipeline([("pre", pre), ("clf", clf)])
    except Exception:
        clf = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=1500,
            random_state=RANDOM_STATE,
        )
        return "Histgb_classifier", Pipeline([("pre", pre), ("clf", clf)])

def split_train_test(ids: np.ndarray, y_classification: np.ndarray, test_size: float) -> Tuple[np.ndarray, np.ndarray]:
    innersplit_within_training_for_threshold = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    tr, te = next(innersplit_within_training_for_threshold.split(ids.reshape(-1, 1), y_classification))
    return tr, te

def repeated_cv_on_train_pool(
    ds: Dataset,
    train_idx: np.ndarray,
    out_dir: Path,
    n_splits: int,
    n_repeats: int,
    conformal_alpha: float,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    X = ds.X.iloc[train_idx].reset_index(drop=True)
    y_regression = ds.y_regression[train_idx]
    y_classification = ds.y_classification[train_idx]
    ids = ds.ids[train_idx]
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    base_class_weight_ratio = float((y_classification == 0).sum() / max((y_classification == 1).sum(), 1)) # negative / positive in training labels
    string_label_of_regressor, estimator_object_for_regression = xgb_regressor(ds.feature_cols)
    string_label_of_classifier, _ = xgb_classifier(ds.feature_cols, scale_pos_weight=base_class_weight_ratio)
    pred_r_sum = np.zeros(len(y_regression), dtype=float)
    pred_r_cnt = np.zeros(len(y_regression), dtype=float)
    p_h_sum = np.zeros(len(y_regression), dtype=float)
    p_h_cnt = np.zeros(len(y_regression), dtype=float)
    probability_cutoff_for_classifying_hard_list: List[float] = []
    mae_list: List[float] = []
    rmse_list: List[float] = []
    ap_list: List[float] = []
    roc_list: List[float] = []
    brier_list: List[float] = []
    f1_list: List[float] = []

    for split_idx, (tr, te) in enumerate(rskf.split(X, y_classification), start=1):
        X_train, X_test = X.iloc[tr], X.iloc[te]
        y_train_regressor, true_RMSD = y_regression[tr], y_regression[te]
        y_train_classifier, hard_risk_01_labels = y_classification[tr], y_classification[te]

        reg = clone(estimator_object_for_regression)
        reg.fit(X_train, y_train_regressor)
        pred_r = reg.predict(X_test)

        mae_list.append(float(mean_absolute_error(true_RMSD, pred_r)))
        rmse_list.append(float(np.sqrt(mean_squared_error(true_RMSD, pred_r))))
        pred_r_sum[te] += pred_r
        pred_r_cnt[te] += 1

        fold_class_weight_ratio = float((y_train_classifier == 0).sum() / max((y_train_classifier == 1).sum(), 1))
        _, uncalibrated_classifier_template = xgb_classifier(ds.feature_cols, scale_pos_weight=fold_class_weight_ratio)

        innersplit_within_training_for_threshold = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE + split_idx)
        inner_training_subset, inner_calibration_subset = next(innersplit_within_training_for_threshold.split(X_train, y_train_classifier)) # calibration here means best F1 decides the threshold fo inner splits

        cal_inner = CalibratedClassifierCV(clone(uncalibrated_classifier_template), method=CALIBRATION_METHOD, cv=CALIBRATION_INNER_CV)
        cal_inner.fit(X_train.iloc[inner_training_subset], y_train_classifier[inner_training_subset])
        predicted_probability_1_hard = cal_inner.predict_proba(X_train.iloc[inner_calibration_subset])[:, 1]
        probability_cutoff_for_classifying_hard = choose_threshold_on_train(y_train_classifier[inner_calibration_subset], predicted_probability_1_hard)
        probability_cutoff_for_classifying_hard_list.append(float(probability_cutoff_for_classifying_hard))

        cal_full = CalibratedClassifierCV(clone(uncalibrated_classifier_template), method=CALIBRATION_METHOD, cv=CALIBRATION_INNER_CV)
        cal_full.fit(X_train, y_train_classifier)
        predicted_probability_test = np.clip(cal_full.predict_proba(X_test)[:, 1], 0.0, 1.0)

        ap_list.append(float(average_precision_score(hard_risk_01_labels, predicted_probability_test)) if int(np.sum(hard_risk_01_labels)) > 0 else float("nan"))
        roc_list.append(float(roc_auc_score(hard_risk_01_labels, predicted_probability_test)) if np.unique(hard_risk_01_labels).size == 2 else float("nan"))
        brier_list.append(float(brier_score_loss(hard_risk_01_labels, predicted_probability_test)))

        yhat = (predicted_probability_test >= probability_cutoff_for_classifying_hard).astype(int)
        tp = np.sum((yhat == 1) & (hard_risk_01_labels == 1))
        fp = np.sum((yhat == 1) & (hard_risk_01_labels == 0))
        fn = np.sum((yhat == 0) & (hard_risk_01_labels == 1))
        precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        f1_list.append(float(f1))

        p_h_sum[te] += predicted_probability_test
        p_h_cnt[te] += 1

    pred_r_avg = pred_r_sum / np.maximum(pred_r_cnt, 1.0)
    p_h_avg = p_h_sum / np.maximum(p_h_cnt, 1.0)

    resid = y_regression - pred_r_avg
    interval_radius = conformal_radius(resid, alpha=conformal_alpha)

    rank_metrics = precision_recall_at_k(y_true_01=y_classification, scores=p_h_avg, ks=[10, 20, 50])

    pred_df = pd.DataFrame(
        {
            "id": ids,
            "y_regression_true": y_regression,
            "y_classification_true_01": y_classification,
            "pred_rmsd_cv_avg": pred_r_avg,
            "pred_rmsd_lower_prediction_interval_bounds_cv": pred_r_avg - interval_radius,
            "pred_rmsd_upper_prediction_interval_bounds_cv": pred_r_avg + interval_radius,
            "p_hard_cv_avg": p_h_avg,
            "abs_err_cv": np.abs(resid),
        }
    )
    pred_df.to_csv(out_dir / "predictions_train_cv_avg.csv", index=False)

    def mean_std(xs: List[float]) -> Dict[str, float]:
        xs2 = np.asarray(xs, dtype=float)
        xs2 = xs2[~np.isnan(xs2)]
        if xs2.size == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(np.mean(xs2)), "std": float(np.std(xs2, ddof=1)) if xs2.size > 1 else 0.0}

    return {
        "train_cv": {
            "n_samples": int(len(y_regression)),
            "n_splits": int(n_splits),
            "n_repeats": int(n_repeats),
            "total_splits": int(n_splits * n_repeats),
        },
        "models": {
            "regressor": string_label_of_regressor,
            "hard_risk_base": string_label_of_classifier,
            "calibration": CALIBRATION_METHOD,
        },
        "metrics": {
            "reg_mae": mean_std(mae_list),
            "reg_rmse": mean_std(rmse_list),
            "hard_pr_auc_ap": mean_std(ap_list),
            "hard_roc_auc": mean_std(roc_list),
            "hard_brier": mean_std(brier_list),
            "hard_f1_thresholded": mean_std(f1_list),
            "decision_threshold_for_hard": mean_std(probability_cutoff_for_classifying_hard_list),
            "hard_rank_metrics_from_p_hard_cv_avg": rank_metrics,
            "reg_conformal_alpha": float(conformal_alpha),
            "reg_conformal_radius": float(interval_radius),
        },
        "artifacts": {
            "train_cv_predictions_csv": "predictions_train_cv_avg.csv",
        },
    }

def train_validation_test_results(
    ds: Dataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    out_dir: Path,
    conformal_alpha: float,
    conformal_radius_train: float,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = ds.X.iloc[train_idx].reset_index(drop=True)
    y_train_regressor = ds.y_regression[train_idx]
    y_train_classifier = ds.y_classification[train_idx]

    X_test = ds.X.iloc[test_idx].reset_index(drop=True)
    true_RMSD = ds.y_regression[test_idx]
    hard_risk_01_labels = ds.y_classification[test_idx]
    ids_test = ds.ids[test_idx]

    string_label_of_regressor, estimator_object_for_regression = xgb_regressor(ds.feature_cols)
    class_weight_ratio = float((y_train_classifier == 0).sum() / max((y_train_classifier == 1).sum(), 1))
    string_label_of_classifier, uncalibrated_classifier_template = xgb_classifier(ds.feature_cols, scale_pos_weight=class_weight_ratio)

    reg = clone(estimator_object_for_regression)
    reg.fit(X_train, y_train_regressor)
    predicted_RMSD_test = reg.predict(X_test)

    mae = float(mean_absolute_error(true_RMSD, predicted_RMSD_test))
    rmse = float(np.sqrt(mean_squared_error(true_RMSD, predicted_RMSD_test)))

    interval_radius = float(conformal_radius_train)
    lower_prediction_interval_bounds = predicted_RMSD_test - interval_radius
    upper_prediction_interval_bounds = predicted_RMSD_test + interval_radius
    coverage = float(np.mean((true_RMSD >= lower_prediction_interval_bounds) & (true_RMSD <= upper_prediction_interval_bounds)))

    innersplit_within_training_for_threshold = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    inner_training_subset, inner_calibration_subset = next(innersplit_within_training_for_threshold.split(X_train, y_train_classifier))

    cal_inner = CalibratedClassifierCV(clone(uncalibrated_classifier_template), method=CALIBRATION_METHOD, cv=CALIBRATION_INNER_CV)
    cal_inner.fit(X_train.iloc[inner_training_subset], y_train_classifier[inner_training_subset])
    predicted_probability_1_hard = cal_inner.predict_proba(X_train.iloc[inner_calibration_subset])[:, 1]
    probability_cutoff_for_classifying_hard = choose_threshold_on_train(y_train_classifier[inner_calibration_subset], predicted_probability_1_hard)

    cal = CalibratedClassifierCV(clone(uncalibrated_classifier_template), method=CALIBRATION_METHOD, cv=CALIBRATION_INNER_CV)
    cal.fit(X_train, y_train_classifier)
    predicted_probability_test = np.clip(cal.predict_proba(X_test)[:, 1], 0.0, 1.0)

    ap = float(average_precision_score(hard_risk_01_labels, predicted_probability_test)) if int(np.sum(hard_risk_01_labels)) > 0 else float("nan")
    roc = float(roc_auc_score(hard_risk_01_labels, predicted_probability_test)) if np.unique(hard_risk_01_labels).size == 2 else float("nan")
    brier = float(brier_score_loss(hard_risk_01_labels, predicted_probability_test))

    yhat = (predicted_probability_test >= probability_cutoff_for_classifying_hard).astype(int)
    tp = int(np.sum((yhat == 1) & (hard_risk_01_labels == 1)))
    fp = int(np.sum((yhat == 1) & (hard_risk_01_labels == 0)))
    tn = int(np.sum((yhat == 0) & (hard_risk_01_labels == 0)))
    fn = int(np.sum((yhat == 0) & (hard_risk_01_labels == 1)))

    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    test_predictions = pd.DataFrame(
        {
            "id": ids_test,
            "y_regression_true": true_RMSD,
            "y_classification_true_01": hard_risk_01_labels,
            "pred_rmsd": predicted_RMSD_test,
            "pred_rmsd_lower_prediction_interval_bounds": lower_prediction_interval_bounds,
            "pred_rmsd_upper_prediction_interval_bounds": upper_prediction_interval_bounds,
            "p_hard": predicted_probability_test,
        }
    )
    test_predictions.to_csv(out_dir / "predictions_test.csv", index=False)

    plot_pr_curve(hard_risk_01_labels, predicted_probability_test, out_dir / "test_hard_pr.png", title=f"Test hard-risk PR (hard = RMSD≥{ds.hard_ge:.1f}Å)")
    plot_roc_curve(hard_risk_01_labels, predicted_probability_test, out_dir / "test_hard_roc.png", title="Test hard-risk ROC")
    plot_reliability(hard_risk_01_labels, predicted_probability_test, out_dir / "test_hard_reliability.png", title="Test hard-risk reliability")
    plot_reg_scatter(true_RMSD, predicted_RMSD_test, out_dir / "test_reg_scatter.png", title="Test regression: predicted vs true")
    plot_reg_error_hist(predicted_RMSD_test - true_RMSD, out_dir / "test_reg_error_hist.png", title="Test regression error distribution")

    rank_metrics = precision_recall_at_k(y_true_01=hard_risk_01_labels, scores=predicted_probability_test, ks=[10, 20, 50])

    return {
        "test_set": {"n_samples": int(len(true_RMSD)), "hard_rate": float(np.mean(hard_risk_01_labels))},
        "models": {"regressor": string_label_of_regressor, "hard_risk_base": string_label_of_classifier, "calibration": CALIBRATION_METHOD},
        "metrics": {
            "reg_mae": mae,
            "reg_rmse": rmse,
            "reg_conformal_alpha": float(conformal_alpha),
            "reg_conformal_radius": float(interval_radius),
            "reg_pi_coverage": float(coverage),
            "hard_pr_auc_ap": ap,
            "hard_roc_auc": roc,
            "hard_brier": brier,
            "decision_threshold_for_hard": float(probability_cutoff_for_classifying_hard),
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "hard_rank_metrics_from_p_hard": rank_metrics,
        },
        "artifacts": {
            "test_predictions_csv": "predictions_test.csv",
            "test_pr_png": "test_hard_pr.png",
            "test_roc_png": "test_hard_roc.png",
            "test_reliability_png": "test_hard_reliability.png",
            "test_reg_scatter_png": "test_reg_scatter.png",
            "test_reg_error_hist_png": "test_reg_error_hist.png",
        },
    }


def run_one_setting(args: argparse.Namespace, out_dir: Path, feature_set: str) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(
        csv_path=Path(args.csv),
        target_rmsd_col=args.target_rmsd_col,
        agree_tol=args.agree_tol,
        hard_ge=args.hard_ge,
        feature_set=feature_set,
        use_esm=args.use_esm,
        esm_model=args.esm_model,
        esm_cache_path=Path(args.esm_cache),
    )

    train_idx, test_idx = split_train_test(ds.ids, ds.y_classification, test_size=args.test_size)

    dataset_summary = {
        "feature_set": feature_set,
        "n_targets": int(len(ds.ids)),
        "agree_tol": float(ds.agree_tol),
        "hard_ge": float(ds.hard_ge),
        "target_rmsd_col": ds.target_rmsd_col,
        "feature_count": int(ds.X.shape[1]),
        "hard_rate_overall": float(np.mean(ds.y_classification)),
        "counts_overall": {
            "not_hard": int(np.sum(ds.y_classification == 0)),
            "hard": int(np.sum(ds.y_classification == 1)),
        },
        "split": {
            "test_size": float(args.test_size),
            "n_train_pool": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "hard_rate_train_pool": float(np.mean(ds.y_classification[train_idx])),
            "hard_rate_test": float(np.mean(ds.y_classification[test_idx])),
        },
        "leakage_guard": {
            "feature_policy": "only h3_* engineered features (excluding h3_seq) + ESM embeddings from h3_seq",
            "explicitly_excluded_patterns": BANNED_FEATURE_RE.pattern,
            "note": "RMSD is used ONLY for filtering and labels, never as an input feature",
        },
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2))

    cv_summary = repeated_cv_on_train_pool(
        ds=ds,
        train_idx=train_idx,
        out_dir=out_dir,
        n_splits=args.splits,
        n_repeats=args.repeats,
        conformal_alpha=args.conformal_alpha,
    )

    interval_radius = float(cv_summary["metrics"]["reg_conformal_radius"])
    test_summary = train_validation_test_results(
        ds=ds,
        train_idx=train_idx,
        test_idx=test_idx,
        out_dir=out_dir,
        conformal_alpha=args.conformal_alpha,
        conformal_radius_train=interval_radius,
    )

    results = {"dataset": dataset_summary, "train_cv": cv_summary, "test": test_summary}
    (out_dir / "results_summary.json").write_text(json.dumps(results, indent=2))

    print(json.dumps(dataset_summary, indent=2))
    print("\nTrain pool repeated-CV metrics (mean±std)")
    print(json.dumps(cv_summary["metrics"], indent=2))
    print("\nHeld-out test metrics")
    print(json.dumps(test_summary["metrics"], indent=2))
    print(f"\nSaved outputs to: {out_dir.resolve()}")

    return results

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=CSV_PATH_DEFAULT)
    parser.add_argument("--out", type=str, default=OUT_DIR_DEFAULT)
    parser.add_argument("--target_rmsd_col", type=str, default=TARGET_RMSD_COL_DEFAULT)

    parser.add_argument("--agree_tol", type=float, default=AGREE_TOL_DEFAULT)
    parser.add_argument("--hard_ge", type=float, default=HARD_GE_DEFAULT)
    parser.add_argument("--test_size", type=float, default=TEST_SIZE_DEFAULT)

    parser.add_argument("--splits", type=int, default=N_SPLITS_DEFAULT)
    parser.add_argument("--repeats", type=int, default=N_REPEATS_DEFAULT)

    parser.add_argument("--use_esm", action="store_true")
    parser.add_argument("--esm_model", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--esm_cache", type=str, default="outputs/esm_cache.csv")

    parser.add_argument("--feature_set", type=str, default=FEATURE_SET_DEFAULT, choices=FEATURE_SET_CHOICES)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument(
        "--include_kmer",
        action="store_true",
        help="Also run appendix-only kmer_rarity baseline during --ablation.",
    )

    parser.add_argument("--conformal_alpha", type=float, default=CONFORMAL_ALPHA_DEFAULT)

    args = parser.parse_args()

    if not (0.0 < args.conformal_alpha < 1.0):
        raise ValueError("--conformal_alpha must be between 0 and 1 (e.g. 0.10)")

    out_dir = Path(args.out)

    if args.ablation:
        out_dir.mkdir(parents=True, exist_ok=True)
        ablation_results: Dict[str, Any] = {}

        run_sets = list(MAIN_FEATURE_SETS)
        if args.include_kmer:
            run_sets.append("kmer_rarity")

        for fs in run_sets:
            if fs == "engineered_esm" and not args.use_esm:
                raise ValueError("--use_esm")
            sub_out = out_dir / fs
            ablation_results[fs] = run_one_setting(args, out_dir=sub_out, feature_set=fs)

        (out_dir / "ablation_summary.json").write_text(json.dumps(ablation_results, indent=2))
        print(f"\{(out_dir / 'ablation_summary.json').resolve()}")

    else:
        run_one_setting(args, out_dir=out_dir, feature_set=args.feature_set)


if __name__ == "__main__":
    main()