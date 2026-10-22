# scripts/3_train_lgb.py
import os
import json
import argparse
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

RANDOM_STATE = 42


# -------------------------
# Metrics & threshold helpers
# -------------------------
def confusion_and_rates(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    y_true = y_true.astype(np.int8)
    y_pred = (y_prob >= thr).astype(np.int8)

    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    fpr = fp / (fp + tn + 1e-12)
    fnr = fn / (fn + tp + 1e-12)

    return {
        "thr": float(thr),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "ACC": float(acc),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "Pred_Pos_Rate": float((tp + fp) / (tp + tn + fp + fn + 1e-12)),
        "True_Pos_Rate": float((tp + fn) / (tp + tn + fp + fn + 1e-12)),
    }


def find_best_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray, n_grid: int = 400):
    y_true = y_true.astype(np.int8)
    lo, hi = float(np.min(y_prob)), float(np.max(y_prob))
    if lo == hi:
        return lo, 0.0

    thrs = np.linspace(lo, hi, n_grid, dtype=np.float64)
    best_thr, best_f1 = thrs[0], -1.0

    for thr in thrs:
        m = confusion_and_rates(y_true, y_prob, float(thr))
        if m["F1"] > best_f1:
            best_f1 = m["F1"]
            best_thr = float(thr)

    return best_thr, float(best_f1)


def recall_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float):
    """
    Return (recall, thr, achieved_fpr) at the largest threshold such that FPR<=target_fpr.
    Efficient implementation by sorting scores descending.
    """
    y_true = y_true.astype(np.int8)
    order = np.argsort(-y_prob)  # desc
    y_sorted = y_true[order]
    p_sorted = y_prob[order]

    P = int(np.sum(y_sorted == 1))
    N = int(np.sum(y_sorted == 0))

    tp = 0
    fp = 0
    best_thr = float(p_sorted[0]) if len(p_sorted) else 0.5
    best_rec = 0.0
    best_fpr = 0.0

    for i in range(len(p_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1

        fpr = fp / (N + 1e-12)
        if fpr <= target_fpr:
            rec = tp / (P + 1e-12)
            best_rec = rec
            best_thr = float(p_sorted[i])
            best_fpr = float(fpr)
        else:
            break

    return float(best_rec), float(best_thr), float(best_fpr)


# -------------------------
# Data utils
# -------------------------
def align_train_test_columns(X_train: pd.DataFrame, X_test: pd.DataFrame):
    tr_cols = set(X_train.columns)
    te_cols = set(X_test.columns)

    missing_in_test = sorted(list(tr_cols - te_cols))
    missing_in_train = sorted(list(te_cols - tr_cols))

    if missing_in_test:
        print(f"[Align] add {len(missing_in_test)} cols to test, e.g. {missing_in_test[:8]}")
        for c in missing_in_test:
            X_test[c] = np.nan
    if missing_in_train:
        print(f"[Align] add {len(missing_in_train)} cols to train, e.g. {missing_in_train[:8]}")
        for c in missing_in_train:
            X_train[c] = np.nan

    cols = sorted(X_train.columns.tolist())
    X_train = X_train[cols]
    X_test = X_test[cols]
    return X_train, X_test


def ensure_categorical_consistency(train: pd.DataFrame, test: pd.DataFrame, cat_cols: list[str]):
    train = train.copy()
    test = test.copy()

    for c in cat_cols:
        if c not in train.columns:
            continue
        if c not in test.columns:
            test[c] = np.nan

        tr = train[c].astype("object").fillna("Unknown")
        te = test[c].astype("object").fillna("Unknown")
        all_cats = pd.Index(tr.unique()).union(pd.Index(te.unique()))
        train[c] = pd.Categorical(tr, categories=all_cats)
        test[c] = pd.Categorical(te, categories=all_cats)

    return train, test


def pick_time_column(df: pd.DataFrame) -> str | None:
    if "TransactionDT" in df.columns:
        return "TransactionDT"
    if "DT_WEEK" in df.columns:
        return "DT_WEEK"
    return None


# -------------------------
# CV split helpers
# -------------------------
def make_cv_splits(y: np.ndarray, groups=None, n_splits=5, seed=42, cv_mode="stratified"):
    cv_mode = cv_mode.lower()

    if cv_mode == "group":
        if groups is None:
            raise ValueError("cv_mode=group requires groups (entity.npy).")
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return list(sgkf.split(np.zeros(len(y)), y, groups))
        except Exception:
            from sklearn.model_selection import GroupKFold
            gkf = GroupKFold(n_splits=n_splits)
            return list(gkf.split(np.zeros(len(y)), y, groups))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def temporal_split_indices(dt: np.ndarray, train_frac=0.8):
    order = np.argsort(dt)
    cut = int(train_frac * len(dt))
    tr_idx = order[:cut]
    te_idx = order[cut:]
    return tr_idx, te_idx


# -------------------------
# Calibration (decoupling)
# -------------------------
def fit_calibrator(method: str, y_true: np.ndarray, p_raw: np.ndarray):
    method = method.lower()
    if method == "none":
        return lambda p: p

    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    y_true = y_true.astype(np.int8)
    p_raw = p_raw.astype(np.float64)

    if method == "platt":
        eps = 1e-12
        x = np.log(np.clip(p_raw, eps, 1 - eps) / np.clip(1 - p_raw, eps, 1 - eps)).reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs", max_iter=200)
        lr.fit(x, y_true)

        def f(p):
            p = p.astype(np.float64)
            x2 = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)).reshape(-1, 1)
            return lr.predict_proba(x2)[:, 1].astype(np.float32)

        return f

    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_raw, y_true)

        def f(p):
            return iso.transform(p.astype(np.float64)).astype(np.float32)

        return f

    raise ValueError(f"Unknown calibration method: {method}")


# -------------------------
# Risk-aware thresholding (risk-weighted FP allocation)
# -------------------------
def fit_risk_bins_edges(risk: np.ndarray, n_bins=5) -> np.ndarray:
    s = pd.Series(risk.astype(np.float64))
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = s.quantile(qs).to_numpy(dtype=np.float64)
    edges = np.unique(edges)
    if len(edges) <= 2:
        mn, mx = float(s.min()), float(s.max())
        edges = np.array([mn - 1e-9, mx + 1e-9], dtype=np.float64)
    else:
        edges[0] -= 1e-9
        edges[-1] += 1e-9
    return edges


def apply_risk_bins(risk: np.ndarray, edges: np.ndarray):
    r = risk.astype(np.float64)
    b = np.digitize(r, edges[1:-1], right=True).astype(np.int32)
    n_bins_eff = int(len(edges) - 1)
    b = np.clip(b, 0, n_bins_eff - 1)
    return b, n_bins_eff


def _allocate_fp_budget(
    FP_total: int,
    neg_counts: np.ndarray,
    bin_risk_mean: np.ndarray,
    mode: str = "risk_weighted",
    risk_power: float = 1.0,
):
    """
    Allocate integer FP budget across bins, sum(fp_alloc)=FP_total.
    mode:
      - "size": proportional to neg_counts
      - "risk_weighted": proportional to neg_counts * (risk_mean ** risk_power)
      - "risk_only": proportional to (risk_mean ** risk_power)
    """
    neg_counts = neg_counts.astype(np.float64)
    r = np.clip(bin_risk_mean.astype(np.float64), 0.0, None)
    r = np.power(r + 1e-12, float(risk_power))

    if mode == "size":
        w = neg_counts
    elif mode == "risk_only":
        w = r
    elif mode == "risk_weighted":
        w = neg_counts * r
    else:
        raise ValueError(f"Unknown allocation mode: {mode}")

    if np.all(w <= 0):
        w = neg_counts  # fallback

    w = w / (np.sum(w) + 1e-12)
    raw = FP_total * w
    fp_alloc = np.floor(raw).astype(np.int64)

    remain = FP_total - int(fp_alloc.sum())
    if remain > 0:
        frac = raw - fp_alloc
        order = np.argsort(-frac)
        for i in range(remain):
            fp_alloc[order[i % len(fp_alloc)]] += 1

    return fp_alloc.astype(np.int64)


def fit_bucket_thresholds_global_fpr(
    y_true: np.ndarray,
    p: np.ndarray,
    risk: np.ndarray,
    n_bins=5,
    target_fpr=0.01,
    edges: np.ndarray | None = None,
    alloc_mode: str = "risk_weighted",
    risk_power: float = 1.0,
):
    """
    Risk-aware bucket thresholds under GLOBAL FPR:
      1) fixed quantile bins (edges)
      2) allocate FP_total across bins using alloc_mode (risk-weighted)
      3) per bin threshold from top fp_b negative scores
    Returns: (bins, thr_per_bin, edges, fp_alloc, bin_risk_mean)
    """
    y_true = y_true.astype(np.int8)
    p = p.astype(np.float64)

    if edges is None:
        edges = fit_risk_bins_edges(risk, n_bins=n_bins)
    bins, n_bins_eff = apply_risk_bins(risk, edges)

    neg_mask = (y_true == 0)
    N_neg = int(np.sum(neg_mask))
    FP_total = int(np.floor(target_fpr * N_neg + 1e-9))

    neg_counts = np.array([int(np.sum(neg_mask & (bins == b))) for b in range(n_bins_eff)], dtype=np.int64)

    # risk mean per bin (on ALL samples, you can also use negatives only; keep simple and stable)
    bin_risk_mean = np.array([float(np.mean(risk[bins == b])) if np.any(bins == b) else 0.0
                              for b in range(n_bins_eff)], dtype=np.float64)

    if FP_total <= 0 or neg_counts.sum() == 0:
        thr_per_bin = {b: 1.1 for b in range(n_bins_eff)}
        fp_alloc = np.zeros(n_bins_eff, dtype=np.int64)
        return bins, thr_per_bin, edges, fp_alloc, bin_risk_mean

    fp_alloc = _allocate_fp_budget(
        FP_total=FP_total,
        neg_counts=neg_counts,
        bin_risk_mean=bin_risk_mean,
        mode=alloc_mode,
        risk_power=risk_power,
    )

    thr_per_bin = {}
    for b in range(n_bins_eff):
        fp_b = int(fp_alloc[b])
        idx_neg = (bins == b) & (y_true == 0)
        neg_scores = p[idx_neg]

        if fp_b <= 0 or len(neg_scores) == 0:
            thr_per_bin[b] = 1.1
            continue

        neg_scores_sorted = np.sort(neg_scores)[::-1]
        k = min(fp_b, len(neg_scores_sorted)) - 1
        thr_per_bin[b] = float(neg_scores_sorted[k])

    return bins, thr_per_bin, edges, fp_alloc, bin_risk_mean


def predict_with_bucket_thresholds(p: np.ndarray, bins: np.ndarray, thr_per_bin: dict[int, float]):
    p = p.astype(np.float64)
    pred = np.zeros(len(p), dtype=np.int8)
    for b, thr in thr_per_bin.items():
        idx = (bins == b)
        if np.any(idx):
            pred[idx] = (p[idx] >= thr).astype(np.int8)
    return pred


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--proc_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--cv_mode", type=str, default="stratified", choices=["stratified", "group"])
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    # LGB params
    ap.add_argument("--num_leaves", type=int, default=256)
    ap.add_argument("--learning_rate", type=float, default=0.02)
    ap.add_argument("--n_estimators", type=int, default=5000)
    ap.add_argument("--min_child_samples", type=int, default=40)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.8)
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument("--early_stopping_rounds", type=int, default=200)

    # Calibration
    ap.add_argument("--calibration", type=str, default="platt", choices=["none", "platt", "isotonic"])

    # Bucket thresholding
    ap.add_argument("--do_bucket_thresholding", action="store_true")
    ap.add_argument("--risk_col", type=str, default="auto")
    ap.add_argument("--n_bins", type=int, default=5)
    ap.add_argument("--target_fpr", type=float, default=0.01)
    ap.add_argument("--alloc_mode", type=str, default="risk_weighted",
                    choices=["size", "risk_weighted", "risk_only"])
    ap.add_argument("--risk_power", type=float, default=1.0)

    # Temporal
    ap.add_argument("--do_temporal", action="store_true")
    ap.add_argument("--temporal_train_frac", type=float, default=0.8)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train_path = os.path.join(args.proc_dir, "train_features.pkl")
    test_path = os.path.join(args.proc_dir, "test_features.pkl")
    y_path = os.path.join(args.proc_dir, "y.npy")
    meta_path = os.path.join(args.proc_dir, "meta.json")

    X_train = pd.read_pickle(train_path)
    X_test = pd.read_pickle(test_path)
    y = np.load(y_path).astype(np.int8)

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # Align
    X_train, X_test = align_train_test_columns(X_train, X_test)

    # Categoricals
    cat_cols = meta.get("cat_cols", None)
    if not cat_cols:
        cat_cols = [
            c for c in X_train.columns
            if pd.api.types.is_string_dtype(X_train[c])
            or pd.api.types.is_object_dtype(X_train[c])
            or isinstance(X_train[c].dtype, pd.CategoricalDtype)
        ]
    else:
        cat_cols = [c for c in cat_cols if c in X_train.columns]

    if cat_cols:
        X_train, X_test = ensure_categorical_consistency(X_train, X_test, cat_cols)

    obj_left = [c for c in X_train.columns if pd.api.types.is_object_dtype(X_train[c])]
    if obj_left:
        for c in obj_left:
            X_train[c] = X_train[c].astype("object").fillna("Unknown").astype("category")
            X_test[c] = X_test[c].astype("object").fillna("Unknown").astype("category")
        cat_cols = sorted(set(cat_cols) | set(obj_left))

    # Groups
    groups = None
    if args.cv_mode == "group":
        ent_path = os.path.join(args.proc_dir, "entity.npy")
        if not os.path.exists(ent_path):
            raise FileNotFoundError(
                f"cv_mode=group requires entity.npy at: {ent_path}\n"
                f"Please generate entity.npy aligned with train_features.pkl."
            )
        groups = np.load(ent_path, allow_pickle=True)
        if len(groups) != len(y):
            raise ValueError(f"entity.npy length {len(groups)} != y length {len(y)}")

    splits = make_cv_splits(y, groups=groups, n_splits=args.n_splits, seed=args.seed, cv_mode=args.cv_mode)

    import lightgbm as lgb

    feature_cols = X_train.columns.tolist()

    oof_raw = np.zeros(len(y), dtype=np.float32)
    test_pred_raw = np.zeros(len(X_test), dtype=np.float32)
    aucs, pr_aucs = [], []

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        t_fold = time.time()
        X_tr = X_train.iloc[tr_idx][feature_cols]
        y_tr = y[tr_idx]
        X_va = X_train.iloc[va_idx][feature_cols]
        y_va = y[va_idx]

        model = lgb.LGBMClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            min_child_samples=args.min_child_samples,
            reg_lambda=args.reg_lambda,
            random_state=args.seed,
            n_jobs=-1,
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            categorical_feature=cat_cols if cat_cols else "auto",
            callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False)],
        )

        p_va = model.predict_proba(X_va)[:, 1].astype(np.float32)
        oof_raw[va_idx] = p_va

        auc = roc_auc_score(y_va, p_va)
        pr = average_precision_score(y_va, p_va)
        aucs.append(float(auc))
        pr_aucs.append(float(pr))

        p_te = model.predict_proba(X_test[feature_cols])[:, 1].astype(np.float32)
        test_pred_raw += p_te / args.n_splits

        print(f"[Fold {fold}] AUC={auc:.6f} PR-AUC={pr:.6f} | best_iter={model.best_iteration_} | elapsed={time.time() - t_fold:.1f}s")

    print(f"\n[CV raw] mean AUC={np.mean(aucs):.6f} std={np.std(aucs):.6f}")
    print(f"[CV raw] mean PR-AUC={np.mean(pr_aucs):.6f} std={np.std(pr_aucs):.6f}")

    # Calibration on OOF
    cal_method = args.calibration.lower()
    calibrator = fit_calibrator(cal_method, y, oof_raw)
    oof_cal = calibrator(oof_raw)
    test_pred_cal = calibrator(test_pred_raw)

    auc_all_raw = float(roc_auc_score(y, oof_raw))
    pr_all_raw = float(average_precision_score(y, oof_raw))
    auc_all_cal = float(roc_auc_score(y, oof_cal))
    pr_all_cal = float(average_precision_score(y, oof_cal))

    print(f"\n[OOF raw] AUC={auc_all_raw:.6f} PR-AUC={pr_all_raw:.6f}")
    print(f"[OOF cal] AUC={auc_all_cal:.6f} PR-AUC={pr_all_cal:.6f} | method={cal_method}")

    thr_f1, best_f1 = find_best_threshold_f1(y, oof_cal)
    cm_f1 = confusion_and_rates(y, oof_cal, thr_f1)
    print(f"\n[F1] thr={thr_f1:.6f} F1={best_f1:.6f} Precision={cm_f1['Precision']:.6f} Recall={cm_f1['Recall']:.6f} (calibrated)")

    rec1, thr1, fpr1 = recall_at_fpr(y, oof_cal, target_fpr=args.target_fpr)
    rec5, thr5, fpr5 = recall_at_fpr(y, oof_cal, target_fpr=0.05)
    print(f"[FPR constraint] target={args.target_fpr:.4f} thr={thr1:.6f} fpr={fpr1:.6f} recall={rec1:.6f} (calibrated)")
    print(f"[FPR constraint] target=0.0500 thr={thr5:.6f} fpr={fpr5:.6f} recall={rec5:.6f} (calibrated)")

    # -------------------------
    # Candidate 1: Risk-weighted bucket thresholds
    # -------------------------
    bucket_metrics = None
    if args.do_bucket_thresholding:
        if args.risk_col.lower() == "auto":
            candidates = ["ER_HIER_RATE", "GR_MAX_WRATE", "ER_WRATE", "ER_RATE", "ER_TREND"]
            risk_col = next((c for c in candidates if c in X_train.columns), None)
            if risk_col is None:
                raise ValueError("risk_col=auto cannot find a risk feature. Please pass --risk_col ...")
        else:
            risk_col = args.risk_col
            if risk_col not in X_train.columns:
                raise ValueError(f"risk_col={risk_col} not found in train_features.pkl")

        risk = X_train[risk_col].to_numpy(dtype=np.float32)
        edges = fit_risk_bins_edges(risk, n_bins=args.n_bins)

        bins_oof, thr_per_bin, _, fp_alloc, bin_risk_mean = fit_bucket_thresholds_global_fpr(
            y_true=y,
            p=oof_cal,
            risk=risk,
            n_bins=args.n_bins,
            target_fpr=args.target_fpr,
            edges=edges,
            alloc_mode=args.alloc_mode,
            risk_power=args.risk_power,
        )

        pred_bucket = predict_with_bucket_thresholds(oof_cal, bins_oof, thr_per_bin)

        tn = int(np.sum((pred_bucket == 0) & (y == 0)))
        fp = int(np.sum((pred_bucket == 1) & (y == 0)))
        fn = int(np.sum((pred_bucket == 0) & (y == 1)))
        tp = int(np.sum((pred_bucket == 1) & (y == 1)))

        fpr = fp / (fp + tn + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        prec = tp / (tp + fp + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)

        bucket_metrics = {
            "risk_col": risk_col,
            "n_bins": int(args.n_bins),
            "target_fpr": float(args.target_fpr),
            "achieved_fpr": float(fpr),
            "recall": float(rec),
            "precision": float(prec),
            "f1": float(f1),
            "alloc_mode": args.alloc_mode,
            "risk_power": float(args.risk_power),
            "fp_alloc": [int(x) for x in fp_alloc.tolist()],
            "bin_risk_mean": [float(x) for x in bin_risk_mean.tolist()],
            "thr_per_bin": {str(k): float(v) for k, v in thr_per_bin.items()},
            "edges": [float(x) for x in edges.tolist()],
        }

        print(f"\n[BucketThr RISK-W] risk_col={risk_col} bins={args.n_bins} "
              f"alloc={args.alloc_mode} power={args.risk_power:.2f} "
              f"target_fpr={args.target_fpr:.4f} achieved_fpr={fpr:.6f} "
              f"recall={rec:.6f} precision={prec:.6f} f1={f1:.6f}")
        print(f"[BucketThr RISK-W] fp_alloc={bucket_metrics['fp_alloc']} | bin_risk_mean={[round(x,6) for x in bucket_metrics['bin_risk_mean']]}")

    # Save CV artifacts
    np.save(os.path.join(args.out_dir, "oof_raw.npy"), oof_raw)
    np.save(os.path.join(args.out_dir, "oof_cal.npy"), oof_cal)
    np.save(os.path.join(args.out_dir, "test_pred_raw.npy"), test_pred_raw)
    np.save(os.path.join(args.out_dir, "test_pred_cal.npy"), test_pred_cal)

    metrics = {
        "cv_mode": args.cv_mode,
        "n_splits": args.n_splits,
        "calibration": cal_method,
        "OOF_raw": {"AUC": auc_all_raw, "PR_AUC": pr_all_raw},
        "OOF_cal": {"AUC": auc_all_cal, "PR_AUC": pr_all_cal},
        "F1_on_cal": {
            "thr": float(thr_f1),
            "F1": float(best_f1),
            "Precision": float(cm_f1["Precision"]),
            "Recall": float(cm_f1["Recall"]),
            "FPR": float(cm_f1["FPR"]),
        },
        "Recall_at_FPR": {
            f"{args.target_fpr:.4f}": {"recall": float(rec1), "thr": float(thr1), "fpr": float(fpr1)},
            "0.0500": {"recall": float(rec5), "thr": float(thr5), "fpr": float(fpr5)},
        },
        "cv_fold_auc": aucs,
        "cv_fold_pr_auc": pr_aucs,
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "cv_pr_auc_mean": float(np.mean(pr_aucs)),
        "cv_pr_auc_std": float(np.std(pr_aucs)),
        "bucket_thresholding": bucket_metrics,
    }
    with open(os.path.join(args.out_dir, "metrics_cv.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # -------------------------
    # Candidate 2: Temporal split experiment
    # -------------------------
    if args.do_temporal:
        time_col = pick_time_column(X_train)
        if time_col is None:
            raise ValueError("Temporal split requested but no TransactionDT/DT_WEEK found in features.")

        dt = X_train[time_col].to_numpy(dtype=np.float64)
        tr_idx, te_idx = temporal_split_indices(dt, train_frac=args.temporal_train_frac)

        # inner val = last 10% by time inside temporal-train
        dt_tr = dt[tr_idx]
        order_tr = np.argsort(dt_tr)
        tr_idx_sorted = tr_idx[order_tr]
        cut2 = int(0.9 * len(tr_idx_sorted))
        tr2_idx = tr_idx_sorted[:cut2]
        va2_idx = tr_idx_sorted[cut2:]

        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            min_child_samples=args.min_child_samples,
            reg_lambda=args.reg_lambda,
            random_state=args.seed,
            n_jobs=-1,
        )

        model.fit(
            X_train.iloc[tr2_idx][feature_cols], y[tr2_idx],
            eval_set=[(X_train.iloc[va2_idx][feature_cols], y[va2_idx])],
            eval_metric="auc",
            categorical_feature=cat_cols if cat_cols else "auto",
            callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False)],
        )

        p_va2_raw = model.predict_proba(X_train.iloc[va2_idx][feature_cols])[:, 1].astype(np.float32)
        p_te2_raw = model.predict_proba(X_train.iloc[te_idx][feature_cols])[:, 1].astype(np.float32)

        cal2 = fit_calibrator(cal_method, y[va2_idx], p_va2_raw)
        p_te2_cal = cal2(p_te2_raw)

        y_te2 = y[te_idx]

        auc_te_raw = float(roc_auc_score(y_te2, p_te2_raw))
        pr_te_raw = float(average_precision_score(y_te2, p_te2_raw))
        auc_te_cal = float(roc_auc_score(y_te2, p_te2_cal))
        pr_te_cal = float(average_precision_score(y_te2, p_te2_cal))

        rec_te_1, thr_te_1, fpr_te_1 = recall_at_fpr(y_te2, p_te2_cal, args.target_fpr)
        rec_te_5, thr_te_5, fpr_te_5 = recall_at_fpr(y_te2, p_te2_cal, 0.05)

        print("\n[Temporal] time_col=", time_col, f"train_frac={args.temporal_train_frac:.2f}")
        print(f"[Temporal raw] AUC={auc_te_raw:.6f} PR-AUC={pr_te_raw:.6f}")
        print(f"[Temporal cal] AUC={auc_te_cal:.6f} PR-AUC={pr_te_cal:.6f} | method={cal_method}")
        print(f"[Temporal FPR] target={args.target_fpr:.4f} thr={thr_te_1:.6f} fpr={fpr_te_1:.6f} recall={rec_te_1:.6f} (cal)")
        print(f"[Temporal FPR] target=0.0500 thr={thr_te_5:.6f} fpr={fpr_te_5:.6f} recall={rec_te_5:.6f} (cal)")

        # Temporal bucket thr: edges + thresholds learned on temporal-val, applied on future test
        if args.do_bucket_thresholding:
            if args.risk_col.lower() == "auto":
                candidates = ["ER_HIER_RATE", "GR_MAX_WRATE", "ER_WRATE", "ER_RATE", "ER_TREND"]
                risk_col = next((c for c in candidates if c in X_train.columns), None)
                if risk_col is None:
                    raise ValueError("Temporal bucket thresholding needs risk_col but auto not found.")
            else:
                risk_col = args.risk_col

            risk_va = X_train.iloc[va2_idx][risk_col].to_numpy(dtype=np.float32)
            edges_va = fit_risk_bins_edges(risk_va, n_bins=args.n_bins)

            bins_va, thr_per_bin_va, _, fp_alloc_va, bin_risk_mean_va = fit_bucket_thresholds_global_fpr(
                y_true=y[va2_idx],
                p=cal2(p_va2_raw),
                risk=risk_va,
                n_bins=args.n_bins,
                target_fpr=args.target_fpr,
                edges=edges_va,
                alloc_mode=args.alloc_mode,
                risk_power=args.risk_power,
            )

            risk_te = X_train.iloc[te_idx][risk_col].to_numpy(dtype=np.float32)
            bins_te, _ = apply_risk_bins(risk_te, edges_va)
            pred_bucket_te = predict_with_bucket_thresholds(p_te2_cal, bins_te, thr_per_bin_va)

            tn = int(np.sum((pred_bucket_te == 0) & (y_te2 == 0)))
            fp = int(np.sum((pred_bucket_te == 1) & (y_te2 == 0)))
            fn = int(np.sum((pred_bucket_te == 0) & (y_te2 == 1)))
            tp = int(np.sum((pred_bucket_te == 1) & (y_te2 == 1)))

            fpr = fp / (fp + tn + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            prec = tp / (tp + fp + 1e-12)
            f1 = 2 * prec * rec / (prec + rec + 1e-12)

            print(f"[Temporal BucketThr RISK-W] risk_col={risk_col} bins={args.n_bins} "
                  f"alloc={args.alloc_mode} power={args.risk_power:.2f} "
                  f"target_fpr={args.target_fpr:.4f} achieved_fpr={fpr:.6f} "
                  f"recall={rec:.6f} precision={prec:.6f} f1={f1:.6f}")
            print(f"[Temporal BucketThr RISK-W] fp_alloc={fp_alloc_va.tolist()} | bin_risk_mean={[round(x,6) for x in bin_risk_mean_va.tolist()]}")

    print("\n[Saved]", args.out_dir)
    print(" - oof_raw.npy / oof_cal.npy / test_pred_raw.npy / test_pred_cal.npy")
    print(" - metrics_cv.json")


if __name__ == "__main__":
    main()
