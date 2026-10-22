import argparse
import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd


def load_train_module(path):
    spec = importlib.util.spec_from_file_location("train_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "03_train_lgbm.py"
    )
    mod = load_train_module(base_script)

    t0 = time.time()
    X = pd.read_pickle(os.path.join(args.proc_dir, "train_features.pkl"))
    y = np.load(os.path.join(args.proc_dir, "y.npy")).astype(np.int8)
    with open(os.path.join(args.proc_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    cat_cols = [c for c in meta.get("cat_cols", []) if c in X.columns]
    feature_cols = X.columns.tolist()

    dt = X["TransactionDT"].to_numpy(np.float64)
    tr_idx, te_idx = mod.temporal_split_indices(dt, train_frac=args.train_frac)

    dt_tr = dt[tr_idx]
    order = np.argsort(dt_tr)
    tr_sorted = tr_idx[order]
    cut = int(0.9 * len(tr_sorted))
    tr2_idx = tr_sorted[:cut]
    va2_idx = tr_sorted[cut:]

    print(f"[Temporal] rows train={len(tr2_idx)} val={len(va2_idx)} test={len(te_idx)}")

    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=256,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=40,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=-1,
    )

    t_fit = time.time()
    model.fit(
        X.iloc[tr2_idx][feature_cols],
        y[tr2_idx],
        eval_set=[(X.iloc[va2_idx][feature_cols], y[va2_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols if cat_cols else "auto",
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
    )
    fit_sec = time.time() - t_fit

    p_va_raw = model.predict_proba(X.iloc[va2_idx][feature_cols])[:, 1].astype(np.float32)
    p_te_raw = model.predict_proba(X.iloc[te_idx][feature_cols])[:, 1].astype(np.float32)
    cal = mod.fit_calibrator("platt", y[va2_idx], p_va_raw)
    p_te_cal = cal(p_te_raw)

    y_te = y[te_idx]
    auc_raw = float(mod.roc_auc_score(y_te, p_te_raw))
    pr_raw = float(mod.average_precision_score(y_te, p_te_raw))
    auc_cal = float(mod.roc_auc_score(y_te, p_te_cal))
    pr_cal = float(mod.average_precision_score(y_te, p_te_cal))
    rec1, thr1, fpr1 = mod.recall_at_fpr(y_te, p_te_cal, 0.01)
    rec5, thr5, fpr5 = mod.recall_at_fpr(y_te, p_te_cal, 0.05)

    metrics = {
        "protocol": "chronological_temporal_holdout",
        "train_frac": args.train_frac,
        "n_train": int(len(tr2_idx)),
        "n_val": int(len(va2_idx)),
        "n_test": int(len(te_idx)),
        "seed": args.seed,
        "best_iteration": int(model.best_iteration_),
        "elapsed_fit_sec": float(fit_sec),
        "elapsed_total_sec": float(time.time() - t0),
        "OOF_raw": {"AUC": auc_raw, "PR_AUC": pr_raw},
        "OOF_cal": {"AUC": auc_cal, "PR_AUC": pr_cal},
        "Recall_at_FPR": {
            "0.0100": {"recall": float(rec1), "thr": float(thr1), "fpr": float(fpr1)},
            "0.0500": {"recall": float(rec5), "thr": float(thr5), "fpr": float(fpr5)},
        },
    }
    with open(os.path.join(args.out_dir, "metrics_temporal.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(args.out_dir, "temporal_preds.npz"),
        y=y_te,
        p_raw=p_te_raw,
        p_cal=p_te_cal,
    )

    print(
        f"[Temporal done] AUC={auc_cal:.6f} PR-AUC={pr_cal:.6f} "
        f"R@1%={rec1:.6f} best_iter={model.best_iteration_} fit={fit_sec:.1f}s"
    )


if __name__ == "__main__":
    main()
