import argparse
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_CSV = os.path.join("data", "paysim", "PS_20174392719_1491204439457_log.csv")
DEFAULT_OUT = os.path.join("outputs", "paysim_experiment", "results")
RAW = os.environ.get("PAYSIM_CSV", DEFAULT_CSV)
OUT = os.environ.get("PAYSIM_PROBE_OUT", DEFAULT_OUT)


def recall_at_fpr(y, p, target=0.01):
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    pos = int(ys.sum())
    neg = len(ys) - pos
    fp = np.cumsum(1 - ys)
    tp = np.cumsum(ys)
    ok = np.where(fp / neg <= target)[0]
    i = int(ok[-1]) if len(ok) else 0
    thr = float(ps[i])
    pred = p >= thr
    return (
        float(np.sum((pred == 1) & (y == 1)) / pos),
        float(np.sum((pred == 1) & (y == 0)) / neg),
    )


def main():
    global RAW, OUT
    ap = argparse.ArgumentParser(
        description="Probe candidate weak-feature sets for the PaySim experiment"
    )
    ap.add_argument("--csv", default=RAW, help="path to PS_20174392719_1491204439457_log.csv")
    ap.add_argument("--out_dir", default=OUT, help="directory for the probe log")
    args = ap.parse_args()
    RAW, OUT = args.csv, args.out_dir
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    df = pd.read_csv(RAW)
    df["hour"] = (df["step"] % 24).astype(np.int8)
    df["day"] = (df["step"] // 24).astype(np.int16)
    df["amount_log"] = np.log1p(df["amount"]).astype(np.float32)
    df["dest_is_merchant"] = df["nameDest"].str.startswith("M").astype(np.int8)
    df["type"] = df["type"].astype("category")

    step_counts = df.groupby("step", sort=True).size().cumsum()
    cut = int(step_counts[step_counts >= 0.8 * len(df)].index[0])
    train = df[df["step"] < cut].copy()
    test = df[df["step"] >= cut].copy()

    steps = np.sort(train["step"].unique())
    val_start = int(steps[int(0.9 * len(steps))])
    inner = train[train["step"] < val_start]
    val = train[train["step"] >= val_start]

    sets = {
        "type_amount_step": ["type", "amount", "step"],
        "amount_step": ["amount", "step"],
        "type_amount": ["type", "amount"],
        "type_amount_time": ["type", "amount", "step", "hour", "day"],
        "type_amount_merchant": ["type", "amount", "step", "dest_is_merchant"],
        "type_amount_balances": [
            "type", "amount", "step", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest",
        ],
    }

    y_i = inner["isFraud"].to_numpy(np.int8)
    y_v = val["isFraud"].to_numpy(np.int8)
    y_t = test["isFraud"].to_numpy(np.int8)

    for name, cols in sets.items():
        t1 = time.time()
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=64,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            inner[cols], y_i,
            eval_set=[(val[cols], y_v)],
            eval_metric="auc",
            categorical_feature=["type"] if "type" in cols else "auto",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pv = model.predict_proba(val[cols])[:, 1]
        pt = model.predict_proba(test[cols])[:, 1]
        r, f = recall_at_fpr(y_t, pt, 0.01)
        print(
            f"{name}: val_AUC={roc_auc_score(y_v, pv):.4f} "
            f"test_AUC={roc_auc_score(y_t, pt):.4f} "
            f"test_PR={average_precision_score(y_t, pt):.4f} "
            f"R@1%={r:.4f} FPR={f:.4f} iter={model.best_iteration_} "
            f"time={time.time()-t1:.1f}s",
            flush=True,
        )

    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
