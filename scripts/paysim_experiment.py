import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_CSV = os.path.join("data", "paysim", "PS_20174392719_1491204439457_log.csv")
DEFAULT_OUT = os.path.join("outputs", "paysim_experiment")
RAW = os.environ.get("PAYSIM_CSV", DEFAULT_CSV)
ROOT = os.environ.get("PAYSIM_OUT", DEFAULT_OUT)
RESULTS = os.path.join(ROOT, "results")

SEED = 42
HALF_LIFE_STEPS = 168.0  # 7 days, PaySim step = 1 hour
SMOOTH = 10.0
TARGET_FPR = 0.01


class EntityStats:
    """Causal entity statistics with recursive exponential decay."""

    def __init__(self, n_codes, global_rate, half_life_steps=168.0, smooth=10.0):
        self.cnt = np.zeros(n_codes, dtype=np.float64)
        self.pos = np.zeros(n_codes, dtype=np.float64)
        self.wsum = np.zeros(n_codes, dtype=np.float64)
        self.wpos = np.zeros(n_codes, dtype=np.float64)
        self.last_t = np.full(n_codes, -1.0, dtype=np.float64)
        self.global_rate = float(global_rate)
        self.smooth = float(smooth)
        self.log2_over_hl = np.log(2.0) / float(half_life_steps)

    def _decay(self, code, t_now):
        last = self.last_t[code]
        if last < 0:
            self.last_t[code] = t_now
            return
        dt = t_now - last
        if dt > 0:
            decay = np.exp(-self.log2_over_hl * dt)
            self.wsum[code] *= decay
            self.wpos[code] *= decay
            self.last_t[code] = t_now

    def query(self, code, t_now):
        cnt = self.cnt[code]
        if cnt <= 0:
            return self.global_rate, self.global_rate, 1
        self._decay(code, t_now)
        rate = (self.pos[code] + self.smooth * self.global_rate) / (
            cnt + self.smooth
        )
        wrate = (self.wpos[code] + self.smooth * self.global_rate) / (
            self.wsum[code] + self.smooth
        )
        return float(rate), float(wrate), 0

    def update(self, code, t_now, y):
        self._decay(code, t_now)
        self.cnt[code] += 1.0
        self.pos[code] += float(y)
        self.wsum[code] += 1.0
        self.wpos[code] += float(y)
        self.last_t[code] = t_now


def log(msg):
    print(msg, flush=True)


def add_basic_features(df):
    df["hour"] = (df["step"] % 24).astype(np.int8)
    df["day"] = (df["step"] // 24).astype(np.int16)
    df["amount_log"] = np.log1p(df["amount"]).astype(np.float32)
    df["orig_balance_delta"] = (
        df["oldbalanceOrg"] - df["newbalanceOrig"]
    ).astype(np.float32)
    df["dest_balance_delta"] = (
        df["newbalanceDest"] - df["oldbalanceDest"]
    ).astype(np.float32)
    df["orig_zero_after"] = (df["newbalanceOrig"] == 0).astype(np.int8)
    df["dest_zero_before"] = (df["oldbalanceDest"] == 0).astype(np.int8)
    df["amount_to_orig"] = (
        df["amount"] / (df["oldbalanceOrg"] + 1.0)
    ).astype(np.float32)
    df["amount_to_dest"] = (
        df["amount"] / (df["oldbalanceDest"] + 1.0)
    ).astype(np.float32)
    df["dest_is_merchant"] = df["nameDest"].str.startswith("M").astype(np.int8)
    df["type"] = df["type"].astype("category")
    return df


def build_er_train(codes, y, steps, n_codes, global_rate):
    n = len(y)
    rate = np.empty(n, dtype=np.float32)
    wrate = np.empty(n, dtype=np.float32)
    trend = np.empty(n, dtype=np.float32)
    is_new = np.empty(n, dtype=np.int8)
    stats = EntityStats(n_codes, global_rate, HALF_LIFE_STEPS, SMOOTH)

    order = np.argsort(steps, kind="mergesort")
    sorted_steps = steps[order]
    unique_steps, first = np.unique(sorted_steps, return_index=True)
    bounds = np.append(first, n)

    processed = 0
    for g in range(len(unique_steps)):
        group = order[bounds[g]:bounds[g + 1]]
        t_now = float(unique_steps[g])
        for idx in group:
            r, w, new = stats.query(codes[idx], t_now)
            rate[idx] = r
            wrate[idx] = w
            trend[idx] = w - r
            is_new[idx] = new
        for idx in group:
            stats.update(codes[idx], t_now, y[idx])
        processed += len(group)
        if processed // 500000 > (processed - len(group)) // 500000:
            log(f"  ER train {processed}/{n}")
    return rate, wrate, trend, is_new, stats


def build_er_test(codes, steps, stats):
    n = len(steps)
    rate = np.empty(n, dtype=np.float32)
    wrate = np.empty(n, dtype=np.float32)
    trend = np.empty(n, dtype=np.float32)
    is_new = np.empty(n, dtype=np.int8)
    order = np.argsort(steps, kind="mergesort")
    for idx in order:
        r, w, new = stats.query(codes[idx], float(steps[idx]))
        rate[idx] = r
        wrate[idx] = w
        trend[idx] = w - r
        is_new[idx] = new
    return rate, wrate, trend, is_new


def threshold_at_fpr(y, p, target):
    y = y.astype(np.int8)
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    ps = p[order]
    pos = int(ys.sum())
    neg = len(ys) - pos
    fp = np.cumsum(1 - ys)
    ok = np.where(fp / neg <= target)[0]
    i = int(ok[-1]) if len(ok) else 0
    thr = float(ps[i])

    for _ in range(20):
        pred = p >= thr
        fpr = float(np.sum((pred == 1) & (y == 0)) / neg)
        if fpr <= target:
            break
        thr = float(np.nextafter(thr, np.inf))

    pred = p >= thr
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp_count = int(np.sum((pred == 1) & (y == 0)))
    return tp / pos, thr, fp_count / neg


def recall_with_threshold(y, p, thr):
    pred = p >= thr
    pos = int(y.sum())
    neg = len(y) - pos
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    return tp / pos, fp / neg


def evaluate(name, model, X_val, y_val, X_test, y_test, feature_cols):
    t0 = time.time()
    p_val = model.predict_proba(X_val[feature_cols])[:, 1]
    p_test = model.predict_proba(X_test[feature_cols])[:, 1]

    auc = float(roc_auc_score(y_test, p_test))
    pr = float(average_precision_score(y_test, p_test))
    rec_test, thr_test, fpr_test = threshold_at_fpr(y_test, p_test, TARGET_FPR)
    rec_val_thr, thr_val, _ = threshold_at_fpr(y_val, p_val, TARGET_FPR)
    rec_on_test, fpr_on_test = recall_with_threshold(y_test, p_test, thr_val)

    result = {
        "AUC": auc,
        "PR_AUC": pr,
        "test_threshold": {
            "threshold": thr_test,
            "recall": rec_test,
            "fpr": fpr_test,
        },
        "val_threshold": {
            "threshold": thr_val,
            "recall_on_val": rec_val_thr,
            "recall_on_test": rec_on_test,
            "fpr_on_test": fpr_on_test,
        },
        "best_iteration": int(model.best_iteration_),
        "eval_seconds": time.time() - t0,
    }
    log(
        f"{name}: AUC={auc:.6f} PR-AUC={pr:.6f} "
        f"R@1%test={rec_test:.6f} R@1%val={rec_on_test:.6f} "
        f"best_iter={model.best_iteration_}"
    )
    return result, p_val, p_test


def fit_model(name, X_tr, y_tr, X_val, y_val, feature_cols):
    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=256,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X_tr[feature_cols],
        y_tr,
        eval_set=[(X_val[feature_cols], y_val)],
        eval_metric="auc",
        categorical_feature=["type"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    elapsed = time.time() - t0
    log(f"{name}: fitted in {elapsed:.1f}s, best_iter={model.best_iteration_}")
    return model, elapsed


def main():
    global RAW, ROOT, RESULTS
    ap = argparse.ArgumentParser(
        description="PaySim supplementary experiment"
    )
    ap.add_argument("--csv", default=RAW, help="path to PS_20174392719_1491204439457_log.csv")
    ap.add_argument("--out_dir", default=ROOT, help="output directory for metrics and predictions")
    args = ap.parse_args()
    RAW, ROOT = args.csv, args.out_dir
    RESULTS = os.path.join(ROOT, "results")
    os.makedirs(RESULTS, exist_ok=True)
    t_start = time.time()
    log("loading PaySim")
    df = pd.read_csv(RAW)
    log(f"loaded {len(df):,} rows, fraud={int(df.isFraud.sum()):,}")

    df["orig_code"] = pd.factorize(df["nameOrig"])[0].astype(np.int32)
    df["dest_code"] = pd.factorize(df["nameDest"])[0].astype(np.int32)
    n_orig = int(df["orig_code"].max()) + 1
    n_dest = int(df["dest_code"].max()) + 1
    df = add_basic_features(df)
    df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"])

    # Cut at a timestamp boundary so the training set contains approximately
    # the first 80% of transactions, matching the transaction-count temporal
    # holdout used for Table 13 rather than the first 80% of distinct steps.
    step_counts = df.groupby("step", sort=True).size().cumsum()
    cut_step = int(step_counts[step_counts >= 0.8 * len(df)].index[0])
    train = df[df["step"] < cut_step].copy()
    test = df[df["step"] >= cut_step].copy()
    log(
        f"split: train<= {cut_step} rows={len(train):,} fraud={int(train.isFraud.sum()):,}; "
        f"test> {cut_step} rows={len(test):,} fraud={int(test.isFraud.sum()):,}"
    )

    train_steps = np.sort(train["step"].unique())
    val_start = int(train_steps[int(0.9 * len(train_steps))])
    inner = train[train["step"] < val_start].copy()
    val = train[train["step"] >= val_start].copy()
    if val["isFraud"].sum() == 0:
        raise RuntimeError("inner validation split contains no fraud cases")
    log(
        f"inner train rows={len(inner):,} fraud={int(inner.isFraud.sum()):,}; "
        f"val rows={len(val):,} fraud={int(val.isFraud.sum()):,}"
    )

    global_rate = float(train["isFraud"].mean())
    y_train = train["isFraud"].to_numpy(np.int8)
    y_test = test["isFraud"].to_numpy(np.int8)

    log("building causal entity features on train")
    er_orig = build_er_train(
        train["orig_code"].to_numpy(), y_train, train["step"].to_numpy(),
        n_orig, global_rate,
    )
    er_dest = build_er_train(
        train["dest_code"].to_numpy(), y_train, train["step"].to_numpy(),
        n_dest, global_rate,
    )
    for prefix, arrays in (("orig", er_orig[:4]), ("dest", er_dest[:4])):
        train[f"ER_RATE_{prefix}"] = arrays[0]
        train[f"ER_WRATE_{prefix}"] = arrays[1]
        train[f"ER_TREND_{prefix}"] = arrays[2]
        train[f"IS_NEW_{prefix}"] = arrays[3]

    log("building frozen causal entity features on test")
    te_orig = build_er_test(test["orig_code"].to_numpy(), test["step"].to_numpy(), er_orig[4])
    te_dest = build_er_test(test["dest_code"].to_numpy(), test["step"].to_numpy(), er_dest[4])
    for prefix, arrays in (("orig", te_orig), ("dest", te_dest)):
        test[f"ER_RATE_{prefix}"] = arrays[0]
        test[f"ER_WRATE_{prefix}"] = arrays[1]
        test[f"ER_TREND_{prefix}"] = arrays[2]
        test[f"IS_NEW_{prefix}"] = arrays[3]

    # Re-create the inner/validation splits after the ER columns are attached.
    inner = train[train["step"] < val_start].copy()
    val = train[train["step"] >= val_start].copy()

    # Deliberately weak transaction-level baseline: transaction type and amount
    # only. Balance-derived fields make PaySim's synthetic fraud rule almost
    # deterministic (AUC ~1.0) and are therefore excluded from the weak setting.
    base_cols = ["type", "amount"]
    er_cols = [
        f"{kind}_{prefix}"
        for prefix in ("orig", "dest")
        for kind in ("ER_RATE", "ER_WRATE", "ER_TREND", "IS_NEW")
    ]
    full_cols = base_cols + er_cols

    y_inner = inner["isFraud"].to_numpy(np.int8)
    y_val = val["isFraud"].to_numpy(np.int8)

    log("training baseline")
    model_b, time_b = fit_model("baseline", inner, y_inner, val, y_val, base_cols)
    res_b, pv_b, pt_b = evaluate("baseline", model_b, val, y_val, test, y_test, base_cols)
    res_b["fit_seconds"] = time_b

    log("training full causal framework")
    model_f, time_f = fit_model("full", inner, y_inner, val, y_val, full_cols)
    res_f, pv_f, pt_f = evaluate("full", model_f, val, y_val, test, y_test, full_cols)
    res_f["fit_seconds"] = time_f

    np.save(os.path.join(RESULTS, "y_test.npy"), y_test)
    np.save(os.path.join(RESULTS, "baseline_test_pred.npy"), pt_b)
    np.save(os.path.join(RESULTS, "full_test_pred.npy"), pt_f)
    np.save(os.path.join(RESULTS, "baseline_val_pred.npy"), pv_b)
    np.save(os.path.join(RESULTS, "full_val_pred.npy"), pv_f)

    imp = pd.DataFrame({
        "feature": full_cols,
        "importance": model_f.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp.to_csv(os.path.join(RESULTS, "full_feature_importance.csv"), index=False)

    summary = {
        "dataset": {
            "rows": int(len(df)),
            "fraud": int(df.isFraud.sum()),
            "fraud_rate": float(df.isFraud.mean()),
            "steps": [int(df.step.min()), int(df.step.max())],
        },
        "split": {
            "cut_step": cut_step,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_fraud": int(train.isFraud.sum()),
            "test_fraud": int(test.isFraud.sum()),
            "inner_rows": int(len(inner)),
            "val_rows": int(len(val)),
            "val_start_step": val_start,
        },
        "settings": {
            "half_life_steps": HALF_LIFE_STEPS,
            "smooth_strength": SMOOTH,
            "target_fpr": TARGET_FPR,
            "seed": SEED,
            "entity_keys": ["nameOrig", "nameDest"],
            "baseline_features": base_cols,
            "full_features": full_cols,
        },
        "baseline": res_b,
        "full": res_f,
        "reference_table13": {
            "baseline": {"AUC": 0.9195, "PR_AUC": 0.3319, "Recall_at_1pct_FPR": 0.3077},
            "full": {"AUC": 0.9485, "PR_AUC": 0.4570, "Recall_at_1pct_FPR": 0.3283},
        },
        "total_seconds": time.time() - t_start,
    }
    with open(os.path.join(RESULTS, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
