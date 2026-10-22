# scripts/2_build_feats_causal_fast.py
import os
import gc
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd


def make_key_series(df: pd.DataFrame, cols: list[str], fill="NA") -> pd.Series:
    parts = []
    for c in cols:
        s = df[c] if c in df.columns else pd.Series(fill, index=df.index)
        parts.append(s.astype("object").fillna(fill).astype(str))
    out = parts[0]
    for s in parts[1:]:
        out = out + "_" + s
    return out


def bayes_smooth_rate(pos, cnt, global_rate, strength=50.0):
    return (pos + strength * global_rate) / (cnt + strength + 1e-12)


def get_time_seconds(df: pd.DataFrame):
    if "TransactionDT" in df.columns:
        return df["TransactionDT"].to_numpy(np.float64)
    if "DT_WEEK" in df.columns:
        return df["DT_WEEK"].to_numpy(np.float64) * 7 * 24 * 3600
    raise ValueError("Need TransactionDT or DT_WEEK for causal feature construction.")


class FastCausalEntityStats:
    """
    Causal entity statistics:
    - static count / positive count
    - exponentially decayed count / positive count updated recursively

    Query first, then update, so current sample label is not used in its own feature.
    """

    def __init__(self, global_rate: float, smooth_strength: float = 10.0, half_life_days: float = 7.0):
        self.global_rate = float(global_rate)
        self.smooth_strength = float(smooth_strength)
        self.half_life_seconds = float(half_life_days) * 24.0 * 3600.0

        self.cnt = defaultdict(float)
        self.pos = defaultdict(float)

        self.wsum = defaultdict(float)
        self.wpos = defaultdict(float)
        self.last_t = {}

    def _decay_to(self, key: str, t_now: float):
        if key not in self.last_t:
            self.last_t[key] = float(t_now)
            return

        dt = max(0.0, float(t_now) - float(self.last_t[key]))
        if dt > 0:
            decay = np.exp(-np.log(2.0) * dt / (self.half_life_seconds + 1e-12))
            self.wsum[key] *= decay
            self.wpos[key] *= decay
            self.last_t[key] = float(t_now)

    def query(self, key: str, t_now: float):
        cnt = self.cnt[key]
        pos = self.pos[key]

        if cnt <= 0:
            return self.global_rate, self.global_rate, 1

        self._decay_to(key, t_now)

        rate = bayes_smooth_rate(pos, cnt, self.global_rate, self.smooth_strength)
        wrate = bayes_smooth_rate(self.wpos[key], self.wsum[key], self.global_rate, self.smooth_strength)

        return float(rate), float(wrate), 0

    def update(self, key: str, t_now: float, y_now: int):
        self._decay_to(key, t_now)

        self.cnt[key] += 1.0
        self.pos[key] += float(y_now)

        self.wsum[key] += 1.0
        self.wpos[key] += float(y_now)

        self.last_t[key] = float(t_now)


def add_graph_agg(df: pd.DataFrame, node_keys: list[str], prefix="GR"):
    rates, wrates = [], []

    for k in node_keys:
        rates.append(df[f"ER_RATE_{k}"].astype(np.float32).to_numpy())
        wrates.append(df[f"ER_WRATE_{k}"].astype(np.float32).to_numpy())

    R = np.vstack(rates)
    W = np.vstack(wrates)

    df[f"{prefix}_MEAN_RATE"] = R.mean(axis=0).astype(np.float32)
    df[f"{prefix}_MAX_RATE"] = R.max(axis=0).astype(np.float32)
    df[f"{prefix}_STD_RATE"] = R.std(axis=0).astype(np.float32)

    df[f"{prefix}_MEAN_WRATE"] = W.mean(axis=0).astype(np.float32)
    df[f"{prefix}_MAX_WRATE"] = W.max(axis=0).astype(np.float32)
    df[f"{prefix}_STD_WRATE"] = W.std(axis=0).astype(np.float32)

    W_sorted = np.sort(W, axis=0)
    top2 = W_sorted[-2:, :] if W_sorted.shape[0] >= 2 else W_sorted
    df[f"{prefix}_TOP2_MEAN_WRATE"] = top2.mean(axis=0).astype(np.float32)
    df[f"{prefix}_SPREAD_WRATE"] = (W.max(axis=0) - W.min(axis=0)).astype(np.float32)

    return df


def add_hier_backoff(df: pd.DataFrame, chain: list[str], global_rate: float, out_prefix="ER_HIER"):
    n = len(df)
    rate = np.full(n, global_rate, dtype=np.float32)
    wrate = np.full(n, global_rate, dtype=np.float32)
    filled = np.zeros(n, dtype=bool)

    for k in chain:
        is_new = df[f"IS_NEW_{k}"].to_numpy(np.int8)
        seen = is_new == 0

        r = df[f"ER_RATE_{k}"].to_numpy(np.float32)
        w = df[f"ER_WRATE_{k}"].to_numpy(np.float32)

        to_fill = (~filled) & seen
        rate[to_fill] = r[to_fill]
        wrate[to_fill] = w[to_fill]
        filled[to_fill] = True

    df[f"{out_prefix}_RATE"] = rate
    df[f"{out_prefix}_WRATE"] = wrate
    df[f"{out_prefix}_TREND"] = (wrate - rate).astype(np.float32)

    return df


def build_causal_er_train(X_train, y, keys, smooth_strength=10.0, half_life_days=7.0):
    n = len(X_train)
    global_rate = float(np.mean(y))
    dt = get_time_seconds(X_train)

    key_values = {
        name: make_key_series(X_train, cols).astype(str).to_numpy()
        for name, cols in keys.items()
    }

    out = {}
    for name in keys:
        out[f"ER_RATE_{name}"] = np.full(n, global_rate, dtype=np.float32)
        out[f"ER_WRATE_{name}"] = np.full(n, global_rate, dtype=np.float32)
        out[f"ER_TREND_{name}"] = np.zeros(n, dtype=np.float32)
        out[f"IS_NEW_{name}"] = np.ones(n, dtype=np.int8)

    stats = {
        name: FastCausalEntityStats(global_rate, smooth_strength, half_life_days)
        for name in keys
    }

    # Transactions sharing a timestamp are scored as a block: all queries are
    # issued against the pre-timestamp state, then all labels in the block are
    # applied together. This prevents same-second label leakage.
    order = np.argsort(dt, kind="mergesort")
    sorted_dt = dt[order]
    unique_t, first_pos = np.unique(sorted_dt, return_index=True)
    bounds = np.append(first_pos, n)

    processed = 0
    for g in range(len(unique_t)):
        group = order[bounds[g]:bounds[g + 1]]
        t_now = float(unique_t[g])

        # Query every row in the timestamp block before any label update.
        for idx in group:
            for name in keys:
                k = key_values[name][idx]
                rate, wrate, is_new = stats[name].query(k, t_now)

                out[f"ER_RATE_{name}"][idx] = rate
                out[f"ER_WRATE_{name}"][idx] = wrate
                out[f"ER_TREND_{name}"][idx] = wrate - rate
                out[f"IS_NEW_{name}"][idx] = is_new

        # Apply labels only after the whole block has been scored.
        for idx in group:
            y_now = int(y[idx])
            for name in keys:
                k = key_values[name][idx]
                stats[name].update(k, t_now, y_now)

        processed += len(group)
        if processed // 50000 > (processed - len(group)) // 50000:
            print(f"[Train causal ER] {processed}/{n} rows processed")

    return out, stats


def build_frozen_er_test(X_test, keys, stats, global_rate):
    n = len(X_test)
    dt = get_time_seconds(X_test)

    key_values = {
        name: make_key_series(X_test, cols).astype(str).to_numpy()
        for name, cols in keys.items()
    }

    out = {}
    for name in keys:
        out[f"ER_RATE_{name}"] = np.full(n, global_rate, dtype=np.float32)
        out[f"ER_WRATE_{name}"] = np.full(n, global_rate, dtype=np.float32)
        out[f"ER_TREND_{name}"] = np.zeros(n, dtype=np.float32)
        out[f"IS_NEW_{name}"] = np.ones(n, dtype=np.int8)

    # test has no labels, so only query frozen train history
    order = np.argsort(dt)

    for step, idx in enumerate(order, start=1):
        t_now = float(dt[idx])

        for name in keys:
            k = key_values[name][idx]
            rate, wrate, is_new = stats[name].query(k, t_now)

            out[f"ER_RATE_{name}"][idx] = rate
            out[f"ER_WRATE_{name}"][idx] = wrate
            out[f"ER_TREND_{name}"][idx] = wrate - rate
            out[f"IS_NEW_{name}"][idx] = is_new

        if step % 50000 == 0:
            print(f"[Test frozen ER] {step}/{n} rows processed")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--half_life_days", type=float, default=7.0)
    ap.add_argument("--smooth_strength", type=float, default=10.0)
    ap.add_argument("--train_only", action="store_true",
                    help="Build causal features for the training matrix only.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_train = pd.read_pickle(os.path.join(args.proc_dir, "train_features.pkl"))
    X_test = None
    if not args.train_only:
        X_test = pd.read_pickle(os.path.join(args.proc_dir, "test_features.pkl"))
    y = np.load(os.path.join(args.proc_dir, "y.npy")).astype(np.int8)

    er_keys = {
        "card1": ["card1"],
        "addr1": ["addr1"],
        "P_emaildomain": ["P_emaildomain"],
        "R_emaildomain": ["R_emaildomain"],
        "DeviceType": ["DeviceType"],

        "c1_a1": ["card1", "addr1"],
        "c1_pe": ["card1", "P_emaildomain"],
        "c1_re": ["card1", "R_emaildomain"],
        "a1_pe": ["addr1", "P_emaildomain"],
        "a1_dt": ["addr1", "DeviceType"],
        "pe_dt": ["P_emaildomain", "DeviceType"],
        "c4_c1": ["card4", "card1"],

        "c1_a1_pe": ["card1", "addr1", "P_emaildomain"],

        "full": [
            "card1", "card2", "card3", "card4", "card5", "card6",
            "P_emaildomain", "R_emaildomain", "addr1", "addr2", "DeviceType"
        ],
    }

    global_rate = float(np.mean(y))

    print("[Build] fast causal ER train")
    er_tr, stats = build_causal_er_train(
        X_train,
        y,
        er_keys,
        smooth_strength=args.smooth_strength,
        half_life_days=args.half_life_days,
    )

    X_train2 = X_train.copy()
    X_test2 = None
    er_te = None
    if not args.train_only:
        print("[Build] frozen causal ER test")
        er_te = build_frozen_er_test(X_test, er_keys, stats, global_rate)
        X_test2 = X_test.copy()

    for k, v in er_tr.items():
        X_train2[k] = v
    if er_te is not None:
        for k, v in er_te.items():
            X_test2[k] = v

    node_keys = ["card1", "addr1", "P_emaildomain", "R_emaildomain", "DeviceType"]
    X_train2 = add_graph_agg(X_train2, node_keys, prefix="GR")
    if X_test2 is not None:
        X_test2 = add_graph_agg(X_test2, node_keys, prefix="GR")

    backoff_chain = ["full", "c1_a1_pe", "c1_a1", "c1_pe", "card1"]
    X_train2 = add_hier_backoff(X_train2, backoff_chain, global_rate, out_prefix="ER_HIER")
    if X_test2 is not None:
        X_test2 = add_hier_backoff(X_test2, backoff_chain, global_rate, out_prefix="ER_HIER")

    # keep only two IS_NEW indicators
    keep_new = {"IS_NEW_full", "IS_NEW_c1_a1"}
    for c in [c for c in X_train2.columns if c.startswith("IS_NEW_")]:
        if c not in keep_new:
            X_train2.drop(columns=[c], inplace=True)
            if X_test2 is not None:
                X_test2.drop(columns=[c], inplace=True)

    meta_path = os.path.join(args.proc_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    new_cols = [c for c in X_train2.columns if c not in X_train.columns]

    meta["num_cols"] = sorted(list(set(meta.get("num_cols", [])) | set(new_cols)))
    meta["causal_entity_features"] = True
    meta["causal_entity_algorithm"] = "recursive_exponential_decay"
    meta["half_life_days"] = args.half_life_days
    meta["smooth_strength"] = args.smooth_strength
    meta["entity_keys"] = er_keys

    X_train2.to_pickle(os.path.join(args.out_dir, "train_features.pkl"))
    if X_test2 is not None:
        X_test2.to_pickle(os.path.join(args.out_dir, "test_features.pkl"))
    np.save(os.path.join(args.out_dir, "y.npy"), y)

    for fn in ["train_txid.npy", "test_txid.npy"]:
        src = os.path.join(args.proc_dir, fn)
        if os.path.exists(src):
            arr = np.load(src, allow_pickle=True)
            np.save(os.path.join(args.out_dir, fn), arr)

    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[Done] Saved to", args.out_dir)
    print("Base cols:", X_train.shape[1], "| New cols:", len(new_cols), "| Total:", X_train2.shape[1])
    print("New col examples:", new_cols[:20])


if __name__ == "__main__":
    main()
