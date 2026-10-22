import os
import gc
import json
import argparse
import numpy as np
import pandas as pd

def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            continue
        if pd.api.types.is_integer_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], downcast="integer")
        elif pd.api.types.is_float_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], downcast="float")
    return out

def read_csv(path: str) -> pd.DataFrame:
    print("[Load]", path)
    df = pd.read_csv(path, low_memory=False)
    df = reduce_mem(df)
    print("[Shape]", os.path.basename(path), df.shape)
    return df

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "TransactionDT" in out.columns:
        dt = out["TransactionDT"].astype(np.float32)
        out["DT_HOUR"] = (dt / 3600.0) % 24
        out["DT_DAY"]  = (dt / (3600.0 * 24.0)) % 7
        out["DT_WEEK"] = (dt / (3600.0 * 24.0 * 7.0)).astype(np.float32)

    if "TransactionAmt" in out.columns:
        amt = out["TransactionAmt"].astype(np.float32)
        out["AMT_LOG1P"] = np.log1p(amt)
        out["AMT_CENTS"] = ((amt - np.floor(amt)) * 1000).round().astype(np.int32)

    out["MISS_CNT"] = out.isna().sum(axis=1).astype(np.int16)
    return reduce_mem(out)

def align_columns(train: pd.DataFrame, test: pd.DataFrame):
    tr_cols = set(train.columns)
    te_cols = set(test.columns)

    only_tr = sorted(list(tr_cols - te_cols))
    only_te = sorted(list(te_cols - tr_cols))

    if only_tr:
        print(f"[Align] Add {len(only_tr)} cols to test (missing in test), e.g. {only_tr[:8]}")
        for c in only_tr:
            test[c] = np.nan
    if only_te:
        print(f"[Align] Add {len(only_te)} cols to train (missing in train), e.g. {only_te[:8]}")
        for c in only_te:
            train[c] = np.nan

    cols = sorted(train.columns.tolist())
    train = train[cols]
    test = test[cols]
    return train, test, only_tr, only_te

def unify_categories(train: pd.DataFrame, test: pd.DataFrame, cat_cols: list[str]):
    train = train.copy()
    test = test.copy()
    for c in cat_cols:
        tr = train[c].astype("object").fillna("Unknown")
        te = test[c].astype("object").fillna("Unknown")
        all_cats = pd.Index(tr.unique()).union(pd.Index(te.unique()))
        train[c] = pd.Categorical(tr, categories=all_cats)
        test[c] = pd.Categorical(te, categories=all_cats)
    return train, test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./ieee")
    ap.add_argument("--out_dir", type=str, default="./ieee_proc_v2")
    ap.add_argument("--miss_thresh", type=float, default=0.99)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tr_trx = read_csv(os.path.join(args.data_dir, "train_transaction.csv"))
    te_trx = read_csv(os.path.join(args.data_dir, "test_transaction.csv"))
    tr_id  = read_csv(os.path.join(args.data_dir, "train_identity.csv"))
    te_id  = read_csv(os.path.join(args.data_dir, "test_identity.csv"))

    train = tr_trx.merge(tr_id, on="TransactionID", how="left")
    test  = te_trx.merge(te_id, on="TransactionID", how="left")
    del tr_trx, te_trx, tr_id, te_id
    gc.collect()

    y = train["isFraud"].astype(np.int8).to_numpy()
    train = train.drop(columns=["isFraud"], errors="ignore")

    # drop huge-missing cols based on train
    miss = train.isna().mean()
    drop_cols = miss[miss >= args.miss_thresh].index.tolist()
    if drop_cols:
        print(f"[Drop] {len(drop_cols)} cols with missing >= {args.miss_thresh}")
        train = train.drop(columns=drop_cols, errors="ignore")
        test  = test.drop(columns=drop_cols, errors="ignore")

    train = add_basic_features(train)
    test  = add_basic_features(test)

    # keep TransactionID arrays (for later join / debugging)
    train_txid = train["TransactionID"].astype(np.int64).to_numpy()
    test_txid  = test["TransactionID"].astype(np.int64).to_numpy()

    # align before dropping TransactionID
    train, test, only_tr, only_te = align_columns(train, test)

    # drop TransactionID from model features
    train = train.drop(columns=["TransactionID"], errors="ignore")
    test  = test.drop(columns=["TransactionID"], errors="ignore")

    # categorical cols: object / str / string / category (pandas 2.x and 3.x)
    cat_cols = [
        c for c in train.columns
        if pd.api.types.is_string_dtype(train[c])
        or pd.api.types.is_object_dtype(train[c])
        or isinstance(train[c].dtype, pd.CategoricalDtype)
    ]
    if cat_cols:
        train, test = unify_categories(train, test, cat_cols)

    # numeric cols
    num_cols = [c for c in train.columns if c not in cat_cols]

    # numeric coercion
    for c in num_cols:
        train[c] = pd.to_numeric(train[c], errors="coerce")
        test[c]  = pd.to_numeric(test[c], errors="coerce")

    train[num_cols] = train[num_cols].replace([np.inf, -np.inf], np.nan).fillna(-999.0)
    test[num_cols]  = test[num_cols].replace([np.inf, -np.inf], np.nan).fillna(-999.0)

    train = reduce_mem(train)
    test  = reduce_mem(test)

    const_cols = [c for c in train.columns if train[c].nunique(dropna=False) <= 1]
    print(f"[Check] constant columns: {len(const_cols)} | categorical columns: {len(cat_cols)}")
    for key in ["P_emaildomain", "DeviceType"]:
        if key in train.columns and train[key].nunique(dropna=False) <= 1:
            raise RuntimeError(f"{key} collapsed to a single value; categorical handling failed")

    # save
    train.to_pickle(os.path.join(args.out_dir, "train_features.pkl"))
    test.to_pickle(os.path.join(args.out_dir, "test_features.pkl"))
    np.save(os.path.join(args.out_dir, "y.npy"), y)
    np.save(os.path.join(args.out_dir, "train_txid.npy"), train_txid)
    np.save(os.path.join(args.out_dir, "test_txid.npy"), test_txid)

    meta = {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "drop_cols": drop_cols,
        "only_in_train_added_to_test": only_tr[:50],
        "only_in_test_added_to_train": only_te[:50],
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[Done] Saved to", args.out_dir)
    print(" - train_features.pkl / test_features.pkl / y.npy / meta.json")
    print(" - train_txid.npy / test_txid.npy")

if __name__ == "__main__":
    main()
