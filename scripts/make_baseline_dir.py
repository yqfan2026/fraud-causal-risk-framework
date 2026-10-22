import argparse
import json
import os
import shutil

import pandas as pd


_DEFAULT_SRC = os.path.join("outputs", "processed_features_full_h7")
_DEFAULT_DST = os.path.join("outputs", "processed_features_full_h7_baseline")
SRC = os.environ.get("FCRF_SRC", _DEFAULT_SRC)
DST = os.environ.get("FCRF_DST", _DEFAULT_DST)

PREFIXES = ("ER_RATE_", "ER_WRATE_", "ER_TREND_", "ER_HIER_", "GR_", "IS_NEW_")


def main():
    global SRC, DST
    ap = argparse.ArgumentParser(
        description="Create the weak-feature baseline copy of the processed feature matrix"
    )
    ap.add_argument("--src", default=SRC, help="processed feature directory (with causal features)")
    ap.add_argument("--dst", default=DST, help="output directory for the baseline copy")
    args = ap.parse_args()
    SRC, DST = args.src, args.dst
    os.makedirs(DST, exist_ok=True)

    for name in ["train_features.pkl", "test_features.pkl"]:
        src_path = os.path.join(SRC, name)
        dst_path = os.path.join(DST, name)
        df = pd.read_pickle(src_path)
        drop = [c for c in df.columns if c.startswith(PREFIXES)]
        df = df.drop(columns=drop)
        df.to_pickle(dst_path)
        print(f"{name}: {df.shape[1]} cols kept, {len(drop)} causal cols dropped")
        del df

    for name in ["y.npy", "entity.npy", "train_txid.npy", "test_txid.npy"]:
        shutil.copy2(os.path.join(SRC, name), os.path.join(DST, name))

    with open(os.path.join(SRC, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    keep = [c for c in meta["num_cols"] if not c.startswith(PREFIXES)]
    meta["num_cols"] = keep
    with open(os.path.join(DST, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("baseline dir ready:", DST, "| num_cols:", len(keep))


if __name__ == "__main__":
    main()
