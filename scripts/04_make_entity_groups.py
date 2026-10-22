import os
import argparse
import numpy as np
import pandas as pd

def make_key_series(df: pd.DataFrame, cols, fill="NA"):
    parts = []
    for c in cols:
        s = df[c] if c in df.columns else pd.Series(fill, index=df.index)
        parts.append(s.astype("object").fillna(fill).astype(str))
    out = parts[0]
    for s in parts[1:]:
        out = out + "_" + s
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc_dir", type=str, required=True)
    ap.add_argument("--out_path", type=str, default=None)
    args = ap.parse_args()

    X_train = pd.read_pickle(os.path.join(args.proc_dir, "train_features.pkl"))

    cols = ["card1","card2","card3","card4","card5","card6","P_emaildomain","R_emaildomain","addr1","addr2","DeviceType"]
    ent = make_key_series(X_train, cols).astype(str).to_numpy()

    out_path = args.out_path or os.path.join(args.proc_dir, "entity.npy")
    np.save(out_path, ent, allow_pickle=True)
    print("[Saved]", out_path, "len=", len(ent))

if __name__ == "__main__":
    main()
