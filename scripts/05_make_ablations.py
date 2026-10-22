
import os, json, shutil, argparse
from pathlib import Path
import pandas as pd

SRC = "./processed_features/ieee_proc_v2_plus_causal_h7_s10"

ABLATIONS = {
    "ieee_ab_causal_er_rate": [
        "drop_prefix:ER_WRATE_",
        "drop_prefix:ER_TREND_",
        "drop_prefix:GR_",
        "drop_prefix:ER_HIER_",
        "drop_prefix:IS_NEW_",
    ],

    "ieee_ab_causal_er_time": [
        "drop_prefix:GR_",
        "drop_prefix:ER_HIER_",
        "drop_prefix:IS_NEW_",
    ],

    "ieee_ab_causal_full_nohier": [
        "drop_prefix:ER_HIER_",
        "drop_prefix:IS_NEW_",
    ],
}

def apply_rules(df, rules):
    cols = df.columns
    to_drop = set()
    for r in rules:
        if r.startswith("drop_prefix:"):
            p = r.split(":", 1)[1]
            to_drop |= {c for c in cols if c.startswith(p)}
        else:
            raise ValueError(f"Unknown rule: {r}")
    return df.drop(columns=sorted(to_drop), errors="ignore")

def update_meta(meta_path, train_cols):
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    meta["num_cols"] = sorted(list(train_cols))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def make_one(out_dir, rules, src, force=False):
    src_path = Path(src).resolve()
    out_path = Path(out_dir).resolve()
    out_root = out_path.parent

    if not src_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_path}")
    if out_path == src_path or src_path in out_path.parents:
        raise ValueError(f"Refusing to write inside the source directory: {out_path}")
    if out_root == Path(out_root.anchor) or out_root == Path.home():
        raise ValueError(f"Refusing to use a filesystem root or home directory: {out_root}")
    if out_path.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {out_path}. "
                f"Pass --force only after verifying the path."
            )
        shutil.rmtree(out_path)

    shutil.copytree(src_path, out_path)

    tr_path = os.path.join(out_dir, "train_features.pkl")
    te_path = os.path.join(out_dir, "test_features.pkl")
    meta_path = os.path.join(out_dir, "meta.json")

    Xtr = pd.read_pickle(tr_path)
    Xte = pd.read_pickle(te_path)

    Xtr2 = apply_rules(Xtr, rules)
    Xte2 = apply_rules(Xte, rules)

    Xtr2.to_pickle(tr_path)
    Xte2.to_pickle(te_path)

    update_meta(meta_path, Xtr2.columns)

    print(f"[OK] {out_dir}: {Xtr.shape[1]} -> {Xtr2.shape[1]} cols")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=SRC)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--force", action="store_true",
                    help="Allow replacing existing variant directories.")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    if out_root == Path(out_root.anchor) or out_root == Path.home():
        raise ValueError(f"Refusing to use a filesystem root or home directory: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    for name, rules in ABLATIONS.items():
        out_dir = out_root / name
        make_one(out_dir, rules, args.src, force=args.force)
