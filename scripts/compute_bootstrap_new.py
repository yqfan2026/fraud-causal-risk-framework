import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


_DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
ROOT = os.environ.get("FCRF_ROOT", _DEFAULT_ROOT)
OUT = os.path.join(ROOT, "outputs", "figures")
N_BOOT = 1000


def recall_at_fpr(y, p, target=0.01):
    order = np.argsort(-p)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    pos = int(ys.sum())
    neg = len(ys) - pos
    ok = np.where(fp / neg <= target)[0]
    i = ok[-1] if len(ok) else 0
    return float(tp[i] / pos)


def main():
    os.makedirs(OUT, exist_ok=True)
    y = np.load(os.path.join(ROOT, "processed_features_full_h7", "y.npy")).astype(np.int8)
    rng = np.random.default_rng(42)

    result = {}
    for name in ("baseline", "full"):
        p = np.load(os.path.join(ROOT, "runs_new_entities", name, "oof_cal.npy")).astype(np.float64)
        aucs, prs, recs = [], [], []
        for b in range(N_BOOT):
            idx = rng.integers(0, len(y), len(y))
            yb, pb = y[idx], p[idx]
            aucs.append(float(roc_auc_score(yb, pb)))
            prs.append(float(average_precision_score(yb, pb)))
            recs.append(recall_at_fpr(yb, pb, 0.01))
            if (b + 1) % 100 == 0:
                print(f"{name}: {b + 1}/{N_BOOT}", flush=True)
        result[name] = {}
        for metric, vals in (("AUC", aucs), ("PR-AUC", prs), ("Recall@1%FPR", recs)):
            vals = np.asarray(vals)
            result[name][metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)),
                "ci_lower": float(np.quantile(vals, 0.025)),
                "ci_upper": float(np.quantile(vals, 0.975)),
            }
    with open(os.path.join(OUT, "bootstrap_ci_new.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
