import argparse
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter
from PIL import Image

_DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
ROOT = os.environ.get("FCRF_ROOT", _DEFAULT_ROOT)
RUNS = os.path.join(ROOT, "runs_new_entities")
OUT = os.path.join(ROOT, "outputs", "figures")
DATA_DIR = OUT

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.prop_cycle": plt.cycler(color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]),
})
C0, C1, C2, C3 = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"
GRAY = "#444444"
DEEP_BLUE = "#1F4E79"
METRICS4 = ["AUC", "PR-AUC", "Recall@1%FPR", "Recall@5%FPR"]

def results_dir():
    """Directory holding the result files; follows --root unless FCRF_RESULTS is set.

    The directory is created when missing so that a fresh clone without the
    results folder does not fail before the first figure is drawn.
    """
    path = os.environ.get("FCRF_RESULTS", os.path.join(ROOT, "results"))
    os.makedirs(path, exist_ok=True)
    return path


def require_result(path, hint):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required result file not found: {path}\n{hint}")
    return path


def load_paysim():
    """PaySim results, read from results/metrics_paysim.json."""
    path = require_result(
        os.path.join(results_dir(), "metrics_paysim.json"),
        "Run scripts/paysim_experiment.py first to generate this file.",
    )
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {
        "baseline": [d["baseline"]["AUC"], d["baseline"]["PR_AUC"],
                     d["baseline"]["test_threshold"]["recall"]],
        "full": [d["full"]["AUC"], d["full"]["PR_AUC"],
                 d["full"]["test_threshold"]["recall"]],
    }


def load_runtime():
    """Stage runtimes, read from results/runtime_table_for_paper.csv."""
    import csv
    path = require_result(
        os.path.join(results_dir(), "runtime_table_for_paper.csv"),
        "Run the training scripts first to generate this file.",
    )
    labels = ["Causal feature\nconstruction", "Group entity\nindex",
              "Full causal\ntraining (5-fold)", "Baseline\ntraining (5-fold)"]
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("Wall_time_minutes")]
    if len(rows) != len(labels):
        raise ValueError(f"Expected {len(labels)} pipeline stages in {path}, found {len(rows)}.")
    return [(lab, float(r["Wall_time_minutes"])) for lab, r in zip(labels, rows)]


def load_hp():
    """Hyperparameter sweep, read from results/hyperparameter_sensitivity.csv."""
    import csv
    path = require_result(
        os.path.join(results_dir(), "hyperparameter_sensitivity.csv"),
        "Run scripts/02_build_causal_features.py and scripts/03_train_lgbm.py for each "
        "setting first to generate this file.",
    )
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((f"{int(float(r['half_life_days']))}/{int(float(r['smooth_strength']))}",
                         float(r["Recall@1%FPR"])))
    if not rows:
        raise ValueError(f"No configurations found in {path}.")
    return rows


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.15)
    for ext, kw in (("png", {}), ("tif", {"pil_kwargs": {"compression": "tiff_lzw"}})):
        buf = io.BytesIO()
        fig.savefig(buf, format="png" if ext == "png" else "tif",
                    bbox_inches="tight", pad_inches=0.15, **kw)
        buf.seek(0)
        im = Image.open(buf).convert("RGBA")
        flat = Image.new("RGB", im.size, (255, 255, 255))
        flat.paste(im, mask=im.split()[3])
        if ext == "tif":
            flat.save(os.path.join(OUT, f"{name}.{ext}"), dpi=(300, 300),
                      compression="tiff_lzw")
        else:
            flat.save(os.path.join(OUT, f"{name}.{ext}"), dpi=(300, 300))
    plt.close(fig)
    print("[saved]", name)


def panel_label(ax, letter, text=""):
    ax.set_title(f"{letter}  {text}".strip(), loc="left", fontsize=9.5,
                 fontweight="bold", pad=6)


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required data file not found: {path}\n"
            f"Run compute_bootstrap_new.py first, or place bootstrap_ci_new.json "
            f"in {os.path.dirname(path)}."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_metrics():
    names = {
        "baseline": "baseline",
        "full": "full",
        "static": "ab_er_rate",
        "time": "ab_er_time",
        "nohier": "ab_full_nohier",
    }
    out = {}
    for key, folder in names.items():
        m = load_json(os.path.join(RUNS, folder, "metrics_cv.json"))
        out[key] = {
            "AUC": m["OOF_cal"]["AUC"],
            "PR-AUC": m["OOF_cal"]["PR_AUC"],
            "Recall@1%FPR": m["Recall_at_FPR"]["0.0100"]["recall"],
            "Recall@5%FPR": m["Recall_at_FPR"]["0.0500"]["recall"],
        }
    return out


def temporal_metrics():
    out = {}
    for key, folder in (("baseline", "temporal_baseline"), ("full", "temporal_full")):
        m = load_json(os.path.join(RUNS, folder, "metrics_temporal.json"))
        out[key] = {
            "AUC": m["OOF_cal"]["AUC"],
            "PR-AUC": m["OOF_cal"]["PR_AUC"],
            "Recall@1%FPR": m["Recall_at_FPR"]["0.0100"]["recall"],
            "Recall@5%FPR": m["Recall_at_FPR"]["0.0500"]["recall"],
        }
    return out


def fixed_fpr_table():
    y = np.load(os.path.join(ROOT, "processed_features_full_h7", "y.npy")).astype(np.int8)
    out = {}
    for key in ("baseline", "full"):
        p = np.load(os.path.join(RUNS, key, "oof_cal.npy")).astype(np.float64)
        rows = []
        for target in (0.005, 0.01, 0.02):
            order = np.argsort(-p)
            ys = y[order]
            tp = np.cumsum(ys)
            fp = np.cumsum(1 - ys)
            pos = int(ys.sum())
            neg = len(ys) - pos
            ok = np.where(fp / neg <= target)[0]
            i = ok[-1]
            rows.append({
                "target": target,
                "fpr": float(fp[i] / neg),
                "tp": int(tp[i]),
                "fp": int(fp[i]),
                "fn": int(pos - tp[i]),
                "recall": float(tp[i] / pos),
                "precision": float(tp[i] / max(tp[i] + fp[i], 1)),
            })
        out[key] = rows
    return out


def cost_table():
    tab = fixed_fpr_table()
    out = {}
    for key in ("baseline", "full"):
        row = next(r for r in tab[key] if abs(r["target"] - 0.01) < 1e-12)
        out[key] = {"fn": row["fn"], "fp": row["fp"], "ratio": {}}
        for ratio in (10, 50, 100):
            fn_cost = ratio * row["fn"]
            fp_cost = row["fp"]
            out[key]["ratio"][ratio] = {
                "fn_cost": fn_cost,
                "fp_cost": fp_cost,
                "total": fn_cost + fp_cost,
            }
    return out


def relative_gains():
    g = group_metrics()
    t = temporal_metrics()
    rows = []
    for base, full in ((g["baseline"], g["full"]), (t["baseline"], t["full"])):
        rows.append([
            100.0 * (full[m] - base[m]) / base[m]
            for m in ("AUC", "PR-AUC", "Recall@1%FPR")
        ])
    paysim = load_paysim()
    rows.append([
        100.0 * (paysim["full"][i] - paysim["baseline"][i]) / paysim["baseline"][i]
        for i in range(3)
    ])
    return rows


def fig1():
    fig, ax = plt.subplots(figsize=(4.6, 8.2), layout="constrained")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 21)
    ax.axis("off")
    boxes = [
        (18.0, "Transaction Data\n(Transaction & Identity Features)", "#EAF1F8"),
        (15.6, "Baseline Feature Engineering", "#EAF1F8"),
        (13.2, "Entity Construction\n(Card, Email, Device, Address)", "#E6F2E9"),
        (10.3, "OOF Entity Risk Encoding\n(No Label Leakage)", "#FDEBD9"),
        (7.9, "Time-Decayed Entity Risk Modeling", "#FDEBD9"),
        (5.5, "Risk Aggregation\n& Hierarchical Backoff", "#FDEBD9"),
        (3.0, "LightGBM Classifier", "#EAF1F8"),
        (0.9, "Risk-Aware Thresholding\n(Global FPR Constraint)", "#EFE7F5"),
    ]
    for y, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((0.9, y), 8.2, 1.8, boxstyle="round,pad=0.12",
                                    fc=col, ec="#404040", lw=0.9))
        ax.text(5.0, y + 0.9, txt, ha="center", va="center", fontsize=8.5)
    for (y0, _, _), (y1, _, _) in zip(boxes[:-1], boxes[1:]):
        ax.add_patch(FancyArrowPatch((5.0, y0), (5.0, y1 + 1.8), arrowstyle="-|>",
                                     mutation_scale=8, color="#404040", lw=0.9))
    ax.add_patch(Rectangle((0.55, 5.05), 8.9, 7.35, fill=False, ec="#B30000",
                           lw=1.1, ls=(0, (4, 2.6))))
    ax.text(0.62, 12.55, "Leakage-aware entity-risk modeling", fontsize=8.5, color="#B30000")
    ax.add_patch(FancyArrowPatch((5.0, 0.9), (5.0, 0.42), arrowstyle="-|>",
                                 mutation_scale=8, color="#404040", lw=0.9))
    ax.text(5.0, 0.20, "Fraud Detection Output", ha="center", va="bottom", fontsize=8.5)
    save(fig, "Fig1")


def fig2():
    boot = load_json(os.path.join(DATA_DIR, "bootstrap_ci_new.json"))
    metrics = ["AUC", "PR-AUC", "Recall@1%FPR"]
    labels = ["Baseline", "Full causal framework"]
    keys = ["baseline", "full"]
    colors = [C0, C1]

    widths = [
        boot[k][m]["ci_upper"] - boot[k][m]["ci_lower"]
        for k in keys for m in metrics
    ]
    # Horizontal CI glyphs are magnified by a common factor so that the
    # narrow intervals remain visible; the exact interval is printed inside
    # each bar.
    scale = 0.30 / max(widths)

    fig, ax = plt.subplots(figsize=(6.83, 4.3), layout="constrained")
    x = np.arange(len(metrics))
    w = 0.34
    offset = 0.18

    for i, (key, label, col) in enumerate(zip(keys, labels, colors)):
        xc = x + (i - 0.5) * 2 * offset
        means = np.array([boot[key][m]["mean"] for m in metrics])
        los = np.array([boot[key][m]["ci_lower"] for m in metrics])
        his = np.array([boot[key][m]["ci_upper"] for m in metrics])
        ax.bar(xc, means, w, color=col, edgecolor="black", linewidth=0.8,
               label=label, zorder=2)
        for cx, m, lo, hi in zip(xc, means, los, his):
            half = 0.5 * (hi - lo) * scale
            ax.plot([cx - half, cx + half], [m, m], color="#222222", lw=1.2, zorder=5)
            for xv in (cx - half, cx + half):
                ax.plot([xv, xv], [m - 0.012, m + 0.012], color="#222222", lw=1.2, zorder=5)
            ax.text(cx, m + 0.075, f"{m:.4f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="black", zorder=6)
            ax.text(cx, m + 0.030, f"{lo:.3f}-{hi:.3f}", ha="center", va="bottom",
                    fontsize=8, color=GRAY, zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 1.10)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Metric value")
    ax.set_xlabel("Metric")
    ax.grid(axis="y", color="#E6E6E6", lw=0.6, ls="--")
    ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2, frameon=False,
               handletextpad=0.4, columnspacing=1.6)
    save(fig, "Fig2")


def fig3():
    m = group_metrics()
    order = ["baseline", "static", "time", "nohier", "full"]
    labels = ["Baseline", "+Static", "+Time-decay", "No backoff", "Full"]
    fig, ax = plt.subplots(figsize=(6.83, 3.9), layout="constrained")
    x = np.arange(len(order))
    vals = [m[k]["Recall@1%FPR"] for k in order]
    vals_arr = np.asarray(vals, dtype=float)
    ax.bar(x, vals_arr - 0.04, 0.55, bottom=0.04, color=DEEP_BLUE,
           edgecolor="black", linewidth=0.8)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 0.82)
    ax.set_yticks(np.arange(0, 0.81, 0.1))
    ax.set_ylabel("Recall@1%FPR")
    ax.grid(axis="y", color="#E6E6E6", lw=0.6, ls="--")
    ax.set_axisbelow(True)
    save(fig, "Fig3")


def fig4():
    tab = fixed_fpr_table()
    x = [0.5, 1.0, 2.0]
    fig, ax = plt.subplots(figsize=(6.83, 4.1), layout="constrained")
    series = [
        ("baseline", "recall", "Baseline Recall", C0, "-", "o", 0.012),
        ("full", "recall", "Full Recall", C1, "-", "s", 0.012),
        ("baseline", "precision", "Baseline Precision", C0, "--", "^", -0.014),
        ("full", "precision", "Full Precision", C1, "--", "D", -0.014),
    ]
    for key, field, label, col, ls, marker, dy in series:
        vals = [r[field] for r in tab[key]]
        ax.plot(x, vals, color=col, ls=ls, marker=marker, lw=1.5, ms=4.2, label=label)
        for xi, v in zip(x, vals):
            ax.text(xi, v + dy, f"{v:.3f}", ha="center",
                    va="bottom" if dy > 0 else "top", fontsize=8, color=col)
    ax.set_xlabel("Target FPR (%)")
    ax.set_ylabel("Metric value")
    ax.set_xticks(x)
    ax.set_xlim(0.2, 2.3)
    ax.set_ylim(0.45, 0.88)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6, ls="--")
    ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=4, frameon=False,
               handletextpad=0.4, columnspacing=1.2)
    save(fig, "Fig4")


def fig5():
    tab = fixed_fpr_table()
    x = np.arange(3)
    w = 0.34
    rb = [r["recall"] for r in tab["baseline"]]
    rf = [r["recall"] for r in tab["full"]]
    fig, axes = plt.subplots(1, 2, figsize=(6.83, 3.8), layout="constrained")

    ax = axes[0]
    ax.plot(x, rb, "o-", color=C0, lw=1.5, ms=4.2, label="Baseline")
    ax.plot(x, rf, "o-", color=C1, lw=1.5, ms=4.2, label="Full causal framework")
    for xi, v in list(zip(x, rb)) + list(zip(x, rf)):
        ax.text(xi, v + 0.010, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["0.5%", "1%", "2%"])
    ax.set_xlabel("Target FPR")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 0.90)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6, ls="--")
    ax.set_axisbelow(True)
    panel_label(ax, "A", "Recall at matched FPR")

    ax = axes[1]
    gains = [f - b for b, f in zip(rb, rf)]
    ax.bar(x, gains, 0.5, color=DEEP_BLUE, edgecolor="black", linewidth=0.8)
    for xi, v in zip(x, gains):
        ax.text(xi, v + 0.006, f"+{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["0.5%", "1%", "2%"])
    ax.set_xlabel("Target FPR")
    ax.set_ylabel("Absolute recall gain")
    ax.set_ylim(0, 0.30)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6, ls="--")
    ax.set_axisbelow(True)
    panel_label(ax, "B", "Gain over baseline")

    save(fig, "Fig5")


def fig6():
    tab = fixed_fpr_table()
    rb = next(r for r in tab["baseline"] if abs(r["target"] - 0.01) < 1e-12)
    rf = next(r for r in tab["full"] if abs(r["target"] - 0.01) < 1e-12)
    x = np.arange(2)
    w = 0.34
    fn_b, fn_f, fp_b, fp_f = rb["fn"], rf["fn"], rb["fp"], rf["fp"]
    pct = 100.0 * (fn_b - fn_f) / fn_b
    fig, ax = plt.subplots(figsize=(6.83, 4.0), layout="constrained")
    ax.bar(x - w / 2, [fn_b, fn_f], w, color=C0, edgecolor="black", linewidth=0.8,
           label="False negatives")
    ax.bar(x + w / 2, [fp_b, fp_f], w, color=C1, edgecolor="black", linewidth=0.8,
           label="False positives")
    for xi, v in list(zip(x - w / 2, [fn_b, fn_f])) + list(zip(x + w / 2, [fp_b, fp_f])):
        ax.text(xi, v + 160, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.text(0.5, 6600, f"{pct:.1f}% fewer\nfalse negatives",
            ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["LightGBM baseline", "Full causal framework"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Count")
    ax.set_ylim(0, 11000)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6, ls="--")
    ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2, frameon=False,
               handletextpad=0.4, columnspacing=1.6)
    save(fig, "Fig6")


def fig7():
    tab = fixed_fpr_table()
    ratios = [10, 50, 100]
    red = np.zeros((3, 3))
    for i in range(3):
        b = tab["baseline"][i]
        f = tab["full"][i]
        for j, ratio in enumerate(ratios):
            cb = ratio * b["fn"] + b["fp"]
            cf = ratio * f["fn"] + f["fp"]
            red[i, j] = 100.0 * (cb - cf) / cb

    fig, ax = plt.subplots(figsize=(6.83, 3.9), layout="constrained")
    x = np.arange(3)
    width = 0.25
    offsets = [-0.27, 0.0, 0.27]
    colors = ["#0B3C6E", "#2E75B6", "#9DC3E6"]
    for j, (ratio, off, col) in enumerate(zip(ratios, offsets, colors)):
        vals = red[:, j]
        ax.bar(x + off, vals, width, color=col, edgecolor="black", linewidth=0.8,
               label=f"{ratio}:1")
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.6, f"{v:.2f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["0.5%", "1.0%", "2.0%"])
    ax.set_xlabel("Target FPR")
    ax.set_ylabel("Relative cost reduction (%)")
    ax.set_ylim(0, 62)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6, ls="--")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=3, frameon=False, title="Cost ratio",
              handletextpad=0.4, columnspacing=1.2, fontsize=8.5, title_fontsize=8.5)
    save(fig, "Fig7")


def fig8():
    # Values are the five configurations reported in Table 11.
    # The y-axis is truncated so that the 0.737-0.754 range stays readable.
    hp = load_hp()
    labels = [x for x, _ in hp]
    vals = [v for _, v in hp]
    fig, ax = plt.subplots(figsize=(6.83, 3.8), layout="constrained")
    x = np.arange(len(vals))
    ax.plot(x, vals, "-o", color=DEEP_BLUE, lw=1.5, ms=4.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.0004, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.4, len(vals) - 0.6)
    ax.set_xlabel("Half-life (days) / smoothing strength")
    ax.set_ylabel("Recall@1%FPR")
    ax.set_ylim(0.730, 0.760)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6, ls="--")
    ax.set_axisbelow(True)
    ax.text(0.99, 0.02, "y-axis truncated", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=GRAY)
    save(fig, "Fig8")


def fig9():
    # Wall-clock runtimes reported in Table 12.
    names = [
        "Causal feature construction",
        "Group entity index",
        "Full causal training (5-fold)",
        "Baseline training (5-fold)",
    ]
    vals = [v for _, v in load_runtime()]
    fig, ax = plt.subplots(figsize=(6.83, 3.9), layout="constrained")
    x = np.arange(len(vals))
    ax.bar(x, vals, 0.55, color=DEEP_BLUE, edgecolor="black", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylim(0.04, 60)
    ax.set_yticks([0.05, 0.1, 0.5, 1, 5, 10, 50])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for xi, v in zip(x, vals):
        ax.text(xi, v * 1.12, f"{v:.2f} min", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Wall-clock time (minutes, log scale)")
    ax.grid(axis="y", which="major", color="#E6E6E6", lw=0.6, ls="--")
    ax.set_axisbelow(True)
    save(fig, "Fig9")


def fig10():
    # PaySim results for Table 13, produced by scripts/paysim_experiment.py.
    paysim = load_paysim()
    metrics = ["AUC", "PR-AUC", "Recall@FPR<=1%"]
    x = np.arange(3)
    w = 0.34
    fig, ax = plt.subplots(figsize=(6.83, 4.0), layout="constrained")
    ax.bar(x - w / 2, paysim["baseline"], w, color=C0, edgecolor="black",
           linewidth=0.8, label="Baseline (weak features)")
    ax.bar(x + w / 2, paysim["full"], w, color=C1, edgecolor="black",
           linewidth=0.8, label="Time-aware entity risk")
    for xi, v in list(zip(x - w / 2, paysim["baseline"])) + list(zip(x + w / 2, paysim["full"])):
        ax.text(xi, v + 0.012, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6, ls="--")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "Fig10")


def fig11():
    g = group_metrics()
    t = temporal_metrics()
    fig, axes = plt.subplots(1, 2, figsize=(6.83, 3.9), layout="constrained")
    for ax, data, letter, desc in (
        (axes[0], g, "A", "Group cold-start"),
        (axes[1], t, "B", "Temporal holdout"),
    ):
        x = np.arange(4)
        w = 0.30
        off = 0.20
        base = [data["baseline"][m] for m in METRICS4]
        full = [data["full"][m] for m in METRICS4]
        ax.bar(x - off, base, w, color=C0, edgecolor="black", linewidth=0.8,
               label="Baseline")
        ax.bar(x + off, full, w, color=C1, edgecolor="black", linewidth=0.8,
               label="Full causal framework")
        for xi, v in list(zip(x - off, base)) + list(zip(x + off, full)):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["AUC", "PR-AUC", "Recall@1%", "Recall@5%"],
                           rotation=18, ha="right")
        ax.set_xlim(-0.6, 3.6)
        ax.set_ylim(0, 1.02)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", color="#E6E6E6", lw=0.6, ls="--")
        ax.set_axisbelow(True)
        panel_label(ax, letter, desc)
    axes[0].set_ylabel("Metric value")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2, frameon=False,
               handletextpad=0.4, columnspacing=1.6)
    save(fig, "Fig11")


def fig12():
    vals = relative_gains()
    rows = ["Group cold-start", "Temporal holdout", "PaySim"]
    cols = ["AUC", "PR-AUC", "Recall@1%FPR"]
    fig, ax = plt.subplots(figsize=(6.83, 3.8), layout="constrained")
    im = ax.imshow(vals, cmap="Blues", vmin=0, vmax=65, aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(rows)
    for i in range(3):
        for j in range(3):
            v = vals[i][j]
            ax.text(j, i, f"{v:.2f}%", ha="center", va="center",
                    fontsize=8.5, color="white" if v > 32 else "black")
    cb = fig.colorbar(im, ax=ax, ticks=[0, 10, 20, 30, 40, 50, 60], shrink=0.95)
    cb.set_label("Relative improvement (%)")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Evaluation protocol")
    save(fig, "Fig12")


def fig13():
    g = group_metrics()
    t = temporal_metrics()
    x = np.arange(3)
    w = 0.34
    group_gain = [g["full"][m] - g["baseline"][m] for m in ("AUC", "PR-AUC", "Recall@1%FPR")]
    temp_gain = [t["full"][m] - t["baseline"][m] for m in ("AUC", "PR-AUC", "Recall@1%FPR")]
    fig, ax = plt.subplots(figsize=(6.83, 3.9), layout="constrained")
    ax.bar(x - w / 2, group_gain, w, color=C0, edgecolor="black", linewidth=0.8,
           label="Group cold-start")
    ax.bar(x + w / 2, temp_gain, w, color=C1, edgecolor="black", linewidth=0.8,
           label="Temporal holdout")
    for xi, v in list(zip(x - w / 2, group_gain)) + list(zip(x + w / 2, temp_gain)):
        ax.text(xi, v + 0.006, f"+{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["AUC", "PR-AUC", "Recall@1%FPR"])
    ax.set_ylabel("Absolute gain over baseline")
    ax.set_ylim(0, 0.32)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6, ls="--")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "Fig13")


def fig14():
    ct = cost_table()
    ratios = [10, 50, 100]
    fig, axes = plt.subplots(1, 2, figsize=(6.83, 3.8), layout="constrained")
    for ax, key, letter, desc in (
        (axes[0], "baseline", "A", "Baseline"),
        (axes[1], "full", "B", "Full causal framework"),
    ):
        fn = [ct[key]["ratio"][r]["fn_cost"] for r in ratios]
        fp = [ct[key]["ratio"][r]["fp_cost"] for r in ratios]
        x = np.arange(3)
        ax.bar(x, fn, 0.5, color=C0, edgecolor="black", linewidth=0.8,
               label="False negatives")
        ax.bar(x, fp, 0.5, bottom=fn, color=C1, edgecolor="black", linewidth=0.8,
               label="False positives")
        for xi, a, b in zip(x, fn, fp):
            ax.text(xi, a + b + 18000, f"{(a + b) / 1000:.1f}k", ha="center",
                    va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}:1" for r in ratios])
        ax.set_xlabel("Cost ratio")
        ax.set_ylim(0, 1_100_000)
        ax.set_yticks([0, 200000, 400000, 600000, 800000, 1000000])
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: "1M" if v >= 1_000_000 else f"{v / 1000:.0f}k")
        )
        ax.grid(axis="y", color="#E6E6E6", lw=0.6, ls="--")
        ax.set_axisbelow(True)
        panel_label(ax, letter, desc)
    axes[0].set_ylabel("Total cost")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2, frameon=False,
               handletextpad=0.4, columnspacing=1.6)
    save(fig, "Fig14")


def fig15():
    ct = cost_table()
    fn_b = ct["baseline"]["fn"]
    fn_f = ct["full"]["fn"]
    drop = fn_b - fn_f
    pct = 100.0 * drop / fn_b
    fig, ax = plt.subplots(figsize=(6.83, 3.7), layout="constrained")
    ax.bar([0, 1], [fn_b, fn_f], 0.5, color=[C0, C1],
           edgecolor="black", linewidth=0.8)
    ax.text(0, fn_b + 180, f"{fn_b:,}", ha="center", va="bottom", fontsize=8)
    ax.text(1, fn_f + 180, f"{fn_f:,}", ha="center", va="bottom", fontsize=8)
    bracket_y = 11000
    tick = 300
    ax.plot([0, 0, 1, 1], [bracket_y - tick, bracket_y, bracket_y, bracket_y - tick],
            color="black", lw=1.2, solid_capstyle="butt", clip_on=False)
    ax.text(0.5, bracket_y + 120, f"Reduction = {pct:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Full causal framework"])
    ax.set_ylabel("False negatives at 1% FPR")
    ax.set_ylim(0, 12200)
    ax.set_yticks([0, 2000, 4000, 6000, 8000, 10000, 12000])
    ax.grid(axis="y", color="#E6E6E6", lw=0.6, ls="--")
    ax.set_axisbelow(True)
    save(fig, "Fig15")


def fig16():
    # Both panels use truncated y-axes; the truncation is stated in the caption.
    tab = fixed_fpr_table()
    x = [0.5, 1.0, 2.0]
    fig, axes = plt.subplots(1, 2, figsize=(6.83, 3.8), layout="constrained")
    for ax, field, letter, desc, ylim in (
        (axes[0], "recall", "A", "Recall", (0.45, 0.85)),
        (axes[1], "precision", "B", "Precision", (0.50, 0.88)),
    ):
        for key, label, col, marker in (("baseline", "Baseline", C0, "o"),
                                        ("full", "Full causal framework", C1, "s")):
            vals = [r[field] for r in tab[key]]
            ax.plot(x, vals, marker=marker, color=col, lw=1.5, ms=4.6, label=label)
            for xi, v in zip(x, vals):
                ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8, color=col)
        ax.set_xticks(x)
        ax.set_xlim(0.2, 2.3)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Target FPR (%)")
        ax.set_ylabel(desc)
        ax.grid(axis="y", color="#E6E6E6", lw=0.6, ls="--")
        ax.set_axisbelow(True)
        ax.text(0.99, 0.02, "y-axis truncated", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=GRAY)
        panel_label(ax, letter, desc)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2, frameon=False,
               handletextpad=0.4, columnspacing=1.6)
    save(fig, "Fig16")


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7,
        8: fig8, 9: fig9, 10: fig10, 11: fig11, 12: fig12, 13: fig13,
        14: fig14, 15: fig15, 16: fig16}


def main():
    global ROOT, RUNS, OUT, DATA_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", default="1,2,3,4")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ROOT = os.path.abspath(args.root)
    RUNS = os.path.join(ROOT, "runs_new_entities")
    OUT = args.out or os.path.join(ROOT, "outputs", "figures")
    DATA_DIR = OUT
    os.makedirs(OUT, exist_ok=True)
    for n in [int(x) for x in args.figs.split(",") if x.strip()]:
        FIGS[n]()


if __name__ == "__main__":
    main()
