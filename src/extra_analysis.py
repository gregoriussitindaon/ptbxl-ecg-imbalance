"""Analisis tambahan untuk slide: uji berpasangan antar fold, analisis kesalahan, contoh sinyal EKG.

Jalankan dari folder code:  python src/extra_analysis.py
Keluaran: results/extra_analysis.json dan results/figures/ekg_contoh.png
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from metrics import best_thresholds

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "ptbxl_sub_100hz_min150"
RUNS = ROOT / "results" / "runs" / "ptbxl_sub_100hz_min150"
OUT = ROOT / "results"
rng = np.random.default_rng(0)


def paired(summary):
    rot = summary["rotation"]
    a, b = rot["baseline_cnn1d_s0"]["per_fold"], rot["cnn1d_pos_weight_s0"]["per_fold"]
    res = {}
    for m in ("macro_auprc", "macro_auroc", "rare_auprc"):
        d = np.array(b[m]) - np.array(a[m])
        boots = [rng.choice(d, len(d)).mean() for _ in range(10000)]
        res[m] = {
            "mean_diff": float(d.mean()), "std": float(d.std(ddof=1)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "wins": int((d > 0).sum()), "n": int(len(d)),
            "wilcoxon_p": float(wilcoxon(d).pvalue),
        }
    return res


def error_analysis(classes, hard=("LAO/LAE", "NST_", "IVCD")):
    """Gabungkan prediksi uji 10 fold (CNN1D + BCE, seed 0), threshold per kelas dari validasi fold itu."""
    ys, preds, probs = [], [], []
    for f in range(1, 11):
        r = RUNS / f"baseline_cnn1d_f{f}_s0"
        y, p = np.load(r / "test_y.npy"), np.load(r / "test_prob.npy")
        thr = best_thresholds(np.load(r / "val_y.npy"), np.load(r / "val_prob.npy"))
        ys.append(y); probs.append(p); preds.append(p >= thr)
    y, pred, prob = (np.concatenate(v) for v in (ys, preds, probs))
    ci = {c: i for i, c in enumerate(classes)}
    out = {}
    for c in hard:
        i = ci[c]
        pos = y[:, i] == 1
        fn = pos & ~pred[:, i]
        others = [k for k in classes if k != c]
        # kelas lain yang sering diprediksi pada rekaman positif c yang terlewat
        miss_pred = pred[fn][:, [ci[k] for k in others]].mean(0)
        top = sorted(zip(others, miss_pred), key=lambda t: -t[1])[:3]
        # kelas lain yang menyertai label c pada rekaman positif
        co = y[pos][:, [ci[k] for k in others]].mean(0)
        cotop = sorted(zip(others, co), key=lambda t: -t[1])[:3]
        out[c] = {
            "n_pos": int(pos.sum()), "sens": float((pos & pred[:, i]).sum() / pos.sum()),
            "n_missed": int(fn.sum()),
            "missed_predicted_norm": float(pred[fn][:, ci["NORM"]].mean()),
            "missed_no_prediction": float((pred[fn].sum(1) == 0).mean()),
            "missed_top_pred": [(k, float(v)) for k, v in top],
            "co_labels": [(k, float(v)) for k, v in cotop],
            "precision": float((pos & pred[:, i]).sum() / max(pred[:, i].sum(), 1)),
        }
    return out


def ekg_figure(classes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X = np.load(DATA / "X.npy", mmap_mode="r")
    m = np.load(DATA / "meta.npz", allow_pickle=True)
    y, folds = m["y"], m["strat_fold"]
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    picks = [("NORM", "Normal (NORM)"), ("CLBBB", "Blok cabang berkas kiri (CLBBB)"), ("LVH", "Hipertrofi ventrikel kiri (LVH)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    t = np.arange(1000) / 100
    for ax, (c, title) in zip(axes, picks):
        i = classes.index(c)
        only = (y[:, i] == 1) & (y.sum(1) == 1) & (folds == 10)
        idx = np.where(only)[0][3] if only.sum() > 3 else np.where(y[:, i] == 1)[0][0]
        sig = np.asarray(X[idx])
        sig = (sig - sig.mean(0)) / (sig.std(0) + 1e-6)
        for k in range(12):
            ax.plot(t, sig[:, k] * 0.38 - k, lw=0.7, color="#1E2761")
            ax.text(-0.15, -k, leads[k], ha="right", va="center", fontsize=8, color="#6B7280")
        ax.set_title(title, fontsize=12, color="#1E2761", fontweight="bold")
        ax.set_yticks([]); ax.set_xlim(0, 10); ax.set_xlabel("detik")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "ekg_contoh.png", dpi=170)


if __name__ == "__main__":
    s = json.loads((OUT / "summary.json").read_text())
    res = {"paired_rotation": paired(s), "error_analysis": error_analysis(s["classes"])}
    (OUT / "extra_analysis.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    ekg_figure(s["classes"])
    print(json.dumps(res, indent=1, ensure_ascii=False))
