"""Metrik multi-label: AUROC/AUPRC per kelas, threshold per kelas, statistik biner."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def per_class_scores(y, p):
    """AUROC dan AUPRC per kelas. NaN jika kelas tidak punya contoh positif atau negatif."""
    n = y.shape[1]
    auroc, auprc = np.full(n, np.nan), np.full(n, np.nan)
    for c in range(n):
        pos = y[:, c].sum()
        if 0 < pos < len(y):
            auroc[c] = roc_auc_score(y[:, c], p[:, c])
            auprc[c] = average_precision_score(y[:, c], p[:, c])
    return auroc, auprc


def macro(x):
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def binary_stats(y, pred):
    """Presisi, sensitivitas, spesifisitas, F1 per kelas dari prediksi biner."""
    y, pred = y.astype(bool), pred.astype(bool)
    tp = (y & pred).sum(0).astype(float)
    fp = (~y & pred).sum(0).astype(float)
    fn = (y & ~pred).sum(0).astype(float)
    tn = (~y & ~pred).sum(0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        sens = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        spec = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    return {"precision": prec, "sensitivity": sens, "specificity": spec, "f1": f1}


def best_thresholds(y, p, grid=None):
    """Threshold per kelas yang memaksimalkan F1. Dipanggil hanya dengan data validasi."""
    grid = np.linspace(0.02, 0.98, 49) if grid is None else grid
    n = y.shape[1]
    thr = np.full(n, 0.5)
    for c in range(n):
        if y[:, c].sum() == 0:
            continue
        f1s = [binary_stats(y[:, [c]], (p[:, [c]] >= t))["f1"][0] for t in grid]
        thr[c] = grid[int(np.nanargmax(f1s))]
    return thr
