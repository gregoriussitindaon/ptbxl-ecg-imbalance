"""Pemuatan data hasil preprocess.py, split per fold resmi, normalisasi, bobot kelas.

Contoh:
    from data_utils import load_processed, make_split, fit_norm, apply_norm

    d = load_processed("data/processed/ptbxl_super_100hz")
    split = make_split(d)                      # train fold 1-8, val 9, uji 10
    mean, std = fit_norm(d["X"], split["train"])
    Xtr = apply_norm(d["X"][split["train"]], mean, std)
"""
import json
from pathlib import Path

import numpy as np


def load_processed(path):
    path = Path(path)
    meta = np.load(path / "meta.npz")
    d = {k: meta[k] for k in meta.files}
    d["X"] = np.load(path / "X.npy", mmap_mode="r")
    d["info"] = json.loads((path / "info.json").read_text())
    return d


def make_split(d, train_folds=range(1, 9), val_fold=9, test_fold=10):
    """Indeks split berdasarkan strat_fold resmi PTB-XL.

    Fold resmi sudah dipisah per pasien. Pengecekan di bawah memastikan tidak ada
    pasien yang muncul di lebih dari satu split.
    """
    f = d["strat_fold"]
    split = {
        "train": np.where(np.isin(f, list(train_folds)))[0],
        "val": np.where(f == val_fold)[0],
        "test": np.where(f == test_fold)[0],
    }
    pid = d["patient_id"]
    sets = {k: set(pid[v].tolist()) for k, v in split.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        if overlap:
            raise ValueError(f"{len(overlap)} pasien ada di {a} dan {b}")
    return split


def default_val_fold(test_fold):
    """Fold validasi untuk fold uji tertentu: 10 -> 9, 1 -> 10, 2 -> 1, dst."""
    return (test_fold - 2) % 10 + 1


def rotating_splits(d):
    """10 rotasi: tiap fold jadi uji sekali, fold sebelumnya jadi validasi.

    Uji 10 memakai validasi 9, sama dengan split standar PTB-XL.
    """
    for test_fold in range(1, 11):
        val_fold = default_val_fold(test_fold)
        train = [k for k in range(1, 11) if k not in (test_fold, val_fold)]
        yield test_fold, make_split(d, train, val_fold, test_fold)


def fit_norm(X, idx):
    """Mean dan std per lead dari fold latih saja. Mengembalikan dua array (12,)."""
    sub = np.asarray(X[idx], dtype=np.float64)
    return sub.mean(axis=(0, 1)).astype(np.float32), (sub.std(axis=(0, 1)) + 1e-8).astype(np.float32)


def apply_norm(X, mean, std):
    return ((np.asarray(X) - mean) / std).astype(np.float32)


def class_counts(y):
    return y.sum(axis=0).astype(int)


def pos_weights(y):
    """Rasio negatif/positif per kelas, untuk BCE berbobot. Hitung dari data latih saja."""
    pos = y.sum(axis=0).astype(np.float64)
    neg = len(y) - pos
    return (neg / np.maximum(pos, 1)).astype(np.float32)
