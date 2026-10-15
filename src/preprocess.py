"""Preprocessing PTB-XL: baca sinyal 12-lead, bentuk label multi-hot, simpan array.

Dua tugas label:
  super  5 superclass diagnostik (NORM, MI, STTC, CD, HYP)
  sub    subclass diagnostik (jumlahnya lebih banyak dan lebih tidak seimbang)

Aturan label mengikuti benchmark Strodthoff et al.: tiap kode SCP yang
berstatus diagnostik dipetakan ke kelas induknya, rekaman tanpa label
diagnostik dibuang.

Sinyal disimpan apa adanya (opsional band-pass). Normalisasi TIDAK dilakukan
di sini. Statistik z-score dihitung dari fold latih saja pada saat training
(lihat data_utils.py), supaya tidak ada kebocoran dari fold validasi/uji.

Contoh:
    python src/preprocess.py --task super
    python src/preprocess.py --task sub --sr 100 --bandpass
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

SUPER_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TASK_COLUMN = {"super": "diagnostic_class", "sub": "diagnostic_subclass"}


# ---- Fungsi inti (mudah diuji) -------------------------------------------

def find_root(raw: Path) -> Path:
    hits = sorted(raw.rglob("ptbxl_database.csv"))
    if not hits:
        raise SystemExit(f"ptbxl_database.csv tidak ditemukan di bawah {raw}. "
                         f"Unduh PTB-XL dan jalankan: python src/check_data.py")
    return hits[0].parent


def load_tables(root: Path):
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1]  # hanya pernyataan diagnostik
    return db, scp


def code_to_class(scp: pd.DataFrame, task: str) -> dict:
    return scp[TASK_COLUMN[task]].dropna().to_dict()


def aggregate_labels(codes: dict, mapping: dict) -> set:
    return {mapping[c] for c in codes if c in mapping}


def build_labels(db: pd.DataFrame, mapping: dict, task: str):
    """Kembalikan (label_set per rekaman, daftar kelas)."""
    sets = db["scp_codes"].apply(lambda c: aggregate_labels(c, mapping))
    if task == "super":
        classes = SUPER_CLASSES
    else:
        classes = sorted(set().union(*sets.tolist()))
    return sets, classes


def multi_hot(sets, classes) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    y = np.zeros((len(sets), len(classes)), dtype=np.uint8)
    for r, s in enumerate(sets):
        for c in s:
            if c in idx:
                y[r, idx[c]] = 1
    return y


def bandpass(sig, fs, low=0.5, high=40.0, order=4):
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, sig, axis=0)


def read_signal(root: Path, rel: str, bp: bool):
    import wfdb
    sig, fields = wfdb.rdsamp(str(root / rel))
    sig = sig.astype(np.float32)
    bad = int((~np.isfinite(sig)).sum())
    if bad:
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    if bp:
        sig = bandpass(sig, fields["fs"]).astype(np.float32)
    return sig, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw", help="folder induk hasil unduhan manual")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--task", default="super", choices=["super", "sub"])
    ap.add_argument("--sr", type=int, default=100, choices=[100, 500])
    ap.add_argument("--bandpass", action="store_true", help="band-pass 0.5-40 Hz (default mati)")
    ap.add_argument("--min-count", type=int, default=0,
                    help="buang kelas dengan contoh lebih sedikit dari ini (disarankan 150 untuk --task sub)")
    ap.add_argument("--limit", type=int, default=0, help="batasi jumlah rekaman, untuk uji cepat")
    args = ap.parse_args()

    root = find_root(Path(args.raw))
    db, scp = load_tables(root)
    mapping = code_to_class(scp, args.task)
    sets, classes = build_labels(db, mapping, args.task)

    col = "filename_lr" if args.sr == 100 else "filename_hr"
    has_file = db[col].apply(lambda f: (root / f"{f}.dat").exists() and (root / f"{f}.hea").exists())
    n_total = len(db)
    n_missing_file = int((~has_file).sum())
    db, sets = db[has_file], sets[has_file]

    keep = sets.apply(len) > 0
    dropped = int((~keep).sum())
    db, sets = db[keep], sets[keep]

    # Buang kelas yang terlalu jarang (hitungan dari rekaman yang tersedia), lalu
    # buang rekaman yang labelnya habis. Hitungan dibuat sebelum --limit.
    dropped_classes, dropped_rare_only = [], 0
    if args.min_count > 0:
        tot = {c: sum(c in s for s in sets) for c in classes}
        dropped_classes = [c for c in classes if tot[c] < args.min_count]
        if dropped_classes:
            classes = [c for c in classes if c not in dropped_classes]
            sets = sets.apply(lambda s: s - set(dropped_classes))
            keep2 = sets.apply(len) > 0
            dropped_rare_only = int((~keep2).sum())
            db, sets = db[keep2], sets[keep2]
            print(f"Kelas dibuang (< {args.min_count} contoh): {dropped_classes}; "
                  f"rekaman yang hanya punya kelas itu dibuang: {dropped_rare_only}")
    if args.limit:
        db, sets = db.iloc[:args.limit], sets.iloc[:args.limit]
    y = multi_hot(sets.tolist(), classes)
    print(f"Rekaman di tabel: {n_total}, file sinyal tidak ada: {n_missing_file}")
    print(f"Rekaman dipakai: {len(db)} (dibuang tanpa label diagnostik: {dropped})")

    try:
        from tqdm import tqdm
        it = tqdm(db[col].tolist(), desc="baca sinyal")
    except ImportError:
        it = db[col].tolist()
    sigs, bad_total = [], 0
    for rel in it:
        s, bad = read_signal(root, rel, args.bandpass)
        sigs.append(s)
        bad_total += bad
    X = np.stack(sigs).astype(np.float32)  # (N, T, 12)
    print(f"X: {X.shape}, nilai non-finite diganti 0: {bad_total}")

    suffix = f"_min{args.min_count}" if args.min_count else ""
    out = Path(args.out) / f"ptbxl_{args.task}_{args.sr}hz{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", X)
    np.savez_compressed(
        out / "meta.npz",
        y=y,
        ecg_id=db.index.to_numpy().astype(np.int64),
        patient_id=db["patient_id"].to_numpy().astype(np.int64),
        strat_fold=db["strat_fold"].to_numpy().astype(np.int8),
        age=db["age"].to_numpy().astype(np.float32),
        sex=db["sex"].to_numpy().astype(np.float32),
    )
    counts = {c: int(y[:, i].sum()) for i, c in enumerate(classes)}
    info = {
        "task": args.task, "classes": classes, "leads": LEADS, "sr": args.sr,
        "n_records": int(len(db)), "n_in_table": int(n_total),
        "n_missing_signal_file": n_missing_file, "n_dropped_no_label": dropped,
        "min_count": args.min_count, "dropped_rare_classes": dropped_classes,
        "n_dropped_rare_only": dropped_rare_only,
        "bandpass": args.bandpass, "normalization": "tidak ada, z-score dari fold latih saat training",
        "class_counts": counts, "folds": "strat_fold resmi PTB-XL (per pasien)",
    }
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print("Distribusi kelas (multi-label):", counts)
    print(f"Selesai. Keluaran ada di {out}")


if __name__ == "__main__":
    main()
