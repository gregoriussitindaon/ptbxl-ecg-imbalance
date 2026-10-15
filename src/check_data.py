"""Verifikasi folder PTB-XL yang diunduh manual.

Contoh:
    python src/check_data.py
    python src/check_data.py --raw data/raw --sr 100

Unduh dari PhysioNet (halaman "PTB-XL, a large publicly available
electrocardiography dataset"), ekstrak, dan taruh di data/raw/. Folder bersarang
tidak masalah, skrip mencari ptbxl_database.csv di bawah --raw.

File yang dibutuhkan:
    ptbxl_database.csv, scp_statements.csv, dan records100/ (atau records500/)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def find_root(raw: Path) -> Path:
    hits = sorted(raw.rglob("ptbxl_database.csv"))
    if not hits:
        raise FileNotFoundError(f"ptbxl_database.csv tidak ditemukan di bawah {raw}")
    return hits[0].parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--sr", type=int, default=100, choices=[100, 500], help="sampling rate yang dicek")
    args = ap.parse_args()

    try:
        root = find_root(Path(args.raw))
    except FileNotFoundError as e:
        sys.exit(f"{e}\nUnduh PTB-XL dari PhysioNet dan ekstrak ke {args.raw}")
    print(f"Root dataset: {root}")

    ok = True
    for name in ("ptbxl_database.csv", "scp_statements.csv"):
        exists = (root / name).exists()
        print(f"  {name}: {'ada' if exists else 'HILANG'}")
        ok &= exists
    if not ok:
        sys.exit(1)

    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    col = "filename_lr" if args.sr == 100 else "filename_hr"
    print(f"Rekaman di tabel: {len(db)}, pasien unik: {db.patient_id.nunique()}")
    print("Jumlah rekaman per fold:", db.strat_fold.value_counts().sort_index().to_dict())

    has = db[col].apply(lambda f: (root / f"{f}.dat").exists() and (root / f"{f}.hea").exists())
    n = int(has.sum())
    print(f"Rekaman dengan sinyal {args.sr} Hz lengkap (.dat dan .hea): {n} dari {len(db)}")
    if n == 0:
        sys.exit("Tidak ada file sinyal yang ditemukan.")
    sub = db[has]
    print("Rekaman tersedia per fold:", sub.strat_fold.value_counts().sort_index().to_dict())
    print(f"Pasien unik pada rekaman tersedia: {sub.patient_id.nunique()}")
    if n < len(db):
        print("PERINGATAN: dataset hanya sebagian. preprocess.py hanya memakai rekaman yang tersedia.")
    else:
        print("Data lengkap.")


if __name__ == "__main__":
    main()
