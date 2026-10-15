"""Jalankan banyak konfigurasi berurutan, lalu evaluasi dan bandingkan.

Contoh:
    python src/run_all.py --dry-run                         # lihat daftar run, tanpa menjalankan
    python src/run_all.py                                   # semua config, fold uji 10, seed 0
    python src/run_all.py --configs b0_bce b1_pos_weight    # sebagian config
    python src/run_all.py --configs b0_bce b3_asl --folds 1-10     # rotasi 10 fold
    python src/run_all.py --seeds 0 1 2 --configs b0_bce b3_asl
    python src/run_all.py --train-args "--epochs 5 --max-train 500"   # uji cepat

Perilaku:
  - Run yang sudah selesai (ada config.json) dilewati, jadi aman dihentikan dengan
    Ctrl-C lalu dijalankan ulang. Pakai --force untuk mengulang.
  - Nama run: <config>_f<fold>_s<seed>. Log tiap run di results/logs/.
  - Satu fold: semua config dibandingkan di satu tabel.
    Banyak fold: tiap config digabung lintas fold (--pool), lalu dibuat tabel perbandingan.
"""
import argparse
import os
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

DEFAULT_CONFIGS = [
    "b0_bce", "b1_pos_weight", "b2_focal", "b3_asl", "b4_class_balanced",
    "b5_oversample", "b6_augment", "baseline_mlp", "baseline_cnn1d",
]
HERE = Path(__file__).resolve().parent


def parse_folds(spec):
    folds = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            folds += list(range(int(a), int(b) + 1))
        else:
            folds.append(int(part))
    if not folds or any(not 1 <= f <= 10 for f in folds):
        raise SystemExit("--folds harus berisi angka 1-10, contoh: 10 atau 1-10 atau 1,5,10")
    return folds


def run_name(cfg, fold, seed):
    return f"{cfg}_f{fold}_s{seed}"


def call(cmd, log=None):
    if log is None:
        return subprocess.run(cmd).returncode
    with open(log, "w", encoding="utf-8") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, help="nama file di configs/ tanpa .yaml")
    ap.add_argument("--folds", default="10", help="fold uji: 10, 1-10, atau 1,5,10")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--data", default="data/processed/ptbxl_sub_100hz_min150")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--logs", default="results/logs")
    ap.add_argument("--train-args", default="", help='argumen tambahan untuk train.py, contoh "--epochs 5"')
    ap.add_argument("--force", action="store_true", help="ulangi run yang sudah selesai")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--no-eval", action="store_true", help="hanya training")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    folds = parse_folds(args.folds)
    for c in args.configs:
        if not Path(f"configs/{c}.yaml").exists():
            raise SystemExit(f"configs/{c}.yaml tidak ada. Jalankan dari folder code/.")
    data_name = Path(args.data).name
    runs_dir = Path(args.out) / data_name
    logs_dir = Path(args.logs) / data_name
    extra = shlex.split(args.train_args)

    plan = [(s, f, c) for s in args.seeds for c in args.configs for f in folds]
    todo = [(s, f, c) for s, f, c in plan
            if args.force or not (runs_dir / run_name(c, f, s) / "config.json").exists()]
    print(f"Total run: {len(plan)}, perlu dijalankan: {len(todo)}, sudah selesai: {len(plan) - len(todo)}")
    if args.dry_run:
        for s, f, c in plan:
            done = (runs_dir / run_name(c, f, s) / "config.json").exists()
            print(f"  {run_name(c, f, s):32} {'selesai' if done and not args.force else 'akan dijalankan'}")
        return

    logs_dir.mkdir(parents=True, exist_ok=True)
    failed, t_start = [], time.time()
    for i, (s, f, c) in enumerate(todo, 1):
        name = run_name(c, f, s)
        log = logs_dir / f"{name}.log"
        cmd = [sys.executable, str(HERE / "train.py"), "--config", f"configs/{c}.yaml",
               "--data", args.data, "--out", args.out, "--test-fold", str(f),
               "--seed", str(s), "--run-name", name] + extra
        t0 = time.time()
        print(f"[{i}/{len(todo)}] {name} ...", end=" ", flush=True)
        try:
            rc = call(cmd, log)
        except KeyboardInterrupt:
            print("\nDihentikan. Jalankan ulang perintah yang sama untuk melanjutkan.")
            sys.exit(130)
        mins = (time.time() - t0) / 60
        if rc == 0:
            cfgp = runs_dir / name / "config.json"
            best = json.loads(cfgp.read_text())["best_val_auprc"] if cfgp.exists() else float("nan")
            print(f"selesai {mins:.1f} mnt, val AUPRC {best:.4f}")
        else:
            failed.append(name)
            print(f"GAGAL (kode {rc}), lihat {log}")
            if args.stop_on_error:
                break
    print(f"\nTraining selesai dalam {(time.time() - t_start) / 60:.1f} menit. Gagal: {failed or 'tidak ada'}")

    if args.no_eval:
        return
    py = [sys.executable, str(HERE / "evaluate.py")]
    seed_tables = []
    for s in args.seeds:
        have = {c: [str(runs_dir / run_name(c, f, s)) for f in folds
                    if (runs_dir / run_name(c, f, s) / "config.json").exists()] for c in args.configs}
        have = {c: r for c, r in have.items() if r}
        if not have:
            continue
        if len(folds) == 1:
            allruns = [r for rs in have.values() for r in rs]
            name = f"perbandingan_f{folds[0]}_s{s}"
            call(py + allruns + ["--name", name, "--out", args.tables, "--no-aggregate"],
                 log=logs_dir / f"eval_{name}.log")
            print(f"Tabel: {Path(args.tables) / (name + '_summary.csv')}")
            t = pd.read_csv(Path(args.tables) / f"{name}_summary.csv")
            t["config"] = t["run"].str.replace(rf"_f{folds[0]}_s{s}$", "", regex=True)
            seed_tables.append(t.assign(seed=s))
            continue
        rows = []
        for c, rs in have.items():
            name = f"{c}_rotasi_s{s}"
            call(py + rs + ["--pool", "--name", name, "--out", args.tables], log=logs_dir / f"eval_{name}.log")
            sm = pd.read_csv(Path(args.tables) / f"{name}_summary.csv").set_index("run")
            if "POOLED" not in sm.index:
                continue
            pooled = sm.loc["POOLED"]
            row = {"config": c, "n_fold": len(rs), "macro_auroc": pooled["macro_auroc"],
                   "macro_auprc": pooled["macro_auprc"], "rare_auprc": pooled["rare_auprc"],
                   "macro_f1_tuned": pooled["macro_f1_tuned"]}
            if "MEAN" in sm.index:
                row["auprc_mean_antar_fold"] = sm.loc["MEAN", "macro_auprc"]
                row["auprc_std_antar_fold"] = sm.loc["STD", "macro_auprc"]
            rows.append(row)
        if rows:
            comp = pd.DataFrame(rows).sort_values("macro_auprc", ascending=False)
            path = Path(args.tables) / f"perbandingan_rotasi_s{s}.csv"
            comp.to_csv(path, index=False)
            pd.set_option("display.width", 200)
            print(f"\nPerbandingan strategi (gabungan {len(folds)} fold, seed {s}):\n" + comp.round(4).to_string(index=False))
            print(f"Tabel: {path}")

    if seed_tables and len(seed_tables) > 1:
        allt = pd.concat(seed_tables, ignore_index=True)
        cols = ["macro_auroc", "macro_auprc", "rare_auprc", "macro_f1_tuned"]
        g = allt.groupby("config")[cols].agg(["mean", "std"])
        g.columns = [f"{a}_{b}" for a, b in g.columns]
        g.insert(0, "n_seed", allt.groupby("config")["seed"].nunique())
        g = g.sort_values("macro_auprc_mean", ascending=False).reset_index()
        path = Path(args.tables) / f"perbandingan_f{folds[0]}_antar_seed.csv"
        g.to_csv(path, index=False)
        pd.set_option("display.width", 250)
        print(f"\nRata-rata dan simpangan baku antar seed (fold uji {folds[0]}):\n" + g.round(4).to_string(index=False))
        print(f"Tabel: {path}")


if __name__ == "__main__":
    main()
