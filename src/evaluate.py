"""Evaluasi satu atau beberapa run hasil train.py.

Contoh:
    python src/evaluate.py results/runs/ptbxl_sub_100hz_min150/b0_bce_f10_s0
    python src/evaluate.py results/runs/*/b0_bce_f* --pool --name b0_rotasi
    python src/evaluate.py results/runs/*/b*_f10_s0 --name banding_strategi --no-aggregate --plot

Yang dihitung dari prediksi uji:
  - macro AUROC dan macro AUPRC (tidak butuh threshold)
  - AUPRC rata-rata pada kelas jarang (prevalensi < --rare-prev)
  - macro F1 pada threshold 0.5 dan pada threshold per kelas yang dipilih di
    fold VALIDASI (strategi B7), bukan di fold uji
  - per kelas: AUROC, AUPRC, presisi, sensitivitas, spesifisitas, F1
Dengan --pool, prediksi uji dari beberapa run (misalnya 10 rotasi fold) digabung
sehingga tiap rekaman dihitung sekali dan metrik per kelas memakai semua contoh.
Dengan --subgroups, macro AUROC juga dihitung per jenis kelamin dan kelompok usia.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import best_thresholds, binary_stats, macro, per_class_scores


def load_run(path):
    p = Path(path)
    cfg = json.loads((p / "config.json").read_text())
    r = {"name": p.name, "cfg": cfg, "classes": cfg["classes"]}
    for k in ("val_prob", "val_y", "test_prob", "test_y", "test_idx"):
        r[k] = np.load(p / f"{k}.npy")
    r["thr"] = best_thresholds(r["val_y"], r["val_prob"])
    r["pred_tuned"] = r["test_prob"] >= r["thr"]
    r["pred_05"] = r["test_prob"] >= 0.5
    return r


def evaluate(y, prob, pred05, predtuned, classes, rare_mask):
    auroc, auprc = per_class_scores(y, prob)
    s05, st = binary_stats(y, pred05), binary_stats(y, predtuned)
    per_class = pd.DataFrame({
        "class": classes, "n_pos": y.sum(0).astype(int),
        "auroc": auroc, "auprc": auprc,
        "precision": st["precision"], "sensitivity": st["sensitivity"],
        "specificity": st["specificity"], "f1_tuned": st["f1"], "f1_at_0.5": s05["f1"],
    })
    summary = {
        "macro_auroc": macro(auroc), "macro_auprc": macro(auprc),
        "rare_auprc": macro(np.where(rare_mask, auprc, np.nan)),
        "macro_f1_0.5": macro(s05["f1"]), "macro_f1_tuned": macro(st["f1"]),
        "n_test": int(len(y)),
    }
    return per_class, summary


def subgroup_table(y, prob, ids, data_dir):
    meta = np.load(Path(data_dir) / "meta.npz")
    age, sex = meta["age"][ids], meta["sex"][ids]
    groups = {"sex=0": sex == 0, "sex=1": sex == 1,
              "usia<40": age < 40, "usia 40-59": (age >= 40) & (age < 60),
              "usia 60-79": (age >= 60) & (age < 80), "usia>=80": age >= 80}
    rows = []
    for g, m in groups.items():
        if m.sum() < 30:
            continue
        au, ap_ = per_class_scores(y[m], prob[m])
        rows.append({"grup": g, "n": int(m.sum()), "macro_auroc": macro(au), "macro_auprc": macro(ap_)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="folder run dari train.py")
    ap.add_argument("--pool", action="store_true", help="gabungkan prediksi uji semua run")
    ap.add_argument("--name", default=None, help="nama berkas keluaran")
    ap.add_argument("--rare-prev", type=float, default=0.05, help="prevalensi di bawah ini dianggap kelas jarang")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="jangan tambah baris MEAN/STD (untuk membandingkan konfigurasi berbeda)")
    ap.add_argument("--subgroups", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()

    runs = [load_run(r) for r in args.runs]
    classes = runs[0]["classes"]
    if any(r["classes"] != classes for r in runs):
        raise SystemExit("Semua run harus memakai kelas yang sama.")
    counts, n = runs[0]["cfg"]["class_counts"], runs[0]["cfg"]["n_records"]
    rare = np.array([counts[c] / n < args.rare_prev for c in classes])
    print(f"Kelas jarang (prevalensi < {args.rare_prev:.0%}): {[c for c, r in zip(classes, rare) if r]}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or (runs[0]["name"] if len(runs) == 1 else "ringkasan")

    rows = []
    for r in runs:
        pc, sm = evaluate(r["test_y"], r["test_prob"], r["pred_05"], r["pred_tuned"], classes, rare)
        pc.to_csv(out / f"{r['name']}_per_class.csv", index=False)
        rows.append({"run": r["name"], **sm})
    summary = pd.DataFrame(rows)
    num = summary.drop(columns=["run", "n_test"])
    if len(runs) > 1 and not args.no_aggregate:
        extra = pd.DataFrame([{"run": "MEAN", **num.mean().to_dict()}, {"run": "STD", **num.std(ddof=1).to_dict()}])
        summary = pd.concat([summary, extra], ignore_index=True)

    pooled = None
    if args.pool:
        y = np.concatenate([r["test_y"] for r in runs])
        prob = np.concatenate([r["test_prob"] for r in runs])
        p05 = np.concatenate([r["pred_05"] for r in runs])
        ptu = np.concatenate([r["pred_tuned"] for r in runs])
        ids = np.concatenate([r["test_idx"] for r in runs])
        if len(set(ids.tolist())) != len(ids):
            print("PERINGATAN: ada rekaman uji yang muncul lebih dari sekali pada gabungan run.")
        pc, sm = evaluate(y, prob, p05, ptu, classes, rare)
        pc.to_csv(out / f"{name}_pooled_per_class.csv", index=False)
        summary = pd.concat([summary, pd.DataFrame([{"run": "POOLED", **sm}])], ignore_index=True)
        pooled = (pc, y, prob, ids)

    pd.set_option("display.width", 200)
    print("\n" + summary.round(4).to_string(index=False))
    summary.to_csv(out / f"{name}_summary.csv", index=False)

    if len(runs) == 1 or pooled is not None:
        pc = pooled[0] if pooled else pd.read_csv(out / f"{runs[0]['name']}_per_class.csv")
        print("\nPer kelas:\n" + pc.round(3).to_string(index=False))

    if args.subgroups:
        y, prob, ids = (pooled[1], pooled[2], pooled[3]) if pooled else (
            runs[0]["test_y"], runs[0]["test_prob"], runs[0]["test_idx"])
        if len(runs) > 1 and not pooled:
            print("\n--subgroups pada banyak run butuh --pool. Dilewati.")
        else:
            sg = subgroup_table(y, prob, ids, runs[0]["cfg"].get("data_abs", runs[0]["cfg"]["data"]))
            sg.to_csv(out / f"{name}_subgroups.csv", index=False)
            print("\nSubkelompok (sex: 0/1 sesuai kode PTB-XL):\n" + sg.round(4).to_string(index=False))

    if args.plot and (len(runs) == 1 or pooled is not None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pc = pooled[0] if pooled else pd.read_csv(out / f"{runs[0]['name']}_per_class.csv")
        pc = pc.sort_values("n_pos", ascending=False)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(pc["class"], pc["auprc"], color="#1d5fd1")
        ax.set_ylabel("AUPRC")
        ax.set_title(f"AUPRC per kelas (urut jumlah contoh): {name}")
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
        fig.tight_layout()
        figs = Path("results/figures")
        figs.mkdir(parents=True, exist_ok=True)
        fig.savefig(figs / f"{name}_auprc.png", dpi=150)
        print(f"Gambar: {figs / (name + '_auprc.png')}")


if __name__ == "__main__":
    main()
