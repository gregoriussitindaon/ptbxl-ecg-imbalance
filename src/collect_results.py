"""Kumpulkan semua run di results/runs menjadi tabel ringkas dan satu JSON untuk slide.

Contoh:
    python src/collect_results.py
    python src/collect_results.py --data ptbxl_sub_100hz_min150 --json results/summary.json

Keluaran:
    results/tables/all_runs.csv            satu baris per run
    results/tables/by_config_fold10.csv    rata-rata dan simpangan baku antar seed (fold uji 10)
    results/tables/rotation_<config>.csv   metrik per fold uji untuk config yang punya banyak fold
    results/summary.json                   semua di atas dalam satu berkas
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import evaluate, load_run, subgroup_table

NAME_RE = re.compile(r"^(?P<cfg>.+)_f(?P<fold>\d+)_s(?P<seed>\d+)$")
METRICS = ["macro_auroc", "macro_auprc", "rare_auprc", "macro_f1_0.5", "macro_f1_tuned"]


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="ptbxl_sub_100hz_min150", help="nama folder data di results/runs")
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--json", default="results/summary.json")
    ap.add_argument("--rare-prev", type=float, default=0.05)
    ap.add_argument("--min-folds", type=int, default=5, help="minimal fold agar config dianggap rotasi")
    args = ap.parse_args()

    root = Path(args.runs) / args.data
    out = Path(args.tables)
    out.mkdir(parents=True, exist_ok=True)

    loaded, rows, per_class = {}, [], {}
    for d in sorted(p for p in root.iterdir() if (p / "config.json").exists()):
        m = NAME_RE.match(d.name)
        if not m:
            continue
        r = load_run(d)
        classes = r["classes"]
        counts, n = r["cfg"]["class_counts"], r["cfg"]["n_records"]
        rare = np.array([counts[c] / n < args.rare_prev for c in classes])
        pc, sm = evaluate(r["test_y"], r["test_prob"], r["pred_05"], r["pred_tuned"], classes, rare)
        key = (m["cfg"], int(m["fold"]), int(m["seed"]))
        loaded[key] = (r, rare)
        per_class[key] = pc
        rows.append({"config": m["cfg"], "fold": int(m["fold"]), "seed": int(m["seed"]),
                     "best_epoch": r["cfg"]["best_epoch"], "val_auprc": r["cfg"]["best_val_auprc"],
                     "n_params": r["cfg"]["n_params"], **sm})
    if not rows:
        raise SystemExit(f"Tidak ada run di {root}")
    allr = pd.DataFrame(rows).sort_values(["config", "fold", "seed"])
    allr.to_csv(out / "all_runs.csv", index=False)

    # ---- fold uji 10: rata-rata antar seed -----------------------------------
    f10 = allr[allr.fold == 10]
    agg = f10.groupby("config").agg(
        n_seed=("seed", "nunique"), n_params=("n_params", "first"), val_auprc_mean=("val_auprc", "mean"),
        **{f"{m}_mean": (m, "mean") for m in METRICS}, **{f"{m}_std": (m, "std") for m in METRICS})
    agg = agg.sort_values("macro_auprc_mean", ascending=False).reset_index()
    agg.to_csv(out / "by_config_fold10.csv", index=False)

    classes = next(iter(loaded.values()))[0]["classes"]
    pcs = {}
    for cfg in f10.config.unique():
        frames = [per_class[(c, f, s)] for (c, f, s) in per_class if c == cfg and f == 10]
        pcs[cfg] = {
            cl: {"n_pos": int(frames[0].loc[frames[0]["class"] == cl, "n_pos"].iloc[0]),
                 "auprc": float(np.nanmean([fr.loc[fr["class"] == cl, "auprc"].iloc[0] for fr in frames])),
                 "auroc": float(np.nanmean([fr.loc[fr["class"] == cl, "auroc"].iloc[0] for fr in frames])),
                 "sens": float(np.nanmean([fr.loc[fr["class"] == cl, "sensitivity"].iloc[0] for fr in frames])),
                 "f1": float(np.nanmean([fr.loc[fr["class"] == cl, "f1_tuned"].iloc[0] for fr in frames]))}
            for cl in classes}

    # ---- rotasi fold -----------------------------------------------------------
    rotation = {}
    for cfg in allr.config.unique():
        sub = allr[allr.config == cfg]
        for seed in sorted(sub.seed.unique()):
            ss = sub[sub.seed == seed]
            if ss.fold.nunique() < args.min_folds:
                continue
            keys = [(cfg, f, seed) for f in sorted(ss.fold.unique())]
            rs = [loaded[k][0] for k in keys]
            rare = loaded[keys[0]][1]
            y = np.concatenate([r["test_y"] for r in rs])
            prob = np.concatenate([r["test_prob"] for r in rs])
            p05 = np.concatenate([r["pred_05"] for r in rs])
            ptu = np.concatenate([r["pred_tuned"] for r in rs])
            ids = np.concatenate([r["test_idx"] for r in rs])
            pc, sm = evaluate(y, prob, p05, ptu, classes, rare)
            sg = subgroup_table(y, prob, ids, rs[0]["cfg"].get("data_abs", rs[0]["cfg"]["data"]))
            per_fold = ss.sort_values("fold")
            rotation[f"{cfg}_s{seed}"] = {
                "config": cfg, "seed": int(seed), "folds": per_fold.fold.tolist(),
                "per_fold": {m: per_fold[m].tolist() for m in ("macro_auprc", "macro_auroc", "rare_auprc")},
                "mean": {m: float(per_fold[m].mean()) for m in METRICS},
                "std": {m: float(per_fold[m].std(ddof=1)) for m in METRICS},
                "pooled": sm, "n_pooled": int(len(y)),
                "pooled_per_class": pc.set_index("class")[["n_pos", "auroc", "auprc", "sensitivity", "specificity", "f1_tuned"]].to_dict("index"),
                "subgroups": sg.to_dict("records"),
            }
            pd.DataFrame({"fold": per_fold.fold, **{m: per_fold[m] for m in METRICS}}).to_csv(
                out / f"rotation_{cfg}_s{seed}.csv", index=False)

    info = json.loads((Path("data/processed") / args.data / "info.json").read_text())
    summary = {"data": args.data, "classes": classes, "dataset": info,
               "by_config_fold10": agg.to_dict("records"), "per_class_fold10": pcs,
               "rotation": rotation, "runs": allr.to_dict("records")}
    Path(args.json).write_text(json.dumps(clean(summary), indent=1))
    print(f"{len(allr)} run, {len(agg)} config di fold 10, {len(rotation)} rotasi. JSON: {args.json}")
    pd.set_option("display.width", 220)
    print(agg[["config", "n_seed", "macro_auroc_mean", "macro_auprc_mean", "macro_auprc_std", "rare_auprc_mean"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
