"""Training satu konfigurasi pada satu fold uji.

Contoh:
    python src/train.py --config configs/b0_bce.yaml
    python src/train.py --config configs/b1_pos_weight.yaml --test-fold 3 --seed 1
    python src/train.py --model mlp --loss bce --epochs 20

Nilai di file YAML menjadi default, argumen CLI menimpanya.
Keluaran per run di results/runs/<nama-data>/<nama-run>/:
    config.json, history.json, best.pt (di-ignore git),
    val_prob.npy, val_y.npy, test_prob.npy, test_y.npy, test_idx.npy
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import data_utils as u
from losses import make_loss
from metrics import macro, per_class_scores
from models import make_model


def get_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="file YAML berisi default argumen")
    ap.add_argument("--data", default="data/processed/ptbxl_sub_100hz_min150")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--model", default="resnet1d", choices=["mlp", "cnn1d", "resnet1d"])
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=1, help="blok per stage (resnet1d)")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--loss", default="bce", choices=["bce", "pos_weight", "focal", "asl", "cb"])
    ap.add_argument("--pw-power", type=float, default=1.0, help="pangkat untuk pos_weight")
    ap.add_argument("--sampler", default="none", choices=["none", "oversample"])
    ap.add_argument("--os-power", type=float, default=0.5, help="pangkat bobot oversampling")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--test-fold", type=int, default=10)
    ap.add_argument("--val-fold", type=int, default=None, help="default: lihat data_utils.default_val_fold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-train", type=int, default=0, help="batasi data latih, untuk uji cepat")

    pre, _ = ap.parse_known_args()
    if pre.config:
        import yaml
        cfg = yaml.safe_load(Path(pre.config).read_text()) or {}
        ap.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items()})
    return ap.parse_args()


def augment(x, noise=0.03, scale=0.1, shift=50):
    """Augmentasi ringan pada batch latih (B, 12, T) yang sudah dinormalisasi."""
    b = x.shape[0]
    x = x * (1 + (torch.rand(b, 1, 1, device=x.device) * 2 - 1) * scale)
    x = x + torch.randn_like(x) * noise
    shifts = torch.randint(-shift, shift + 1, (b,)).tolist()
    return torch.stack([torch.roll(x[i], s, dims=-1) for i, s in enumerate(shifts)])


@torch.no_grad()
def predict(model, X, device, bs=256):
    model.eval()
    out = [torch.sigmoid(model(X[i:i + bs].to(device))).cpu() for i in range(0, len(X), bs)]
    return torch.cat(out).numpy()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))

    d = u.load_processed(args.data)
    classes = d["info"]["classes"]
    y_all = d["y"]
    test = args.test_fold
    val = args.val_fold or u.default_val_fold(test)
    train_folds = [k for k in range(1, 11) if k not in (test, val)]
    split = u.make_split(d, train_folds, val, test)
    tr_idx = split["train"]
    if args.max_train:
        tr_idx = np.random.default_rng(args.seed).choice(tr_idx, min(args.max_train, len(tr_idx)), replace=False)

    X = np.asarray(d["X"])
    mean, std = u.fit_norm(X, tr_idx)

    def prep(idx):
        x = u.apply_norm(X[idx], mean, std).transpose(0, 2, 1).copy()
        return torch.from_numpy(x), torch.from_numpy(y_all[idx].astype(np.float32))

    Xtr, ytr = prep(tr_idx)
    Xva, yva = prep(split["val"])
    Xte, yte = prep(split["test"])
    del X

    name = args.run_name or "_".join(filter(None, [
        args.model, args.loss, "os" if args.sampler == "oversample" else "",
        "aug" if args.augment else "", f"f{test}", f"s{args.seed}"]))
    run_dir = Path(args.out) / Path(args.data).name / name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = make_model(args.model, len(classes), args.width, args.blocks, args.dropout, Xtr.shape[-1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = make_loss(args.loss, y_all[tr_idx], args.pw_power).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    ds = TensorDataset(Xtr, ytr)
    if args.sampler == "oversample":
        n_c = np.maximum(y_all[tr_idx].sum(0), 1)
        w = np.array([((1.0 / n_c[row.astype(bool)]) ** args.os_power).max() for row in y_all[tr_idx]])
        loader = DataLoader(ds, batch_size=args.batch_size, drop_last=True,
                            sampler=WeightedRandomSampler(w, len(ds), replacement=True))
    else:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print(f"run: {name} | {device} | params {n_params:,} | latih {len(tr_idx)} val {len(split['val'])} "
          f"uji {len(split['test'])} | kelas {len(classes)}")

    best, best_state, best_ep, history, bad = -1.0, None, 0, [], 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if args.augment:
                xb = augment(xb)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        pv = predict(model, Xva, device)
        au, ap_ = per_class_scores(yva.numpy(), pv)
        rec = {"epoch": ep, "train_loss": tot / len(loader), "val_auroc": macro(au),
               "val_auprc": macro(ap_), "lr": sched.get_last_lr()[0], "sec": time.time() - t0}
        history.append(rec)
        print(f"ep {ep:02d} loss {rec['train_loss']:.4f} val AUROC {rec['val_auroc']:.4f} "
              f"AUPRC {rec['val_auprc']:.4f} ({rec['sec']:.0f}s)")
        if rec["val_auprc"] > best:
            best, best_ep, bad = rec["val_auprc"], ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop di epoch {ep}")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "best.pt")
    np.save(run_dir / "val_prob.npy", predict(model, Xva, device))
    np.save(run_dir / "val_y.npy", yva.numpy().astype(np.uint8))
    np.save(run_dir / "test_prob.npy", predict(model, Xte, device))
    np.save(run_dir / "test_y.npy", yte.numpy().astype(np.uint8))
    np.save(run_dir / "test_idx.npy", split["test"])
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    cfg = {**vars(args), "data_abs": str(Path(args.data).resolve()), "classes": classes, "best_epoch": best_ep, "best_val_auprc": best,
           "n_params": n_params, "val_fold_used": val, "train_folds": train_folds,
           "class_counts": d["info"]["class_counts"], "n_records": d["info"]["n_records"],
           "train_counts": dict(zip(classes, y_all[tr_idx].sum(0).astype(int).tolist()))}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"selesai. best epoch {best_ep}, val AUPRC {best:.4f}. Keluaran: {run_dir}")


if __name__ == "__main__":
    main()
