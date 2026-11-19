#!/usr/bin/env python3
"""Reviewer run (paper 266): pooling heads on an alternative backbone.

One run = one (dataset, backbone, head, seed). Heads: gap, gmp, se, gcp.
Backbones: tcn and resnet1d are imported unchanged from src/models.py (the
Gap 5 backbones used by scripts/run_gap5.py); mlp is a small all-linear
fallback defined here. GMP (global max pooling) is defined here because the
repo has no GMP head; GAP, SE, and GCP heads are imported unchanged.

Data: the materialized splits written by src/data.py (materialize_dataset)
are loaded with src.data.load_split. The split is NOT re-created here. The
loaded train/val/test arrays are verified against the sha256 checksums in
meta.json before training.

Protocol mirrors src/engine.train_model + evaluate_model: Adam lr 1e-3,
batch 512, 20 epochs, grad clip 1.0, plain cross-entropy, best epoch chosen
on VAL macro-F1 only, TEST read exactly once after training.

Outputs:
  <out-dir>/<head>_<dataset>_<seed>.json         (smoke: smoke_ prefix)
  <out-dir>/logs/<run_id>_<YYYYMMDD_hhmmssAM|PM>.log
Skip rule: if the result JSON exists and parses, log "skipped (exists)" and
exit 0. A result JSON is written last, via temp file plus no-overwrite link.
Results for different backbones must go to different --out-dir values (the
default out-dir already includes the backbone); a backbone, label-mode,
c_last, epochs, batch-size, or lr mismatch with an existing result is an
error, never a silent skip.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import platform
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

HEADS = ("gap", "gmp", "se", "gcp")
BACKBONES = ("tcn", "resnet1d", "mlp")
SEEDS = (42, 123, 456, 789, 1337)
DATASETS = ("NF-UNSW-NB15-v2", "NF-CSE-CIC-IDS2018-v2",
            "NF-ToN-IoT-v2", "NF-BoT-IoT-v2")   # same list as src.data.DATASETS
SMOKE_N = {"train": 2000, "val": 500, "test": 500}
TS_FMT = "%Y-%m-%d %I:%M:%S %p"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train one pooling head on one alternative backbone, "
                    "one NetFlow dataset, one seed (paper 266 reviewer run).")
    # head/backbone/seed/dataset are validated AFTER logging starts, so a bad
    # value produces a logged ERROR and a non-zero exit.
    p.add_argument("--seed", type=int, required=True,
                   help=f"one of {list(SEEDS)}")
    p.add_argument("--dataset", required=True, help=f"one of {list(DATASETS)}")
    p.add_argument("--head", required=True, help=f"one of {list(HEADS)}")
    p.add_argument("--backbone", default="tcn", help=f"one of {list(BACKBONES)}")
    p.add_argument("--out-dir", default=None,
                   help="default: reviewer_runs_2026-11-18/results/<backbone>")
    p.add_argument("--data-dir", default=None,
                   help="materialized split root (default: src.data.DEFAULT_OUT)")
    p.add_argument("--smoke", action="store_true",
                   help=f"1 epoch on a seeded subset {SMOKE_N}; smoke_ prefix")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--c-last", type=int, default=32)
    p.add_argument("--label-mode", default="binary",
                   choices=("binary", "multiclass"))
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                   help="default cpu; cpu also hides all GPUs from torch")
    p.add_argument("--skip-checksum", action="store_true",
                   help="do not verify split checksums against meta.json")
    return p


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------
def make_logger(log_dir: str, run_id: str) -> tuple[logging.Logger, str]:
    os.makedirs(log_dir, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%I%M%S%p")
    path = os.path.join(log_dir, f"{run_id}_{stamp}.log")
    k = 2
    while os.path.exists(path):                     # never overwrite a log
        path = os.path.join(log_dir, f"{run_id}_{stamp}_{k}.log"); k += 1
    log = logging.getLogger(f"alt_arch.{run_id}.{stamp}.{k}")
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", TS_FMT)
    fh = logging.FileHandler(path, mode="x", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt); log.addHandler(h)
    return log, path


class Step:
    """Context manager: logs START <step> / END <step> elapsed=<s>s."""
    def __init__(self, log, name):
        self.log, self.name = log, name

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.log.info(f"START {self.name}")
        return self

    def __exit__(self, et, ev, tb):
        dt = time.perf_counter() - self.t0
        status = "" if et is None else " status=FAILED"
        self.log.info(f"END {self.name} elapsed={dt:.3f}s{status}")
        return False


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _git_head() -> str:
    try:
        with open(os.path.join(REPO, ".git", "HEAD")) as fh:
            ref = fh.read().strip()
        if ref.startswith("ref: "):
            with open(os.path.join(REPO, ".git", ref[5:])) as fh:
                return fh.read().strip()
        return ref
    except OSError:
        return "unknown"


def log_header(log, args, argv):
    import numpy, sklearn, torch
    try:
        import pyarrow; pa_v = pyarrow.__version__
    except ImportError:
        pa_v = "not installed"
    log.info(f"argv: {' '.join(argv)}")
    log.info(f"config: {json.dumps(vars(args), sort_keys=True)}")
    log.info(f"python: {platform.python_version()} ({sys.executable})")
    log.info(f"os: {platform.platform()}")
    log.info(f"cpu: {_cpu_name()} logical_cores={os.cpu_count()} "
             f"torch_threads={torch.get_num_threads()}")
    log.info(f"versions: torch={torch.__version__} numpy={numpy.__version__} "
             f"scikit-learn={sklearn.__version__} pyarrow={pa_v}")
    log.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
             f"cuda_available={torch.cuda.is_available()}")
    log.info(f"repo: {REPO} git_head={_git_head()}")


# --------------------------------------------------------------------------
# model pieces not present in the repo
# --------------------------------------------------------------------------
def _torch_modules():
    import torch
    import torch.nn as nn

    class GMPHead(nn.Module):
        """Global max pooling over positions: (B, C, L') -> (B, C)."""
        def __init__(self, channels: int):
            super().__init__()
            self.out_dim = channels

        def forward(self, x):
            return x.amax(dim=2)

    class _MixerBlock(nn.Module):
        def __init__(self, n_pos: int, channels: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(channels)
            self.tok = nn.Sequential(nn.Linear(n_pos, n_pos), nn.GELU(),
                                     nn.Linear(n_pos, n_pos))
            self.ln2 = nn.LayerNorm(channels)
            self.chn = nn.Sequential(nn.Linear(channels, channels), nn.GELU(),
                                     nn.Linear(channels, channels))

        def forward(self, z):                          # z: (B, L, C)
            z = z + self.tok(self.ln1(z).transpose(1, 2)).transpose(1, 2)
            return z + self.chn(self.ln2(z))

    class MLPBackbone(nn.Module):
        """All-linear fallback backbone (MLP-Mixer style, no convolution).

        Each of the F standardized features becomes one position with a
        learned per-feature embedding of width C; blocks then mix across
        features (token MLP) and across channels (channel MLP). Output keeps
        the (B, C, L') contract of the conv backbones with L' = F, so the GCP
        rank condition L' >= C holds whenever F >= C (39 >= 32 on NF-v2).
        """
        def __init__(self, in_features: int, channels=(16, 32, 32)):
            super().__init__()
            c = channels[-1]
            self.out_channels = c
            self.embed_w = nn.Parameter(torch.randn(in_features, c) * 0.1)
            self.embed_b = nn.Parameter(torch.zeros(in_features, c))
            self.blocks = nn.Sequential(*[_MixerBlock(in_features, c)
                                          for _ in range(len(channels) - 1)])
            self.ln = nn.LayerNorm(c)

        def forward(self, x):                          # (B, 1, F) or (B, F)
            if x.dim() == 3:
                x = x.squeeze(1)
            z = x.unsqueeze(2) * self.embed_w + self.embed_b   # (B, F, C)
            z = self.ln(self.blocks(z))
            return z.transpose(1, 2)                           # (B, C, F)

    return GMPHead, MLPBackbone


def build_model(backbone: str, head: str, n_classes: int, in_features: int,
                c_last: int = 32):
    """Backbone (repo or MLP) -> head (repo or GMP) -> linear classifier."""
    import torch.nn as nn
    from src.models import build_backbone, build_head
    GMPHead, MLPBackbone = _torch_modules()
    channels = (16, 32, c_last)

    if backbone == "mlp":
        bb = MLPBackbone(in_features, channels)
    else:
        bb = build_backbone(backbone, channels, 3)      # repo code, unchanged
    hd = GMPHead(bb.out_channels) if head == "gmp" else build_head(head, bb.out_channels)

    class AltModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone, self.head = bb, hd
            self.classifier = nn.Linear(hd.out_dim, n_classes)
            self.lprime = in_features          # tcn/resnet1d/mlp keep L' = F

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            return self.classifier(self.head(self.backbone(x)))

    return AltModel()


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_data(log, dataset, data_dir, label_mode, smoke, seed, verify):
    import numpy as np
    from src import data as D
    meta_path = os.path.join(data_dir, dataset, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"materialized split not found: {meta_path}. Run "
            f"src.data.materialize_dataset on the machine that holds the "
            f"NF-UQ-NIDS-v2 parquet (this script never re-creates splits).")
    with open(meta_path) as fh:
        meta = json.load(fh)
    out = {}
    for split in ("train", "val", "test"):
        X, y = D.load_split(dataset, split, data_dir, label_mode)
        if verify:
            got = hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()[:16]
            want = meta.get("checksums", {}).get(f"{split}_X")
            if want is None or got != want:
                raise ValueError(f"{split}_X checksum mismatch: got {got}, "
                                 f"meta.json has {want}")
            log.info(f"checksum ok {split}_X={got}")
        n_full = len(X)
        if smoke:
            rng = np.random.default_rng(seed)
            k = min(SMOKE_N[split], n_full)
            idx = np.sort(rng.choice(n_full, size=k, replace=False))
            X, y = X[idx], y[idx]
        log.info(f"split {split}: n={len(X)} (full={n_full}) F={X.shape[1]} "
                 f"class_counts={np.bincount(y).tolist()}")
        out[split] = (X, y)
    n_classes = 2 if label_mode == "binary" else len(meta["classes"])
    return out, meta, n_classes


# --------------------------------------------------------------------------
# result I/O
# --------------------------------------------------------------------------
def result_path(out_dir, head, dataset, seed, smoke):
    prefix = "smoke_" if smoke else ""
    return os.path.join(out_dir, f"{prefix}{head}_{dataset}_{seed}.json")


def write_json_no_overwrite(path: str, obj: dict) -> None:
    """Temp file + os.link: atomic, and fails instead of overwriting."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "x", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, allow_nan=True)   # repr floats = full precision
        fh.flush(); os.fsync(fh.fileno())
    try:
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


# --------------------------------------------------------------------------
# train / evaluate
# --------------------------------------------------------------------------
def run(args, log):
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                                 roc_auc_score)
    from src.engine import set_seed
    _loader = _fast_loader
    from src.metrics import macro_f1, per_family_f1

    t_run = time.perf_counter()
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    log.info(f"device: {device}"
             + (f" ({torch.cuda.get_device_name(0)})" if args.device == "cuda" else ""))
    epochs = 1 if args.smoke else args.epochs

    with Step(log, "seed"):
        set_seed(args.seed)
        # Speed setting (2026-11-18): cuDNN deterministic mode made each conv ~100x slower
        # on this workload. Turned off after seeding; all seeds, splits and batch order are unchanged.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        log.info("cudnn: deterministic=False benchmark=True (set after set_seed)")

    with Step(log, "load_data"):
        splits, meta, n_classes = load_data(
            log, args.dataset, args.data_dir, args.label_mode, args.smoke,
            args.seed, verify=not args.skip_checksum)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = (splits["train"], splits["val"],
                                              splits["test"])
        in_features = Xtr.shape[1]

    with Step(log, "build_model"):
        model = build_model(args.backbone, args.head, n_classes, in_features,
                            args.c_last).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"model: backbone={args.backbone} head={args.head} "
                 f"pooled_dim={model.head.out_dim} n_classes={n_classes} "
                 f"n_params={n_params} lprime={model.lprime} C={args.c_last} "
                 f"gcp_rank_ok={model.lprime >= args.c_last}")
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        crit = nn.CrossEntropyLoss()
        tr_loader = _loader(Xtr, ytr, args.batch_size, shuffle=True)
        va_loader = _loader(Xva, yva, args.batch_size, shuffle=False)

    best_val, best_epoch, best_state, history = -1.0, -1, None, []
    with Step(log, "train"):
        for epoch in range(epochs):
            t_ep = time.perf_counter()
            model.train()
            loss_sum, n_seen = 0.0, 0
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch}")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                loss_sum += float(loss.item()) * len(yb); n_seen += len(yb)
            model.eval()
            preds, gts = [], []
            with torch.no_grad():                    # VAL only for selection
                for xb, yb in va_loader:
                    preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
                    gts.append(yb.numpy())
            vf1 = macro_f1(np.concatenate(gts), np.concatenate(preds))
            tr_loss = loss_sum / max(n_seen, 1)
            history.append({"epoch": epoch, "train_loss": tr_loss,
                            "val_macro_f1": vf1})
            if vf1 > best_val:
                best_val, best_epoch = vf1, epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            log.info(f"epoch {epoch + 1}/{epochs} train_loss={tr_loss!r} "
                     f"val_macro_f1={vf1!r} best={best_val!r} "
                     f"elapsed={time.perf_counter() - t_ep:.3f}s")

    with Step(log, "evaluate_test"):
        model.load_state_dict(best_state)
        model.eval()
        te_loader = _loader(Xte, yte, 512, shuffle=False)
        preds, gts, probs = [], [], []
        with torch.no_grad():                        # TEST touched once, here
            for xb, yb in te_loader:
                logits = model(xb.to(device))
                preds.append(logits.argmax(1).cpu().numpy())
                probs.append(torch.softmax(logits, 1).cpu().numpy())
                gts.append(yb.numpy())
        y_true, y_pred = np.concatenate(gts), np.concatenate(preds)
        y_prob = np.concatenate(probs)
        auc = None
        if len(np.unique(y_true)) >= 2:
            try:
                auc = (float(roc_auc_score(y_true, y_prob[:, 1]))
                       if y_prob.shape[1] == 2 else
                       float(roc_auc_score(y_true, y_prob, multi_class="ovr",
                                           average="macro")))
            except ValueError:
                auc = None
        metrics = {
            "macro_f1": macro_f1(y_true, y_pred),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "auc": auc,
            "per_family_f1": {str(k): v for k, v in
                              per_family_f1(y_true, y_pred).items()},
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "n_test": int(len(y_true)),
        }
        for k in ("macro_f1", "balanced_accuracy", "auc"):
            log.info(f"metric {k}={metrics[k]!r}")

    nan_events = int(model.head.gcp.nan_events) if args.head == "gcp" else 0
    return {
        "run_id": args.run_id, "dataset": args.dataset,
        "backbone": args.backbone, "head": args.head, "seed": args.seed,
        "label_mode": args.label_mode, "smoke": args.smoke,
        "epochs": epochs, "batch_size": args.batch_size, "lr": args.lr,
        "c_last": args.c_last, "n_params": n_params,
        "lprime": model.lprime, "pooled_dim": model.head.out_dim,
        "best_val_macro_f1": best_val, "best_epoch": best_epoch,
        "gcp_nan_events": nan_events, "history": history,
        "test": metrics,
        "split_checksums": meta.get("checksums"),
        "split_meta": {k: meta.get(k) for k in
                       ("n_train", "n_val", "n_test", "n_features", "seed",
                        "dedup_policy")},
        "data_dir": os.path.abspath(args.data_dir),
        "device": str(device), "python": platform.python_version(),
        "torch": torch.__version__, "git_head": _git_head(),
        "runtime_seconds": time.perf_counter() - t_run,
        "finished_at": _dt.datetime.now().strftime(TS_FMT),
    }


def validate(args):
    errs = []
    if args.head not in HEADS:
        errs.append(f"--head {args.head!r} not in {list(HEADS)}")
    if args.backbone not in BACKBONES:
        errs.append(f"--backbone {args.backbone!r} not in {list(BACKBONES)}")
    if args.seed not in SEEDS:
        errs.append(f"--seed {args.seed} not in {list(SEEDS)}")
    if args.dataset not in DATASETS:
        errs.append(f"--dataset {args.dataset!r} not in {list(DATASETS)}")
    if errs:
        raise ValueError("; ".join(errs))


def _fast_loader(X, y, batch_size, shuffle):
    """Same batch contents and order as src.engine._loader (DataLoader with
    num_workers=0, drop_last rule copied), but slices whole batches from the
    tensors instead of collating one sample at a time. Data stays on the device."""
    import numpy as np
    import torch
    from torch.utils.data import RandomSampler
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.from_numpy(X.astype(np.float32)).to(dev)
    yt = torch.from_numpy(y.astype(np.int64))          # labels stay on CPU, as in src.engine._loader
    n = len(Xt)
    drop_last = shuffle and n > batch_size and (n % batch_size == 1)

    class _Batches:
        def __len__(self):
            full = n // batch_size
            return full if (drop_last or n % batch_size == 0) else full + 1

        def __iter__(self):
            order = list(RandomSampler(range(n))) if shuffle else list(range(n))
            order = torch.as_tensor(order)
            for i in range(0, n, batch_size):
                idx = order[i:i + batch_size]
                if drop_last and len(idx) < batch_size:
                    break
                yield Xt[idx.to(dev)], yt[idx]
    return _Batches()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    if args.device == "cpu":
        # hide every GPU before torch is imported: CPU-only is guaranteed
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "results", str(args.backbone))
    args.run_id = (("smoke_" if args.smoke else "")
                   + f"{args.dataset}_{args.backbone}_{args.head}_seed{args.seed}")
    log, log_path = make_logger(os.path.join(args.out_dir, "logs"),
                                args.run_id.replace("/", "_"))
    t0 = time.perf_counter()
    try:
        log.info(f"run_id: {args.run_id}")
        log.info(f"log_file: {log_path}")
        validate(args)
        sys.path.insert(0, REPO)
        from src import data as D
        if args.data_dir is None:
            args.data_dir = D.DEFAULT_OUT
        log_header(log, args, [sys.argv[0]] + list(argv))

        rp = result_path(args.out_dir, args.head, args.dataset, args.seed,
                         args.smoke)
        log.info(f"result_path: {rp}")
        if os.path.exists(rp):
            with open(rp) as fh:
                prev = json.load(fh)              # raises if corrupt -> ERROR
            # the file name does not encode these, so any mismatch with the
            # existing result is an error, never a silent skip
            want = {"backbone": args.backbone, "label_mode": args.label_mode,
                    "c_last": args.c_last, "smoke": args.smoke,
                    "epochs": 1 if args.smoke else args.epochs,
                    "batch_size": args.batch_size, "lr": args.lr}
            diff = {k: (prev.get(k), v) for k, v in want.items()
                    if prev.get(k) != v}
            if diff:
                raise RuntimeError(
                    f"{rp} exists with a different config (existing, "
                    f"requested): {diff}; use a different --out-dir")
            log.info(f"skipped (exists): {rp}")
            log.info(f"DONE status=SKIP total_elapsed="
                     f"{time.perf_counter() - t0:.3f}s")
            return 0

        res = run(args, log)
        with Step(log, "write_result"):
            os.makedirs(args.out_dir, exist_ok=True)
            res["log_file"] = log_path
            write_json_no_overwrite(rp, res)
            log.info(f"wrote {rp}")
        log.info(f"DONE status=OK total_elapsed="
                 f"{time.perf_counter() - t0:.3f}s")
        return 0
    except Exception as e:  # noqa: BLE001
        log.error(f"ERROR {type(e).__name__}: {e}")
        log.error("traceback:\n" + traceback.format_exc())
        log.info(f"DONE status=ERROR total_elapsed="
                 f"{time.perf_counter() - t0:.3f}s")
        return 2 if isinstance(e, ValueError) and "not in" in str(e) else 1


if __name__ == "__main__":
    sys.exit(main())
