"""T5 — training / evaluation engine (plan v3 C.2, T5d, T5e).

Design guarantees baked in:
- T5d val-only model selection: the checkpoint is the epoch with best VAL macro-F1;
  the TEST split is never read during training, only once in evaluate_model.
- T5e identical input: every head loads the SAME materialised npy split
  (src/data.py), so GAP/SE/GCP consume byte-identical tensors. The head is the
  only variable.
- Reproducibility: full seeding (random/numpy/torch CPU+CUDA, cudnn deterministic).
- Memory: splits are in-RAM tensors (TensorDataset, num_workers=0) -> no streaming
  loader, no pyarrow/DataLoader leak. RSS is structurally flat across epochs.
"""
from __future__ import annotations

import os
import json
import random
import platform

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src import data as D
from src.models import NIDSModel, count_params
from src.metrics import macro_f1, per_family_f1
from sklearn.metrics import (confusion_matrix, balanced_accuracy_score,
                             roc_auc_score)


def _shuffle_labels(y, seed: int):
    """Return a deterministic permutation of y (breaks the feature-label link).

    Used only for the Gap 3 H5 label-shuffled control: an identically-trained
    model whose labels carry no real signal. Permutes (preserves class counts),
    it does not relabel.
    """
    rng = np.random.default_rng(seed)
    return np.asarray(y)[rng.permutation(len(y))]


def _atomic_json_dump(obj, path: str):
    """Write JSON via a temp file plus os.replace.

    Result JSONs double as RESUME MARKERS: every driver skips a cell whose
    result file exists. A non-atomic dump interrupted by a kernel panic, an
    OOM kill or `systemctl stop` leaves a truncated file that the resume check
    trusts, so the cell is never recomputed and the run silently loses it.
    scripts/run_gap1.py hit exactly this and was hardened on 2026-07-20; this
    puts the same guarantee under every driver that goes through
    evaluate_model. Pair it with a load-check on resume, since a file written
    before this fix may already be truncated.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _feature_permutation(n_features: int, perm_seed: int):
    """A fixed reordering of the NetFlow feature columns.

    The backbone presents the feature vector as a 1-channel sequence, so a
    kernel-3 conv slides over adjacent COLUMNS of the parquet schema. That
    adjacency is an artefact of the column order, and every learned channel --
    hence the covariance the GCP head pools -- inherits it. This hook lets us
    measure how much of the GCP-GAP delta survives a different ordering.

    Keyed on perm_seed, NOT on the training seed: the ablation varies ordering
    while holding the seed protocol fixed, so the two sources of variance stay
    separable. Draws until the permutation is not the identity, otherwise a
    cell would silently re-run the baseline and read as a pass.
    """
    rng = np.random.default_rng(perm_seed)
    ident = np.arange(n_features)
    for _ in range(100):
        perm = rng.permutation(n_features)
        if not np.array_equal(perm, ident):
            return perm
    raise RuntimeError(f"no non-identity permutation for n={n_features}")


def _apply_feature_permutation(X, perm):
    """Reorder feature columns of X. Rows are untouched."""
    X = np.asarray(X)
    perm = np.asarray(perm)
    assert X.ndim == 2, f"expected (N, F), got {X.shape}"
    assert X.shape[1] == len(perm), (
        f"permutation width {len(perm)} does not match feature count {X.shape[1]}")
    return X[:, perm]


def _class_weights(y, n_classes):
    """Inverse-frequency class weights for CrossEntropyLoss (Gap 4 / G1)."""
    counts = np.bincount(np.asarray(y), minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rng_state() -> dict:
    """Snapshot every RNG the training loop draws from.

    Captured: torch CPU, numpy, python `random`, and the per-device CUDA
    generators when CUDA is present. Needed so a resumed run continues the
    random stream instead of replaying it from the epoch-0 state that
    set_seed installs (batch order comes from the torch CPU generator via
    DataLoader's RandomSampler).
    """
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "cuda": (torch.cuda.get_rng_state_all()
                 if torch.cuda.is_available() else []),
    }


def _restore_rng_state(state: dict) -> None:
    """Inverse of _rng_state. CUDA restore is guarded so a snapshot taken on a
    GPU box still loads on a CPU-only box (and vice versa): the CUDA states are
    applied only when CUDA is available and the device count matches.

    Every generator state is forced back to CPU first: the snapshot is loaded
    with map_location=device, which on a GPU box hands back CUDA ByteTensors,
    and both set_rng_state and set_rng_state_all reject those."""
    if not state:
        return
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    cuda_states = state.get("cuda") or []
    if cuda_states and torch.cuda.is_available() \
            and len(cuda_states) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)),
                       torch.from_numpy(y.astype(np.int64)))
    # B1 fix: drop a trailing size-1 batch only when training (BatchNorm1d needs
    # >1 sample per channel in train mode). Never drop during eval, and never
    # drop if the whole split is <= one batch.
    drop_last = shuffle and len(X) > batch_size and (len(X) % batch_size == 1)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, drop_last=drop_last)


def _num_classes(name, label_mode, out_dir):
    if label_mode == "binary":
        return 2
    with open(os.path.join(out_dir, name, "meta.json")) as fh:
        return len(json.load(fh)["classes"])


def train_model(name: str, head: str, seed: int = 42,
                label_mode: str = "binary", epochs: int = 20,
                batch_size: int = 512, lr: float = 1e-3,
                out_dir: str = D.DEFAULT_OUT, ckpt_dir: str = "checkpoints",
                c_last: int = 32, class_weight: bool = False,
                backbone: str = "cnn", shuffle_labels: bool = False,
                resume: bool = False, permute_features: int | None = None):
    """Train one (dataset, head, seed). Returns (best_val_f1, ckpt_path).

    VAL-only selection: best epoch chosen by val macro-F1. Test untouched.
    c_last sets the final conv channel count C (Gap 6); backbone in
    {cnn,tcn,resnet1d} (Gap 5); class_weight enables inverse-freq CE (Gap 4).

    resume (opt-in, default False -> behaviour identical for all existing
    callers): when True, an interrupted run continues from the last completed
    epoch instead of restarting. A per-epoch snapshot {model, optimizer, epoch,
    best_val, history, rng_state} is written to <tag>__resume.pt and removed
    once the run finishes; the best-val checkpoint contract (what
    evaluate_model reads) is unchanged. rng_state carries the torch CPU, numpy,
    python and CUDA generators so a resumed run is bit-identical to an
    uninterrupted one.

    permute_features (opt-in, default None -> behaviour identical for all
    existing callers): reorder the feature columns by a permutation keyed on
    this integer, applied identically to train and val. The permutation is
    written into the checkpoint so evaluate_model applies the same ordering to
    test; scoring a permuted model on unpermuted columns would be silently
    wrong. Keyed on its own seed, independent of `seed`, so ordering variance
    and seed variance stay separable (R2+ feature-order ablation).
    """
    set_seed(seed)
    device = _device()
    Xtr, ytr = D.load_split(name, "train", out_dir, label_mode)
    if shuffle_labels:
        # H5 control: destroy the feature-label association with a per-seed
        # permutation so any covariance concentration cannot be label-driven.
        ytr = _shuffle_labels(ytr, seed)
    Xva, yva = D.load_split(name, "val", out_dir, label_mode)
    in_features = Xtr.shape[1]
    feature_perm = None
    if permute_features is not None:
        # R2+: the conv slides over adjacent feature COLUMNS, so the parquet
        # schema order is a free parameter nothing in the paper varies.
        feature_perm = _feature_permutation(in_features, permute_features)
        Xtr = _apply_feature_permutation(Xtr, feature_perm)
        Xva = _apply_feature_permutation(Xva, feature_perm)
    n_classes = _num_classes(name, label_mode, out_dir)

    model = NIDSModel(head, n_classes, in_features=in_features,
                      channels=(16, 32, c_last), backbone=backbone).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if class_weight:
        crit = nn.CrossEntropyLoss(weight=_class_weights(ytr, n_classes).to(device))
    else:
        crit = nn.CrossEntropyLoss()
    tr_loader = _loader(Xtr, ytr, batch_size, shuffle=True)
    va_loader = _loader(Xva, yva, batch_size, shuffle=False)

    os.makedirs(ckpt_dir, exist_ok=True)
    # Tag includes c_last/backbone ONLY when non-default, so existing default
    # runs (T6b) keep their names while Gap 5/6 configs get distinct artifacts
    # (no checkpoint/result/env collision across C values or backbones).
    suffix = ""
    if c_last != 32:
        suffix += f"__C{c_last}"
    if backbone != "cnn":
        suffix += f"__{backbone}"
    if shuffle_labels:
        suffix += "__shuffled"
    if permute_features is not None:
        suffix += f"__perm{permute_features}"
    tag = f"{name}__{head}__{label_mode}__seed{seed}{suffix}"
    ckpt_path = os.path.join(ckpt_dir, f"{tag}.pt")
    best_val = -1.0
    history = []
    start_epoch = 0
    resume_path = os.path.join(ckpt_dir, f"{tag}__resume.pt")
    if resume and os.path.exists(resume_path):
        r = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(r["model_state_dict"])
        opt.load_state_dict(r["opt_state_dict"])
        start_epoch = r["epoch"] + 1
        best_val = r["best_val"]
        history = r["history"]
        # Continue the random stream rather than replaying it: without this the
        # set_seed above leaves the RNG at its epoch-0 state and the resumed run
        # repeats epoch 0's shuffle order. Snapshots written before this fix
        # carry no rng_state, so the old (non-identical) behaviour is kept for
        # them instead of crashing.
        rng = r.get("rng_state")
        if rng is None:
            print(f"[resume] {tag}: snapshot has no rng_state (pre-fix file); "
                  f"batch order will NOT be bit-identical", flush=True)
        else:
            _restore_rng_state(rng)
        print(f"[resume] {tag}: continuing from epoch {start_epoch} "
              f"(best_val={best_val:.4f})", flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # ---- VAL ONLY for selection (test never read here) ----
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                p = model(xb.to(device)).argmax(1).cpu().numpy()
                preds.append(p); gts.append(yb.numpy())
        vf1 = macro_f1(np.concatenate(gts), np.concatenate(preds))
        history.append({"epoch": epoch, "val_macro_f1": vf1})
        if vf1 > best_val:
            best_val = vf1
            torch.save({"model_state_dict": model.state_dict(),
                        "head": head, "n_classes": n_classes,
                        "in_features": in_features, "epoch": epoch,
                        "val_macro_f1": vf1, "c_last": c_last,
                        "backbone": backbone,
                        # None for every pre-existing checkpoint; evaluate_model
                        # treats a missing key as the identity ordering.
                        "feature_perm": (None if feature_perm is None
                                         else feature_perm.tolist())}, ckpt_path)
        # ---- per-epoch resume snapshot (opt-in; removed on completion) ----
        if resume:
            torch.save({"model_state_dict": model.state_dict(),
                        "opt_state_dict": opt.state_dict(), "epoch": epoch,
                        "best_val": best_val, "history": history,
                        "rng_state": _rng_state()}, resume_path)

    if resume and os.path.exists(resume_path):
        os.remove(resume_path)

    nan_events = int(model.head.gcp.nan_events) if head == "gcp" else 0
    _write_environment(ckpt_dir, tag, seed, head, name, label_mode,
                       epochs, batch_size, lr, best_val,
                       count_params(model), model.lprime, history,
                       nan_events=nan_events)
    return best_val, ckpt_path


@torch.no_grad()
def evaluate_model(ckpt_path: str, name: str, label_mode: str = "binary",
                   out_dir: str = D.DEFAULT_OUT,
                   results_dir: str = "results") -> dict:
    """Evaluate a checkpoint on the TEST split (touched exactly once here)."""
    device = _device()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = NIDSModel(ckpt["head"], ckpt["n_classes"],
                      in_features=ckpt["in_features"],
                      channels=(16, 32, ckpt.get("c_last", 32)),
                      backbone=ckpt.get("backbone", "cnn")).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    Xte, yte = D.load_split(name, "test", out_dir, label_mode)
    # A model trained under a permuted feature ordering must be scored on the
    # SAME ordering. Missing key == identity, so every pre-existing checkpoint
    # is unaffected.
    ckpt_perm = ckpt.get("feature_perm")
    if ckpt_perm is not None:
        Xte = _apply_feature_permutation(Xte, np.asarray(ckpt_perm))
    loader = _loader(Xte, yte, 512, shuffle=False)
    preds, gts, probs = [], [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        gts.append(yb.numpy())
    y_pred = np.concatenate(preds); y_true = np.concatenate(gts)
    y_prob = np.concatenate(probs)

    # M1 fix: prior-insensitive metrics for the benign-minority / label-shift
    # cross-dataset case (plan v3 R1-7). AUC needs >=2 classes present in y_true.
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    auc = None
    classes_present = np.unique(y_true)
    if len(classes_present) >= 2:
        try:
            if y_prob.shape[1] == 2:
                auc = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                auc = float(roc_auc_score(y_true, y_prob,
                                          multi_class="ovr", average="macro"))
        except ValueError:
            auc = None

    res = {
        "checkpoint": ckpt_path, "dataset": name, "head": ckpt["head"],
        "label_mode": label_mode,
        "macro_f1": macro_f1(y_true, y_pred),
        "balanced_accuracy": bal_acc,
        "auc": auc,
        "per_family_f1": {str(k): v for k, v in
                          per_family_f1(y_true, y_pred).items()},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_test": int(len(y_true)),
    }
    os.makedirs(results_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    _atomic_json_dump(res, os.path.join(results_dir, f"{tag}.json"))
    return res


def _write_environment(ckpt_dir, tag, seed, head, name, label_mode,
                       epochs, batch_size, lr, best_val, n_params,
                       lprime, history, nan_events=0):
    env = {
        "tag": tag, "seed": seed, "head": head, "dataset": name,
        "label_mode": label_mode, "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "best_val_macro_f1": best_val, "n_params": n_params,
        "lprime": lprime, "nan_events": nan_events,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None),
        "history": history,
    }
    _atomic_json_dump(env, os.path.join(ckpt_dir, f"environment__{tag}.json"))
