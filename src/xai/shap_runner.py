"""Gap 3 — SHAP attribution + H5 covariance control (plan v3 D.3).

DeepExplainer often breaks on the Newton-Schulz custom op; we fall back to
GradientExplainer automatically. The H5 control compares GCP-layer covariance of
top-SHAP feature pairs against random pairs / GAP feature map / label-shuffled
(the label-shuffled baseline is a GPU training step, deferred — see plan Task 14).
"""
from __future__ import annotations

import itertools

import numpy as np
import torch


def _shap_values(model, X_np, background_n=50):
    import shap
    model.eval()
    # SHAP calls model(*data) internally, so the background and input tensors
    # must sit on the model's device or conv1d raises a CPU/CUDA type mismatch.
    dev = next(model.parameters()).device
    bg = torch.from_numpy(X_np[:background_n]).to(dev)
    X = torch.from_numpy(X_np).to(dev)
    try:
        explainer = shap.DeepExplainer(model, bg)
        sv = explainer.shap_values(X)
    except Exception:                      # custom op breaks DeepExplainer
        explainer = shap.GradientExplainer(model, bg)
        sv = explainer.shap_values(X)
    # sv may be a list per class; take attack class (index 1) when present
    if isinstance(sv, list):
        arr = sv[1] if len(sv) > 1 else sv[0]
    else:
        arr = sv
    arr = np.asarray(arr)
    # collapse any class axis: keep the per-feature mean-abs attribution
    return np.abs(arr).reshape(arr.shape[0], -1).mean(axis=0)[:X_np.shape[1]]


def top_shap_features(model, X_np, k=8):
    imp = _shap_values(model, X_np)
    return [int(i) for i in np.argsort(imp)[::-1][:k]]


def top_shap_pairs(model, X_np, k=5):
    feats = top_shap_features(model, X_np, k=max(2, k))
    return list(itertools.combinations(feats, 2))[:k]


def top_channel_features(model, X, k=8):
    """Rank GCP-layer conv channels by gradient x activation attribution.

    The H5 mechanism test lives in the GCP layer's channel space: it must rank
    the C conv channels (which the covariance is formed over), not the input
    NetFlow features that input-SHAP would attribute to. Importance for channel c
    is mean over batch and spatial positions of |grad_c * activation_c|, where the
    gradient is d(attack-logit)/d(feature-map) -- the standard Grad-CAM-style
    channel weighting, evaluated on the differentiable iSQRT-COV head.
    """
    model.eval()
    dev = next(model.parameters()).device
    x = X if torch.is_tensor(X) else torch.from_numpy(X)
    x = x.to(dev).float()
    x = x.unsqueeze(1) if x.dim() == 2 else x
    feat = model.backbone(x)                       # (B, C, L')
    feat.retain_grad()
    logits = model.classifier(model.head(feat))    # (B, num_classes)
    score = logits[:, 1].sum() if logits.shape[1] > 1 else logits.sum()
    model.zero_grad(set_to_none=True)
    score.backward()
    imp = (feat.grad * feat).abs().mean(dim=(0, 2))  # (C,)
    return [int(i) for i in torch.argsort(imp, descending=True)[:k].cpu().numpy()]


def top_channel_pairs(model, X, n_channels=5):
    """All mutual pairs among the top-`n_channels` attributed conv channels.

    Using every pair among the top channels (not the first k pairs of a
    combinations() enumeration) keeps the set from being dominated by the single
    highest-ranked channel: it is a genuine "do the top channels co-activate with
    each other" statistic. Top-5 channels -> C(5,2)=10 pairs.
    """
    feats = top_channel_features(model, X, k=n_channels)
    return list(itertools.combinations(feats, 2))


def layer_covariance(model, X):
    """Mean (over batch) covariance of the conv feature map -> (C, C)."""
    model.eval()
    with torch.no_grad():
        x = X.unsqueeze(1) if X.dim() == 2 else X
        feat = model.backbone(x)                       # (B, C, L')
        feat = feat - feat.mean(dim=2, keepdim=True)
        cov = torch.einsum("bcl,bdl->bcd", feat, feat) / feat.shape[2]
    return cov.mean(dim=0)                              # (C, C)
