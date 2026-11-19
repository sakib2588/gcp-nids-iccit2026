"""T3 — 1D-CNN backbone with swappable pooling heads (GAP / SE / GCP / GMP).

The pooling HEAD is the only variable in the core ablation (plan v3 C.1):
backbone and classifier interface stay fixed; only the head changes. This keeps
the GCP-vs-GAP comparison clean.

Input convention: a tabular NetFlow vector of F features is presented as a
1-channel sequence of length F -> (B, 1, F). Three Conv1d layers (kernel=3,
stride=1, no padding) shrink the length: L' = F - 2*3 = F - 6. For F=39 -> L'=33.

GCP rank condition: covariance_pool centres across the L' positions, so the
covariance rank is bounded by min(C, L'-1) and full rank requires L' >= C + 1,
not L' >= C. With C_last=32 this needs L' >= 33, i.e. F >= 39. The three
39-feature corpora clear it exactly (F=39 -> L'=33). NF-CSE-CIC-IDS2018-v2 has
F=38 -> L'=32, one short: its covariance has rank 31 and is singular, and the
GCP runs on it are flagged rank-deficient. See Gap 6 and
notes/20260720-bug-prepublication-code-audit.md defect B7.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gcp_layer import GlobalCovariancePooling

HEADS = ("gap", "se", "gcp", "gap_wide", "gmp", "gmp_wide")


def measured_lprime(in_features: int, n_layers: int = 3, kernel: int = 3) -> int:
    """Analytic L' for kernel-k, stride-1, no-pad conv stack."""
    return in_features - n_layers * (kernel - 1)


class CNN1DBackbone(nn.Module):
    def __init__(self, in_ch: int = 1, channels=(16, 32, 32), kernel: int = 3):
        super().__init__()
        self.out_channels = channels[-1]
        blocks = []
        c_prev = in_ch
        for c in channels:
            blocks += [
                nn.Conv1d(c_prev, c, kernel_size=kernel, stride=1, padding=0),
                nn.BatchNorm1d(c),
                nn.ReLU(),   # not inplace: keeps SHAP GradientExplainer hooks valid
            ]
            c_prev = c
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, F) -> (B, C_last, L')
        return self.net(x)


class _TCNResBlock(nn.Module):
    """Dilated conv pair with a residual skip (canonical TCN, Bai et al. 2018).
    SAME padding keeps L'=F so L' >= C + 1 for the GCP rank condition
    whenever F > C."""
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.proj = (nn.Conv1d(c_in, c_out, 1) if c_in != c_out
                     else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.proj(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)


class TCNBackbone(nn.Module):
    """Dilated residual TCN; SAME padding keeps L'=F (so L' >= C + 1 for GCP
    rank whenever F > C)."""
    def __init__(self, in_ch: int = 1, channels=(16, 32, 32), kernel: int = 3,
                 dilations=(1, 2, 4)):
        super().__init__()
        self.out_channels = channels[-1]
        blocks, c_prev = [], in_ch
        for c, d in zip(channels, dilations):
            blocks.append(_TCNResBlock(c_prev, c, kernel, d))
            c_prev = c
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResBlock1D(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int = 3):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, padding=pad)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=pad)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.proj = (nn.Conv1d(c_in, c_out, 1) if c_in != c_out
                     else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.proj(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)


class ResNet1DBackbone(nn.Module):
    """Residual 1D conv stack; SAME padding keeps L'=F (so L' >= C + 1
    whenever F > C)."""
    def __init__(self, in_ch: int = 1, channels=(16, 32, 32), kernel: int = 3):
        super().__init__()
        self.out_channels = channels[-1]
        blocks, c_prev = [], in_ch
        for c in channels:
            blocks.append(_ResBlock1D(c_prev, c, kernel)); c_prev = c
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_backbone(name: str = "cnn", channels=(16, 32, 32), kernel: int = 3):
    name = name.lower()
    if name == "cnn":
        return CNN1DBackbone(1, channels, kernel)
    if name == "tcn":
        return TCNBackbone(1, channels, kernel)
    if name == "resnet1d":
        return ResNet1DBackbone(1, channels, kernel)
    raise ValueError(f"unknown backbone {name!r}; choose from cnn/tcn/resnet1d")


class GAPHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.out_dim = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=2)                       # (B, C)


class SEHead(nn.Module):
    """Squeeze-excitation channel reweighting, then global average pool."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.out_dim = channels
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=2)                          # (B, C) squeeze
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        x = x * s.unsqueeze(2)                     # channel-wise scale
        return x.mean(dim=2)                       # (B, C)


class GCPHead(nn.Module):
    def __init__(self, channels: int, iters: int = 3):
        super().__init__()
        self.gcp = GlobalCovariancePooling(channels, iters=iters)
        self.out_dim = self.gcp.out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gcp(x)                         # (B, C*(C+1)/2)


class GAPWideHead(nn.Module):
    """Global average pooling widened to GCP's output dimension.

    Capacity control for the pooling ablation (audit 0.1). GCP emits
    C(C+1)/2 features against GAP's C, so at C=32 the classifier carries 528
    inputs for GCP and 32 for GAP, and Methods' claim that the classifier is
    held fixed across heads is false. This head projects GAP's C outputs to
    C(C+1)/2 with one linear layer, so the classifier sees an identical input
    width and any remaining gap is attributable to what the pooling encodes.

    The projection adds C*out_dim + out_dim parameters, so this head has more
    total parameters than GCP, not fewer. That is deliberate: a GCP win against
    a strictly larger competitor cannot be explained by capacity.
    """
    def __init__(self, channels: int, out_dim: int | None = None):
        super().__init__()
        self.out_dim = (out_dim if out_dim is not None
                        else channels * (channels + 1) // 2)
        self.proj = nn.Linear(channels, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=2))            # (B, C) -> (B, C(C+1)/2)


class GMPHead(nn.Module):
    """Global max pooling: a first-order ORDER statistic instead of the mean.

    Answers a question GAP alone cannot: is GCP better than *first-order*
    pooling, or only better than the mean? GMP emits the same C features as
    GAP, so the classifier is identical in size and the GMP-vs-GAP contrast
    carries no capacity confound.

    Against GCP it does carry one -- 32 classifier inputs against 528 -- which
    is the confound GAPWideHead exists to answer for GAP. GMPWideHead below is
    the matching control; without it, a GCP win over GMP is not attributable to
    what the pooling encodes.

    Max is a NONLINEAR statistic, so unlike GAP it is not strictly blind to the
    correlation structure feeding the conv activations. A GMP gain over GAP in
    the synthetic covariance regime is therefore a finding about first-order
    pooling, not a refutation of GCP's mechanism (see
    notes/20260811-decision-maxpool-control.md, H-MAX-2).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.out_dim = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.amax(dim=2)                       # (B, C)


class GMPWideHead(nn.Module):
    """Global max pooling widened to GCP's output dimension.

    Capacity control for GMP, exactly as GAPWideHead is for GAP: projects the C
    max-pooled outputs to C(C+1)/2 with one linear layer so the classifier sees
    GCP's input width. The projection makes this head strictly larger than GCP
    in total parameters, so a GCP win here cannot be a capacity artefact.
    """
    def __init__(self, channels: int, out_dim: int | None = None):
        super().__init__()
        self.out_dim = (out_dim if out_dim is not None
                        else channels * (channels + 1) // 2)
        self.proj = nn.Linear(channels, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.amax(dim=2))            # (B, C) -> (B, C(C+1)/2)


def build_head(head: str, channels: int) -> nn.Module:
    head = head.lower()
    if head == "gap":
        return GAPHead(channels)
    if head == "se":
        return SEHead(channels)
    if head == "gcp":
        return GCPHead(channels)
    if head == "gap_wide":
        return GAPWideHead(channels)
    if head == "gmp":
        return GMPHead(channels)
    if head == "gmp_wide":
        return GMPWideHead(channels)
    raise ValueError(f"unknown head {head!r}; choose from {HEADS}")


class NIDSModel(nn.Module):
    def __init__(self, head: str, num_classes: int, in_features: int = 39,
                 channels=(16, 32, 32), kernel: int = 3, backbone: str = "cnn"):
        super().__init__()
        self.backbone = build_backbone(backbone, channels, kernel)
        self.head = build_head(head, self.backbone.out_channels)
        self.classifier = nn.Linear(self.head.out_dim, num_classes)
        self.in_features = in_features
        # CNN (no pad) shrinks L'; TCN/ResNet1D keep L'=F via SAME padding.
        self.lprime = (measured_lprime(in_features, len(channels), kernel)
                       if backbone == "cnn" else in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # accept (B, F) or (B, 1, F)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.backbone(x)                    # (B, C, L')
        pooled = self.head(feat)                   # (B, head.out_dim)
        return self.classifier(pooled)             # (B, num_classes)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
