#!/usr/bin/env python3
"""R3 -- inference cost / latency harness for the GAP vs GCP pooling heads.

WHY THIS EXISTS
---------------
The paper reports a GCP-over-GAP in-distribution gain of +0.0046 macro-F1
(95 percent CI [0.0027, 0.0061], n=5 seeds; results/repro/t6b_summary.json) and
offers practitioner guidance on the back of it, without ever stating what that
gain costs per flow. A reviewer is entitled to ask. This script measures the
price in latency, throughput and parameters, using the checkpoints that produced
the accuracy numbers, so the cost and the benefit come from the same artifacts.

WHAT IS MEASURED
----------------
Four conditions, because one number is not interpretable:
  batch=1   on GPU and on CPU  -- per-flow latency, the line-rate question
  batch=512 on GPU and on CPU  -- throughput, the batch-scoring question
On CUDA each cell is measured twice: compute-only (input already resident on the
device) and H2D-inclusive (the host-to-device copy inside the timed region). At
batch=1 the copy is a real share of the per-flow cost, so hiding it would
flatter the GPU. On CPU there is no transfer and that field is null.

Timing uses time.perf_counter with torch.cuda.synchronize() on both sides of
every CUDA iteration; without the synchronise the numbers are meaningless
because the launch returns before the kernel runs. The first `--warmup`
iterations are discarded: cold CUDA kernels, lazy module init and the cuDNN
algorithm pick are not representative of steady state.

Inputs are real rows drawn from the actual test split via src.data.load_split,
not random tensors of a guessed shape.

HOW A NUMBER FROM THIS FILE CAN MISLEAD THE PAPER
-------------------------------------------------
Eight caveats. The load-bearing four are re-emitted into the JSON under
`_caveats` so nobody can quote a number without meeting them.

1. FORWARD PASS ONLY. Flow export, NetFlow record assembly, the 39-feature
   extraction and the z-score standardisation all sit outside the timed region
   and in a real sensor they dominate. Every number here is a LOWER BOUND on
   per-flow cost, not the cost.
2. BATCH=1 ON GPU IS A KERNEL-LAUNCH FLOOR, NOT AN ARITHMETIC MEASUREMENT. At
   C=32 both heads are far below the launch-overhead floor, and a
   cuda.synchronize per iteration adds tens of microseconds on top. GAP and GCP
   will look nearly identical there. That understates GCP's arithmetic cost; it
   does not show the cost is zero. The CPU batch=1 cell is the honest per-flow
   figure.
3. THE COST GAP IS QUADRATIC IN C. GCP is O(C^2 L') for the covariance plus
   three Newton-Schulz 32x32 matmul pairs plus a C(C+1)/2 = 528-input
   classifier, against GAP's O(C L') and a 32-input classifier. At C=32 the
   absolute gap is small. It grows quadratically in C, so a measurement at C=32
   licenses NO claim about any wider backbone.
4. FLOWS PER SECOND IS NOT PACKETS PER SECOND. Line rate is a packets-per-second
   property of a link; this model consumes flow records. The throughput figure
   is comparable only to a flow-export rate. Converting it to a link speed needs
   a packets-per-flow assumption the paper does not have.
5. THIS IS A DESKTOP, NOT AN EDGE DEVICE. RTX 3060 Ti plus a multi-thread
   desktop CPU. An "edge IDS may be CPU-only" claim wants an ARM or embedded
   number, which this box cannot produce. Report the hardware next to the
   figure; the `environment` block carries it.
6. PYTORCH EAGER MODE IS A FRAMEWORK FLOOR. A deployed detector would run
   TorchScript / ONNX / TensorRT. Part of the measured latency is Python
   dispatch, not the pooling operator.
7. cudnn.benchmark IS LEFT AT THE TRAINING DEFAULT (False) and recorded in the
   JSON. Turning the autotuner on changes conv timings; pinning it keeps the
   number reproducible instead of accidentally optimistic.
8. CLOCK AND THERMAL STATE. Warmup discards cold kernels but not sustained
   throttling. A short run on an idle card reports near-boost clocks that a
   saturated sensor would not hold.

USAGE
    .venv/bin/python3 scripts/run_latency.py --device both --batch-sizes 1 512 \
        --n-iters 200 --warmup 20 --out-dir results/latency
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import platform
import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import data as D                                    # noqa: E402
from src.engine import _atomic_json_dump                     # noqa: E402
from src.models import NIDSModel, build_head, count_params   # noqa: E402

DATASET = "NF-UNSW-NB15-v2"
HEADS = ["gap", "gcp"]
SEEDS = [42, 123, 456, 789, 1337]
CKPT_DIR = "checkpoints/repro"
OUT_DIR = "results/latency"
BATCH_SIZES = [1, 512]
ROW_DRAW_SEED = 0          # fixed, so every cell of a given batch size sees the
                           # same real rows and cells stay comparable


# --------------------------------------------------------------------------
# pure helpers (the tested surface)
# --------------------------------------------------------------------------
def timing_stats(times_s, batch_size: int) -> dict:
    """Summarise a list of per-batch wall times (seconds) into a cell record.

    median is the headline (robust to the occasional scheduler hiccup), p99 is
    the tail a line-rate deployment actually has to survive. Throughput is
    derived from the MEDIAN, so it is a steady-state rate and not a best case.
    """
    a = np.asarray(list(times_s), dtype=float)
    if a.size == 0:
        raise ValueError("timing_stats needs at least one timed iteration; "
                         "an empty list means every iteration was discarded as "
                         "warmup, which would silently produce a NaN cell")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    median_s = float(np.median(a))
    return {
        "n_iters": int(a.size),
        "median_ms": median_s * 1e3,
        "p99_ms": float(np.percentile(a, 99)) * 1e3,
        "mean_ms": float(a.mean()) * 1e3,
        "min_ms": float(a.min()) * 1e3,
        "max_ms": float(a.max()) * 1e3,
        "throughput_flows_per_s": (batch_size / median_s
                                   if median_s > 0 else float("inf")),
    }


def resolve_devices(spec: str, cuda_available: bool) -> list:
    """Turn --device into a concrete list, refusing a silent CPU fallback.

    Asking for cuda on a box without cuda and getting CPU numbers labelled
    "cuda" is the kind of error that reaches a paper table, so it raises.
    """
    spec = spec.lower()
    if spec == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "--device cuda requested but torch.cuda.is_available() is "
                "False. Refusing to time the CPU and label it cuda.")
        return ["cuda"]
    if spec == "cpu":
        return ["cpu"]
    if spec == "both":
        if not cuda_available:
            print("[latency] no CUDA on this box; measuring CPU only",
                  flush=True)
            return ["cpu"]
        return ["cuda", "cpu"]
    raise ValueError(f"unknown device spec {spec!r}; choose cuda/cpu/both")


def build_conditions(devices, batch_sizes, heads) -> list:
    """Deterministic full grid of measurement cells."""
    return [{"device": d, "batch_size": int(b), "head": h}
            for d in devices for b in batch_sizes for h in heads]


def head_output_dims(c_last: int) -> dict:
    """Pooled feature width per head, read off src.models rather than hardcoded.

    GAP emits C; GCP emits C(C+1)/2. At C=32 that is 32 against 528, which is
    where GCP's extra classifier parameters and its quadratic cost scaling come
    from.
    """
    return {h: int(build_head(h, c_last).out_dim) for h in HEADS}


def median_across_seeds(per_seed: list, key: str, block: str = "compute_only"):
    """Median of one statistic over the per-seed rows of a cell.

    Returns None when the block is absent (the with_transfer block on CPU), so
    the caller never has to special-case device type.
    """
    vals = [r[block][key] for r in per_seed
            if r.get(block) is not None and key in r[block]]
    return float(np.median(vals)) if vals else None


def assemble_summary(config: dict, environment: dict, model_info: dict,
                     cell_records: list) -> dict:
    """Fold per-cell per-seed rows into the published summary.

    Adds, per cell, the across-seed median of median_ms / p99_ms / throughput
    (requirement 5: not a one-checkpoint number), and a `comparison` block
    giving GCP-vs-GAP latency ratio and delta at each (device, batch_size).
    """
    conditions = []
    for cell in cell_records:
        rec = {k: cell[k] for k in ("device", "batch_size", "head")}
        rec["per_seed"] = cell["per_seed"]
        for block in ("compute_only", "with_transfer"):
            rec[block] = {
                "median_ms_across_seeds": median_across_seeds(
                    cell["per_seed"], "median_ms", block),
                "p99_ms_across_seeds": median_across_seeds(
                    cell["per_seed"], "p99_ms", block),
                "throughput_flows_per_s_across_seeds": median_across_seeds(
                    cell["per_seed"], "throughput_flows_per_s", block),
            }
            if rec[block]["median_ms_across_seeds"] is None:
                rec[block] = None
        conditions.append(rec)

    by_key = {(c["device"], c["batch_size"], c["head"]): c for c in conditions}
    comparison = []
    pairs = sorted({(c["device"], c["batch_size"]) for c in conditions})
    for device, batch in pairs:
        gap = by_key.get((device, batch, "gap"))
        gcp = by_key.get((device, batch, "gcp"))
        if gap is None or gcp is None:
            continue
        g = gap["compute_only"]["median_ms_across_seeds"]
        c = gcp["compute_only"]["median_ms_across_seeds"]
        comparison.append({
            "device": device, "batch_size": batch,
            "gap_median_ms": g, "gcp_median_ms": c,
            "delta_ms_gcp_minus_gap": c - g,
            "ratio_gcp_over_gap": (c / g) if g else None,
            "gap_throughput_flows_per_s":
                gap["compute_only"]["throughput_flows_per_s_across_seeds"],
            "gcp_throughput_flows_per_s":
                gcp["compute_only"]["throughput_flows_per_s_across_seeds"],
        })

    return {
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "environment": environment,
        "model": model_info,
        "conditions": conditions,
        "comparison": comparison,
        "_caveats": _caveats(),
        "_deployment_point": _deployment_point(),
    }


def _caveats() -> dict:
    """The four load-bearing caveats, carried inside the JSON.

    They travel with the data so a paper sentence cannot be written from this
    file without meeting them. The full list of eight is in the module
    docstring.
    """
    return {
        "forward_pass_only":
            "Timed region covers the classifier forward pass ONLY. Flow "
            "export, NetFlow record assembly, 39-feature extraction and "
            "z-score standardisation are excluded and dominate in a real "
            "sensor. Every latency here is a LOWER BOUND on per-flow cost.",
        "batch1_gpu_is_a_launch_floor":
            "At batch=1 on GPU the measurement is dominated by kernel-launch "
            "and synchronisation overhead, not arithmetic. GAP and GCP looking "
            "equal there understates GCP's cost; it does not show the cost is "
            "zero. Quote the CPU batch=1 cell for a per-flow claim.",
        "cost_gap_is_quadratic_in_C":
            "GCP is O(C^2 L') covariance plus three Newton-Schulz CxC matmul "
            "pairs plus a C(C+1)/2-input classifier, against GAP's O(C L') and "
            "a C-input classifier. These numbers are measured at C=32 and "
            "license NO claim about wider backbones.",
        "flows_per_second_is_not_packets_per_second":
            "Line rate is a packets-per-second property of a link. This model "
            "consumes flow records, so throughput is comparable only to a "
            "flow-export rate; converting it to a link speed needs a "
            "packets-per-flow assumption the paper does not have.",
        "see_module_docstring_for_the_other_four":
            "desktop hardware is not an edge device; PyTorch eager mode is a "
            "framework floor a TensorRT deployment would not pay; "
            "cudnn.benchmark is pinned to the training default False; warmup "
            "removes cold kernels but not sustained thermal throttling.",
    }


def _deployment_point() -> dict:
    """Which single cell the paper should quote, and why.

    Recommendation: CPU, batch=1, compute_only. For an edge-IDS framing the
    honest per-flow figure is a CPU one, because an edge sensor may have no
    accelerator, and it is batch=1 because a line-rate detector scores a flow
    when the flow closes rather than accumulating 512 of them. On CPU there is
    no host-to-device copy, so the compute_only / with_transfer distinction is
    irrelevant at this cell -- which is a second reason to prefer it: it is the
    one cell whose number carries no transfer-accounting ambiguity.
    """
    return {
        "cell": {"device": "cpu", "batch_size": 1,
                 "measurement": "compute_only"},
        "recommendation":
            "Quote the CPU batch=1 cell as the per-flow cost of the pooling "
            "head, and the CPU batch=512 cell as the batch-scoring cost.",
        "reasoning":
            "An edge IDS may have no accelerator, so CPU is the conservative "
            "deployment assumption. batch=1 is the line-rate case: a detector "
            "scores a flow when the flow record closes, it does not wait for "
            "512 of them. On CPU there is no host-to-device copy, so this cell "
            "has no transfer-accounting ambiguity, unlike every CUDA cell. The "
            "GPU cells are reported for completeness but at batch=1 they "
            "measure kernel-launch overhead rather than the pooling operator.",
        "do_not_quote_alone":
            "The GPU batch=1 cell. It is a launch-overhead floor and will show "
            "GAP and GCP as near-identical for reasons that have nothing to do "
            "with pooling.",
    }


# --------------------------------------------------------------------------
# measurement path
# --------------------------------------------------------------------------
def load_model_from_ckpt(ckpt_path: str, device: str):
    """Reconstruct a trained model exactly the way evaluation does.

    Mirrors src/engine.py:L250-L256 (evaluate_model): same keys, same channel
    tuple, same backbone fallback. Kept byte-for-byte equivalent so the model
    being timed is the model that produced the accuracy numbers.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = NIDSModel(ckpt["head"], ckpt["n_classes"],
                      in_features=ckpt["in_features"],
                      channels=(16, 32, ckpt.get("c_last", 32)),
                      backbone=ckpt.get("backbone", "cnn")).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def draw_batch(X: np.ndarray, batch_size: int, seed: int = ROW_DRAW_SEED):
    """Real rows from the real test split -- never a random tensor of a guessed
    shape. Sampling with replacement only when the split is smaller than the
    batch, which it is not for any dataset here."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=batch_size, replace=batch_size > len(X))
    return torch.from_numpy(np.ascontiguousarray(X[idx])).float()


@torch.no_grad()
def measure(model, x_host: torch.Tensor, device: str, n_iters: int,
            warmup: int, include_transfer: bool) -> list:
    """Return per-iteration wall times in seconds, warmup discarded.

    CUDA correctness: a launch returns before the kernel finishes, so the timed
    region is bracketed by torch.cuda.synchronize() on both sides. Without that
    the numbers measure Python dispatch and nothing else.
    """
    dev = torch.device(device)
    is_cuda = dev.type == "cuda"
    x_dev = None if include_transfer else x_host.to(dev)
    times = []
    for i in range(warmup + n_iters):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        xb = x_host.to(dev) if include_transfer else x_dev
        model(xb)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= warmup:
            times.append(t1 - t0)
    return times


def collect_model_info(models_by_head: dict, c_last_by_head: dict) -> dict:
    """Parameter counts and pooled-feature widths -- requirement 4."""
    dims = head_output_dims(c_last_by_head.get("gap", 32))
    info = {}
    for head, per_seed in models_by_head.items():
        counts = sorted({n for _, n in per_seed})
        c_last = c_last_by_head[head]
        info[head] = {
            "n_params_per_seed": {str(s): int(n) for s, n in per_seed},
            "n_params": int(counts[0]),
            "n_params_identical_across_seeds": len(counts) == 1,
            "c_last": int(c_last),
            "pooled_feature_dim": int(dims[head]),
            "pooled_feature_dim_formula": ("C" if head == "gap"
                                           else "C(C+1)/2"),
        }
    if "gap" in info and "gcp" in info:
        info["_ratio_gcp_over_gap"] = {
            "pooled_feature_dim": (info["gcp"]["pooled_feature_dim"]
                                   / info["gap"]["pooled_feature_dim"]),
            "n_params": info["gcp"]["n_params"] / info["gap"]["n_params"],
        }
    return info


def run(dataset=DATASET, heads=None, seeds=None, devices=None,
        batch_sizes=None, n_iters=200, warmup=20, ckpt_dir=CKPT_DIR,
        out_dir=OUT_DIR, data_dir=None) -> dict:
    heads = list(heads) if heads else list(HEADS)
    seeds = list(seeds) if seeds else list(SEEDS)
    batch_sizes = list(batch_sizes) if batch_sizes else list(BATCH_SIZES)
    devices = list(devices) if devices else resolve_devices(
        "both", torch.cuda.is_available())
    data_dir = data_dir or D.DEFAULT_OUT

    X, _ = D.load_split(dataset, "test", data_dir, "binary")
    print(f"[latency] test pool {X.shape[0]} rows x {X.shape[1]} features",
          flush=True)
    inputs = {b: draw_batch(X, b) for b in batch_sizes}

    # cells keyed by (device, batch, head) so per-seed rows accumulate in place
    cells = {(c["device"], c["batch_size"], c["head"]): dict(c, per_seed=[])
             for c in build_conditions(devices, batch_sizes, heads)}
    models_by_head = {h: [] for h in heads}
    c_last_by_head = {}

    total = len(heads) * len(seeds)
    done = 0
    for head in heads:
        for seed in seeds:
            tag = f"{dataset}__{head}__binary__seed{seed}"
            ckpt_path = os.path.join(ckpt_dir, f"{tag}.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"missing checkpoint {ckpt_path}; this harness times "
                    "TRAINED models, it does not train anything")
            done += 1
            for device in devices:
                model, ckpt = load_model_from_ckpt(ckpt_path, device)
                if device == devices[0]:
                    models_by_head[head].append((seed, count_params(model)))
                    c_last_by_head[head] = int(ckpt.get("c_last", 32))
                for batch in batch_sizes:
                    x = inputs[batch]
                    row = {"seed": seed}
                    row["compute_only"] = timing_stats(
                        measure(model, x, device, n_iters, warmup, False),
                        batch)
                    row["with_transfer"] = (
                        timing_stats(
                            measure(model, x, device, n_iters, warmup, True),
                            batch)
                        if device == "cuda" else None)
                    cells[(device, batch, head)]["per_seed"].append(row)
                    print(f"[{done}/{total}] {head} seed{seed} {device} "
                          f"b={batch} median="
                          f"{row['compute_only']['median_ms']:.4f} ms "
                          f"thr="
                          f"{row['compute_only']['throughput_flows_per_s']:.0f}"
                          " flows/s", flush=True)
                del model

    config = {
        "dataset": dataset, "label_mode": "binary", "heads": heads,
        "seeds": seeds, "devices": devices, "batch_sizes": batch_sizes,
        "n_iters": int(n_iters), "warmup": int(warmup),
        "ckpt_dir": ckpt_dir, "data_dir": data_dir,
        "n_test_pool": int(X.shape[0]), "in_features": int(X.shape[1]),
        "row_draw_seed": ROW_DRAW_SEED,
        "timer": "time.perf_counter, cuda.synchronize on both sides of every "
                 "CUDA iteration",
    }
    environment = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "torch_num_threads": int(torch.get_num_threads()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    ordered = [cells[(c["device"], c["batch_size"], c["head"])]
               for c in build_conditions(devices, batch_sizes, heads)]
    summary = assemble_summary(
        config, environment,
        collect_model_info(models_by_head, c_last_by_head), ordered)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "latency_summary.json")
    _atomic_json_dump(summary, path)
    print(f"[latency] wrote {path}", flush=True)
    for c in summary["comparison"]:
        print(f"  {c['device']:>4} b={c['batch_size']:<4} "
              f"gap {c['gap_median_ms']:.4f} ms | "
              f"gcp {c['gcp_median_ms']:.4f} ms | "
              f"delta {c['delta_ms_gcp_minus_gap']:+.4f} ms | "
              f"x{c['ratio_gcp_over_gap']:.2f}", flush=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="R3: inference cost of the GCP pooling head vs GAP")
    ap.add_argument("--device", default="both", choices=["cuda", "cpu", "both"])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=BATCH_SIZES,
                    metavar="B",
                    help="1 is the per-flow line-rate case, 512 the "
                         "batch-scoring case")
    ap.add_argument("--n-iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20,
                    help="discarded iterations; cold CUDA kernels are not "
                         "representative")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--heads", nargs="+", default=HEADS, choices=HEADS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args(argv)

    devices = resolve_devices(args.device, torch.cuda.is_available())
    run(dataset=args.dataset, heads=args.heads, seeds=args.seeds,
        devices=devices, batch_sizes=args.batch_sizes, n_iters=args.n_iters,
        warmup=args.warmup, ckpt_dir=args.ckpt_dir, out_dir=args.out_dir,
        data_dir=args.data_dir)


if __name__ == "__main__":
    main()
