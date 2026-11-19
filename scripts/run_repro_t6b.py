#!/usr/bin/env python3
"""T6b — GCP-vs-GAP reproduction run on UNSW, n=5 (plan v3 Phase 3).

Trains GAP and GCP heads on NF-UNSW-NB15-v2, binary, seeds 42/123/456/789/1337.
Resumable: a (head, seed) whose results JSON already exists is skipped, so a run
that died mid-way (e.g. at seed 1337) picks up exactly where it stopped.

Writes:
  checkpoints/repro/{tag}.pt + environment__{tag}.json
  results/repro/{tag}.json
  results/repro/t6b_summary.json   (GCP-vs-GAP delta, bootstrap CI, d_z -> T6c gate)
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.engine import train_model, evaluate_model  # noqa: E402
from src.metrics import bootstrap_ci, cohen_dz       # noqa: E402

DATASET = "NF-UNSW-NB15-v2"
LABEL_MODE = "binary"
HEADS = ["gap", "se", "gcp"]
SEEDS = [42, 123, 456, 789, 1337]
EPOCHS = 20
CKPT_DIR = "checkpoints/repro"
RESULTS_DIR = "results/repro"


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    scores = {h: {} for h in HEADS}

    for head in HEADS:
        for seed in SEEDS:
            tag = f"{DATASET}__{head}__{LABEL_MODE}__seed{seed}"
            res_path = os.path.join(RESULTS_DIR, f"{tag}.json")
            if os.path.exists(res_path):
                with open(res_path) as fh:
                    res = json.load(fh)
                print(f"[skip] {tag} -> macro_f1={res['macro_f1']:.4f}", flush=True)
                scores[head][seed] = res["macro_f1"]
                continue
            print(f"[train] {tag} ...", flush=True)
            _, ckpt = train_model(DATASET, head, seed, LABEL_MODE,
                                  epochs=EPOCHS, ckpt_dir=CKPT_DIR)
            res = evaluate_model(ckpt, DATASET, LABEL_MODE,
                                 results_dir=RESULTS_DIR)
            scores[head][seed] = res["macro_f1"]
            print(f"[done]  {tag} -> macro_f1={res['macro_f1']:.4f}", flush=True)

    # ---- T6c: GCP-vs-GAP paired delta on shared seeds ----
    common = [s for s in SEEDS if s in scores["gap"] and s in scores["gcp"]]
    gap = [scores["gap"][s] for s in common]
    gcp = [scores["gcp"][s] for s in common]
    delta = [g - a for g, a in zip(gcp, gap)]
    summary = {
        "dataset": DATASET, "label_mode": LABEL_MODE,
        "seeds": common,
        "gap_macro_f1": gap, "gcp_macro_f1": gcp,
        "delta_gcp_minus_gap": delta,
        "gap_mean_ci": bootstrap_ci(gap) if gap else None,
        "gcp_mean_ci": bootstrap_ci(gcp) if gcp else None,
        "delta_mean_ci": bootstrap_ci(delta) if len(delta) >= 1 else None,
        "cohen_dz": cohen_dz(gcp, gap) if len(common) >= 2 else None,
    }
    with open(os.path.join(RESULTS_DIR, "t6b_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n=== T6b summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
