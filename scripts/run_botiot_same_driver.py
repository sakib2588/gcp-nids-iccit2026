"""Re-run NF-BoT-IoT-v2's in-distribution cells through the headline driver.

Why this exists
---------------
Table II's NF-BoT-IoT-v2 row currently comes from ``results/gap4/*__none__*`` --
the untreated control arm of the class-imbalance experiment -- while the other
three corpora come from ``run_repro_t6b.py`` (UNSW) and ``run_gateC_headline.py``
(CSE-CIC, ToN-IoT). The manuscript states the heads "differ only in the pooling
head" under an "identical protocol", and a reviewer who asks which script
produced each cell currently gets three different answers plus a foreign
experiment.

``run_gap4.py`` also never persists a checkpoint (it loads from an in-memory
``best_state``), so the BoT-IoT weights do not exist and nothing downstream can
be re-scored from them.

This script re-runs all four heads on that corpus through the SAME call path as
``run_gateC_headline.py``, writing checkpoints. That puts three of four corpora
on one driver and makes the fourth re-scorable.

It writes to NEW directories on purpose. The existing gap4 cells are left
untouched so the old and new numbers can be compared -- if they agree, the
provenance concern is answered with evidence rather than asserted away.

Resumable per cell: any (head, seed) whose result JSON already exists is skipped,
so an interrupted run is restarted by re-invoking with no arguments.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.engine import train_model, evaluate_model   # noqa: E402

DATASET = "NF-BoT-IoT-v2"
HEADS = ["gap", "gmp", "se", "gcp"]
SEEDS = [42, 123, 456, 789, 1337]
CKPT_DIR = "checkpoints/botredo"
RESULTS_DIR = "results/botredo"


def main() -> int:
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    total = len(HEADS) * len(SEEDS)
    done = 0
    failed = []
    for head in HEADS:
        for seed in SEEDS:
            tag = f"{DATASET}__{head}__binary__seed{seed}"
            res_path = os.path.join(RESULTS_DIR, f"{tag}.json")
            done += 1
            if os.path.exists(res_path):
                print(f"[{done}/{total}] {tag} SKIP (exists)", flush=True)
                continue
            try:
                # Identical to run_gateC_headline.py: 20 epochs, binary labels,
                # val-only checkpoint selection, all other hyperparameters left
                # at the engine defaults the paper describes.
                _, ckpt_path = train_model(
                    DATASET, head, seed=seed, label_mode="binary",
                    epochs=20, ckpt_dir=CKPT_DIR,
                )
                res = evaluate_model(ckpt_path, DATASET, label_mode="binary",
                                     results_dir=RESULTS_DIR)
                print(f"[{done}/{total}] {tag} macro_f1={res['macro_f1']:.4f}",
                      flush=True)
            except Exception as exc:                      # noqa: BLE001
                failed.append((tag, repr(exc)))
                print(f"[{done}/{total}] {tag} FAILED {exc!r}", flush=True)
    if failed:
        print(f"botiot same-driver rerun finished with {len(failed)} FAILED cells",
              flush=True)
        for tag, err in failed:
            print(f"  FAILED {tag}: {err}", flush=True)
        return 1
    print("botiot same-driver rerun done, 20/20 cells", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
