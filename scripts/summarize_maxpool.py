#!/usr/bin/env python3
"""Summarise the max-pooling (GMP) control, real data and synthetic.

Reports every pre-registered contrast from
notes/20260811-decision-maxpool-control.md with a t(df=4) 95% interval, a
bootstrap interval for comparability with the published numbers, and paired d_z.

Three things this does that a generic summariser would not:

1. PROVENANCE. The published GAP/SE/GCP baselines live in three directories and
   came from two different training paths (src.engine.train_model for UNSW /
   CSE-CIC / ToN-IoT, run_gap4's own loop for BoT-IoT). Every arm records the
   file it was read from and the script that wrote it, so a paired delta can be
   audited without re-deriving where the numbers came from.

2. CAPACITY FLAG. GMP feeds the classifier C=32 features, GCP feeds it
   C(C+1)/2=528. Any contrast whose two arms differ in classifier input width is
   stamped capacity_confounded=true, so the caveat travels with the number into
   the paper instead of being remembered separately. gmp_wide is the arm that
   clears it.

3. VERDICTS, NOT NUMBERS. Each pre-registered hypothesis prints
   PASS / FAIL / UNRESOLVED against the criterion fixed before the run, and no
   verdict is printed for a cell with fewer than 5 complete seeds.

Writes results/maxpool/maxpool_summary.json atomically.
"""
from __future__ import annotations

import os
import sys
import json
import glob
import math
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import bootstrap_ci, t_ci, cohen_dz    # noqa: E402
from src.models import NIDSModel, count_params          # noqa: E402

SEEDS = [42, 123, 456, 789, 1337]
DZ_GATE = 0.8
OUT_DIR = "results/maxpool"
OUT = os.path.join(OUT_DIR, "maxpool_summary.json")
SYNTH_DIR = "results/gateB_synthetic_n40k"

# dataset -> (directory, filename pattern, writing script). The GMP arm for a
# dataset MUST be read from the same path as its baselines; see module docstring.
SOURCES = {
    "NF-UNSW-NB15-v2": [
        (OUT_DIR, "{ds}__{head}__binary__seed{seed}.json", "run_maxpool_control.py"),
        ("results/repro", "{ds}__{head}__binary__seed{seed}.json", "run_repro_t6b.py"),
    ],
    "NF-CSE-CIC-IDS2018-v2": [
        (OUT_DIR, "{ds}__{head}__binary__seed{seed}.json", "run_maxpool_control.py"),
        ("results/gateC", "{ds}__{head}__binary__seed{seed}.json", "run_gateC_headline.py"),
    ],
    "NF-ToN-IoT-v2": [
        (OUT_DIR, "{ds}__{head}__binary__seed{seed}.json", "run_maxpool_control.py"),
        ("results/gateC", "{ds}__{head}__binary__seed{seed}.json", "run_gateC_headline.py"),
    ],
    "NF-BoT-IoT-v2": [
        ("results/botredo", "{ds}__{head}__binary__seed{seed}.json", "run_botiot_same_driver.py"),
        ("results/gap4", "{ds}__{head}__none__seed{seed}.json", "run_gap4.py"),
    ],
}

# (a, b) contrasts, in reporting order. gmp - gap first: it is the only real-data
# contrast that is width-matched, so it is the one that carries weight.
CONTRASTS = [("gmp", "gap"), ("gcp", "gmp"), ("gcp", "gap"), ("gcp", "gmp_wide")]


def head_width(head: str, channels: int = 32) -> int:
    """Classifier input width for a head, used for the capacity flag."""
    return int(NIDSModel(head, 2, in_features=39,
                         channels=(16, 32, channels)).head.out_dim)


def _read_arm(dataset, head, seeds=SEEDS):
    """Return ({seed: macro_f1}, provenance) for one (dataset, head).

    Searches the source list in order and takes the FIRST directory that holds a
    complete set of the requested seeds, so a partial directory cannot silently
    contribute half an arm.
    """
    partial = None
    for d, pattern, script in SOURCES.get(dataset, []):
        vals = {}
        for s in seeds:
            p = os.path.join(d, pattern.format(ds=dataset, head=head, seed=s))
            if os.path.exists(p):
                with open(p) as fh:
                    vals[s] = float(json.load(fh)["macro_f1"])
        if len(vals) == len(seeds):
            return vals, {"dir": d, "script": script, "n_seeds": len(vals)}
        if vals and partial is None:
            partial = (vals, {"dir": d, "script": script, "n_seeds": len(vals),
                              "incomplete": True})
    return partial if partial else ({}, None)


def _contrast(a_name, a, b_name, b):
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    av = np.array([a[s] for s in seeds])
    bv = np.array([b[s] for s in seeds])
    d = av - bv
    dmean, blo, bhi = bootstrap_ci(d)
    _, tlo, thi = t_ci(d)
    dz = cohen_dz(av, bv)
    wa, wb = head_width(a_name), head_width(b_name)
    return {"contrast": f"{a_name} - {b_name}",
            "seeds": seeds,
            "per_seed_delta": {str(s): float(x) for s, x in zip(seeds, d)},
            "delta_mean": float(dmean),
            "delta_bootstrap_ci": [blo, bhi],
            "delta_t_ci": [tlo, thi],
            "d_z": dz if math.isfinite(dz) else None,
            "n_seeds_positive": int((d > 0).sum()),
            "t_ci_excludes_zero": bool(tlo > 0 or thi < 0),
            "bootstrap_ci_excludes_zero": bool(blo > 0 or bhi < 0),
            "passes_dz_gate": bool(abs(dz) >= DZ_GATE) if math.isfinite(dz) else None,
            "classifier_width": {a_name: wa, b_name: wb},
            # The caveat travels with the number: a GCP win over a 32-wide head
            # is not attributable to what the pooling encodes.
            "capacity_confounded": bool(wa != wb),
            "complete_n5": bool(len(seeds) == 5)}


def _load_synth(d=SYNTH_DIR):
    """(regime, strength, head) -> {seed: macro_f1}, same key layout as
    summarize_gateB_v2._load so the two stay comparable."""
    out = defaultdict(dict)
    for path in glob.glob(os.path.join(d, "*.json")):
        parts = os.path.basename(path)[:-5].split("__")
        if len(parts) != 4:
            continue
        regime, s, head, seed = parts
        rec = json.load(open(path))
        key = next((k for k in ("test_f1", "test_macro_f1", "macro_f1")
                    if k in rec), None)
        if key is None:
            continue
        out[(regime, float(s[1:]), head)][int(seed[4:])] = float(rec[key])
    return out


def _verdict(c, direction):
    """PASS / FAIL / UNRESOLVED against a pre-registered criterion.

    direction 'excludes_zero_positive': hypothesis predicts a positive interval
    excluding zero. direction 'covers_zero': hypothesis predicts no effect.
    UNRESOLVED whenever the cell is not a complete n=5.
    """
    if c is None:
        return "MISSING"
    if not c["complete_n5"]:
        return f"UNRESOLVED (n={len(c['seeds'])}, need 5)"
    excl = c["t_ci_excludes_zero"]
    if direction == "excludes_zero_positive":
        return "PASS" if (excl and c["delta_mean"] > 0) else "FAIL"
    if direction == "covers_zero":
        return "PASS" if not excl else "FAIL"
    return "REPORTED"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Summarise the GMP control")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--synth-dir", default=SYNTH_DIR,
                    help="synthetic run to read. Only a directory whose cells "
                         "carry an n_samples key was written by the current "
                         "script; results/gateB_synthetic_v2 does not and its "
                         "protocol is not established")
    args = ap.parse_args(argv)

    summary = {"pre_registration": "notes/20260811-decision-maxpool-control.md",
               "seeds": SEEDS, "dz_gate": DZ_GATE,
               "head_widths": {h: head_width(h)
                               for h in ("gap", "se", "gcp", "gmp", "gmp_wide")},
               "params": {h: int(count_params(NIDSModel(h, 2, in_features=39)))
                          for h in ("gap", "gcp", "gmp", "gmp_wide")},
               "real": {}, "synthetic": {}, "verdicts": {},
               "drift_check": None}

    # ---------------- real data ----------------
    print("REAL DATA (in-distribution, binary, n=5 seeds)\n")
    hdr = (f"{'dataset':22} {'contrast':16} {'delta':>9} {'t(df=4) 95% CI':>21} "
           f"{'d_z':>6} {'+/n':>5} {'cap?':>5}")
    print(hdr); print("-" * len(hdr))
    for ds in SOURCES:
        arms, prov = {}, {}
        for head in ("gap", "se", "gcp", "gmp", "gmp_wide"):
            v, p = _read_arm(ds, head)
            if v:
                arms[head], prov[head] = v, p
        entry = {"provenance": prov,
                 "per_head": {h: {"per_seed": {str(k): v for k, v in vals.items()},
                                  "mean": float(np.mean(list(vals.values()))),
                                  "t_ci": list(t_ci(list(vals.values()))[1:])}
                              for h, vals in arms.items()},
                 "contrasts": []}
        for a, b in CONTRASTS:
            if a not in arms or b not in arms:
                continue
            c = _contrast(a, arms[a], b, arms[b])
            if c is None:
                continue
            c["source"] = {a: prov.get(a), b: prov.get(b)}
            # A contrast whose arms came from different training paths is
            # comparable only as far as those paths are equivalent. Say so.
            c["cross_path"] = bool(prov.get(a, {}) and prov.get(b, {}) and
                                   prov[a]["script"] != prov[b]["script"])
            entry["contrasts"].append(c)
            print(f"{ds:22} {c['contrast']:16} {c['delta_mean']:+9.4f} "
                  f"[{c['delta_t_ci'][0]:+.4f},{c['delta_t_ci'][1]:+.4f}] "
                  f"{(c['d_z'] if c['d_z'] is not None else float('nan')):6.2f} "
                  f"{c['n_seeds_positive']:d}/{len(c['seeds']):d} "
                  f"{'YES' if c['capacity_confounded'] else '-':>5}"
                  # An incomplete cell must not read as a finished row: the
                  # interval is real but the protocol says n=5, so it carries no
                  # verdict.
                  f"{'' if c['complete_n5'] else '   INCOMPLETE, no verdict'}")
        summary["real"][ds] = entry

    # ---------------- synthetic ----------------
    print("\nSYNTHETIC (regime x strength, n=5 seeds)\n")
    cells = _load_synth(args.synth_dir)
    n_samples = next((json.load(open(p)).get("n_samples")
                      for p in glob.glob(os.path.join(args.synth_dir, "*.json"))
                      if "summary" not in p and "analysis" not in p), None)
    summary["synthetic"] = {"dir": args.synth_dir, "n_samples": n_samples,
                            "current_script_provenance": n_samples is not None,
                            "contrasts": []}
    if n_samples is None:
        print(f"WARNING: cells in {args.synth_dir} carry no n_samples key, so "
              f"they predate the current script. Contrasts below pair today's "
              f"code against an unestablished protocol.")
    hdr = (f"{'regime':6} {'str':>5} {'contrast':12} {'delta':>9} "
           f"{'t(df=4) 95% CI':>21} {'d_z':>6} {'+/n':>5}")
    print(hdr); print("-" * len(hdr))
    for regime in ("mean", "cov"):
        for strength in (0.0, 0.25, 0.5, 0.75, 1.0):
            for a, b in (("gmp", "gap"), ("gcp", "gmp"), ("gcp", "gap")):
                av = cells.get((regime, strength, a), {})
                bv = cells.get((regime, strength, b), {})
                if not av or not bv:
                    continue
                c = _contrast(a, av, b, bv)
                if c is None:
                    continue
                c.update({"regime": regime, "strength": strength})
                summary["synthetic"]["contrasts"].append(c)
                print(f"{regime:6} {strength:5.2f} {c['contrast']:12} "
                      f"{c['delta_mean']:+9.4f} "
                      f"[{c['delta_t_ci'][0]:+.4f},{c['delta_t_ci'][1]:+.4f}] "
                      f"{(c['d_z'] if c['d_z'] is not None else float('nan')):6.2f} "
                      f"{c['n_seeds_positive']:d}/{len(c['seeds']):d}"
                      f"{'' if c['complete_n5'] else '   INCOMPLETE, no verdict'}")

    # ---------------- drift check ----------------
    local = os.path.join(OUT_DIR, "NF-UNSW-NB15-v2__gap__binary__seed42.json")
    pub = "results/repro/NF-UNSW-NB15-v2__gap__binary__seed42.json"
    if os.path.exists(local) and os.path.exists(pub):
        a = float(json.load(open(local))["macro_f1"])
        b = float(json.load(open(pub))["macro_f1"])
        summary["drift_check"] = {"local": local, "published": pub,
                                  "local_macro_f1": a, "published_macro_f1": b,
                                  "bit_identical": bool(a == b),
                                  "abs_diff": abs(a - b)}
        print(f"\ndrift check GAP UNSW seed42: bit-identical={a == b} "
              f"|diff|={abs(a - b):.2e}")
    else:
        print("\ndrift check: not run (retrained GAP seed-42 cell absent). "
              "Every delta above is reported without evidence that the "
              "published baselines still reproduce on this toolchain.")

    # ---------------- pre-registered verdicts ----------------
    def _find(ds, name):
        return next((c for c in summary["real"].get(ds, {}).get("contrasts", [])
                     if c["contrast"] == name), None)

    def _find_syn(regime, strength, name):
        return next((c for c in summary["synthetic"]["contrasts"]
                     if c["regime"] == regime and c["strength"] == strength
                     and c["contrast"] == name), None)

    v = {
        "H-MAX-1 (UNSW gmp-gap covers zero)":
            _verdict(_find("NF-UNSW-NB15-v2", "gmp - gap"), "covers_zero"),
        "H-MAX-2 (synthetic cov s=1.0 gcp-gmp positive)":
            _verdict(_find_syn("cov", 1.0, "gcp - gmp"), "excludes_zero_positive"),
        "H-MAX-2b (synthetic cov s=1.0 gmp-gap, two-sided, reported only)":
            _verdict(_find_syn("cov", 1.0, "gmp - gap"), "report"),
        "H-MAX-3 (UNSW gcp-gmp_wide positive, capacity-clean)":
            _verdict(_find("NF-UNSW-NB15-v2", "gcp - gmp_wide"),
                     "excludes_zero_positive"),
    }
    summary["verdicts"] = v
    print("\nPRE-REGISTERED VERDICTS")
    for k, res in v.items():
        print(f"  {res:28} {k}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(summary, fh, indent=2)
    os.replace(tmp, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
