#!/usr/bin/env python3
"""Recompute every published interval under Student-t (df = n-1).

Defect 2.2 of notes/20260720-bug-prepublication-code-audit.md: `bootstrap_ci`
is implemented correctly but is the wrong estimator at n=5. A percentile
bootstrap of a mean cannot place an endpoint outside the observed [min, max],
so its coverage is 0.839 against a nominal 0.95. The paper's inferential
standard is "the interval excludes zero", so the real false-positive rate is
near 16 percent.

This script is READ-ONLY with respect to the published record. For every
results/**/*summary*.json it writes a sibling `*_tci.json`; nothing existing is
overwritten, and `src.metrics.bootstrap_ci` is left exactly as it was.

Every interval is rebuilt from the same per-seed values the original summariser
used, and the bootstrap interval is recomputed from those same values and
checked against the published number. A PROVENANCE MISMATCH line means the raw
source was resolved wrongly and the row must not be trusted.

Analysis-only: no GPU, no training, no checkpoint reads.

Usage:  .venv/bin/python scripts/recompute_t_intervals.py
"""
from __future__ import annotations

import copy
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import bootstrap_ci, t_ci  # noqa: E402

SEEDS = [42, 123, 456, 789, 1337]
UNSW = "NF-UNSW-NB15-v2"
PROV_TOL = 1e-9

ROWS = []          # comparison-table rows
MISMATCHES = []    # provenance failures


def _excl(lo, hi):
    return (lo > 0) or (hi < 0)


def record(source, label, values, old_ci):
    """Compare published bootstrap interval against t, and log the row.

    `old_ci` is (lo, hi) as published, or None when no interval was published
    for this quantity (then the t interval is an addition, not a correction).
    """
    mean, t_lo, t_hi = t_ci(values)
    b_mean, b_lo, b_hi = bootstrap_ci(values)
    if old_ci is not None:
        if (abs(b_lo - old_ci[0]) > PROV_TOL or abs(b_hi - old_ci[1]) > PROV_TOL):
            MISMATCHES.append(
                f"{source} :: {label} :: published [{old_ci[0]:.6g}, "
                f"{old_ci[1]:.6g}] != bootstrap from resolved raw values "
                f"[{b_lo:.6g}, {b_hi:.6g}]")
    ROWS.append({
        "source": source, "label": label, "n": len(values), "mean": mean,
        "boot_lo": b_lo if old_ci is None else old_ci[0],
        "boot_hi": b_hi if old_ci is None else old_ci[1],
        "boot_published": old_ci is not None,
        "t_lo": t_lo, "t_hi": t_hi,
        "boot_excludes_zero": _excl(b_lo if old_ci is None else old_ci[0],
                                    b_hi if old_ci is None else old_ci[1]),
        "t_excludes_zero": _excl(t_lo, t_hi),
    })
    return [mean, t_lo, t_hi]


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def write_tci(src_path, payload, extra_note=""):
    out_path = src_path.replace(".json", "_tci.json")
    assert not os.path.exists(src_path.replace(".json", "_tci_tci.json"))
    payload = dict(payload)
    payload["_interval_method"] = (
        "Student-t, df = n-1, two-sided 95 percent, sd with ddof=1")
    payload["_supersedes"] = src_path
    payload["_generated_by"] = "scripts/recompute_t_intervals.py"
    payload["_reason"] = (
        "defect 2.2, notes/20260720-bug-prepublication-code-audit.md: "
        "percentile bootstrap of a mean at n=5 has 0.839 coverage")
    if extra_note:
        payload["_note"] = extra_note
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  wrote {out_path}")
    return out_path


# --------------------------------------------------------------------------
# per-summary handlers
# --------------------------------------------------------------------------

def do_gap1():
    src = "results/gap1/gap1_summary.json"
    from scripts.summarize_gap1 import (CELL_GLOB, METRICS, DELTA_METRICS,
                                        index_cells, single_series,
                                        paired_series)
    rows = [load_json(p) for p in
            sorted(glob.glob(os.path.join("results/gap1", CELL_GLOB)))]
    by = index_cells(rows)
    old = load_json(src)
    out = {}
    for test_ds, heads in by.items():
        block = {"n_cells": {h: len(c) for h, c in heads.items()}}
        o = old.get(test_ds, {})
        for metric in METRICS:
            for h in heads:
                seeds, vals = single_series(heads[h], metric)
                if not vals:
                    continue
                key = f"{h}_{metric}_ci"
                pub = o.get(key)
                block[key] = record(src, f"{test_ds} {h} {metric}", vals,
                                    (pub[1], pub[2]) if pub else None)
                block[f"{h}_{metric}_n"] = len(vals)
                block[f"{h}_{metric}_seeds"] = seeds
        if "gap" in heads and "gcp" in heads:
            for metric in DELTA_METRICS:
                seeds, gap, gcp = paired_series(heads["gap"], heads["gcp"],
                                                metric)
                if not seeds:
                    continue
                delta = [g - a for g, a in zip(gcp, gap)]
                key = f"delta_gcp_minus_gap_{metric}_ci"
                pub = o.get(key)
                block[key] = record(src, f"{test_ds} delta gcp-gap {metric}",
                                    delta, (pub[1], pub[2]) if pub else None)
                block[f"delta_gcp_minus_gap_{metric}_n"] = len(seeds)
                block[f"delta_gcp_minus_gap_{metric}_seeds"] = seeds
                block[f"delta_gcp_minus_gap_{metric}_dz"] = (
                    o.get(f"delta_gcp_minus_gap_{metric}_dz"))
        out[test_ds] = block
    write_tci(src, out)


def do_gap3(src, tag):
    d = load_json(src)
    real = [p["concentration_real"] for p in d["per_seed"]]
    shuf = [p["concentration_shuffled"] for p in d["per_seed"]]
    delta = [r - s for r, s in zip(real, shuf)]
    o = d["h5"]
    out = copy.deepcopy(d)
    out["h5"]["real_mean_ci"] = record(
        src, f"{tag} H5 real concentration", real,
        (o["real_mean_ci"][1], o["real_mean_ci"][2]))
    out["h5"]["shuffled_mean_ci"] = record(
        src, f"{tag} H5 shuffled concentration", shuf,
        (o["shuffled_mean_ci"][1], o["shuffled_mean_ci"][2]))
    out["h5"]["delta_ci"] = record(
        src, f"{tag} H5 delta real-minus-shuffled", delta,
        (o["delta_ci"][1], o["delta_ci"][2]))
    lo = out["h5"]["delta_ci"][1]
    out["h5"]["supported"] = bool(lo > 0 and o["cohen_dz"] >= 0.8)
    write_tci(src, out)


def do_gap5():
    src = "results/gap5/gap5_summary.json"
    old = load_json(src)
    by = defaultdict(dict)
    for f in sorted(glob.glob("results/gap5/*__seed*.json")):
        r = load_json(f)
        by[r["backbone"]].setdefault(r["head"], {})[r["seed"]] = r["macro_f1"]
    out = {}
    for bb, heads in by.items():
        seeds = sorted(set(heads["gap"]) & set(heads["gcp"]))
        gap = [heads["gap"][s] for s in seeds]
        gcp = [heads["gcp"][s] for s in seeds]
        delta = [g - a for g, a in zip(gcp, gap)]
        o = old[bb]
        out[bb] = {
            "seeds": seeds,
            "gap_mean_ci": record(src, f"{bb} gap macro-F1", gap,
                                  (o["gap_mean_ci"][1], o["gap_mean_ci"][2])),
            "gcp_mean_ci": record(src, f"{bb} gcp macro-F1", gcp,
                                  (o["gcp_mean_ci"][1], o["gcp_mean_ci"][2])),
            "delta_mean_ci": record(src, f"{bb} delta gcp-gap", delta,
                                    (o["delta_mean_ci"][1],
                                     o["delta_mean_ci"][2])),
            "cohen_dz": o["cohen_dz"], "sign": o["sign"],
        }
    write_tci(src, out)


def do_gap6():
    src = "results/gap6/gap6_summary.json"
    old = load_json(src)
    # seed order matters: bootstrap_ci resamples a fixed-RNG index stream, so
    # the published interval only reproduces under run_gap6.py's append order
    # (c-major, seeds in SEEDS order), not under a lexical glob sort
    by = defaultdict(list)
    for f in glob.glob(f"results/gap6/{UNSW}__gcp__binary__C*__seed*.json"):
        r = load_json(f)
        by[str(r["c"])].append((SEEDS.index(r["seed"]), r["macro_f1"]))
    by = {c: [f1 for _, f1 in sorted(v)] for c, v in by.items()}
    out = copy.deepcopy(old)
    for c, f1s in sorted(by.items(), key=lambda kv: int(kv[0])):
        o = old[c]["mean_ci"]
        out[c]["mean_ci"] = record(src, f"C={c} gcp macro-F1", f1s,
                                   (o[1], o[2]))
    write_tci(src, out)


def do_gateB():
    src = "results/gateB_synthetic/gateB_summary.json"
    old = load_json(src)
    out = copy.deepcopy(old)
    for cell, o in old.items():
        signal, strength = cell.split("_")
        vals = {}
        for head in ("gap", "gcp"):
            vals[head] = [load_json(
                f"results/gateB_synthetic/{signal}__s{strength}__{head}"
                f"__seed{s}.json")["test_macro_f1"] for s in SEEDS]
        out[cell]["gap_ci"] = record(src, f"{cell} gap macro-F1", vals["gap"],
                                     tuple(o["gap_ci"]))[1:]
        out[cell]["gcp_ci"] = record(src, f"{cell} gcp macro-F1", vals["gcp"],
                                     tuple(o["gcp_ci"]))[1:]
        delta = [g - a for g, a in zip(vals["gcp"], vals["gap"])]
        out[cell]["delta_ci"] = record(src, f"{cell} delta gcp-gap [ADDED]",
                                       delta, None)[1:]
    write_tci(src, out, extra_note="delta_ci is new; the original published "
                                   "only a point delta for the contrast")


def do_gateC():
    src = "results/gateC/gateC_summary.json"
    out = copy.deepcopy(load_json(src))
    out["delta_ci"] = {}
    for ds in ("NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v2"):
        vals = {h: [load_json(f"results/gateC/{ds}__{h}__binary__seed{s}.json")
                    ["macro_f1"] for s in SEEDS] for h in ("gap", "gcp")}
        delta = [g - a for g, a in zip(vals["gcp"], vals["gap"])]
        out["delta_ci"][ds] = record(src, f"{ds} delta gcp-gap [ADDED]",
                                     delta, None)
    write_tci(src, out, extra_note="delta_ci is new; gateC_summary.json "
                                   "published point deltas with no interval")


def do_repro():
    src = "results/repro/t6b_summary.json"
    d = load_json(src)
    out = copy.deepcopy(d)
    out["gap_mean_ci"] = record(src, "UNSW gap macro-F1", d["gap_macro_f1"],
                                (d["gap_mean_ci"][1], d["gap_mean_ci"][2]))
    out["gcp_mean_ci"] = record(src, "UNSW gcp macro-F1", d["gcp_macro_f1"],
                                (d["gcp_mean_ci"][1], d["gcp_mean_ci"][2]))
    out["delta_mean_ci"] = record(src, "UNSW delta gcp-gap",
                                  d["delta_gcp_minus_gap"],
                                  (d["delta_mean_ci"][1],
                                   d["delta_mean_ci"][2]))
    write_tci(src, out)


def do_task1():
    """Correlation CIs here are not means, so a t interval for a mean does not
    apply. The per-width delta means are, and they carried no interval at all,
    so add them."""
    src = "results/task1/effrank_summary.json"
    d = load_json(src)
    out = copy.deepcopy(d)
    by = defaultdict(list)
    for r in d["per_checkpoint"]:
        by[str(r["C"])].append(r["delta"])
    for c in sorted(by, key=int):
        out["per_width"][c]["delta_ci"] = record(
            src, f"C={c} delta gcp-gap [ADDED]", by[c], None)
    for key in ("corr_pointlevel", "corr_widthmean"):
        if key in out:
            out[key]["_interval_note"] = (
                "pearson_ci / spearman_ci left as published: they are bootstrap "
                "intervals for a CORRELATION over paired points, not for a mean "
                "of n=5 seeds, so the t interval for a mean does not apply and "
                "defect 2.2 does not describe them")
    write_tci(src, out, extra_note="per_width delta_ci is new; the original "
                                   "published only delta_mean")


def do_tier1():
    src = "results/tier1/tier1_summary.json"
    d = load_json(src)
    out = copy.deepcopy(d)
    for c, o in d["per_c"].items():
        gap, gcp = o["gap"], o["gcp"]
        delta = [g - a for g, a in zip(gcp, gap)]
        out["per_c"][c]["gap_mean_ci"] = record(
            src, f"C={c} gap macro-F1", gap,
            (o["gap_mean_ci"][1], o["gap_mean_ci"][2]))
        out["per_c"][c]["gcp_mean_ci"] = record(
            src, f"C={c} gcp macro-F1", gcp,
            (o["gcp_mean_ci"][1], o["gcp_mean_ci"][2]))
        out["per_c"][c]["delta_mean_ci"] = record(
            src, f"C={c} delta gcp-gap", delta,
            (o["delta_mean_ci"][1], o["delta_mean_ci"][2]))
    write_tci(src, out)


def main():
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    print("recomputing intervals under Student-t (df = n-1)")
    do_gap1()
    do_gap3("results/gap3/gap3_summary.json", "UNSW")
    do_gap3("results/gap3_ton/gap3_summary.json", "ToN-IoT")
    do_gap5()
    do_gap6()
    do_gateB()
    do_gateC()
    do_repro()
    do_task1()
    do_tier1()

    covered = {r["source"] for r in ROWS}
    all_summaries = {p for p in glob.glob("results/**/*summary*.json",
                                          recursive=True)
                     if not p.endswith("_tci.json")}
    missing = all_summaries - covered
    print(f"\nsummary files handled: {len(covered)} / {len(all_summaries)}")
    if missing:
        print("NOT HANDLED:", sorted(missing))

    if MISMATCHES:
        print(f"\n!! {len(MISMATCHES)} PROVENANCE MISMATCHES:")
        for m in MISMATCHES:
            print("  " + m)
    else:
        print("provenance check: every published bootstrap interval reproduced "
              "exactly from the resolved per-seed values")

    flips = [r for r in ROWS
             if r["boot_published"]
             and r["boot_excludes_zero"] != r["t_excludes_zero"]]
    print(f"\nrows: {len(ROWS)}  verdict flips among published intervals: "
          f"{len(flips)}")
    for r in flips:
        print(f"  FLIP  {r['source']} :: {r['label']}: "
              f"boot [{r['boot_lo']:+.4f}, {r['boot_hi']:+.4f}] "
              f"excl={r['boot_excludes_zero']} -> "
              f"t [{r['t_lo']:+.4f}, {r['t_hi']:+.4f}] "
              f"excl={r['t_excludes_zero']}")

    with open("results/interval_comparison_rows.json", "w") as fh:
        json.dump(ROWS, fh, indent=2)
    print("\nwrote results/interval_comparison_rows.json")


if __name__ == "__main__":
    main()
