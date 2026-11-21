#!/usr/bin/env python3
"""Check that re-typesetting the three ICCIT tables changed no digit.

Compares the ordered list of decimal numbers inside the table environments of two
CameraReady.tex files (before and after scripts/retypeset_iccit_tables.py). The text
minus sign (\\mn{}) and the math minus are both read as '-'. Exit code 1 on any
difference.

Usage: python3 scripts/check_iccit_table_numbers.py OLD.tex NEW.tex
"""
import re
import sys

NUM = re.compile(r"[+-]?\d+\.\d+")


def table_numbers(path):
    tex = open(path, encoding="utf-8").read()
    out = {}
    for label in ("tab:indist", "tab:contrasts", "tab:loosynth"):
        i = tex.index(r"\label{%s}" % label)
        s = tex.rfind(r"\begin{table*}", 0, i)
        e = tex.index(r"\end{table*}", i)
        env = tex[s:e]
        # data only: drop the caption (its prose may legitimately change)
        env = re.sub(r"\\caption\{.*?\}\n", "", env, flags=re.S)
        env = re.sub(r"\\renewcommand\{\\arraystretch\}\{[0-9.]+\}", "", env)  # layout, not data
        env = env.replace(r"\mn{}", "-").replace("$", "")
        out[label] = NUM.findall(env)  # file order, so a swapped cell is caught
    return out


old, new = table_numbers(sys.argv[1]), table_numbers(sys.argv[2])
bad = False
for k in old:
    same = old[k] == new[k]
    print(f"{k:14s} numbers old={len(old[k]):4d} new={len(new[k]):4d}  "
          f"{'IDENTICAL, SAME ORDER' if same else 'DIFFERENT'}")
    if not same:
        bad = True
        i = next((j for j, (a, b) in enumerate(zip(old[k], new[k])) if a != b), min(len(old[k]), len(new[k])))
        print("  first difference at number", i, "old:", old[k][i:i + 3], "new:", new[k][i:i + 3])
sys.exit(1 if bad else 0)
