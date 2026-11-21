#!/usr/bin/env python3
"""Re-typeset the three result tables of iccit/CameraReady.tex without touching a digit.

It parses every number out of the existing table environments and writes the same
numbers into a cleaner layout: grouped column headers, one aligned column each for
the paired difference, its 95% interval and d_z, a true minus sign, backbone panel
headings instead of "(TCN)" row suffixes, and full-width rules shared by the two
blocks of Table III. Layout only: the multiset of numbers is checked by
scripts/check_iccit_table_numbers.py.

Usage (repository root):  python3 scripts/retypeset_iccit_tables.py iccit/CameraReady.tex
The file is rewritten in place; keep a backup first.
"""
import re
import sys

MN = r"\mn{}"


def sgn(s):
    return "+" if s == "+" else MN


def num(sign_and_digits):
    """'+0.0020' or '-0.0025' (or 'x' already signed) -> typeset with text minus."""
    m = re.fullmatch(r"([+-])([0-9.]+)", sign_and_digits)
    assert m, sign_and_digits
    return sgn(m.group(1)) + m.group(2)


CONTRAST = re.compile(
    r"\$([+-])([0-9.]+)\$ \[([+-][0-9.]+), ([+-][0-9.]+)\] \(([+-][0-9.]+)\)")


def contrast_cells(cell):
    m = CONTRAST.fullmatch(cell.strip())
    assert m, cell
    d = sgn(m.group(1)) + m.group(2)
    return d, f"[{num(m.group(3))}, {num(m.group(4))}]", num(m.group(5))


def split_rows(body):
    rows = []
    for line in body.split("\\\\"):
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        rows.append(line)
    return rows


def block(tex, label):
    """Return (start, end, text) of the table* environment whose \\label is `label`."""
    i = tex.index(r"\label{%s}" % label)
    s = tex.rfind(r"\begin{table*}", 0, i)
    e = tex.index(r"\end{table*}", i) + len(r"\end{table*}")
    return s, e, tex[s:e]


# ---------------------------------------------------------------- Table I
def new_table_indist(old):
    body = old[old.index(r"\midrule") + len(r"\midrule"):old.index(r"\bottomrule")]
    out = []
    for r in split_rows(body):
        cells = [c.strip() for c in r.split("&")]
        name = cells[0]
        pm = []
        for c in cells[1:6]:
            m = re.fullmatch(r"(\\textbf\{[0-9.]+\}|[0-9.]+) \$\\pm\$ ([0-9.]+)", c)
            assert m, c
            pm.append((m.group(1), m.group(2)))
        d = cells[6].strip("$")
        dz = cells[7].strip("$")
        holm = cells[8]
        d = sgn(d[0]) + d[1:]
        dz = sgn(dz[0]) + dz[1:]
        row = name + " & " + " & ".join(f"{a} $\\pm$ {b}" for a, b in pm)
        row += f" & {d} & {dz} & {holm}"
        out.append(row)
    cols = "@{\\extracolsep{\\fill}}l" + "cccc" + "@{\\hspace{14pt}}c" + "@{\\hspace{14pt}}ccc"
    # spacing between groups handled by \tabcolsep plus phantom gaps below
    s = []
    s.append(r"\begin{table*}[t]")
    s.append(r"  \centering")
    s.append(r"  \color{blue}")
    s.append(r"  \scriptsize")
    s.append(r"  \renewcommand{\arraystretch}{1.0}")
    s.append(r"  \setlength{\tabcolsep}{4pt}")
    s.append(r"  \caption{In-distribution test macro-F1, mean $\pm$ SD over five seeds.}")
    s.append(r"  \label{tab:indist}")
    s.append(r"  \begin{tabular*}{\textwidth}{" + cols + "}")
    s.append(r"    \toprule")
    s.append(r"    & \multicolumn{4}{c}{Pooling head} & Tree "
             r"& \multicolumn{3}{c}{Paired GCP$-$GMP} \\")
    s.append(r"    \cmidrule(lr){2-5}\cmidrule(lr){6-6}\cmidrule(l){7-9}")
    s.append(r"    Dataset & GAP & GMP & SE & GCP & HGB & $\Delta$ & $d_z$ & Holm \\")
    s.append(r"    \midrule")
    for r in out:
        s.append("    " + r + r" \\")
    s.append(r"    \bottomrule")
    s.append(r"  \end{tabular*}")
    s.append(r"\end{table*}")
    return "\n".join(s)


# ---------------------------------------------------------------- Table II
PANELS = [("", "CNN"), (" (TCN)", "TCN"), (" (MLP)", "MLP")]


def new_table_contrasts(old):
    body = old[old.index(r"\midrule") + len(r"\midrule"):old.index(r"\bottomrule")]
    panels = {p[0]: [] for p in PANELS}
    for chunk in body.split(r"\midrule"):
        for r in split_rows(chunk):
            cells = [c.strip() for c in r.split("&")]
            name = cells[0]
            suf = ""
            m = re.fullmatch(r"(.*?)( \((?:TCN|MLP)\))", name)
            if m:
                name, suf = m.group(1), m.group(2)
            parts = []
            for c in cells[1:]:
                parts.extend(contrast_cells(c))
            panels[suf].append(name + " & " + " & ".join(parts))
    colgroup = r"@{\extracolsep{\fill}\hspace{7pt}}r@{\extracolsep{0pt}\hspace{3pt}}c@{\extracolsep{0pt}\hspace{3pt}}r"
    cols = "c@{\\extracolsep{0pt}\\hspace{5pt}}l" + colgroup * 4
    s = [r"\begin{table*}[t]", r"  \centering", r"  \color{blue}", r"  \scriptsize",
         r"  \renewcommand{\arraystretch}{0.92}", r"  \setlength{\tabcolsep}{0pt}",
         r"  \caption{Paired in-distribution contrasts: mean difference, 95\% interval, $d_z$.}",
         r"  \label{tab:contrasts}",
         r"  \begin{tabular*}{\textwidth}{" + cols + "}", r"    \toprule",
         r"    Backbone & Dataset & \multicolumn{3}{c}{GCP$-$GAP} & \multicolumn{3}{c}{GCP$-$SE} "
         r"& \multicolumn{3}{c}{GCP$-$GMP} & \multicolumn{3}{c}{GMP$-$GAP} \\",
         r"    \cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(l){12-14}"]
    for i, (suf, title) in enumerate(PANELS):
        if i:
            s.append(r"    \midrule")
        for j, r in enumerate(panels[suf]):
            lab = (r"\multirow{4}{*}{\rotatebox[origin=c]{90}{" + title + "}}") if j == 0 else ""
            s.append("    " + lab + " & " + r + r" \\")
    s += [r"    \bottomrule", r"  \end{tabular*}", r"\end{table*}"]
    return "\n".join(s)


# --------------------------------------------------------------- Table III
def new_table_loosynth(old):
    parts = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", old, flags=re.S)
    assert len(parts) == 2
    top_body = parts[0][parts[0].index(r"\midrule") + 8:parts[0].index(r"\bottomrule")]
    bot_body = parts[1][parts[1].index(r"\midrule") + 8:parts[1].index(r"\bottomrule")]
    top = []
    for r in split_rows(top_body):
        c = [x.strip() for x in r.split("&")]
        row = c[0] + " & " + " & ".join(c[1:5])
        for k in (5, 6):
            row += " & " + " & ".join(contrast_cells(c[k]))
        top.append(row)
    bot = []
    for r in split_rows(bot_body):
        c = [x.strip() for x in r.split("&")]
        # Strength, GAP, GCP-GAP, GCP-GMP, GAP, GCP-GAP, GCP-GMP
        row = c[0] + " & " + c[1] + " & " + " & ".join(contrast_cells(c[2])) \
            + " & " + " & ".join(contrast_cells(c[3])) + " & " + c[4] + " & " \
            + " & ".join(contrast_cells(c[5])) + " & " + " & ".join(contrast_cells(c[6]))
        bot.append(row)
    g = r"@{\extracolsep{\fill}\hspace{7pt}}r@{\extracolsep{0pt}\hspace{3pt}}c@{\extracolsep{0pt}\hspace{3pt}}r"
    f = r"@{\extracolsep{\fill}\hspace{8pt}}r"
    top_cols = "l" + f * 4 + g * 2
    bot_cols = "l" + f + g * 2 + "@{\\extracolsep{\\fill}\\hspace{14pt}}r" + g * 2
    s = [r"\begin{table*}[t]", r"  \centering", r"  \color{blue}", r"  \scriptsize",
         r"  \renewcommand{\arraystretch}{0.92}", r"  \setlength{\tabcolsep}{0pt}",
         r"  \caption{LOO (top) and synthetic (bottom) macro-F1, paired difference, 95\% interval, $d_z$.}",
         r"  \label{tab:loosynth}",
         r"  \begin{tabular*}{\textwidth}{" + top_cols + "}", r"    \toprule",
         r"    Held-out & GAP & SE & GCP & GMP & \multicolumn{3}{c}{GCP$-$GAP} & \multicolumn{3}{c}{GMP$-$GAP} \\",
         r"    \cmidrule(lr){6-8}\cmidrule(l){9-11}"]
    s += ["    " + r + r" \\" for r in top]
    s += [r"    \bottomrule", r"  \end{tabular*}", "", r"  \vspace{1pt}", "",
          r"  \begin{tabular*}{\textwidth}{" + bot_cols + "}", r"    \toprule",
          r"    & \multicolumn{7}{c}{Original covariance regime} & \multicolumn{7}{c}{Far-pair regime (leak-free)} \\",
          r"    \cmidrule(lr){2-8}\cmidrule(l){9-15}",
          r"    Strength & GAP & \multicolumn{3}{c}{GCP$-$GAP} & \multicolumn{3}{c}{GCP$-$GMP}"
          r" & GAP & \multicolumn{3}{c}{GCP$-$GAP} & \multicolumn{3}{c}{GCP$-$GMP} \\",
          r"    \cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){10-12}\cmidrule(l){13-15}"]
    s += ["    " + r + r" \\" for r in bot]
    s += [r"    \bottomrule", r"  \end{tabular*}", r"\end{table*}"]
    return "\n".join(s)


def main(path):
    tex = open(path, newline="").read()
    for label, fn in (("tab:indist", new_table_indist),
                      ("tab:contrasts", new_table_contrasts),
                      ("tab:loosynth", new_table_loosynth)):
        a, b, old = block(tex, label)
        tex = tex[:a] + fn(old) + tex[b:]
    if r"\usepackage{multirow}" not in tex:
        tex = tex.replace(r"\usepackage{booktabs}", r"\usepackage{booktabs}" + "\n" + r"\usepackage{multirow}", 1)
    if r"\newcommand{\mn}" not in tex:
        tex = tex.replace(r"\begin{document}",
                          r"% true minus for the tables, same width as the plus sign in Times" + "\n"
                          r"\newcommand{\mn}{\textminus}" + "\n" + r"\begin{document}", 1)
    open(path, "w", newline="").write(tex)
    print("rewrote tables in", path)


if __name__ == "__main__":
    main(sys.argv[1])
