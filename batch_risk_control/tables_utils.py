"""Shared table-formatting utility: every paper table is saved as both a
CSV (for re-use/inspection) and a booktabs-style .tex fragment (for direct
\\input{} into the manuscript) from the same DataFrame, so there is exactly
one source of truth per table.
"""
import re
from pathlib import Path

_MATH_SPAN = re.compile(r"(\$[^$]*\$)")


def _escape_header(s):
    """Escape underscores for LaTeX text mode, but leave $...$ math spans
    untouched -- headers like "Final $\\alpha_t$" rely on that underscore
    as a real subscript, not literal text.
    """
    parts = _MATH_SPAN.split(str(s))
    return "".join(p if p.startswith("$") else p.replace("_", r"\_") for p in parts)


def save_table(df, path_stem, caption, label, float_format="{:.3f}", index=False,
               column_format=None, notes=None):
    """df: the table content, already in display order/column names.
    path_stem: path without extension (both .csv and .tex are written).
    caption/label: LaTeX table caption and \\label{} key.
    float_format: applied to all float columns.
    notes: optional string rendered as a footnote line below the table.
    """
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(f"{path_stem}.csv", index=index)

    n_cols = len(df.columns) + (1 if index else 0)
    if column_format is None:
        column_format = "l" + "r" * (n_cols - 1)

    def fmt(v):
        if isinstance(v, float):
            return float_format.format(v)
        return str(v).replace("_", r"\_").replace("%", r"\%")

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{column_format}}}")
    lines.append(r"\toprule")

    headers = ([df.index.name or ""] if index else []) + [_escape_header(c) for c in df.columns]
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    for idx, row in df.iterrows():
        cells = ([str(idx).replace("_", r"\_")] if index else []) + [fmt(v) for v in row]
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if notes:
        lines.append(rf"\begin{{tablenotes}}\small\item {notes}\end{{tablenotes}}")
    lines.append(r"\end{table}")

    with open(f"{path_stem}.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
