from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_latex(summary: pd.DataFrame, method_order: list[str], path: Path) -> None:
    weeks = summary[["week_id", "week_label"]].drop_duplicates().itertuples(index=False)
    lines = [
        r"\begin{tabular}{l" + "rr" * len(method_order) + "}",
        r"\toprule",
        "date & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{m.replace('_', '-')}}}" for m in method_order) + r" \\",
        " & " + " & ".join(["obj & sd"] * len(method_order)) + r" \\",
        r"\midrule",
    ]
    for week in weeks:
        frame = summary[summary["week_id"] == week.week_id].set_index("method")
        best = frame["obj"].min()
        cells = [week.week_label]
        for method in method_order:
            obj = float(frame.loc[method, "obj"])
            sd = float(frame.loc[method, "sd"])
            obj_text = f"{obj:.2f}"
            if abs(obj - best) < 5e-12:
                obj_text = rf"\textbf{{{obj_text}}}"
            cells.extend([obj_text, f"{sd:.2f}"])
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
