from __future__ import annotations

import pandas as pd
from scipy import stats


def summarize(final_runs: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for (week_id, week_label), week_frame in final_runs.groupby(["week_id", "week_label"], sort=False):
        pivot = week_frame.pivot(index="run_index", columns="method", values="objective")
        for method in method_order:
            values = pivot[method]
            row = {
                "week_id": week_id,
                "week_label": week_label,
                "method": method,
                "obj": values.mean(),
                "sd": values.std(ddof=1),
                "n": values.size,
                "p_vs_GZO_NS": 0.0 if method == "GZO_NS" else stats.ttest_rel(pivot["GZO_NS"], values).pvalue,
                "p_vs_GZO_HS": 0.0 if method == "GZO_HS" else stats.ttest_rel(pivot["GZO_HS"], values).pvalue,
            }
            rows.append(row)
    return pd.DataFrame(rows)
