from pathlib import Path

import pandas as pd

from ddp_pricing.runner import run_experiment


def test_quick_run_creates_reports(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    outputs = run_experiment(root / "configs" / "quick.yaml", tmp_path)
    for path in outputs.values():
        assert path.exists()
    summary = pd.read_csv(outputs["summary"])
    assert len(summary) == 2 * 5
    assert set(summary["method"]) == {"GZO_NS", "GZO_HS", "ZO_TG", "ZO_OG", "ZO_OGVR"}
