from pathlib import Path

from ddp_pricing.runner import run_experiment


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    run_experiment(project_root / "configs" / "reference.yaml", project_root / "outputs" / "reference")
