"""
Pipeline orchestrator — runs all steps in the correct order.

Usage:
    python run_pipeline.py            # full pipeline
    python run_pipeline.py --predict  # prediction only (skips training)
"""

import subprocess
import sys
from pathlib import Path

VENV_PYTHON = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def run_step(name: str, script: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  STEP: {name}")
    print(f"{'=' * 55}")
    result = subprocess.run([PYTHON, script], check=False)
    if result.returncode != 0:
        print(f"\n[PIPELINE ABORTED] Step '{name}' failed (exit code {result.returncode}).")
        sys.exit(result.returncode)


def main() -> None:
    predict_only = "--predict" in sys.argv

    if predict_only:
        steps = [
            ("Daily Risk Prediction", "predict_daily_risk.py"),
        ]
    else:
        steps = [
            ("Fetch Historical Weather Data", "fetch_data.py"),
            ("Generate Accident Labels",      "generate_labels.py"),
            ("Train and Evaluate Model",      "train_model.py"),
            ("Daily Risk Prediction",         "predict_daily_risk.py"),
        ]

    for step_name, script in steps:
        if not Path(script).exists():
            print(f"[ERROR] Script not found: {script}")
            sys.exit(1)
        run_step(step_name, script)

    print("\n" + "=" * 55)
    print("  Pipeline completed successfully.")
    print("=" * 55)


if __name__ == "__main__":
    main()
