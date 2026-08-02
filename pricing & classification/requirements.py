from __future__ import annotations

import subprocess
import sys

PACKAGES = [
    "numpy==2.3.5",
    "scipy==1.17.0",
    "pandas==2.2.3",
    "matplotlib==3.10.8",
    "PyYAML==6.0.3",
    "tqdm==4.67.1",
]

subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *PACKAGES])
print("Dependencies installed.")
