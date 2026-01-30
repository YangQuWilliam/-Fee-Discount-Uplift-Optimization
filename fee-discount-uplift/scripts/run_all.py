from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_script(path: Path) -> None:
    print(f"\n=== Running {path.name} ===")
    subprocess.run([sys.executable, str(path)], check=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"

    for name in ["run_baselines.py", "run_uplift_v1.py", "run_uplift_v2.py"]:
        run_script(scripts_dir / name)


if __name__ == "__main__":
    main()
