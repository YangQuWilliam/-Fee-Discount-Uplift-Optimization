from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is on sys.path for "src" imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_bank_marketing_df


def main() -> None:
    df = load_bank_marketing_df()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)

    print("=== Columns ===")
    print(list(df.columns))
    print("\n=== Head ===")
    print(df.head(5))
    print("\n=== Info ===")
    print(df.info())


if __name__ == "__main__":
    main()
