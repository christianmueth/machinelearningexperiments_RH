from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml_oracle.frozen_oracle_client import AnchoredOracleClient


def _parse_float_csv(value: str) -> list[float]:
    out: list[float] = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export a stable anchored-pipeline oracle feature table for the canonical A3 frontend and anchored completion artifacts."
    )
    ap.add_argument("--us", default="0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24")
    ap.add_argument("--out_csv", default="out/anchored_oracle_feature_table.csv")
    args = ap.parse_args()

    client = AnchoredOracleClient()
    table = client.export_feature_table(us=_parse_float_csv(args.us))
    out_path = Path(str(args.out_csv))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())