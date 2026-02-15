"""Lightweight progress monitor for preregistered scan folders.

Usage (PowerShell):
  & .\.venv\Scripts\python.exe tools\scan_progress.py out_phase3B_preregistered_uniqueness_64_127_v5

This only inspects emitted per-job rows.csv files; it does not touch the
Phase-3B measurement spine.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python tools/scan_progress.py <out_root>")
        return 2

    out_root = Path(sys.argv[1])
    if not out_root.exists():
        print(f"ERROR: not found: {out_root}")
        return 2

    rows = list(out_root.rglob("rows.csv"))

    by_backend = Counter()
    by_backend_N = Counter()
    newest = None
    newest_mtime = -1.0

    for p in rows:
        # Expected path fragment: out_root/backend_<label>/N<...>/seed.../rows.csv
        try:
            rel = p.relative_to(out_root)
            parts = rel.parts
            backend = parts[0]
            n_dir = parts[1] if len(parts) > 1 else "?"
        except Exception:
            backend, n_dir = "?", "?"

        by_backend[backend] += 1
        by_backend_N[(backend, n_dir)] += 1

        try:
            mt = p.stat().st_mtime
        except OSError:
            continue
        if mt > newest_mtime:
            newest_mtime = mt
            newest = p

    print(f"out_root: {out_root}")
    print(f"rows.csv found: {len(rows)}")

    if by_backend:
        print("\nBy backend:")
        for backend, n in sorted(by_backend.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {backend}: {n}")

    # Group by N within backend
    grouped = defaultdict(list)
    for (backend, n_dir), n in by_backend_N.items():
        grouped[backend].append((n_dir, n))

    if grouped:
        print("\nBy backend / N:")
        for backend in sorted(grouped.keys()):
            items = ", ".join(f"{n_dir}={n}" for n_dir, n in sorted(grouped[backend]))
            print(f"  {backend}: {items}")

    if newest is not None:
        print(f"\nNewest: {newest.as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
