import argparse
import os
import shutil
import sys
from typing import Dict, Iterable, Tuple

import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def find_npz_files(src_root: str) -> list[str]:
    hits: list[str] = []
    for root, _dirs, files in os.walk(src_root):
        for name in files:
            if name.startswith("channel_diag") and name.endswith(".npz"):
                hits.append(os.path.join(root, name))
    return sorted(hits)


def parse_rel_metadata(path: str, src_root: str) -> Tuple[str, str, int, int, float, float]:
    rel = os.path.relpath(path, src_root).replace("\\", "/")
    parts = rel.split("/")
    backend_label = parts[0].replace("backend_", "") if parts else ""
    seed = -1
    anchor = -1
    wlo = float("nan")
    whi = float("nan")
    for part in parts:
        if part.startswith("seed") and "_anchor" in part:
            seed_part, anchor_part = part.split("_anchor")
            seed = int(seed_part.replace("seed", ""))
            anchor = int(anchor_part)
        if part.startswith("window_"):
            window_part = part.replace("window_", "")
            wlo_str, whi_str = window_part.split("_", 1)
            wlo = float(wlo_str)
            whi = float(whi_str)
    return rel, backend_label, seed, anchor, wlo, whi


def key_for_pair(seed: int, anchor: int, wlo: float, whi: float) -> Tuple[int, int, float, float]:
    return int(seed), int(anchor), float(wlo), float(whi)


def load_pairs(pairs_csv: str) -> Iterable[Tuple[int, int, float, float]]:
    df = pd.read_csv(pairs_csv)
    needed = {"seed", "anchor_seed", "wlo", "whi"}
    missing = needed.difference(set(df.columns))
    if missing:
        raise SystemExit(f"pairs_csv missing columns: {sorted(missing)}")
    pairs = (
        df[["seed", "anchor_seed", "wlo", "whi"]]
        .drop_duplicates()
        .sort_values(["seed", "anchor_seed", "wlo", "whi"], ascending=True)
    )
    for _, r in pairs.iterrows():
        yield (int(r["seed"]), int(r["anchor_seed"]), float(r["wlo"]), float(r["whi"]))


def build_map(src_root: str) -> Dict[Tuple[int, int, float, float], str]:
    files = find_npz_files(src_root)
    if not files:
        raise SystemExit(f"No channel_diag*.npz under {src_root}")
    out: Dict[Tuple[int, int, float, float], str] = {}
    for p in files:
        _rel, _backend, seed, anchor, wlo, whi = parse_rel_metadata(p, src_root)
        out[key_for_pair(seed, anchor, wlo, whi)] = p
    return out


def copy_pair(src_root: str, dst_root: str, src_path: str) -> None:
    rel = os.path.relpath(src_path, src_root)
    dst_path = os.path.join(dst_root, rel)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True, help="CSV with columns seed,anchor_seed,wlo,whi")
    ap.add_argument("--src_root", required=True, help="Source Phase-3D channel_diag root")
    ap.add_argument("--dst_root", required=True, help="Destination subset root")
    args = ap.parse_args()

    src_root = str(args.src_root)
    dst_root = str(args.dst_root)
    os.makedirs(dst_root, exist_ok=True)

    mp = build_map(src_root)

    copied = 0
    missing = 0
    for (seed, anchor, wlo, whi) in load_pairs(str(args.pairs_csv)):
        kk = key_for_pair(seed, anchor, wlo, whi)
        p = mp.get(kk)
        if p is None:
            missing += 1
            continue
        copy_pair(src_root, dst_root, p)
        copied += 1

    print(f"[make_phase3d_subset_from_pairs] src_root={src_root}")
    print(f"[make_phase3d_subset_from_pairs] dst_root={dst_root}")
    print(f"[make_phase3d_subset_from_pairs] copied={copied} missing={missing}")


if __name__ == "__main__":
    main()
