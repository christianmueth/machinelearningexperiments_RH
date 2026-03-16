from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _feature_value(candidate: dict[str, object], feature_name: str) -> float:
    if feature_name in candidate:
        return float(candidate.get(feature_name, 0.0))
    tiebreak = candidate.get("tiebreak_features", {})
    if isinstance(tiebreak, dict):
        return float(tiebreak.get(feature_name, 0.0))
    return 0.0


def _bucket_edges(values: np.ndarray, n_buckets: int) -> np.ndarray:
    if values.size == 0:
        return np.asarray([0.0, 1.0], dtype=np.float64)
    if np.allclose(values, values[0]):
        return np.asarray([float(values[0]), float(values[0]) + 1e-9], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, max(2, int(n_buckets) + 1))
    edges = np.quantile(values, quantiles)
    deduped = [float(edges[0])]
    for edge in edges[1:]:
        if float(edge) > deduped[-1]:
            deduped.append(float(edge))
    if len(deduped) == 1:
        deduped.append(deduped[0] + 1e-9)
    return np.asarray(deduped, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot correctness calibration for a controller branch feature on low-margin branch candidates.")
    ap.add_argument("--controller_json", required=True)
    ap.add_argument("--feature_name", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--n_buckets", type=int, default=8)
    ap.add_argument("--include_non_branch", action="store_true")
    ap.add_argument("--include_high_margin", action="store_true")
    args = ap.parse_args()

    report = json.loads(Path(str(args.controller_json)).read_text(encoding="utf-8"))
    margin_threshold = float(report.get("controller_config", {}).get("controller_margin_threshold", 0.0))

    rows: list[dict[str, float | int | str]] = []
    for group in report.get("group_reports", []):
        group_margin = float(group.get("oracle_margin", 0.0))
        if not bool(args.include_high_margin) and group_margin > margin_threshold:
            continue
        for candidate in group.get("candidates", []):
            in_branch = bool(candidate.get("in_branch", False))
            if not bool(args.include_non_branch) and not in_branch:
                continue
            rows.append(
                {
                    "problem_id": str(group.get("problem_id", "")),
                    "oracle_margin": group_margin,
                    "in_branch": int(in_branch),
                    "label": float(candidate.get("label", 0.0)),
                    "feature_value": _feature_value(candidate, str(args.feature_name)),
                }
            )

    values = np.asarray([float(row["feature_value"]) for row in rows], dtype=np.float64)
    labels = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    edges = _bucket_edges(values, int(args.n_buckets))

    bucket_rows: list[dict[str, float | int | str]] = []
    for bucket_index in range(len(edges) - 1):
        left = float(edges[bucket_index])
        right = float(edges[bucket_index + 1])
        if bucket_index < len(edges) - 2:
            mask = (values >= left) & (values < right)
        else:
            mask = (values >= left) & (values <= right)
        if not np.any(mask):
            continue
        bucket_rows.append(
            {
                "bucket": f"b{bucket_index + 1}",
                "count": int(np.sum(mask)),
                "feature_min": left,
                "feature_max": right,
                "feature_mean": float(np.mean(values[mask])),
                "accuracy": float(np.mean(labels[mask])),
            }
        )

    order = np.argsort(values)
    sorted_values = values[order]
    sorted_labels = labels[order]
    window = max(7, int(round(len(sorted_labels) * 0.08))) if len(sorted_labels) else 7
    smoothed = np.asarray(
        [
            float(np.mean(sorted_labels[max(0, idx - window // 2) : min(len(sorted_labels), idx + window // 2 + 1)]))
            for idx in range(len(sorted_labels))
        ],
        dtype=np.float64,
    ) if len(sorted_labels) else np.asarray([], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    if len(sorted_labels):
        jitter = np.linspace(-0.03, 0.03, num=len(sorted_labels))
        axes[0].scatter(sorted_values, sorted_labels + jitter, s=18, c="#4c566a", alpha=0.45, label="branch candidates")
        axes[0].plot(sorted_values, smoothed, color="#d95f02", linewidth=2.0, label="moving accuracy")
    axes[0].set_xlabel(str(args.feature_name))
    axes[0].set_ylabel("Correct Candidate")
    axes[0].set_title(f"{args.feature_name} vs Correctness")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    labels_x = [str(row["bucket"]) for row in bucket_rows]
    accuracies = [float(row["accuracy"]) for row in bucket_rows]
    counts = [int(row["count"]) for row in bucket_rows]
    axes[1].bar(labels_x, accuracies, color="#66c2a5")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy by Feature Bucket")
    axes[1].grid(axis="y", alpha=0.2)
    for idx, (accuracy, count) in enumerate(zip(accuracies, counts)):
        axes[1].text(idx, min(0.98, accuracy + 0.03), f"n={count}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_png = Path(str(args.out_png))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    summary = {
        "controller_json": str(args.controller_json),
        "feature_name": str(args.feature_name),
        "rows": int(len(rows)),
        "feature_mean": float(np.mean(values)) if values.size else 0.0,
        "positive_rate": float(np.mean(labels)) if labels.size else 0.0,
        "feature_correctness_correlation": float(np.corrcoef(values, labels)[0, 1]) if values.size >= 2 and not np.allclose(values, values[0]) and not np.allclose(labels, labels[0]) else 0.0,
        "bucket_rows": bucket_rows,
        "out_png": str(out_png),
    }

    if str(args.out_csv).strip():
        out_csv = Path(str(args.out_csv))
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["bucket", "count", "feature_min", "feature_max", "feature_mean", "accuracy"])
            writer.writeheader()
            writer.writerows(bucket_rows)
        summary["out_csv"] = str(out_csv)

    if str(args.out_json).strip():
        out_json = Path(str(args.out_json))
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summary["out_json"] = str(out_json)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())