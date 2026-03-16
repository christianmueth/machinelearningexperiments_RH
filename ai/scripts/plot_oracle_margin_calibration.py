from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _bucket_rows(margins: np.ndarray, correct: np.ndarray) -> list[dict[str, float | int | str]]:
    zero_mask = np.isclose(margins, 0.0)
    positive_margins = margins[~zero_mask]
    positive_correct = correct[~zero_mask]

    rows: list[dict[str, float | int | str]] = []
    rows.append(
        {
            "bucket": "zero",
            "count": int(np.sum(zero_mask)),
            "mean_margin": float(np.mean(margins[zero_mask])) if np.any(zero_mask) else 0.0,
            "accuracy": float(np.mean(correct[zero_mask])) if np.any(zero_mask) else 0.0,
        }
    )
    if positive_margins.size == 0:
        return rows

    quantiles = np.quantile(positive_margins, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    for index in range(3):
        low = float(quantiles[index])
        high = float(quantiles[index + 1])
        if index < 2:
            mask = (positive_margins >= low) & (positive_margins < high)
        else:
            mask = (positive_margins >= low) & (positive_margins <= high)
        label = f"positive_q{index + 1}"
        rows.append(
            {
                "bucket": label,
                "count": int(np.sum(mask)),
                "mean_margin": float(np.mean(positive_margins[mask])) if np.any(mask) else 0.0,
                "accuracy": float(np.mean(positive_correct[mask])) if np.any(mask) else 0.0,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot oracle margin versus correctness calibration from a controller report.")
    ap.add_argument("--controller_json", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--out_csv", default="")
    args = ap.parse_args()

    report = json.loads(Path(str(args.controller_json)).read_text(encoding="utf-8"))
    group_reports = list(report.get("group_reports", []))

    margins = np.asarray([float(row.get("oracle_margin", 0.0)) for row in group_reports], dtype=np.float64)
    correct = np.asarray([1.0 if bool(row.get("baseline_top_correct", False)) else 0.0 for row in group_reports], dtype=np.float64)

    bucket_rows = _bucket_rows(margins, correct)
    sorted_index = np.argsort(margins)
    sorted_margins = margins[sorted_index]
    sorted_correct = correct[sorted_index]
    window = max(5, int(round(len(sorted_correct) * 0.1))) if len(sorted_correct) else 5
    smoothed = np.asarray(
        [
            float(np.mean(sorted_correct[max(0, idx - window // 2) : min(len(sorted_correct), idx + window // 2 + 1)]))
            for idx in range(len(sorted_correct))
        ],
        dtype=np.float64,
    ) if len(sorted_correct) else np.asarray([], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    if len(sorted_correct):
        jitter = np.linspace(-0.03, 0.03, num=len(sorted_correct))
        axes[0].scatter(sorted_margins, sorted_correct + jitter, s=22, c="#4c566a", alpha=0.55, label="eval groups")
        axes[0].plot(sorted_margins, smoothed, color="#d95f02", linewidth=2.0, label="moving accuracy")
    axes[0].set_xlabel("Oracle Margin")
    axes[0].set_ylabel("Correct Top Rank")
    axes[0].set_title("Margin vs Correctness")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    labels = [str(row["bucket"]) for row in bucket_rows]
    accuracies = [float(row["accuracy"]) for row in bucket_rows]
    counts = [int(row["count"]) for row in bucket_rows]
    axes[1].bar(labels, accuracies, color=["#8da0cb", "#66c2a5", "#fc8d62", "#e78ac3"][: len(labels)])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy by Margin Bucket")
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
        "groups_total": int(len(group_reports)),
        "overall_accuracy": float(np.mean(correct)) if correct.size else 0.0,
        "zero_margin_fraction": float(np.mean(np.isclose(margins, 0.0))) if margins.size else 0.0,
        "zero_margin_accuracy": float(np.mean(correct[np.isclose(margins, 0.0)])) if np.any(np.isclose(margins, 0.0)) else 0.0,
        "positive_margin_accuracy": float(np.mean(correct[~np.isclose(margins, 0.0)])) if np.any(~np.isclose(margins, 0.0)) else 0.0,
        "margin_correctness_correlation": _safe_corr(margins, correct),
        "bucket_rows": bucket_rows,
        "out_png": str(out_png),
    }

    if str(args.out_csv).strip():
        out_csv = Path(str(args.out_csv))
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["bucket", "count", "mean_margin", "accuracy"])
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