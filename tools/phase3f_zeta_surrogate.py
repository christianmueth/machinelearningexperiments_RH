import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


def _parse_csv_floats(s: str) -> List[float]:
	out: List[float] = []
	for part in str(s).split(","):
		part = part.strip()
		if not part:
			continue
		out.append(float(part))
	if not out:
		raise ValueError("Expected at least one float")
	return out


def _clamp(x: float, lo: float, hi: float) -> float:
	return max(lo, min(hi, x))


def _load_M(npz_path: Path) -> np.ndarray:
	z = np.load(str(npz_path), allow_pickle=True)
	if "M" not in z.files:
		raise KeyError(f"NPZ missing 'M': {npz_path}")
	M = z["M"]
	if M.ndim != 2 or M.shape[0] != M.shape[1]:
		raise ValueError(f"Bad M shape {getattr(M, 'shape', None)} in {npz_path}")
	return M


def _theta_from_trace(M: np.ndarray) -> float:
	d = M.shape[0]
	tr = np.trace(M)
	x = float(np.real(tr) / float(d))
	x = _clamp(x, -1.0, 1.0)
	return float(math.acos(x))


def _fro_norm_I_minus_M(M: np.ndarray) -> float:
	d = M.shape[0]
	return float(np.linalg.norm(np.eye(d, dtype=M.dtype) - M, ord="fro"))


def _compute_class_stats(npz_paths: Iterable[Path]) -> Tuple[float, float, float, float, int]:
	thetas: List[float] = []
	norms: List[float] = []

	for p in npz_paths:
		M = _load_M(p)
		thetas.append(_theta_from_trace(M))
		norms.append(_fro_norm_I_minus_M(M))

	if not thetas:
		return (float("nan"), float("nan"), float("nan"), float("nan"), 0)

	theta_med = float(np.median(thetas))
	theta_std = float(np.std(thetas))
	a_med = float(np.median(norms))
	a_std = float(np.std(norms))
	return (theta_med, theta_std, a_med, a_std, len(thetas))


def main() -> None:
	ap = argparse.ArgumentParser(
		description=(
			"Build a first Euler-product-like zeta surrogate from Phase-3F primitive classes, "
			"using holonomy matrix dumps to define stable per-class weights." 
		)
	)
	ap.add_argument(
		"--run_dir",
		required=True,
		help="Directory containing phase3f_primitive_dictionary.csv and holonomy dumps (smoke root or a refine/refine2 run)",
	)
	ap.add_argument(
		"--run_rel",
		default=".",
		help="Filter primitive_dictionary rows by run_rel (default: '.') to restrict to a specific run inside a smoke root",
	)
	ap.add_argument(
		"--max_per_class",
		type=int,
		default=25,
		help="Max number of holonomy dumps to sample per primitive class (default: 25)",
	)
	ap.add_argument(
		"--s_values",
		default="0.5,1,2,3,4,5",
		help="Comma-separated s values to evaluate (default: 0.5,1,2,3,4,5)",
	)
	ap.add_argument(
		"--out_prefix",
		default="zeta_surrogate",
		help="Output file prefix (default: zeta_surrogate)",
	)
	args = ap.parse_args()

	run_dir = Path(str(args.run_dir)).resolve()
	prim_dict_path = run_dir / "phase3f_primitive_dictionary.csv"
	if not prim_dict_path.exists():
		raise SystemExit(f"Missing: {prim_dict_path}")

	df = pd.read_csv(prim_dict_path)
	if "canon_id" not in df.columns:
		raise SystemExit("primitive_dictionary missing canon_id column")

	if "run_rel" in df.columns:
		df = df[df["run_rel"].astype(str) == str(args.run_rel)].copy()

	if df.empty:
		raise SystemExit("No primitive_dictionary rows after filtering")

	# Resolve dump paths (prefer absolute column).
	if "rep_holonomy_dump_abs" in df.columns:
		path_col = "rep_holonomy_dump_abs"
	elif "rep_holonomy_dump" in df.columns:
		path_col = "rep_holonomy_dump"
	else:
		raise SystemExit("primitive_dictionary missing rep_holonomy_dump[_abs] column")

	out_rows = []

	for canon_id, g in df.groupby("canon_id", sort=True):
		g = g.head(int(args.max_per_class)).copy()

		npz_paths: List[Path] = []
		for p in g[path_col].astype(str).tolist():
			p = p.strip()
			if not p:
				continue
			pp = Path(p)
			if not pp.is_absolute():
				pp = (run_dir / p).resolve()
			if pp.exists():
				npz_paths.append(pp)

		theta_med, theta_std, a_med, a_std, n_used = _compute_class_stats(npz_paths)
		out_rows.append(
			{
				"canon_id": canon_id,
				"dim": int(g["dim"].iloc[0]) if "dim" in g.columns else np.nan,
				"n_rows_dict": int(len(g)),
				"n_used": int(n_used),
				"theta_median": theta_med,
				"theta_std": theta_std,
				"a_median": a_med,
				"a_std": a_std,
				"tanh_a_median": float(math.tanh(a_med)) if np.isfinite(a_med) else float("nan"),
			}
		)

	wdf = pd.DataFrame(out_rows)
	wdf = wdf.sort_values(["theta_median", "canon_id"], ascending=[True, True]).reset_index(drop=True)

	# Define w(p,s) = exp(-s * theta_p) * tanh(a_p)
	s_values = _parse_csv_floats(str(args.s_values))

	partial_rows = []
	product_rows = []

	theta = pd.to_numeric(wdf["theta_median"], errors="coerce").to_numpy()
	tanh_a = pd.to_numeric(wdf["tanh_a_median"], errors="coerce").to_numpy()

	for s in s_values:
		w = np.exp(-float(s) * theta) * tanh_a
		good = np.isfinite(w)
		w_good = w[good]
		if w_good.size == 0:
			product_rows.append({"s": float(s), "n_primitives": 0, "logZ": float("nan"), "Z": float("nan")})
			continue

		one_minus = 1.0 - w_good
		one_minus = np.maximum(one_minus, 1e-300)
		logZ_total = float(np.sum(-np.log(one_minus)))
		Z_total = float(math.exp(min(logZ_total, 700.0)))  # avoid overflow
		product_rows.append({"s": float(s), "n_primitives": int(w_good.size), "logZ": logZ_total, "Z": Z_total})

		log_terms = -np.log(one_minus)
		cumsum = np.cumsum(log_terms)
		for n in [5, 10, 20, 50, 100, int(w_good.size)]:
			if n <= 0:
				continue
			n = min(n, int(w_good.size))
			logZp = float(cumsum[n - 1]) if n > 0 else 0.0
			partial_rows.append(
				{
					"s": float(s),
					"n": int(n),
					"logZ_partial": logZp,
					"Z_partial": float(math.exp(min(logZp, 700.0))),
				}
			)

	weights_out = run_dir / f"{args.out_prefix}_primitive_weights.csv"
	partial_out = run_dir / f"{args.out_prefix}_partial_sums.csv"
	prod_out = run_dir / f"{args.out_prefix}_products.csv"

	wdf.to_csv(weights_out, index=False)
	pd.DataFrame(partial_rows).sort_values(["s", "n"]).to_csv(partial_out, index=False)
	pd.DataFrame(product_rows).sort_values(["s"]).to_csv(prod_out, index=False)

	print(f"Wrote: {weights_out}")
	print(f"Wrote: {partial_out}")
	print(f"Wrote: {prod_out}")


if __name__ == "__main__":
	main()