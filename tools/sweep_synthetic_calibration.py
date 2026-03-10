import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _run(cmd: list[str], *, cwd: Path) -> None:
	print("\n$", " ".join(str(x) for x in cmd))
	subprocess.run(cmd, cwd=str(cwd), check=True)


def _read_json(p: Path) -> dict[str, Any]:
	try:
		with p.open("r", encoding="utf-8") as f:
			obj = json.load(f)
		return obj if isinstance(obj, dict) else {}
	except Exception:
		return {}


def _parse_floats(s: str) -> list[float]:
	out: list[float] = []
	for tok in str(s).split(","):
		tok = tok.strip()
		if not tok:
			continue
		out.append(float(tok))
	return out


def _safe_get(d: dict[str, Any], path: str) -> Any:
	cur: Any = d
	for key in path.split("."):
		if not isinstance(cur, dict):
			return None
		cur = cur.get(key)
	return cur


def _set_flag(cmd: list[str], flag: str, value: str) -> None:
	"""Set/override a flag value in an argv-style list."""
	try:
		i = cmd.index(flag)
		if i + 1 < len(cmd):
			cmd[i + 1] = value
			return
	except ValueError:
		pass
	cmd += [flag, value]


def _replace_first_csv_value(s: str, first: float) -> str:
	parts = [p.strip() for p in str(s).split(",") if p.strip()]
	if not parts:
		return str(first)
	if len(parts) == 1:
		return f"{first},{parts[0]}"
	parts[0] = str(first)
	return ",".join(parts)


@dataclass(frozen=True)
class SweepRow:
	out_dir: str
	scan: str
	value: str
	quotient: str
	c_star_error_abs: float | None
	c_star_error_angle: float | None
	headline_mode: str | None
	precision: float | None
	recall: float | None
	tp: int | None
	fp: int | None
	fn: int | None
	tn: int | None
	false_accept_rate: float | None
	false_reject_rate: float | None
	corridor_accept_frac: float | None
	swap_accept_frac: float | None
	cond_accept_frac: float | None
	corridor_false_reject_frac: float | None
	swap_false_accept_frac: float | None
	cond_false_accept_frac: float | None
	swap_lam_p95: float | None
	cond_gate_frac_adm: float | None


def main() -> int:
	ap = argparse.ArgumentParser(
		description=(
			"Run parameter sweeps for tools/calibrate_synthetic_projector_csv.py and aggregate key operating metrics. "
			"Intended to map boundary-of-failure thresholds (Tier0/Tier1/Tier2) and lsfit contamination sensitivity."
		)
	)

	ap.add_argument("--out_root", required=True)
	ap.add_argument(
		"--scan",
		required=True,
		choices=[
			"tier1_swap_strength",
			"tier0_cond_tinv",
			"tier0_cond_sep2",
			"tier2_p_err",
			"lsfit_contam_swap",
			"roc_gate_lam_cost_max",
			"roc_gate_tin_inv_max",
			"roc_gate_sep2_min",
			"roc_tau_logabs",
			"roc_tau_arg",
			"observable_bakeoff",
		],
	)
	ap.add_argument(
		"--values",
		default="",
		help=(
			"Comma-separated values for the scan. For observable_bakeoff use quotients like 'Q2_tr,Q2,Q2_phi2'. "
			"For tier0/tier1/tier2/contam use numeric values."
		),
	)

	# Base calibration knobs (passed through).
	ap.add_argument("--level", type=int, default=3, choices=[1, 2, 3])
	ap.add_argument("--N", type=int, default=2048)
	ap.add_argument("--N2", type=int, default=0)
	ap.add_argument("--sigma", type=float, default=1.005)
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--c_star_re", type=float, default=0.96)
	ap.add_argument("--c_star_im", type=float, default=-0.02)
	ap.add_argument("--p_err", type=float, default=1.0)
	ap.add_argument("--eps0", type=float, default=2.0)

	ap.add_argument(
		"--swap_q_strength",
		type=float,
		default=1.0,
		help="When scanning tier1_swap_strength, also set swap_q_strength (0 isolates identity; 1 includes full Q adversary).",
	)

	ap.add_argument("--taus", default="0.3,0.5")
	ap.add_argument("--taus_logabs", default="0.2,0.5")
	ap.add_argument("--taus_arg", default="0.5,1.0")
	ap.add_argument("--gate_tin_inv_max", type=float, default=500.0)
	ap.add_argument("--gate_sep2_min", type=float, default=0.01)
	ap.add_argument("--gate_lam_cost_max", type=float, default=2.0)

	ap.add_argument(
		"--headline_accept",
		default="tier01_logboth",
		choices=["tier01_logboth", "tier01_only", "tier01_rel", "tier01_rel_logboth"],
	)
	ap.add_argument("--force_adversaries", type=int, default=0)
	ap.add_argument("--gauge_phase_wobble", type=float, default=0.0)
	ap.add_argument("--gauge_amp_wobble", type=float, default=0.0)

	args = ap.parse_args()

	out_root = Path(args.out_root)
	out_root.mkdir(parents=True, exist_ok=True)

	repo_root = Path(__file__).resolve().parents[1]
	py = sys.executable
	cal_tool = Path(__file__).parent / "calibrate_synthetic_projector_csv.py"

	scan = str(args.scan)

	if scan == "observable_bakeoff":
		if not str(args.values).strip():
			raise SystemExit("--values is required for observable_bakeoff")
		sweep_vals = [v.strip() for v in str(args.values).split(",") if v.strip()]
	else:
		if not str(args.values).strip():
			raise SystemExit("--values is required for numeric scans")
		sweep_vals = [str(v) for v in _parse_floats(args.values)]

	rows: list[SweepRow] = []

	for v in sweep_vals:
		if scan == "observable_bakeoff":
			quotient = str(v)
			value_label = quotient
		else:
			quotient = "Q2_tr"
			value_label = str(v).replace(".", "p")

		out_dir = out_root / f"{scan}_{value_label}"
		out_dir.mkdir(parents=True, exist_ok=True)

		cmd = [
			py,
			"-u",
			str(cal_tool),
			"--out_dir",
			str(out_dir.resolve()),
			"--overwrite",
			"1",
			"--level",
			str(int(args.level)),
			"--N",
			str(int(args.N)),
			"--N2",
			str(int(args.N2)),
			"--sigma",
			str(float(args.sigma)),
			"--seed",
			str(int(args.seed)),
			"--c_star_re",
			str(float(args.c_star_re)),
			"--c_star_im",
			str(float(args.c_star_im)),
			"--p_err",
			str(float(args.p_err)),
			"--eps0",
			str(float(args.eps0)),
			"--force_adversaries",
			str(int(args.force_adversaries)),
			"--gauge_phase_wobble",
			str(float(args.gauge_phase_wobble)),
			"--gauge_amp_wobble",
			str(float(args.gauge_amp_wobble)),
			"--quotient",
			str(quotient),
			"--taus",
			str(args.taus),
			"--taus_logabs",
			str(args.taus_logabs),
			"--taus_arg",
			str(args.taus_arg),
			"--gate_tin_inv_max",
			str(float(args.gate_tin_inv_max)),
			"--gate_sep2_min",
			str(float(args.gate_sep2_min)),
			"--gate_lam_cost_max",
			str(float(args.gate_lam_cost_max)),
			"--headline_accept",
			str(args.headline_accept),
		]

		# Scan-specific parameter injection.
		if scan == "tier1_swap_strength":
			cmd += ["--swap_strength", str(float(v)), "--swap_q_strength", str(float(args.swap_q_strength))]
		elif scan == "tier0_cond_tinv":
			cmd += ["--cond_tinv", str(float(v))]
		elif scan == "tier0_cond_sep2":
			cmd += ["--cond_sep2", str(float(v))]
		elif scan == "tier2_p_err":
			# Force level 2 unless user overrides.
			cmd[cmd.index("--level") + 1] = str(int(2))
			cmd += ["--p_err", str(float(v)), "--force_adversaries", "1"]
		elif scan == "lsfit_contam_swap":
			# Fit c* using a dedicated contaminated fit side.
			cmd += [
				"--emit_fit_side",
				"1",
				"--fit_contam_swap_frac",
				str(float(v)),
				"--fit_contam_cond_frac",
				"0.0",
				"--lsfit_include_sides",
				"fit",
			]
		elif scan == "roc_gate_lam_cost_max":
			_set_flag(cmd, "--gate_lam_cost_max", str(float(v)))
		elif scan == "roc_gate_tin_inv_max":
			_set_flag(cmd, "--gate_tin_inv_max", str(float(v)))
		elif scan == "roc_gate_sep2_min":
			_set_flag(cmd, "--gate_sep2_min", str(float(v)))
		elif scan == "roc_tau_logabs":
			_set_flag(cmd, "--taus_logabs", _replace_first_csv_value(str(args.taus_logabs), float(v)))
		elif scan == "roc_tau_arg":
			_set_flag(cmd, "--taus_arg", _replace_first_csv_value(str(args.taus_arg), float(v)))
		elif scan == "observable_bakeoff":
			pass

		_run(cmd, cwd=repo_root)

		rep = _read_json(out_dir / "synthetic_calibration_report.json")

		op = rep.get("operating", {}) if isinstance(rep, dict) else {}
		by_side_oper = op.get("by_side", {}) if isinstance(op, dict) else {}
		corridor_accept = _safe_get(by_side_oper, "corridor.accept_frac")
		swap_accept = _safe_get(by_side_oper, "swap.accept_frac")
		cond_accept = _safe_get(by_side_oper, "cond.accept_frac")

		corridor_fr = _safe_get(by_side_oper, "corridor.false_reject_frac")
		swap_fa = _safe_get(by_side_oper, "swap.false_accept_frac")
		cond_fa = _safe_get(by_side_oper, "cond.false_accept_frac")

		conf = op.get("confusion", {}) if isinstance(op, dict) else {}
		tp = _safe_get(conf, "tp")
		fp = _safe_get(conf, "fp")
		fn = _safe_get(conf, "fn")
		tn = _safe_get(conf, "tn")
		try:
			tp_i = int(tp) if tp is not None else None
			fp_i = int(fp) if fp is not None else None
			fn_i = int(fn) if fn is not None else None
			tn_i = int(tn) if tn is not None else None
		except Exception:
			tp_i = fp_i = fn_i = tn_i = None

		fa_rate = None
		fr_rate = None
		if fp_i is not None and tn_i is not None and (fp_i + tn_i) > 0:
			fa_rate = float(fp_i / (fp_i + tn_i))
		if fn_i is not None and tp_i is not None and (fn_i + tp_i) > 0:
			fr_rate = float(fn_i / (fn_i + tp_i))

		dash_by_side = rep.get("by_side", {}) if isinstance(rep, dict) else {}
		swap_lam_p95 = _safe_get(dash_by_side, "swap.lam_match_cost_p95")
		cond_gate_frac_adm = _safe_get(dash_by_side, "cond.gate_frac_of_admissible")

		rows.append(
			SweepRow(
				out_dir=str(out_dir),
				scan=scan,
				value=str(v),
				quotient=str(quotient),
				c_star_error_abs=_safe_get(rep, "recovered.c_star_error_abs"),
				c_star_error_angle=_safe_get(rep, "recovered.c_star_error_angle"),
				headline_mode=_safe_get(rep, "operating.headline_mode"),
				precision=_safe_get(rep, "operating.precision"),
				recall=_safe_get(rep, "operating.recall"),
				tp=tp_i,
				fp=fp_i,
				fn=fn_i,
				tn=tn_i,
				false_accept_rate=fa_rate,
				false_reject_rate=fr_rate,
				corridor_accept_frac=corridor_accept,
				swap_accept_frac=swap_accept,
				cond_accept_frac=cond_accept,
				corridor_false_reject_frac=corridor_fr,
				swap_false_accept_frac=swap_fa,
				cond_false_accept_frac=cond_fa,
				swap_lam_p95=swap_lam_p95,
				cond_gate_frac_adm=cond_gate_frac_adm,
			)
		)

	# Write aggregate.
	out_csv = out_root / "sweep_summary.csv"
	with out_csv.open("w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		fields = list(SweepRow.__annotations__.keys())
		w.writerow(fields)
		for r in rows:
			w.writerow([getattr(r, k) for k in fields])

	(out_root / "sweep_summary.json").write_text(json.dumps([r.__dict__ for r in rows], indent=2), encoding="utf-8")
	print(f"\nWROTE {out_csv}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())