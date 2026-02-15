# ============================================
# CELL 1/5: Base config + utilities
# Self-contained: defines CFG_BASE and helper functions
# ============================================

import os, time, json, math, shutil, hashlib
import numpy as np
import pandas as pd

# -------------------------
# Base config (edit as needed)
# -------------------------
CFG_BASE = dict(
    # operator/template dimension
    d=16,

    # number of energies in "measurement" placeholder
    nE=256,

    # decision threshold
    alpha=0.01,

    # null sample count
    N_null=256,

    # energy windows
    windows=[(0.60, 7.50), (2.0, 5.0)],

    # stride for energies (placeholder)
    dnmap_stride=2,

    # DN-map null batching (purely computational; does not change the statistic)
    # Kept modest to avoid large temporary arrays on memory-fragmented systems.
    dnmap_null_batch=16,

    # Phase-3C FE knobs (used only when phase3_mode in {'dnmap_fe','dnmap_fe_both'})
    #
    # IMPORTANT TERMINOLOGY (per Phase-3C feedback):
    # - fe_calibration_mode uses statistics from the label-scramble null ensemble.
    #   This is a stabilizer/normalizer, NOT an RH-style analytic completion.
    # - fe_completion_mode is a deterministic, null-free counterterm G(E).
    #
    # Null-calibration modes (null-dependent):
    # - 'none': no null-based centering/scaling
    # - 'center_null_mean': subtract per-energy null mean
    # - 'center_null_median': subtract per-energy null median
    # - 'zscore_null': per-energy z-score using null mean/std
    fe_calibration_mode="none",
    fe_calibration_eps=1e-12,

    # Analytic completion modes (null-free):
    # - 'none': no analytic counterterm
    # - 'analytic_harmonic': G(E)=beta*log(1/4 + (alpha_E*E)^2)
    fe_completion_mode="none",
    fe_alpha_E=1.0,
    fe_beta=1.0,
    fe_G_clip=None,

    # FE residual aggregation across the energy grid:
    # - 'mean': mean_E |(logXi+G)(E) - (logXi+G)(-E)|
    # - 'median': median_E |...|
    fe_agg="mean",

    # which k to run
    n_ops_list=[13],
    primes_small_k=13,

    # anchor selection seed (runner overwrites)
    anchor_seed=0,

    # null sampling behavior
    sequential=True,
    early_stop=True,

    # default backend
    tp_backend="legacy",

    # Phase 3 mode control:
    # - 'dnmap_only': p_family is computed ONLY from the frozen DN-map/logdet channel (p_dnmap).
    #   This prevents any closure residuals from contaminating Phase-3B conclusions.
    # - 'dnmap_fe': Phase-3C observable: gate on paired-window involution residual (p_fe).
    #   DN-map/Schur/logdet mechanics stay frozen; only the scalar observable changes.
    # - 'dnmap_fe_both': Phase-3C combined gate: require BOTH FE-consistency and DN-map rigidity.
    #   Implemented as p_family = max(p_fe, p_rigid) where p_rigid = min(p_dnmap(W1), p_dnmap(W2)).
    # - 'full': p_family uses the enabled closure channels (pq/pqr/...) per gate_keys.
    phase3_mode="dnmap_only",

    # Closure residual semantics (only used when phase3_mode != 'dnmap_only'):
    # - 'direct_composite': compares Tp products against a separately-generated composite label matrix.
    # - 'against_true_v5': compares Tp_null products against Tp_true products (as documented in code_updates20/21).
    # IMPORTANT: this is part of the closure-channel definition; keep it fixed within an experiment.
    closure_semantics="against_true_v5",

    # Safety policy for full closure runs:
    # - 'error': hard-fail if closure_semantics is not doc-aligned (prevents accidental toy/migrated semantics).
    # - 'warn': print a warning but continue.
    # - 'allow': no guard.
    closure_semantics_policy="error",

    # Closure residual scaling/normalization:
    # - 'absolute': ||A-B||_F
    # - 'relative_ref': ||A-B||_F / (||B||_F + eps)
    # - 'relative_sum': ||A-B||_F / (||A||_F + ||B||_F + eps)
    # NOTE: this changes the closure statistic (not the DN-map spine). Keep fixed within an experiment.
    closure_residual_mode="relative_ref",
    closure_eps=1e-12,

    # -------------------------
    # Phase 3B discipline controls
    # -------------------------
    freeze_measurement=True,

    # -------------------------
    # Frozen measurement spine params (DN-map / Schur logdet)
    # -------------------------
    meas_sigma=0.75,
    meas_diag=0.8,
    meas_boundary_dim=4,
    meas_boundary_seed=None,
    meas_boundary_idx=None,
    meas_jitter=1e-6,

    # -------------------------
    # Geometry (Tp-only) warp map params
    # -------------------------
    # warp_mode is a chart/gauge choice for encoding the complexified radial continuation.
    # Recommended default (Guidance 30): 'ratio' -> W=(r-rs)/(r+rs) for clean self-dual symmetry.
    # Keep 'plus' -> W=1+r/rs available for RH_proof_beta wording continuity.
    # Other option: 'minus' -> W=1-r/rs (affine critical locus emphasized at r=r_s).
    warp_mode="ratio",
    warp_unwrap=True,
    warp_y_scale=1.0,
    warp_rs_re=1.0,
    warp_rs_im=1.0,

    # geom v2/v3 tuning knobs
    geom_v2_depth_scale=1.0,
    geom_v2_rot_scale=0.25,
    geom_v3_eps=0.01,
    geom_v3_rot_scale=0.25,
    geom_v3_rank=1,

    # -------------------------
    # Geometry v4: gauge-preregistered warp + Dirac-ish (chiral) deformation
    # -------------------------
    # Backend name: 'geom_warp_dirac_v4'
    # v4 is designed to be a controlled deformation of the legacy Tp (for reviewer-proof delta-band tuning).
    geom_v4_u_scale=1.0,
    geom_v4_theta_scale=0.125,
    geom_v4_rank=1,
    geom_v4_eps=0.01,
    # Global deformation scale (used when eps_auto_tune=False, and as the initial/default scale)
    geom_v4_eps_global=0.02,
    # Optional boundedness controls on warp outputs (admissibility proxy)
    geom_v4_u_clip=4.0,
    geom_v4_theta_clip=6.283185307179586,
    # Optional: tune a single global deformation scale to hit a target relative Tp delta vs legacy.
    eps_auto_tune=False,
    eps_target_delta=0.02,
    eps_tune_max_iter=20,
    eps_tune_eps_max=2.0,
    eps_tune_tol=0.001,

    # -------------------------
    # Geometry v5: v4 + self-dual harmonic coupling modulation
    # -------------------------
    # Backend name: 'geom_warp_dirac_v5'
    # The self-dual identity r+r_s=1 is made structural by letting a self-dual
    # harmonic term modulate the chiral coupling amplitude.
    geom_v5_u_scale=1.0,
    geom_v5_theta_scale=0.125,
    geom_v5_rank=1,
    geom_v5_eps=0.01,
    geom_v5_eps_global=0.02,
    geom_v5_u_clip=4.0,
    geom_v5_theta_clip=6.283185307179586,
    # Coupling modulation: h(y) = 1 / (1/4 + y^2), normalized then exponentiated.
    geom_v5_harmonic_gamma=1.0,
    geom_v5_harmonic_strength=1.0,

    # -------------------------
    # Geometry v6: v4 + self-dual harmonic PHASE perturbation (not amplitude)
    # -------------------------
    # Backend name: 'geom_warp_dirac_v6'
    # Uses the same self-dual harmonic h(p)=1/(1/4+y(p)^2) but inserts it as a
    # small, clipped perturbation of the phase theta(p) before driving chirality.
    geom_v6_u_scale=1.0,
    geom_v6_theta_scale=0.125,
    geom_v6_rank=1,
    geom_v6_eps=0.01,
    geom_v6_eps_global=0.02,
    geom_v6_u_clip=4.0,
    geom_v6_theta_clip=6.283185307179586,
    # phase perturbation: theta' = theta + lambda * clip(log(h/median(h)))
    geom_v6_phase_lambda=0.05,
    geom_v6_phase_clip=2.0,

    # -------------------------
    # Geometry v7: v4 + LOCALIZED defect modulation (Phase-3B legal)
    # -------------------------
    # Backend name: 'geom_warp_dirac_v7_localdefect'
    # Goal: embrace the empirical diagnosis that any DN-map rigidity observed so far is
    # seed-conditional and anchor-localized. v7 introduces controlled non-smoothness by
    # modulating the chiral coupling on a localized subset of primes.
    geom_v7_u_scale=1.0,
    geom_v7_theta_scale=0.125,
    geom_v7_rank=1,
    geom_v7_eps=0.01,
    geom_v7_eps_global=0.02,
    geom_v7_u_clip=4.0,
    geom_v7_theta_clip=6.283185307179586,
    # defect selection: 'block' (contiguous subset) or 'bernoulli' (sparse iid)
    geom_v7_defect_mode="block",
    # fraction of primes affected (0..1)
    geom_v7_defect_rho=0.25,
    # multiplier = 1 + strength * sign on defect primes (sign is ±1), then clipped
    geom_v7_defect_strength=1.0,
    geom_v7_defect_clip=0.75,
    # stable salt to decouple defect pattern from other hash uses
    geom_v7_defect_salt=1,
    # defect pattern seed (geometry-class constant; MUST NOT be the observation seed)
    geom_v7_defect_seed=0,

    # -------------------------
    # Geometry v8: v4 + BOUNDARY-TETHERED defect (Phase-3B legal)
    # -------------------------
    # Backend name: 'geom_warp_dirac_v8_boundarydefect'
    # Goal: test whether defects localized in *coordinate space* near the fixed DN-map boundary
    # split can generate robust DN-map rigidity signals, while remaining Tp-only.
    geom_v8_u_scale=1.0,
    geom_v8_theta_scale=0.125,
    geom_v8_rank=1,
    geom_v8_eps=0.01,
    geom_v8_eps_global=0.02,
    geom_v8_u_clip=4.0,
    geom_v8_theta_clip=6.283185307179586,
    # Prime-side activation pattern (which primes get the boundary defect term)
    geom_v8_prime_defect_mode="block",  # block|bernoulli
    geom_v8_prime_defect_rho=0.25,
    geom_v8_defect_strength=1.0,
    geom_v8_defect_clip=0.75,
    geom_v8_defect_salt=1,
    geom_v8_defect_seed=0,
    # Boundary tethering controls (coordinate-space)
    geom_v8_pair_chiral=True,  # expand boundary indices to include chiral partners
    geom_v8_defect_norm_mode="fro_match",  # none|fro_match

    # -------------------------
    # Geometry v9: v4 + GLOBAL phase transport / connection (Phase-3B legal)
    # -------------------------
    # Backend name: 'geom_warp_dirac_v9_phasetransport'
    # Goal: test whether *global* coherence (a transport rule for phase along ordered primes)
    # can create robust DN-map rigidity under the frozen measurement spine.
    geom_v9_u_scale=1.0,
    geom_v9_theta_scale=0.125,
    geom_v9_rank=1,
    geom_v9_eps=0.01,
    geom_v9_eps_global=0.02,
    geom_v9_u_clip=4.0,
    geom_v9_theta_clip=6.283185307179586,
    # Transport knobs (deterministic; MUST NOT depend on observation seed)
    geom_v9_transport_rule="delta_raw_wrapped",  # delta_raw_wrapped
    geom_v9_transport_lambda=1.0,
    geom_v9_transport_clip=0.25,
    # Optional constant in policy digest (kept for audit; transport is deterministic by default)
    geom_v9_transport_seed=0,

    # -------------------------
    # Geometry v10: v4 + GLOBAL phase-field solve (Phase-3B legal)
    # -------------------------
    # Backend name: 'geom_warp_dirac_v10_phasefieldsolve'
    # Goal: a qualitatively stronger global coupling than v9 transport: solve a global phase field
    # (discrete Laplacian smoothing with gauge-fixing tether), not a sequential transport.
    geom_v10_u_scale=1.0,
    geom_v10_theta_scale=0.125,
    geom_v10_rank=1,
    geom_v10_eps=0.01,
    geom_v10_eps_global=0.02,
    geom_v10_u_clip=4.0,
    geom_v10_theta_clip=6.283185307179586,
    # Phase-field solve knobs (deterministic; MUST NOT depend on observation seed)
    geom_v10_phasefield_graph="path",  # path
    geom_v10_phasefield_w=1.0,          # edge weight for path Laplacian
    geom_v10_phasefield_eta=0.25,       # gauge-fixing tether strength (L+eta I) theta = eta theta0
    geom_v10_phasefield_seed=0,         # reserved for audit; solve is deterministic by default

    # Policy / gauge metadata (logged for audit; does not change the measurement spine)
    gauge_policy="none",
    freeze_geometry_policy=False,

    # DN-map uniqueness/collision KPIs (derived from dnmap null scores; does not alter measurement spine)
    dnmap_uniqueness=False,
    dn_collision_tol=0.0,
)

# -------------------------
# Prime helpers
# -------------------------
def first_primes(n: int):
    n = int(n)
    if n <= 0:
        return []
    primes = []
    x = 2
    while len(primes) < n:
        is_p = True
        r = int(math.isqrt(x))
        for p in primes:
            if p > r:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
        x += 1
    return primes

# -------------------------
# Stats helpers
# -------------------------
def empirical_p_lower(null_arr: np.ndarray, obs: float, tie_le=True):
    null_arr = np.asarray(null_arr, dtype=np.float64)
    if null_arr.size == 0:
        return float("inf")
    if tie_le:
        n_extreme = int(np.sum(null_arr <= obs))
    else:
        n_extreme = int(np.sum(null_arr < obs))
    # canonical add-one smoothing
    return float((n_extreme + 1) / (null_arr.size + 1))

def z_effect(null_arr: np.ndarray, obs: float):
    # lightweight signed effect (negative means obs is smaller than typical null)
    null_arr = np.asarray(null_arr, dtype=np.float64)
    if null_arr.size == 0:
        return 0.0
    mu = float(np.mean(null_arr))
    sd = float(np.std(null_arr) + 1e-12)
    return (obs - mu) / sd

def can_stop_now_lower_tail(j_used, n_extreme, N_target, alpha):
    # Simple early stop: if even best-case can't exceed alpha or must reject
    # Lower tail: reject if p <= alpha.
    # Current p_hat_min = (n_extreme+1)/(j_used+1); best-case (if no more extremes) would increase denom only -> p shrinks? Actually with add-one it's monotone in denom.
    p_now = (n_extreme + 1) / (j_used + 1)
    # If already <= alpha, can stop (reject)
    if p_now <= alpha:
        return True, False
    # If even if all remaining are extreme (worst for p), p would be (n_extreme + (N_target-j_used) + 1)/(N_target+1) which increases -> not help rejection.
    # For lower-tail rejection, failing early-stop isn't very strong; we keep conservative:
    return False, False

def sha16(x: bytes):
    return hashlib.sha256(x).hexdigest()[:16]

# ============================================
# CELL 2/5: Closure knobs + anchor selection + gate contract
# Self-contained: apply_closure_knobs, pq/pqr anchor selectors
# ============================================

P_CHANNELS_ALL = ["p_pq", "p_pqr", "p_mult", "p_pow", "p_tower", "p_comm"]

RAW_TO_P = {
    "pq": "p_pq",
    "p_pqr": "p_pqr",
    "mult": "p_mult",
    "p_pow": "p_pow",
    "tower": "p_tower",
    "comm": "p_comm",
}

def apply_closure_knobs(cfg: dict, pqM: int, pqrM: int, kmax: int, use_pow=False, use_tower=False):
    cfg = dict(cfg)
    cfg["pq_anchor_M"] = int(pqM)
    cfg["pqr_anchor_M"] = int(pqrM)
    cfg["p_pow_kmax"] = int(kmax)

    # OFF means OFF flags
    cfg["use_pq"] = bool(int(pqM) > 0)
    cfg["use_pqr"] = bool(int(pqrM) > 0)
    cfg["use_pow"] = bool(use_pow) and (int(kmax) >= 2)
    cfg["use_tower"] = bool(use_tower) and (int(kmax) >= 2)

    # keep other channels off by default in this mini-skeleton
    cfg["use_mult"] = False
    cfg["use_comm"] = False

    # two-path default true unless explicitly turned off
    cfg["pqr_two_path"] = bool(cfg.get("pqr_two_path", True))
    return cfg

def assert_gate_contract(gate_keys, enabled_set):
    # Basic safety check
    for k in gate_keys:
        assert k in enabled_set, f"Gate key {k} not enabled"
    return True

def _rng_from_seed(seed: int):
    return np.random.default_rng(int(seed))

def _hash_array16(x: np.ndarray) -> str:
    x = np.asarray(x)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode("utf-8"))
    h.update(repr(tuple(x.shape)).encode("utf-8"))
    h.update(x.tobytes(order="C"))
    return h.hexdigest()[:16]

def get_measurement_boundary_idx(cfg: dict, d: int) -> np.ndarray:
    """Returns boundary indices for DN-map split.

    Default behavior (no cfg override): use first b indices (original behavior).
    Optional overrides:
      - cfg['meas_boundary_idx'] = list/array of indices (length b)
      - cfg['meas_boundary_seed'] = int seed to sample b unique indices from range(d)
    """
    d = int(d)
    b = int(cfg.get("meas_boundary_dim", max(1, d // 4)))
    b = max(1, min(int(b), d - 1))

    explicit = cfg.get("meas_boundary_idx", None)
    if explicit is not None:
        idx = np.asarray(list(explicit), dtype=np.int64)
        if idx.size != b:
            raise ValueError(f"meas_boundary_idx length {idx.size} != meas_boundary_dim {b}")
        if np.any(idx < 0) or np.any(idx >= d):
            raise ValueError("meas_boundary_idx contains out-of-range indices")
        if len(set(map(int, idx.tolist()))) != int(b):
            raise ValueError("meas_boundary_idx must contain unique indices")
        return idx

    seed = cfg.get("meas_boundary_seed", None)
    if seed is not None:
        rng = _rng_from_seed(int(seed))
        idx = rng.choice(d, size=b, replace=False)
        idx = np.asarray(sorted(map(int, idx.tolist())), dtype=np.int64)
        return idx

    return np.arange(b, dtype=np.int64)

def dnmap_energy_grid(cfg: dict, wlo: float, whi: float, dn_stride: int) -> np.ndarray:
    """Deterministic energy grid used by the DN-map/logdet spine.

    This is part of the *measurement* spine and is intentionally independent of Tp.
    """
    nE = int(cfg.get("nE", 256))
    energies_full = np.linspace(float(wlo), float(whi), int(nE), dtype=np.float64)
    return energies_full[::max(1, int(dn_stride))]

def select_pq_pairs(primes_used, M: int, anchor_seed: int):
    primes = list(map(int, primes_used))
    M = int(M)
    if M <= 0 or len(primes) < 2:
        return []
    rng = _rng_from_seed(anchor_seed)
    pairs = []
    for _ in range(M):
        i, j = rng.integers(0, len(primes), size=2)
        while j == i:
            j = int(rng.integers(0, len(primes)))
        a, b = primes[min(i, j)], primes[max(i, j)]
        pairs.append((a, b))
    return pairs

def select_pqr_triples(primes_used, M: int, anchor_seed: int):
    primes = list(map(int, primes_used))
    M = int(M)
    if M <= 0 or len(primes) < 3:
        return []
    rng = _rng_from_seed(anchor_seed + 777)
    triples = []
    for _ in range(M):
        idx = rng.choice(len(primes), size=3, replace=False)
        p, q, r = [primes[int(k)] for k in idx]
        triples.append((p, q, r))
    return triples

def anchor_hash_and_digest(pq_pairs, pqr_triples):
    b = repr((pq_pairs, pqr_triples)).encode("utf-8")
    h = sha16(b)
    return h, sha16(repr(pq_pairs).encode()), sha16(repr(pqr_triples).encode())

def make_label_scramble_phi(primes_used, scramble_seed: int):
    primes = list(map(int, primes_used))
    rng = _rng_from_seed(scramble_seed)
    perm = primes.copy()
    rng.shuffle(perm)
    return {p: perm[i] for i, p in enumerate(primes)}


def build_Tp_from_phi(Tp_true: dict, primes_used: list, phi: dict) -> dict:
    """Build a Tp dictionary by relabeling Tp_true via phi on primes_used."""
    out = {}
    for p in map(int, primes_used):
        q = int(phi[int(p)])
        out[int(p)] = np.array(Tp_true[int(q)], copy=True)
    return out


def build_composites_true_products(Tp_true: dict, primes_used: list, kmax: int, two_path: bool) -> dict:
    """True composites built as products of Tp_true generators (doc-aligned semantics)."""
    composites = {}
    P = list(map(int, primes_used))

    # pq composites (both orders)
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            p, q = P[i], P[j]
            composites[("pq", (p, q))] = Tp_true[p] @ Tp_true[q]
            composites[("pq", (q, p))] = Tp_true[q] @ Tp_true[p]

    # prime powers (aggregate up to kmax)
    if int(kmax) >= 2:
        for p in P:
            cur = Tp_true[p].copy()
            for m in range(2, int(kmax) + 1):
                cur = cur @ Tp_true[p]
                composites[("ppow", (p, m))] = cur.copy()

    # pqr two-path composites
    if bool(two_path) and len(P) >= 3:
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                for t in range(j + 1, len(P)):
                    p, q, r = P[i], P[j], P[t]
                    composites[("pqr_path1", (p, q, r))] = (Tp_true[p] @ Tp_true[q]) @ Tp_true[r]
                    composites[("pqr_path2", (p, q, r))] = Tp_true[p] @ (Tp_true[q] @ Tp_true[r])

    return composites


def closure_residuals_against_true_products(
    Tp: dict,
    composites_true: dict,
    pq_pairs: list,
    pqr_triples: list,
    kmax: int,
    two_path: bool,
    use_pq: bool,
    use_pqr: bool,
    use_pow: bool,
    cfg: dict,
):
    """Compute closure residual summaries by comparing Tp products against composites_true."""
    def _res(A: np.ndarray, B: np.ndarray) -> float:
        return float(_closure_residual(A, B, cfg))

    out = {"res_pq": float("inf"), "res_pqr": float("inf"), "res_pow": float("inf")}

    if bool(use_pq) and pq_pairs:
        vals = []
        for (p, q) in pq_pairs:
            p = int(p); q = int(q)
            X = Tp[p] @ Tp[q]
            T = composites_true[("pq", (p, q))]
            vals.append(_res(X, T))
        out["res_pq"] = float(np.mean(vals)) if vals else float("inf")

    if bool(use_pow) and int(kmax) >= 2:
        vals = []
        for p in map(int, Tp.keys()):
            A = Tp[int(p)]
            cur = A.copy()
            for m in range(2, int(kmax) + 1):
                cur = cur @ A
                key = ("ppow", (int(p), int(m)))
                if key in composites_true:
                    vals.append(_res(cur, composites_true[key]))
        out["res_pow"] = float(np.mean(vals)) if vals else float("inf")

    if bool(use_pqr) and bool(two_path) and pqr_triples:
        vals = []
        for (p, q, r) in pqr_triples:
            p = int(p); q = int(q); r = int(r)
            X1 = (Tp[p] @ Tp[q]) @ Tp[r]
            X2 = Tp[p] @ (Tp[q] @ Tp[r])
            T1 = composites_true[("pqr_path1", (p, q, r))]
            T2 = composites_true[("pqr_path2", (p, q, r))]
            vals.append(_res(X1, T1))
            vals.append(_res(X2, T2))
            vals.append(_res(X1, X2))
        out["res_pqr"] = float(np.mean(vals)) if vals else float("inf")

    return out


# ============================================
# Closure residual pipeline (pq / pqr / p^k)
#
# Motivation:
# - In Phase 2C/3A-style experiments, pq/pqr channels must be real residuals, not placeholders.
# - We model a closure check by comparing a product of prime generators against a "direct"
#   composite generator indexed by the integer label n (e.g. n=p*q).
# - Label-scramble nulls break the label↔matrix alignment while keeping the composite label fixed,
#   creating a meaningful null distribution.
# ============================================

def _build_Tn_synth(nums, d: int, seed: int = 0):
    """Legacy direct generator for arbitrary integer labels (not just primes)."""
    nums = list(map(int, nums))
    out = {}
    for n in nums:
        rng = np.random.default_rng(int(seed) * 1000003 + int(n) * 9176 + 11)
        A = rng.standard_normal((int(d), int(d)))
        H = _sym(A) / np.sqrt(max(1, d))
        out[n] = _expm_via_eig_sym(0.25 * H)
    return out


def _build_Tn_backend_dispatch(tp_backend: str, nums: list, d: int, seed: int, cfg: dict) -> dict:
    """Direct composite generator at labels n (uses the same backend family as Tp)."""
    tp_backend = str(tp_backend or "legacy")
    alias = {
        "geom_v3": "geom_v3_shared_axis",
        "geom_v3_shared": "geom_v3_shared_axis",
        "geom_v3_graded": "geom_v3_shared_axis_graded",
        "geom_v3_shared_axis_graded": "geom_v3_shared_axis_graded",
        "geom_v4": "geom_warp_dirac_v4",
        "geom_warp_dirac_v4": "geom_warp_dirac_v4",
        "geom_v5": "geom_warp_dirac_v5",
        "geom_warp_dirac_v5": "geom_warp_dirac_v5",
        "geom_v6": "geom_warp_dirac_v6",
        "geom_warp_dirac_v6": "geom_warp_dirac_v6",
        "geom_v7": "geom_warp_dirac_v7_localdefect",
        "geom_warp_dirac_v7_localdefect": "geom_warp_dirac_v7_localdefect",

        "geom_v8": "geom_warp_dirac_v8_boundarydefect",
        "geom_warp_dirac_v8_boundarydefect": "geom_warp_dirac_v8_boundarydefect",

        "geom_v9": "geom_warp_dirac_v9_phasetransport",
        "geom_warp_dirac_v9_phasetransport": "geom_warp_dirac_v9_phasetransport",

        "geom_v10": "geom_warp_dirac_v10_phasefieldsolve",
        "geom_warp_dirac_v10_phasefieldsolve": "geom_warp_dirac_v10_phasefieldsolve",
    }
    tp_backend = alias.get(tp_backend, tp_backend)

    nums = list(map(int, nums))
    if tp_backend == "legacy":
        return _build_Tn_synth(nums, d=int(d), seed=int(seed))

    # Geometry backends: reuse the same templates as Tp but evaluate u/theta at arbitrary labels.
    if tp_backend == "geom_v3_shared_axis":
        u, theta = _warp_u_theta_from_complex_r(nums, cfg)
        rng = np.random.default_rng(int(seed) * 99991 + 123)

        A0 = rng.standard_normal((int(d), int(d)))
        H0 = _sym(A0) / np.sqrt(max(1, d))

        rank = int(cfg.get("geom_v3_rank", 1))
        U = rng.standard_normal((int(d), rank)) / np.sqrt(max(1, d))
        C_shared = _sym(U @ U.T)

        eps = float(cfg.get("geom_v3_eps", 0.01))
        theta_scale_raw = cfg.get("geom_v3_theta_scale", None)
        if theta_scale_raw is None:
            theta_scale_raw = cfg.get("geom_v3_rot_scale", 0.25)
        theta_scale = float(theta_scale_raw)

        out = {}
        for i, n in enumerate(nums):
            H = (float(u[i]) * H0) + (eps * float(u[i]) * C_shared)
            H = H + (theta_scale * float(theta[i])) * np.eye(int(d))
            out[n] = _expm_via_eig_sym(_sym(H))
        return out

    if tp_backend == "geom_v3_shared_axis_graded":
        u, theta = _warp_u_theta_from_complex_r(nums, cfg)
        rng = np.random.default_rng(int(seed) * 99991 + 123)

        A0 = rng.standard_normal((int(d), int(d)))
        H0 = _sym(A0) / np.sqrt(max(1, d))

        rank = int(cfg.get("geom_v3_rank", 1))
        U = rng.standard_normal((int(d), rank)) / np.sqrt(max(1, d))
        C_shared = _sym(U @ U.T)

        eps = float(cfg.get("geom_v3_eps", 0.01))
        theta_scale_raw = cfg.get("geom_v3_theta_scale", None)
        if theta_scale_raw is None:
            theta_scale_raw = cfg.get("geom_v3_rot_scale", 0.25)
        theta_scale = float(theta_scale_raw)

        n2 = int(d) // 2
        I = np.eye(n2, dtype=np.float64)
        Z = np.zeros((n2, n2), dtype=np.float64)
        X_chiral = np.block([[Z, I], [I, Z]])

        out = {}
        for i, n in enumerate(nums):
            H = (float(u[i]) * H0) + (eps * float(u[i]) * C_shared) + (theta_scale * float(theta[i]) * X_chiral)
            out[n] = _expm_via_eig_sym(_sym(H))
        return out

    if tp_backend == "geom_warp_dirac_v4":
        return _build_Tn_geom_warp_dirac_v4(nums, d=int(d), seed=int(seed), cfg=cfg)

    if tp_backend == "geom_warp_dirac_v5":
        return _build_Tn_geom_warp_dirac_v5(nums, d=int(d), seed=int(seed), cfg=cfg)

    if tp_backend == "geom_warp_dirac_v6":
        return _build_Tn_geom_warp_dirac_v6(nums, d=int(d), seed=int(seed), cfg=cfg)

    if tp_backend == "geom_warp_dirac_v7_localdefect":
        return _build_Tn_geom_warp_dirac_v7_localdefect(nums, d=int(d), seed=int(seed), cfg=cfg)

    if tp_backend == "geom_warp_dirac_v8_boundarydefect":
        return _build_Tn_geom_warp_dirac_v8_boundarydefect(nums, d=int(d), seed=int(seed), cfg=cfg)

    if tp_backend == "geom_warp_dirac_v9_phasetransport":
        return _build_Tn_geom_warp_dirac_v9_phasetransport(nums, d=int(d), seed=int(seed), cfg=cfg)

    if tp_backend == "geom_warp_dirac_v10_phasefieldsolve":
        return _build_Tn_geom_warp_dirac_v10_phasefieldsolve(nums, d=int(d), seed=int(seed), cfg=cfg)

    # If user added v2 backends, allow them for direct labels too.
    if tp_backend == "geom_v2a" and "build_Tp_geom_v2a" in globals():
        u, theta = _warp_u_theta_from_complex_r(nums, cfg)
        rng = np.random.default_rng(int(seed) * 99991 + 123)
        A0 = rng.standard_normal((int(d), int(d)))
        H0 = _sym(A0) / np.sqrt(max(1, d))
        depth_scale = float(cfg.get("geom_v2_depth_scale", 1.0))
        out = {}
        for i, n in enumerate(nums):
            out[n] = _expm_via_eig_sym((depth_scale * float(u[i])) * H0)
        return out
    if tp_backend == "geom_v2b" and "build_Tp_geom_v2b" in globals():
        u, theta = _warp_u_theta_from_complex_r(nums, cfg)
        rng = np.random.default_rng(int(seed) * 99991 + 123)
        A0 = rng.standard_normal((int(d), int(d)))
        H0 = _sym(A0) / np.sqrt(max(1, d))
        depth_scale = float(cfg.get("geom_v2_depth_scale", 1.0))
        rot_scale = float(cfg.get("geom_v2_rot_scale", 0.25))
        J0 = _skew(rng.standard_normal((int(d), int(d)))) / np.sqrt(max(1, d))
        out = {}
        for i, n in enumerate(nums):
            out[n] = _expm_via_eig_sym((depth_scale * float(u[i])) * H0) @ _expm_via_eig_skew((rot_scale * float(theta[i])) * J0)
        return out

    raise ValueError(f"Unknown tp_backend={tp_backend!r} for direct composites")


def _relative_fro_residual(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> float:
    num = float(np.linalg.norm(A - B, ord="fro"))
    den = float(np.linalg.norm(B, ord="fro")) + float(eps)
    return num / den


def _closure_residual(A: np.ndarray, B: np.ndarray, cfg: dict) -> float:
    """Closure residual with configurable normalization.

    A, B: matrices to compare.
    """
    mode = str(cfg.get("closure_residual_mode", "relative_ref")).strip().lower()
    eps = float(cfg.get("closure_eps", 1e-12))
    num = float(np.linalg.norm(A - B, ord="fro"))

    if mode in {"absolute", "abs"}:
        return num
    if mode in {"relative_ref", "rel_ref", "relative_b", "relative"}:
        den = float(np.linalg.norm(B, ord="fro")) + eps
        return num / den
    if mode in {"relative_sum", "rel_sum"}:
        den = float(np.linalg.norm(A, ord="fro")) + float(np.linalg.norm(B, ord="fro")) + eps
        return num / den
    raise ValueError(f"Unknown closure_residual_mode={mode!r}. Expected: absolute, relative_ref, relative_sum")


def closure_residuals_from_Tp(
    Tp: dict,
    direct_Tn: dict,
    pq_pairs: list,
    pqr_triples: list,
    kmax: int,
    two_path: bool,
    cfg: dict,
) -> dict:
    """Compute scalar residual summaries for each closure channel."""
    # closure_eps + closure_residual_mode are handled inside _closure_residual.

    pq_res = []
    for (p, q) in pq_pairs:
        n = int(p) * int(q)
        A = Tp[int(p)] @ Tp[int(q)]
        B = direct_Tn[int(n)]
        pq_res.append(_closure_residual(A, B, cfg))

    pqr_res = []
    for (p, q, r) in pqr_triples:
        n = int(p) * int(q) * int(r)
        B = direct_Tn[int(n)]
        path1 = Tp[int(p)] @ Tp[int(q)] @ Tp[int(r)]
        if bool(two_path):
            path2 = Tp[int(p)] @ Tp[int(r)] @ Tp[int(q)]
            rr = min(_closure_residual(path1, B, cfg), _closure_residual(path2, B, cfg))
        else:
            rr = _closure_residual(path1, B, cfg)
        pqr_res.append(float(rr))

    pow_res = []
    kmax = int(kmax)
    if kmax >= 2:
        primes = list(sorted(map(int, Tp.keys())))
        for p in primes:
            for k in range(2, kmax + 1):
                n = int(p) ** int(k)
                if int(n) not in direct_Tn:
                    continue
                A = np.linalg.matrix_power(Tp[int(p)], int(k))
                B = direct_Tn[int(n)]
                pow_res.append(_closure_residual(A, B, cfg))

    # For now, summarize each channel by mean residual across its anchors.
    out = {
        "res_pq": float(np.mean(pq_res)) if len(pq_res) else float("inf"),
        "res_pqr": float(np.mean(pqr_res)) if len(pqr_res) else float("inf"),
        "res_pow": float(np.mean(pow_res)) if len(pow_res) else float("inf"),
    }
    return out

# ============================================
# CELL 2.1 (ADD): WIRING PATCH + GEOM V3 BACKEND
# - Provides missing symbols ONLY if absent
# - Implements complex r and complex r_s warp map inside Tp backend (Phase 3B-legal)
# - Adds geom_v3_shared_axis: non-decomposable shared-axis coupling
# ============================================

import numpy as np
import hashlib

# ----------------------------
# 0) Tiny linear algebra helpers (no scipy required)
# ----------------------------
def _sym(A):
    return 0.5 * (A + A.T)

def _skew(A):
    return 0.5 * (A - A.T)

def _expm_via_eig_sym(H):
    """exp(H) for symmetric/Hermitian-real H using eigendecomposition."""
    w, V = np.linalg.eigh(H)
    return (V * np.exp(w))[...] @ V.T

def _expm_via_eig_skew(K):
    """exp(K) for real skew-symmetric K: eigen trick via complex eig is ok for small d."""
    # For stability: use series for tiny d? We'll do eig in complex.
    w, V = np.linalg.eig(K.astype(np.complex128))
    Vinv = np.linalg.inv(V)
    return (V @ np.diag(np.exp(w)) @ Vinv).real

def _logm_via_eig_sym(S):
    """log(S) for symmetric positive definite S using eigendecomposition."""
    S = np.asarray(S, dtype=np.float64)
    S = _sym(S)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, 1e-15)
    return (V * np.log(w))[...] @ V.T

# ----------------------------
# 1) Legacy Tp (fallback if missing)
# ----------------------------
if "build_Tp_synth" not in globals():
    def build_Tp_synth(primes_used, d: int, seed: int = 0):
        """
        Fallback legacy Tp generator (deterministic).
        Produces SPD-ish matrices per prime (via exp(sym noise)).
        """
        primes_used = list(map(int, primes_used))
        Tp = {}
        for p in primes_used:
            rng = np.random.default_rng(int(seed) * 1000003 + int(p) * 9176 + 11)
            A = rng.standard_normal((int(d), int(d)))
            H = _sym(A) / np.sqrt(max(1, d))
            Tp[p] = _expm_via_eig_sym(0.25 * H)  # mild scale
        return Tp

# ----------------------------
# 2) Fingerprints (fallback if missing)
# ----------------------------
if "tp_fingerprint" not in globals():
    def tp_fingerprint(Tp: dict, primes_used: list, tp_backend: str, u_vals=None, theta_vals=None) -> dict:
        ps = sorted(list(map(int, primes_used)))
        # stable hash from a small canonical slice across primes
        h = hashlib.sha256()
        trace_sum = 0.0
        norm_sum = 0.0
        head = []
        for p in ps:
            X = np.asarray(Tp[p])
            trace_sum += float(np.trace(X))
            norm_sum += float(np.linalg.norm(X, ord="fro"))
            # grab a tiny head signature
            head.extend([float(X[0,0]), float(X[0,1]) if X.shape[1] > 1 else 0.0])
        head_arr = np.asarray(head[:64], dtype=np.float64)
        h.update(head_arr.tobytes())

        out = {
            "tp_hash": h.hexdigest()[:16],
            "tp_sig_trace_sum": float(trace_sum),
            "tp_sig_norm_sum": float(norm_sum),
        }
        if u_vals is not None:
            out["tp_u_min"] = float(np.min(u_vals))
            out["tp_u_mean"] = float(np.mean(u_vals))
            out["tp_u_max"] = float(np.max(u_vals))
        if theta_vals is not None:
            out["tp_theta_min"] = float(np.min(theta_vals))
            out["tp_theta_mean"] = float(np.mean(theta_vals))
            out["tp_theta_max"] = float(np.max(theta_vals))
        return out

# ----------------------------
# 3) Commutator diagnostics (fallback if missing)
# ----------------------------
if "commutator_summary" not in globals():
    def commutator_summary(Tp: dict, primes_used: list, max_pairs: int = 6) -> dict:
        ps = sorted(list(map(int, primes_used)))
        pairs = []
        for i in range(min(len(ps), 4)):
            for j in range(i+1, min(len(ps), 6)):
                pairs.append((ps[i], ps[j]))
        pairs = pairs[:max_pairs]

        vals = []
        for p,q in pairs:
            A = Tp[p]; B = Tp[q]
            C = A @ B - B @ A
            vals.append(float(np.linalg.norm(C, ord="fro")))
        if not vals:
            return {"tp_comm_mean": 0.0, "tp_comm_max": 0.0}
        return {"tp_comm_mean": float(np.mean(vals)), "tp_comm_max": float(np.max(vals))}

# ----------------------------
# 4) GEOM V3: complex warp map + shared-axis coupling (non-decomposable)
# ----------------------------
def _warp_u_theta_from_complex_r(primes_used, cfg: dict):
    """
    Implements: r(p) = x(p) + i y(p), with complex r_s too.
    W(p) = 1 + r(p)/r_s  (completed by complex log; u=Re log W, theta=Im log W).
    """
    ps = list(map(int, primes_used))
    # Real "horn seed"
    x = np.log(np.asarray(ps, dtype=np.float64))

    # Imag displacement y(p): deterministic continuation phase (not mod-4 placeholder)
    # theta_seed = frac(log p / (2π)) mapped to (-π, π]
    twopi = 2.0 * np.pi
    frac = (x / twopi) - np.floor(x / twopi)
    theta0 = (frac * twopi) - np.pi  # in (-pi, pi]

    # Allow y scale control
    y_scale = float(cfg.get("warp_y_scale", 1.0))
    y = y_scale * theta0

    # complex r_s (REQUIRED by your note): rs = a + i b
    rs_re = float(cfg.get("warp_rs_re", 1.0))
    rs_im = float(cfg.get("warp_rs_im", 1.0))
    rs = np.complex128(rs_re + 1j * rs_im)

    # complex r(p)
    r = x.astype(np.complex128) + 1j * y.astype(np.complex128)

    # Warp factor + complex log
    mode = str(cfg.get("warp_mode", "plus")).strip().lower()
    if mode in {"plus", "+"}:
        W = 1.0 + (r / rs)
    elif mode in {"minus", "-"}:
        W = 1.0 - (r / rs)
    elif mode in {"ratio", "mobius", "moebius"}:
        # Self-dual/Möbius flavored warp: critical locus at r=rs, dual at r=-rs.
        denom = (r + rs)
        denom = np.where(np.abs(denom) < 1e-12, denom + np.complex128(1e-12 + 0j), denom)
        W = (r - rs) / denom
    else:
        raise ValueError(f"Unknown warp_mode={mode!r}. Expected: plus, minus, ratio.")

    logW = np.log(W)
    u = np.real(logW).astype(np.float64)
    theta = np.imag(logW).astype(np.float64)

    # Branch convention: unwrap phase along ordered primes to maintain continuity.
    if bool(cfg.get("warp_unwrap", True)) and theta.size:
        theta = np.unwrap(theta).astype(np.float64)
    return u, theta

def _warp_u_theta_ratio_selfdual(primes_used, cfg: dict):
        """Self-dual ratio warp with an explicit branch/unwrap convention.

        Enforces the self-dual identity pointwise:
            r(p) + r_s(p) = 1
        by parameterizing
            r(p)   = 1/2 + i y(p)
            r_s(p) = 1/2 - i y(p)
        so the critical-line locus r=r_s is exactly y(p)=0.

        Warp chart (Möbius/ratio):
            W(p) = (r(p) - r_s(p)) / (r(p) + r_s(p))
        and we define
            log W(p) = u(p) + i theta(p)
        with a fixed unwrap convention if enabled.
        """
        ps = list(map(int, primes_used))
        x = np.log(np.asarray(ps, dtype=np.float64))

        twopi = 2.0 * np.pi
        frac = (x / twopi) - np.floor(x / twopi)
        theta0 = (frac * twopi) - np.pi
        y_scale = float(cfg.get("warp_y_scale", 1.0))
        y = y_scale * theta0

        r = (0.5 + 1j * y.astype(np.complex128)).astype(np.complex128)
        rs = (0.5 - 1j * y.astype(np.complex128)).astype(np.complex128)

        denom = (r + rs)
        denom = np.where(np.abs(denom) < 1e-15, denom + np.complex128(1e-15 + 0j), denom)
        W = (r - rs) / denom
        # controlled zero at y=0 => W=0; avoid -inf from log by jittering magnitude
        W = np.where(np.abs(W) < 1e-15, W + np.complex128(1e-15 + 0j), W)

        logW = np.log(W)
        u = np.real(logW).astype(np.float64)
        theta = np.imag(logW).astype(np.float64)
        if bool(cfg.get("warp_unwrap", True)) and theta.size:
                theta = np.unwrap(theta).astype(np.float64)
        # Optional clipping for bounded admissible class.
        u_clip = cfg.get("geom_v4_u_clip", None)
        if u_clip is not None:
            u = np.clip(u, -float(u_clip), float(u_clip)).astype(np.float64)
        th_clip = cfg.get("geom_v4_theta_clip", None)
        if th_clip is not None:
            theta = np.clip(theta, -float(th_clip), float(th_clip)).astype(np.float64)
        return u, theta


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Map angles to (-pi, pi] deterministically."""
    x = np.asarray(x, dtype=np.float64)
    return np.arctan2(np.sin(x), np.cos(x)).astype(np.float64)


def _v9_transport_theta(primes_used: list[int], cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (u, theta_raw, theta_transport) for v9.

    Transport rule is deterministic and depends only on primes_used + cfg knobs.
    """
    u, theta_raw = _warp_u_theta_ratio_selfdual(primes_used, cfg)
    theta_raw = np.asarray(theta_raw, dtype=np.float64)

    rule = str(cfg.get("geom_v9_transport_rule", "delta_raw_wrapped")).strip().lower()
    lam = float(cfg.get("geom_v9_transport_lambda", 1.0))
    clip = float(cfg.get("geom_v9_transport_clip", 0.25))

    n = int(theta_raw.size)
    theta_tr = np.array(theta_raw, copy=True)
    if n <= 1:
        return np.asarray(u, dtype=np.float64), theta_raw, theta_tr

    if not np.isfinite(lam):
        lam = 0.0
    if not np.isfinite(clip) or clip < 0:
        clip = 0.0

    for i in range(1, n):
        if rule in {"delta_raw_wrapped", "raw_wrapped", "delta_wrapped"}:
            dtheta = float(_wrap_to_pi(theta_raw[i] - theta_raw[i - 1]))
        else:
            raise ValueError(
                f"Unknown geom_v9_transport_rule={rule!r}. Expected: delta_raw_wrapped"
            )
        inc = float(lam) * float(dtheta)
        if clip > 0:
            inc = float(np.clip(inc, -clip, clip))
        theta_tr[i] = float(theta_tr[i - 1]) + float(inc)

    return np.asarray(u, dtype=np.float64), theta_raw, np.asarray(theta_tr, dtype=np.float64)


def _v10_phasefield_theta(primes_used: list[int], cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (u, theta_raw, theta_solved) for v10.

    Uses a global phase-field solve (discrete Laplacian smoothing with tether) along the ordered
    input list. Deterministic given primes_used + cfg knobs.

    Solve:
        (L + eta I) theta = eta theta_raw
    where L is the graph Laplacian of a path on the ordered inputs.
    """
    u, theta_raw = _warp_u_theta_ratio_selfdual(primes_used, cfg)
    u = np.asarray(u, dtype=np.float64)
    theta_raw = np.asarray(theta_raw, dtype=np.float64)

    n = int(theta_raw.size)
    if n <= 1:
        theta_sol = np.array(theta_raw, copy=True)
    else:
        graph = str(cfg.get("geom_v10_phasefield_graph", "path")).strip().lower()
        if graph not in {"path"}:
            raise ValueError(f"Unknown geom_v10_phasefield_graph={graph!r}. Expected: path")

        w = float(cfg.get("geom_v10_phasefield_w", 1.0))
        if not np.isfinite(w) or w <= 0:
            w = 1.0
        eta = float(cfg.get("geom_v10_phasefield_eta", 0.25))
        if not np.isfinite(eta) or eta <= 0:
            eta = 1e-6

        # Path-graph Laplacian (tridiagonal)
        L = np.zeros((n, n), dtype=np.float64)
        for i in range(n - 1):
            L[i, i] += w
            L[i + 1, i + 1] += w
            L[i, i + 1] -= w
            L[i + 1, i] -= w

        A = L + (eta * np.eye(n, dtype=np.float64))
        b = eta * theta_raw
        theta_sol = np.linalg.solve(A, b).astype(np.float64)

    # Optional clipping for bounded admissible class (use v10 clip keys).
    u_clip = cfg.get("geom_v10_u_clip", None)
    if u_clip is not None:
        u = np.clip(u, -float(u_clip), float(u_clip)).astype(np.float64)

    th_clip = cfg.get("geom_v10_theta_clip", None)
    if th_clip is not None:
        theta_raw = np.clip(theta_raw, -float(th_clip), float(th_clip)).astype(np.float64)
        theta_sol = np.clip(theta_sol, -float(th_clip), float(th_clip)).astype(np.float64)

    return u, theta_raw, np.asarray(theta_sol, dtype=np.float64)

# ----------------------------
# 4.1) GEOM V2A/V2B: warp-derived depth/phase without shared coupling
# ----------------------------
if "build_Tp_geom_v2a" not in globals():
    def build_Tp_geom_v2a(primes_used, d: int, seed: int, cfg: dict):
        """Depth-only warp backend: Tp[p] = exp(depth_scale * u(p) * H0)."""
        d = int(d)
        depth_scale = float(cfg.get("geom_v2_depth_scale", 1.0))
        u, theta = _warp_u_theta_from_complex_r(primes_used, cfg)

        rng = np.random.default_rng(int(seed) * 99991 + 123)
        A0 = rng.standard_normal((d, d))
        H0 = _sym(A0) / np.sqrt(max(1, d))

        Tp = {}
        for i, p in enumerate(map(int, primes_used)):
            Hu = (depth_scale * float(u[i])) * H0
            Tp[p] = _expm_via_eig_sym(Hu)
        return Tp, u, theta

if "build_Tp_geom_v2b" not in globals():
    def build_Tp_geom_v2b(primes_used, d: int, seed: int, cfg: dict):
        """Depth+phase warp backend: Tp[p] = exp(depth_scale*u(p)*H0) @ exp(rot_scale*theta(p)*J)."""
        d = int(d)
        depth_scale = float(cfg.get("geom_v2_depth_scale", 1.0))
        rot_scale = float(cfg.get("geom_v2_rot_scale", 0.25))
        u, theta = _warp_u_theta_from_complex_r(primes_used, cfg)

        rng = np.random.default_rng(int(seed) * 99991 + 123)
        A0 = rng.standard_normal((d, d))
        H0 = _sym(A0) / np.sqrt(max(1, d))

        Aj = rng.standard_normal((d, d))
        J = _skew(Aj) / np.sqrt(max(1, d))

        Tp = {}
        for i, p in enumerate(map(int, primes_used)):
            Hu = (depth_scale * float(u[i])) * H0
            E1 = _expm_via_eig_sym(Hu)
            E2 = _expm_via_eig_skew(rot_scale * float(theta[i]) * J)
            Tp[p] = E1 @ E2
        return Tp, u, theta

def build_Tp_geom_v3_shared_axis(primes_used, d: int, seed: int, cfg: dict):
    """
    Phase-3B legal backend:
      Tp[p] = exp( u(p) H0 + eps*u(p) C_shared )  @  exp( theta(p) J )
    - complex r and complex r_s drive (u, theta)
    - C_shared is low-rank shared axis (non-decomposable coupling)
    """
    d = int(d)
    eps = float(cfg.get("geom_v3_eps", 0.01))  # start 0.01 (1–5% delta target)
    # Back-compat: geom_v3_rot_scale was the original knob; geom_v3_theta_scale is the preferred name.
    theta_scale_raw = cfg.get("geom_v3_theta_scale", None)
    if theta_scale_raw is None:
        theta_scale_raw = cfg.get("geom_v3_rot_scale", 0.25)
    theta_scale = float(theta_scale_raw)

    u, theta = _warp_u_theta_from_complex_r(primes_used, cfg)

    rng = np.random.default_rng(int(seed) * 99991 + 123)
    # H0 symmetric "radial/depth" generator template
    A0 = rng.standard_normal((d, d))
    H0 = _sym(A0) / np.sqrt(max(1, d))

    # Shared axis C_shared (rank-1 or rank-2)
    rank = int(cfg.get("geom_v3_rank", 1))
    U = rng.standard_normal((d, rank)) / np.sqrt(max(1, d))
    C_shared = _sym(U @ U.T)  # PSD/symmetric, shared for all primes

    # J skew "chiral" generator (fixed)
    Aj = rng.standard_normal((d, d))
    J = _skew(Aj) / np.sqrt(max(1, d))

    Tp = {}
    ps = list(map(int, primes_used))
    for i, p in enumerate(ps):
        Hu = (u[i] * H0) + (eps * u[i] * C_shared)
        # symmetric exp
        E1 = _expm_via_eig_sym(Hu)
        # skew exp (rotation-like)
        E2 = _expm_via_eig_skew(theta_scale * theta[i] * J)
        Tp[p] = E1 @ E2

    return Tp, u, theta

def build_Tp_geom_warp_dirac_v4(primes_used, d: int, seed: int, cfg: dict):
    """Gauge-preregistered warp + Dirac-ish (chiral) deformation of legacy Tp.

    Key points:
    - Keeps the measurement spine untouched (Tp-only backend).
    - Uses a self-dual ratio warp with fixed unwrap convention.
    - Builds Tp as a controlled deformation of legacy Tp in log-eigen coordinates,
      enabling reviewer-proof delta-band tuning via a single global scale.
    """
    d = int(d)
    if d < 4 or (d % 2) != 0:
        raise ValueError("geom_warp_dirac_v4 requires even d >= 4 for chiral split")

    # v4 warp is self-dual ratio (r+r_s=1) by construction.
    u, theta = _warp_u_theta_ratio_selfdual(primes_used, cfg)

    # Base (legacy) Tp used as the deformation reference.
    Tp_base = build_Tp_synth(primes_used, d=d, seed=int(seed))

    # Templates (seed-fixed)
    rng = np.random.default_rng(int(seed) * 99991 + 456)
    A0 = rng.standard_normal((d, d))
    H0 = _sym(A0) / np.sqrt(max(1, d))

    rank = int(cfg.get("geom_v4_rank", cfg.get("geom_v3_rank", 1)))
    rank = max(1, min(rank, d))
    U = rng.standard_normal((d, rank)) / np.sqrt(max(1, d))
    C_shared = _sym(U @ U.T)

    n = d // 2
    I = np.eye(n, dtype=np.float64)
    Z = np.zeros((n, n), dtype=np.float64)
    X_chiral = np.block([[Z, I], [I, Z]])

    u_scale = float(cfg.get("geom_v4_u_scale", 1.0))
    theta_scale = float(cfg.get("geom_v4_theta_scale", 0.125))
    eps = float(cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01)))
    eps_global = float(cfg.get("geom_v4_eps_global", 0.02))

    Tp = {}
    for i, p in enumerate(map(int, primes_used)):
        Lp = _logm_via_eig_sym(np.asarray(Tp_base[int(p)], dtype=np.float64))
        Dp = (u_scale * float(u[i])) * H0 + (eps * u_scale * float(u[i])) * C_shared + (theta_scale * float(theta[i])) * X_chiral
        G = _sym(Lp + (eps_global * Dp))
        Tp[int(p)] = _expm_via_eig_sym(G)
    return Tp, u, theta

def build_Tp_geom_warp_dirac_v5(primes_used, d: int, seed: int, cfg: dict):
    """v5 = v4 + self-dual harmonic modulation of chiral coupling.

    The goal is to make the self-dual critical-line choice structural: the same
    self-dual term that encodes r+r_s=1 also controls coupling strength.

    Implementation detail:
    - r = 1/2 + i y(p), r_s = 1/2 - i y(p)
    - harmonic term (self-dual): h(p) = Re(1/r + 1/r_s) = 1 / (1/4 + y(p)^2)
    - normalized modulation: m(p) = (h(p) / median(h))**gamma
    - chiral amplitude scales as theta_scale * theta(p) * strength * m(p)
    """
    d = int(d)
    if d < 4 or (d % 2) != 0:
        raise ValueError("geom_warp_dirac_v5 requires even d >= 4 for chiral split")

    # v5 uses the same self-dual ratio warp for (u, theta).
    # To compute the self-dual harmonic modulation, we reconstruct y(p) deterministically.
    ps = list(map(int, primes_used))
    x = np.log(np.asarray(ps, dtype=np.float64))
    twopi = 2.0 * np.pi
    frac = (x / twopi) - np.floor(x / twopi)
    theta0 = (frac * twopi) - np.pi
    y_scale = float(cfg.get("warp_y_scale", 1.0))
    y = y_scale * theta0

    u, theta = _warp_u_theta_ratio_selfdual(primes_used, cfg)

    # Base (legacy) Tp used as the deformation reference.
    Tp_base = build_Tp_synth(primes_used, d=d, seed=int(seed))

    # Templates (seed-fixed)
    rng = np.random.default_rng(int(seed) * 99991 + 457)
    A0 = rng.standard_normal((d, d))
    H0 = _sym(A0) / np.sqrt(max(1, d))

    rank = int(cfg.get("geom_v5_rank", cfg.get("geom_v4_rank", cfg.get("geom_v3_rank", 1))))
    rank = max(1, min(rank, d))
    U = rng.standard_normal((d, rank)) / np.sqrt(max(1, d))
    C_shared = _sym(U @ U.T)

    n = d // 2
    I = np.eye(n, dtype=np.float64)
    Z = np.zeros((n, n), dtype=np.float64)
    X_chiral = np.block([[Z, I], [I, Z]])

    u_scale = float(cfg.get("geom_v5_u_scale", 1.0))
    theta_scale = float(cfg.get("geom_v5_theta_scale", 0.125))
    eps = float(cfg.get("geom_v5_eps", cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01))))
    eps_global = float(cfg.get("geom_v5_eps_global", 0.02))

    gamma = float(cfg.get("geom_v5_harmonic_gamma", 1.0))
    strength = float(cfg.get("geom_v5_harmonic_strength", 1.0))
    h = 1.0 / (0.25 + np.square(y.astype(np.float64)))
    h_med = float(np.median(h)) if h.size else 1.0
    if not np.isfinite(h_med) or h_med <= 0:
        h_med = 1.0
    mod = np.power(h / h_med, gamma, dtype=np.float64)

    Tp = {}
    for i, p in enumerate(ps):
        Lp = _logm_via_eig_sym(np.asarray(Tp_base[int(p)], dtype=np.float64))
        chiral_amp = (theta_scale * float(theta[i])) * (strength * float(mod[i]))
        Dp = (u_scale * float(u[i])) * H0 + (eps * u_scale * float(u[i])) * C_shared + chiral_amp * X_chiral
        G = _sym(Lp + (eps_global * Dp))
        Tp[int(p)] = _expm_via_eig_sym(G)
    return Tp, u, theta


def build_Tp_geom_warp_dirac_v6(primes_used, d: int, seed: int, cfg: dict):
    """v6 = v4 + self-dual harmonic phase perturbation.

    v5 modulated chiral *amplitude* by a self-dual harmonic term h(p).
    v6 instead uses h(p) to slightly reparameterize the *phase* theta(p), aiming
    to target coherence in phase-sensitive windows while avoiding broad null
    variance inflation from amplitude scaling.

    Spec:
      h(p) = 1 / (1/4 + y(p)^2)
      theta'(p) = theta(p) + lambda * clip( log(h(p)/median(h)), [-clip, clip] )
      chiral_amp = theta_scale * theta'(p)
    """
    d = int(d)
    if d < 4 or (d % 2) != 0:
        raise ValueError("geom_warp_dirac_v6 requires even d >= 4 for chiral split")

    ps = list(map(int, primes_used))
    x = np.log(np.asarray(ps, dtype=np.float64))
    twopi = 2.0 * np.pi
    frac = (x / twopi) - np.floor(x / twopi)
    theta0 = (frac * twopi) - np.pi
    y_scale = float(cfg.get("warp_y_scale", 1.0))
    y = y_scale * theta0

    u, theta = _warp_u_theta_ratio_selfdual(primes_used, cfg)

    Tp_base = build_Tp_synth(primes_used, d=d, seed=int(seed))

    rng = np.random.default_rng(int(seed) * 99991 + 458)
    A0 = rng.standard_normal((d, d))
    H0 = _sym(A0) / np.sqrt(max(1, d))

    rank = int(cfg.get("geom_v6_rank", cfg.get("geom_v4_rank", cfg.get("geom_v3_rank", 1))))
    rank = max(1, min(rank, d))
    U = rng.standard_normal((d, rank)) / np.sqrt(max(1, d))
    C_shared = _sym(U @ U.T)

    n = d // 2
    I = np.eye(n, dtype=np.float64)
    Z = np.zeros((n, n), dtype=np.float64)
    X_chiral = np.block([[Z, I], [I, Z]])

    u_scale = float(cfg.get("geom_v6_u_scale", 1.0))
    theta_scale = float(cfg.get("geom_v6_theta_scale", 0.125))
    eps = float(cfg.get("geom_v6_eps", cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01))))
    eps_global = float(cfg.get("geom_v6_eps_global", 0.02))

    phase_lambda = float(cfg.get("geom_v6_phase_lambda", 0.05))
    phase_clip = float(cfg.get("geom_v6_phase_clip", 2.0))

    h = 1.0 / (0.25 + np.square(y.astype(np.float64)))
    h_med = float(np.median(h)) if h.size else 1.0
    if not np.isfinite(h_med) or h_med <= 0:
        h_med = 1.0

    log_ratio = np.log(np.maximum(h, 1e-300) / h_med)
    if np.isfinite(phase_clip) and phase_clip > 0:
        log_ratio = np.clip(log_ratio, -phase_clip, phase_clip)
    theta_p = theta + (phase_lambda * log_ratio)

    Tp = {}
    for i, p in enumerate(ps):
        Lp = _logm_via_eig_sym(np.asarray(Tp_base[int(p)], dtype=np.float64))
        chiral_amp = (theta_scale * float(theta_p[i]))
        Dp = (u_scale * float(u[i])) * H0 + (eps * u_scale * float(u[i])) * C_shared + chiral_amp * X_chiral
        G = _sym(Lp + (eps_global * Dp))
        Tp[int(p)] = _expm_via_eig_sym(G)
    return Tp, u, theta_p


def _v7_defect_mask_and_sign(ps: list[int], seed: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic localized-defect selector.

    Returns:
      mask: float64 array in {0,1}
      sign: float64 array in {-1,+1}

    Notes:
    - Designed to be deterministic across Python/Numpy versions (uses sha256, not RNG streams).
    - Pattern depends only on (seed, primes_used, cfg knobs), and is independent of any measurement.
    """
    ps = list(map(int, ps))
    n = len(ps)
    if n == 0:
        return np.zeros((0,), dtype=np.float64), np.ones((0,), dtype=np.float64)

    mode = str(cfg.get("geom_v7_defect_mode", "block")).strip().lower()
    rho = float(cfg.get("geom_v7_defect_rho", 0.25))
    rho = float(np.clip(rho, 0.0, 1.0))
    salt = int(cfg.get("geom_v7_defect_salt", 1))
    # IMPORTANT: the defect *pattern* is keyed to a geometry-class seed, not the observation seed.
    seed = int(cfg.get("geom_v7_defect_seed", 0))

    mask = np.zeros((n,), dtype=np.float64)

    if mode in {"block", "contiguous"}:
        k = int(max(0, min(n, int(round(rho * n)))))
        if k <= 0 and rho > 0:
            k = 1
        if k > 0:
            # choose a stable start index that keeps the block in-bounds
            h = hashlib.sha256(f"v7|block|{seed}|{salt}".encode("utf-8")).hexdigest()
            hv = int(h[:16], 16)
            start = 0
            if n - k > 0:
                start = int(hv % (n - k + 1))
            mask[start : start + k] = 1.0
    elif mode in {"bernoulli", "sparse", "iid"}:
        # stable per-prime thresholding
        if rho <= 0:
            pass
        elif rho >= 1:
            mask[:] = 1.0
        else:
            for i, p in enumerate(ps):
                h = hashlib.sha256(f"v7|bern|{seed}|{salt}|{p}".encode("utf-8")).hexdigest()
                hv = int(h[:16], 16) / float(16**16)
                if hv < rho:
                    mask[i] = 1.0
    else:
        raise ValueError(f"Unknown geom_v7_defect_mode={mode!r}. Expected: block or bernoulli")

    sign = np.ones((n,), dtype=np.float64)
    for i, p in enumerate(ps):
        if mask[i] == 0.0:
            continue
        h = hashlib.sha256(f"v7|sign|{seed}|{salt}|{p}".encode("utf-8")).hexdigest()
        bit = int(h[-1], 16) & 1
        sign[i] = 1.0 if bit == 1 else -1.0

    return mask, sign


def _v8_prime_mask_and_sign(ps: list[int], cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic prime-side activation selector for v8.

    Returns:
      mask: float64 array in {0,1}
      sign: float64 array in {-1,+1}

    Notes:
    - Pattern is keyed to geometry-class constants (geom_v8_defect_seed/salt), not observation seed.
    """
    ps = list(map(int, ps))
    n = len(ps)
    if n == 0:
        return np.zeros((0,), dtype=np.float64), np.ones((0,), dtype=np.float64)

    mode = str(cfg.get("geom_v8_prime_defect_mode", "block")).strip().lower()
    rho = float(cfg.get("geom_v8_prime_defect_rho", 0.25))
    rho = float(np.clip(rho, 0.0, 1.0))
    salt = int(cfg.get("geom_v8_defect_salt", 1))
    seed = int(cfg.get("geom_v8_defect_seed", 0))

    mask = np.zeros((n,), dtype=np.float64)
    if mode in {"block", "contiguous"}:
        k = int(max(0, min(n, int(round(rho * n)))))
        if k <= 0 and rho > 0:
            k = 1
        if k > 0:
            h = hashlib.sha256(f"v8|block|{seed}|{salt}".encode("utf-8")).hexdigest()
            hv = int(h[:16], 16)
            start = 0
            if n - k > 0:
                start = int(hv % (n - k + 1))
            mask[start : start + k] = 1.0
    elif mode in {"bernoulli", "sparse", "iid"}:
        if rho <= 0:
            pass
        elif rho >= 1:
            mask[:] = 1.0
        else:
            for i, p in enumerate(ps):
                h = hashlib.sha256(f"v8|bern|{seed}|{salt}|{p}".encode("utf-8")).hexdigest()
                hv = int(h[:16], 16) / float(16**16)
                if hv < rho:
                    mask[i] = 1.0
    else:
        raise ValueError(f"Unknown geom_v8_prime_defect_mode={mode!r}. Expected: block or bernoulli")

    sign = np.ones((n,), dtype=np.float64)
    for i, p in enumerate(ps):
        if mask[i] == 0.0:
            continue
        h = hashlib.sha256(f"v8|sign|{seed}|{salt}|{p}".encode("utf-8")).hexdigest()
        bit = int(h[-1], 16) & 1
        sign[i] = 1.0 if bit == 1 else -1.0

    return mask, sign


def _v8_boundary_coord_mask(cfg: dict, d: int) -> np.ndarray:
    """Coordinate-space mask tethered to the fixed DN-map measurement boundary indices."""
    d = int(d)
    if d <= 0:
        return np.zeros((0,), dtype=np.float64)

    bidx = get_measurement_boundary_idx(cfg, d)
    n2 = d // 2
    pair = bool(cfg.get("geom_v8_pair_chiral", True)) and (d % 2 == 0) and (n2 > 0)

    chosen = set(map(int, np.asarray(bidx, dtype=np.int64).tolist()))
    if pair:
        # Ensure chiral coupling can actually be localized: include both halves for each k.
        expanded = set()
        for j in chosen:
            k = int(j) % int(n2)
            expanded.add(int(k))
            expanded.add(int(k + n2))
        chosen = expanded

    m = np.zeros((d,), dtype=np.float64)
    for j in chosen:
        if 0 <= int(j) < d:
            m[int(j)] = 1.0
    return m


def build_Tp_geom_warp_dirac_v7_localdefect(primes_used, d: int, seed: int, cfg: dict):
    """v7 = v4 + localized defect modulation of chiral coupling.

    Concept:
    - Keep v4's self-dual ratio warp for (u, theta).
    - Introduce a *localized* subset of primes where the chiral amplitude is
      multiplied by a bounded factor (1 ± strength), optionally with a contiguous block.

    This is Phase‑3B legal: only Tp construction changes; the DN-map/logdet spine is untouched.
    """
    d = int(d)
    if d < 4 or (d % 2) != 0:
        raise ValueError("geom_warp_dirac_v7_localdefect requires even d >= 4 for chiral split")

    u, theta = _warp_u_theta_ratio_selfdual(primes_used, cfg)

    ps = list(map(int, primes_used))
    mask, sgn = _v7_defect_mask_and_sign(ps, seed=int(seed), cfg=cfg)

    Tp_base = build_Tp_synth(primes_used, d=d, seed=int(seed))

    rng = np.random.default_rng(int(seed) * 99991 + 459)
    A0 = rng.standard_normal((d, d))
    H0 = _sym(A0) / np.sqrt(max(1, d))

    rank = int(cfg.get("geom_v7_rank", cfg.get("geom_v4_rank", cfg.get("geom_v3_rank", 1))))
    rank = max(1, min(rank, d))
    U = rng.standard_normal((d, rank)) / np.sqrt(max(1, d))
    C_shared = _sym(U @ U.T)

    n = d // 2
    I = np.eye(n, dtype=np.float64)
    Z = np.zeros((n, n), dtype=np.float64)
    X_chiral = np.block([[Z, I], [I, Z]])

    u_scale = float(cfg.get("geom_v7_u_scale", 1.0))
    theta_scale = float(cfg.get("geom_v7_theta_scale", 0.125))
    eps = float(cfg.get("geom_v7_eps", cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01))))
    eps_global = float(cfg.get("geom_v7_eps_global", 0.02))

    strength = float(cfg.get("geom_v7_defect_strength", 1.0))
    clip = float(cfg.get("geom_v7_defect_clip", 0.75))
    clip = float(max(0.0, clip))

    # multiplier is bounded to [1-clip, 1+clip], keeping it nonnegative when clip<=1.
    mult = 1.0 + (strength * mask * sgn)
    mult = np.clip(mult, 1.0 - clip, 1.0 + clip).astype(np.float64)

    Tp = {}
    for i, p in enumerate(ps):
        Lp = _logm_via_eig_sym(np.asarray(Tp_base[int(p)], dtype=np.float64))
        chiral_amp = (theta_scale * float(theta[i])) * float(mult[i])
        Dp = (u_scale * float(u[i])) * H0 + (eps * u_scale * float(u[i])) * C_shared + chiral_amp * X_chiral
        G = _sym(Lp + (eps_global * Dp))
        Tp[int(p)] = _expm_via_eig_sym(G)

    # Expose the effective theta used by the chiral channel for fingerprinting.
    theta_eff = theta * mult
    return Tp, u, theta_eff


def build_Tp_geom_warp_dirac_v8_boundarydefect(primes_used, d: int, seed: int, cfg: dict):
    """v8 = v4 + boundary-tethered defect on the chiral channel.

    Mechanism:
    - Keep v4's self-dual ratio warp for (u, theta).
    - Add an *extra* chiral term supported on the (fixed) DN-map measurement boundary indices
      (optionally paired across chiral halves), with a deterministic prime-side activation mask.

    This is Phase‑3B legal: only Tp construction changes; the DN-map/logdet spine is untouched.
    """
    d = int(d)
    if d < 4 or (d % 2) != 0:
        raise ValueError("geom_warp_dirac_v8_boundarydefect requires even d >= 4 for chiral split")

    u, theta = _warp_u_theta_ratio_selfdual(primes_used, cfg)
    ps = list(map(int, primes_used))
    p_mask, p_sgn = _v8_prime_mask_and_sign(ps, cfg=cfg)

    Tp_base = build_Tp_synth(primes_used, d=d, seed=int(seed))

    rng = np.random.default_rng(int(seed) * 99991 + 460)
    A0 = rng.standard_normal((d, d))
    H0 = _sym(A0) / np.sqrt(max(1, d))

    rank = int(cfg.get("geom_v8_rank", cfg.get("geom_v4_rank", cfg.get("geom_v3_rank", 1))))
    rank = max(1, min(rank, d))
    U = rng.standard_normal((d, rank)) / np.sqrt(max(1, d))
    C_shared = _sym(U @ U.T)

    n = d // 2
    I = np.eye(n, dtype=np.float64)
    Z = np.zeros((n, n), dtype=np.float64)
    X_chiral = np.block([[Z, I], [I, Z]])

    # Boundary-tethered coordinate mask -> local chiral template
    coord_mask = _v8_boundary_coord_mask(cfg, d=d)
    X_defect = (coord_mask[:, None] * X_chiral) * coord_mask[None, :]

    norm_mode = str(cfg.get("geom_v8_defect_norm_mode", "fro_match")).strip().lower()
    defect_scale = 1.0
    if norm_mode in {"fro_match", "fro", "match"}:
        nx = float(np.linalg.norm(X_chiral, ord="fro"))
        nd = float(np.linalg.norm(X_defect, ord="fro"))
        defect_scale = (nx / nd) if (nd > 0 and np.isfinite(nd) and np.isfinite(nx)) else 0.0
        X_defect = (defect_scale * X_defect).astype(np.float64)
    elif norm_mode in {"none", "off"}:
        defect_scale = 1.0
    else:
        raise ValueError(f"Unknown geom_v8_defect_norm_mode={norm_mode!r}. Expected: none or fro_match")

    u_scale = float(cfg.get("geom_v8_u_scale", 1.0))
    theta_scale = float(cfg.get("geom_v8_theta_scale", 0.125))
    eps = float(cfg.get("geom_v8_eps", cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01))))
    eps_global = float(cfg.get("geom_v8_eps_global", 0.02))

    strength = float(cfg.get("geom_v8_defect_strength", 1.0))
    clip = float(cfg.get("geom_v8_defect_clip", 0.75))
    clip = float(max(0.0, clip))

    # Defect coefficient is bounded to [-clip, +clip]
    coeff = (strength * p_mask * p_sgn).astype(np.float64)
    if np.isfinite(clip) and clip >= 0:
        coeff = np.clip(coeff, -clip, clip)

    Tp = {}
    for i, p in enumerate(ps):
        Lp = _logm_via_eig_sym(np.asarray(Tp_base[int(p)], dtype=np.float64))
        chiral_term = (theta_scale * float(theta[i])) * (X_chiral + (float(coeff[i]) * X_defect))
        Dp = (u_scale * float(u[i])) * H0 + (eps * u_scale * float(u[i])) * C_shared + chiral_term
        G = _sym(Lp + (eps_global * Dp))
        Tp[int(p)] = _expm_via_eig_sym(G)

    # Fingerprint scalar: encode the prime-side coefficient into theta_eff.
    theta_eff = theta * (1.0 + coeff)
    return Tp, u, theta_eff


def build_Tp_geom_warp_dirac_v9_phasetransport(primes_used, d: int, seed: int, cfg: dict):
    """v9 = v4 + global phase transport along ordered primes.

    Mechanism:
    - Compute theta_raw(p) from the self-dual ratio warp (with unwrap convention).
    - Construct theta_tr(p) sequentially via a bounded transport rule.
    - Use theta_tr(p) in the chiral channel, keeping the rest of v4's log-eigen deformation.

    This is Phase‑3B legal: only Tp construction changes; the DN-map/logdet spine is untouched.
    """
    d = int(d)
    if d < 4 or (d % 2) != 0:
        raise ValueError("geom_warp_dirac_v9_phasetransport requires even d >= 4 for chiral split")

    u, theta_raw, theta_tr = _v9_transport_theta(primes_used, cfg)
    ps = list(map(int, primes_used))

    Tp_base = build_Tp_synth(primes_used, d=d, seed=int(seed))

    rng = np.random.default_rng(int(seed) * 99991 + 461)
    A0 = rng.standard_normal((d, d))
    H0 = _sym(A0) / np.sqrt(max(1, d))

    rank = int(cfg.get("geom_v9_rank", cfg.get("geom_v4_rank", cfg.get("geom_v3_rank", 1))))
    rank = max(1, min(rank, d))
    U = rng.standard_normal((d, rank)) / np.sqrt(max(1, d))
    C_shared = _sym(U @ U.T)

    n = d // 2
    I = np.eye(n, dtype=np.float64)
    Z = np.zeros((n, n), dtype=np.float64)
    X_chiral = np.block([[Z, I], [I, Z]])

    u_scale = float(cfg.get("geom_v9_u_scale", 1.0))
    theta_scale = float(cfg.get("geom_v9_theta_scale", 0.125))
    eps = float(cfg.get("geom_v9_eps", cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01))))
    eps_global = float(cfg.get("geom_v9_eps_global", 0.02))

    Tp = {}
    for i, p in enumerate(ps):
        Lp = _logm_via_eig_sym(np.asarray(Tp_base[int(p)], dtype=np.float64))
        chiral_amp = (theta_scale * float(theta_tr[i]))
        Dp = (u_scale * float(u[i])) * H0 + (eps * u_scale * float(u[i])) * C_shared + chiral_amp * X_chiral
        G = _sym(Lp + (eps_global * Dp))
        Tp[int(p)] = _expm_via_eig_sym(G)

    return Tp, np.asarray(u, dtype=np.float64), np.asarray(theta_tr, dtype=np.float64)


def build_Tp_geom_warp_dirac_v10_phasefieldsolve(primes_used, d: int, seed: int, cfg: dict):
    """v10 = v4 + global phase-field solve along ordered primes.

    Mechanism:
    - Compute theta_raw(p) from the self-dual ratio warp (with unwrap convention).
    - Compute theta_sol by a global Laplacian+tether solve (a deterministic, nonlocal coupling).
    - Use theta_sol(p) in the chiral channel, keeping the rest of v4's log-eigen deformation.

    Phase‑3B legal: only Tp construction changes; the DN-map/logdet spine is untouched.
    """
    d = int(d)
    if d < 4 or (d % 2) != 0:
        raise ValueError("geom_warp_dirac_v10_phasefieldsolve requires even d >= 4 for chiral split")

    u, _, theta_sol = _v10_phasefield_theta(primes_used, cfg)
    ps = list(map(int, primes_used))

    Tp_base = build_Tp_synth(primes_used, d=d, seed=int(seed))

    rng = np.random.default_rng(int(seed) * 99991 + 462)
    A0 = rng.standard_normal((d, d))
    H0 = _sym(A0) / np.sqrt(max(1, d))

    rank = int(cfg.get("geom_v10_rank", cfg.get("geom_v4_rank", cfg.get("geom_v3_rank", 1))))
    rank = max(1, min(rank, d))
    U = rng.standard_normal((d, rank)) / np.sqrt(max(1, d))
    C_shared = _sym(U @ U.T)

    n = d // 2
    I = np.eye(n, dtype=np.float64)
    Z = np.zeros((n, n), dtype=np.float64)
    X_chiral = np.block([[Z, I], [I, Z]])

    u_scale = float(cfg.get("geom_v10_u_scale", 1.0))
    theta_scale = float(cfg.get("geom_v10_theta_scale", 0.125))
    eps = float(cfg.get("geom_v10_eps", cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01))))
    eps_global = float(cfg.get("geom_v10_eps_global", 0.02))

    Tp = {}
    for i, p in enumerate(ps):
        Lp = _logm_via_eig_sym(np.asarray(Tp_base[int(p)], dtype=np.float64))
        chiral_amp = (theta_scale * float(theta_sol[i]))
        Dp = (u_scale * float(u[i])) * H0 + (eps * u_scale * float(u[i])) * C_shared + chiral_amp * X_chiral
        G = _sym(Lp + (eps_global * Dp))
        Tp[int(p)] = _expm_via_eig_sym(G)

    return Tp, np.asarray(u, dtype=np.float64), np.asarray(theta_sol, dtype=np.float64)


def _v7_defect_metadata(primes_used: list[int], cfg: dict) -> dict:
    """Derived-only diagnostics for v7 localized defects (no measurement-spine changes)."""
    ps = list(map(int, primes_used))
    mode = str(cfg.get("geom_v7_defect_mode", "block")).strip().lower()
    rho = float(cfg.get("geom_v7_defect_rho", 0.25))
    strength = float(cfg.get("geom_v7_defect_strength", 1.0))
    clip = float(cfg.get("geom_v7_defect_clip", 0.75))
    salt = int(cfg.get("geom_v7_defect_salt", 1))
    defect_seed = int(cfg.get("geom_v7_defect_seed", 0))

    mask, sgn = _v7_defect_mask_and_sign(ps, seed=defect_seed, cfg=cfg)
    support_primes = [int(p) for (p, m) in zip(ps, mask.tolist()) if float(m) == 1.0]
    support_size = int(len(support_primes))

    # Digest the marked set (small for k=13, but we keep only a digest for audit friendliness)
    support_digest = sha16(
        repr(
            {
                "mode": mode,
                "rho": rho,
                "salt": salt,
                "defect_seed": defect_seed,
                "support_primes": support_primes,
            }
        ).encode("utf-8")
    )

    rule = f"{mode}|rho={rho:.6g}|salt={salt}|defect_seed={defect_seed}"
    return {
        "defect_support_size": support_size,
        "defect_support_rule": rule,
        "defect_primes_marked_digest": support_digest,
        "defect_strength": float(strength),
        "defect_clip": float(clip),
    }


def _v8_boundarydefect_metadata(primes_used: list[int], cfg: dict, d: int) -> dict:
    """Derived-only diagnostics for v8 boundary-tethered defects (no measurement-spine changes)."""
    ps = list(map(int, primes_used))
    mode = str(cfg.get("geom_v8_prime_defect_mode", "block")).strip().lower()
    rho = float(cfg.get("geom_v8_prime_defect_rho", 0.25))
    strength = float(cfg.get("geom_v8_defect_strength", 1.0))
    clip = float(cfg.get("geom_v8_defect_clip", 0.75))
    salt = int(cfg.get("geom_v8_defect_salt", 1))
    defect_seed = int(cfg.get("geom_v8_defect_seed", 0))
    pair = bool(cfg.get("geom_v8_pair_chiral", True))
    norm_mode = str(cfg.get("geom_v8_defect_norm_mode", "fro_match")).strip().lower()

    p_mask, _ = _v8_prime_mask_and_sign(ps, cfg=cfg)
    support_primes = [int(p) for (p, m) in zip(ps, p_mask.tolist()) if float(m) == 1.0]
    support_size = int(len(support_primes))

    bidx = get_measurement_boundary_idx(cfg, int(d))
    coord_mask = _v8_boundary_coord_mask(cfg, d=int(d))
    boundary_support_size = int(np.sum(coord_mask > 0.0))
    boundary_digest = sha16(
        repr(
            {
                "meas_boundary_idx": list(map(int, np.asarray(bidx, dtype=np.int64).tolist())),
                "pair_chiral": bool(pair),
                "boundary_support_size": boundary_support_size,
            }
        ).encode("utf-8")
    )

    support_digest = sha16(
        repr(
            {
                "mode": mode,
                "rho": rho,
                "salt": salt,
                "defect_seed": defect_seed,
                "support_primes": support_primes,
            }
        ).encode("utf-8")
    )

    rule = f"{mode}|rho={rho:.6g}|salt={salt}|defect_seed={defect_seed}|pair_chiral={int(pair)}|norm={norm_mode}"
    return {
        "v8_support_size": support_size,
        "v8_support_rule": rule,
        "v8_primes_marked_digest": support_digest,
        "v8_strength": float(strength),
        "v8_clip": float(clip),
        "v8_boundary_support_size": boundary_support_size,
        "v8_boundary_digest": boundary_digest,
    }


def _v9_phasetransport_metadata(primes_used: list[int], cfg: dict) -> dict:
    """Derived-only diagnostics for v9 phase transport (no measurement-spine changes)."""
    ps = list(map(int, primes_used))
    rule = str(cfg.get("geom_v9_transport_rule", "delta_raw_wrapped")).strip().lower()
    lam = float(cfg.get("geom_v9_transport_lambda", 1.0))
    clip = float(cfg.get("geom_v9_transport_clip", 0.25))
    t_seed = int(cfg.get("geom_v9_transport_seed", 0))

    _, theta_raw, theta_tr = _v9_transport_theta(ps, cfg)
    # Make a stable, compact fingerprint of the transported sequence.
    theta_fp = [float(f"{x:.12g}") for x in np.asarray(theta_tr, dtype=np.float64).tolist()]
    theta_digest = sha16(
        repr({"rule": rule, "lambda": lam, "clip": clip, "seed": t_seed, "theta_tr": theta_fp}).encode("utf-8")
    )

    return {
        "v9_transport_rule": rule,
        "v9_transport_lambda": float(lam),
        "v9_transport_clip": float(clip),
        "v9_transport_seed": int(t_seed),
        "v9_theta_tr_digest": str(theta_digest),
    }


def _v10_phasefieldsolve_metadata(primes_used: list[int], cfg: dict) -> dict:
    """Derived-only diagnostics for v10 phase-field solve (no measurement-spine changes)."""
    ps = list(map(int, primes_used))
    graph = str(cfg.get("geom_v10_phasefield_graph", "path")).strip().lower()
    w = float(cfg.get("geom_v10_phasefield_w", 1.0))
    eta = float(cfg.get("geom_v10_phasefield_eta", 0.25))
    pf_seed = int(cfg.get("geom_v10_phasefield_seed", 0))

    _, theta_raw, theta_sol = _v10_phasefield_theta(ps, cfg)
    theta_fp = [float(f"{x:.12g}") for x in np.asarray(theta_sol, dtype=np.float64).tolist()]
    theta_digest = sha16(
        repr({"graph": graph, "w": w, "eta": eta, "seed": pf_seed, "theta_sol": theta_fp}).encode("utf-8")
    )
    return {
        "v10_phasefield_graph": graph,
        "v10_phasefield_w": float(w),
        "v10_phasefield_eta": float(eta),
        "v10_phasefield_seed": int(pf_seed),
        "v10_theta_sol_digest": str(theta_digest),
    }


def _build_Tn_geom_warp_dirac_v10_phasefieldsolve(nums, d: int, seed: int, cfg: dict) -> dict:
    """Direct-label generator for v10 (used only by closure channels in full mode)."""
    nums = list(map(int, nums))
    d = int(d)
    if d < 4 or (d % 2) != 0:
        raise ValueError("geom_warp_dirac_v10_phasefieldsolve requires even d >= 4 for chiral split")

    u, _, theta_sol = _v10_phasefield_theta(nums, cfg)
    base = _build_Tn_synth(nums, d=d, seed=int(seed))

    rng = np.random.default_rng(int(seed) * 99991 + 462)
    A0 = rng.standard_normal((d, d))
    H0 = _sym(A0) / np.sqrt(max(1, d))

    rank = int(cfg.get("geom_v10_rank", cfg.get("geom_v4_rank", cfg.get("geom_v3_rank", 1))))
    rank = max(1, min(rank, d))
    U = rng.standard_normal((d, rank)) / np.sqrt(max(1, d))
    C_shared = _sym(U @ U.T)

    n = d // 2
    I = np.eye(n, dtype=np.float64)
    Z = np.zeros((n, n), dtype=np.float64)
    X_chiral = np.block([[Z, I], [I, Z]])

    u_scale = float(cfg.get("geom_v10_u_scale", 1.0))
    theta_scale = float(cfg.get("geom_v10_theta_scale", 0.125))
    eps = float(cfg.get("geom_v10_eps", cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01))))
    eps_global = float(cfg.get("geom_v10_eps_global", 0.02))

    out = {}
    for i, nlab in enumerate(nums):
        Lp = _logm_via_eig_sym(np.asarray(base[int(nlab)], dtype=np.float64))
        chiral_amp = (theta_scale * float(theta_sol[i]))
        Dp = (u_scale * float(u[i])) * H0 + (eps * u_scale * float(u[i])) * C_shared + chiral_amp * X_chiral
        G = _sym(Lp + (eps_global * Dp))
        out[int(nlab)] = _expm_via_eig_sym(G)
    return out

def _build_Tn_geom_warp_dirac_v4(nums, d: int, seed: int, cfg: dict) -> dict:
    """Direct-label generator for v4 (used only by closure channels in full mode)."""
    nums = list(map(int, nums))
    d = int(d)
    if d < 4 or (d % 2) != 0:
        raise ValueError("geom_warp_dirac_v4 requires even d >= 4 for chiral split")

    u, theta = _warp_u_theta_ratio_selfdual(nums, cfg)
    base = _build_Tn_synth(nums, d=d, seed=int(seed))

    rng = np.random.default_rng(int(seed) * 99991 + 456)
    A0 = rng.standard_normal((d, d))
    H0 = _sym(A0) / np.sqrt(max(1, d))

    rank = int(cfg.get("geom_v4_rank", cfg.get("geom_v3_rank", 1)))
    rank = max(1, min(rank, d))
    U = rng.standard_normal((d, rank)) / np.sqrt(max(1, d))
    C_shared = _sym(U @ U.T)

    n = d // 2
    I = np.eye(n, dtype=np.float64)
    Z = np.zeros((n, n), dtype=np.float64)
    X_chiral = np.block([[Z, I], [I, Z]])

    u_scale = float(cfg.get("geom_v4_u_scale", 1.0))
    theta_scale = float(cfg.get("geom_v4_theta_scale", 0.125))
    eps = float(cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01)))
    eps_global = float(cfg.get("geom_v4_eps_global", 0.02))

    out = {}
    for i, nlab in enumerate(nums):
        Lp = _logm_via_eig_sym(np.asarray(base[int(nlab)], dtype=np.float64))
        Dp = (u_scale * float(u[i])) * H0 + (eps * u_scale * float(u[i])) * C_shared + (theta_scale * float(theta[i])) * X_chiral
        G = _sym(Lp + (eps_global * Dp))
        out[int(nlab)] = _expm_via_eig_sym(G)
    return out

def _build_Tn_geom_warp_dirac_v5(nums, d: int, seed: int, cfg: dict) -> dict:
    """Direct-label generator for v5 (used only by closure channels in full mode)."""
    nums = list(map(int, nums))
    d = int(d)
    if d < 4 or (d % 2) != 0:
        raise ValueError("geom_warp_dirac_v5 requires even d >= 4 for chiral split")

    # reconstruct y(n) deterministically (same as in build_Tp_geom_warp_dirac_v5)
    x = np.log(np.asarray(nums, dtype=np.float64))
    twopi = 2.0 * np.pi
    frac = (x / twopi) - np.floor(x / twopi)
    theta0 = (frac * twopi) - np.pi
    y_scale = float(cfg.get("warp_y_scale", 1.0))
    y = y_scale * theta0

    u, theta = _warp_u_theta_ratio_selfdual(nums, cfg)
    base = _build_Tn_synth(nums, d=d, seed=int(seed))

    rng = np.random.default_rng(int(seed) * 99991 + 457)
    A0 = rng.standard_normal((d, d))
    H0 = _sym(A0) / np.sqrt(max(1, d))

    rank = int(cfg.get("geom_v5_rank", cfg.get("geom_v4_rank", cfg.get("geom_v3_rank", 1))))
    rank = max(1, min(rank, d))
    U = rng.standard_normal((d, rank)) / np.sqrt(max(1, d))
    C_shared = _sym(U @ U.T)

    n = d // 2
    I = np.eye(n, dtype=np.float64)
    Z = np.zeros((n, n), dtype=np.float64)
    X_chiral = np.block([[Z, I], [I, Z]])

    u_scale = float(cfg.get("geom_v5_u_scale", 1.0))
    theta_scale = float(cfg.get("geom_v5_theta_scale", 0.125))
    eps = float(cfg.get("geom_v5_eps", cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01))))
    eps_global = float(cfg.get("geom_v5_eps_global", 0.02))

    gamma = float(cfg.get("geom_v5_harmonic_gamma", 1.0))
    strength = float(cfg.get("geom_v5_harmonic_strength", 1.0))
    h = 1.0 / (0.25 + np.square(y.astype(np.float64)))
    h_med = float(np.median(h)) if h.size else 1.0
    if not np.isfinite(h_med) or h_med <= 0:
        h_med = 1.0
    mod = np.power(h / h_med, gamma, dtype=np.float64)

    out = {}
    for i, nlab in enumerate(nums):
        Lp = _logm_via_eig_sym(np.asarray(base[int(nlab)], dtype=np.float64))
        chiral_amp = (theta_scale * float(theta[i])) * (strength * float(mod[i]))
        Dp = (u_scale * float(u[i])) * H0 + (eps * u_scale * float(u[i])) * C_shared + chiral_amp * X_chiral
        G = _sym(Lp + (eps_global * Dp))
        out[int(nlab)] = _expm_via_eig_sym(G)
    return out


def _build_Tn_geom_warp_dirac_v6(nums, d: int, seed: int, cfg: dict) -> dict:
    """Direct-label generator for v6 (used only by closure channels in full mode)."""
    nums = list(map(int, nums))
    d = int(d)
    if d < 4 or (d % 2) != 0:
        raise ValueError("geom_warp_dirac_v6 requires even d >= 4 for chiral split")

    x = np.log(np.asarray(nums, dtype=np.float64))
    twopi = 2.0 * np.pi
    frac = (x / twopi) - np.floor(x / twopi)
    theta0 = (frac * twopi) - np.pi
    y_scale = float(cfg.get("warp_y_scale", 1.0))
    y = y_scale * theta0

    u, theta = _warp_u_theta_ratio_selfdual(nums, cfg)
    base = _build_Tn_synth(nums, d=d, seed=int(seed))

    rng = np.random.default_rng(int(seed) * 99991 + 458)
    A0 = rng.standard_normal((d, d))
    H0 = _sym(A0) / np.sqrt(max(1, d))

    rank = int(cfg.get("geom_v6_rank", cfg.get("geom_v4_rank", cfg.get("geom_v3_rank", 1))))
    rank = max(1, min(rank, d))
    U = rng.standard_normal((d, rank)) / np.sqrt(max(1, d))
    C_shared = _sym(U @ U.T)

    n = d // 2
    I = np.eye(n, dtype=np.float64)
    Z = np.zeros((n, n), dtype=np.float64)
    X_chiral = np.block([[Z, I], [I, Z]])

    u_scale = float(cfg.get("geom_v6_u_scale", 1.0))
    theta_scale = float(cfg.get("geom_v6_theta_scale", 0.125))
    eps = float(cfg.get("geom_v6_eps", cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01))))
    eps_global = float(cfg.get("geom_v6_eps_global", 0.02))

    phase_lambda = float(cfg.get("geom_v6_phase_lambda", 0.05))
    phase_clip = float(cfg.get("geom_v6_phase_clip", 2.0))

    h = 1.0 / (0.25 + np.square(y.astype(np.float64)))
    h_med = float(np.median(h)) if h.size else 1.0
    if not np.isfinite(h_med) or h_med <= 0:
        h_med = 1.0

    log_ratio = np.log(np.maximum(h, 1e-300) / h_med)
    if np.isfinite(phase_clip) and phase_clip > 0:
        log_ratio = np.clip(log_ratio, -phase_clip, phase_clip)
    theta_p = theta + (phase_lambda * log_ratio)

    out = {}
    for i, nlab in enumerate(nums):
        Lp = _logm_via_eig_sym(np.asarray(base[int(nlab)], dtype=np.float64))
        chiral_amp = (theta_scale * float(theta_p[i]))
        Dp = (u_scale * float(u[i])) * H0 + (eps * u_scale * float(u[i])) * C_shared + chiral_amp * X_chiral
        G = _sym(Lp + (eps_global * Dp))
        out[int(nlab)] = _expm_via_eig_sym(G)
    return out


def _build_Tn_geom_warp_dirac_v7_localdefect(nums, d: int, seed: int, cfg: dict) -> dict:
    """Direct-label generator for v7 (used only by closure channels in full mode)."""
    nums = list(map(int, nums))
    d = int(d)
    if d < 4 or (d % 2) != 0:
        raise ValueError("geom_warp_dirac_v7_localdefect requires even d >= 4 for chiral split")

    u, theta = _warp_u_theta_ratio_selfdual(nums, cfg)

    # Use the same deterministic defect mask/sign logic, evaluated on the direct-label set.
    mask, sgn = _v7_defect_mask_and_sign(list(map(int, nums)), seed=int(seed), cfg=cfg)

    base = _build_Tn_synth(nums, d=d, seed=int(seed))

    rng = np.random.default_rng(int(seed) * 99991 + 459)
    A0 = rng.standard_normal((d, d))
    H0 = _sym(A0) / np.sqrt(max(1, d))

    rank = int(cfg.get("geom_v7_rank", cfg.get("geom_v4_rank", cfg.get("geom_v3_rank", 1))))
    rank = max(1, min(rank, d))
    U = rng.standard_normal((d, rank)) / np.sqrt(max(1, d))
    C_shared = _sym(U @ U.T)

    n = d // 2
    I = np.eye(n, dtype=np.float64)
    Z = np.zeros((n, n), dtype=np.float64)
    X_chiral = np.block([[Z, I], [I, Z]])

    u_scale = float(cfg.get("geom_v7_u_scale", 1.0))
    theta_scale = float(cfg.get("geom_v7_theta_scale", 0.125))
    eps = float(cfg.get("geom_v7_eps", cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01))))
    eps_global = float(cfg.get("geom_v7_eps_global", 0.02))

    strength = float(cfg.get("geom_v7_defect_strength", 1.0))
    clip = float(cfg.get("geom_v7_defect_clip", 0.75))
    clip = float(max(0.0, clip))

    mult = 1.0 + (strength * mask * sgn)
    mult = np.clip(mult, 1.0 - clip, 1.0 + clip).astype(np.float64)

    out = {}
    for i, nlab in enumerate(nums):
        Lp = _logm_via_eig_sym(np.asarray(base[int(nlab)], dtype=np.float64))
        chiral_amp = (theta_scale * float(theta[i])) * float(mult[i])
        Dp = (u_scale * float(u[i])) * H0 + (eps * u_scale * float(u[i])) * C_shared + chiral_amp * X_chiral
        G = _sym(Lp + (eps_global * Dp))
        out[int(nlab)] = _expm_via_eig_sym(G)
    return out


def _build_Tn_geom_warp_dirac_v8_boundarydefect(nums, d: int, seed: int, cfg: dict) -> dict:
    """Direct-label generator for v8 (used only by closure channels in full mode)."""
    nums = list(map(int, nums))
    d = int(d)
    if d < 4 or (d % 2) != 0:
        raise ValueError("geom_warp_dirac_v8_boundarydefect requires even d >= 4 for chiral split")

    u, theta = _warp_u_theta_ratio_selfdual(nums, cfg)
    base = _build_Tn_synth(nums, d=d, seed=int(seed))

    # v8 uses deterministic prime-side activation keyed to geometry-class constants.
    p_mask, p_sgn = _v8_prime_mask_and_sign(list(map(int, nums)), cfg=cfg)

    rng = np.random.default_rng(int(seed) * 99991 + 460)
    A0 = rng.standard_normal((d, d))
    H0 = _sym(A0) / np.sqrt(max(1, d))

    rank = int(cfg.get("geom_v8_rank", cfg.get("geom_v4_rank", cfg.get("geom_v3_rank", 1))))
    rank = max(1, min(rank, d))
    U = rng.standard_normal((d, rank)) / np.sqrt(max(1, d))
    C_shared = _sym(U @ U.T)

    n = d // 2
    I = np.eye(n, dtype=np.float64)
    Z = np.zeros((n, n), dtype=np.float64)
    X_chiral = np.block([[Z, I], [I, Z]])

    coord_mask = _v8_boundary_coord_mask(cfg, d=d)
    X_defect = (coord_mask[:, None] * X_chiral) * coord_mask[None, :]

    norm_mode = str(cfg.get("geom_v8_defect_norm_mode", "fro_match")).strip().lower()
    if norm_mode in {"fro_match", "fro", "match"}:
        nx = float(np.linalg.norm(X_chiral, ord="fro"))
        nd = float(np.linalg.norm(X_defect, ord="fro"))
        scale = (nx / nd) if (nd > 0 and np.isfinite(nd) and np.isfinite(nx)) else 0.0
        X_defect = (scale * X_defect).astype(np.float64)
    elif norm_mode in {"none", "off"}:
        pass
    else:
        raise ValueError(f"Unknown geom_v8_defect_norm_mode={norm_mode!r}. Expected: none or fro_match")

    u_scale = float(cfg.get("geom_v8_u_scale", 1.0))
    theta_scale = float(cfg.get("geom_v8_theta_scale", 0.125))
    eps = float(cfg.get("geom_v8_eps", cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01))))
    eps_global = float(cfg.get("geom_v8_eps_global", 0.02))

    strength = float(cfg.get("geom_v8_defect_strength", 1.0))
    clip = float(cfg.get("geom_v8_defect_clip", 0.75))
    clip = float(max(0.0, clip))
    coeff = (strength * p_mask * p_sgn).astype(np.float64)
    if np.isfinite(clip) and clip >= 0:
        coeff = np.clip(coeff, -clip, clip)

    out = {}
    for i, nlab in enumerate(nums):
        Lp = _logm_via_eig_sym(np.asarray(base[int(nlab)], dtype=np.float64))
        chiral_term = (theta_scale * float(theta[i])) * (X_chiral + (float(coeff[i]) * X_defect))
        Dp = (u_scale * float(u[i])) * H0 + (eps * u_scale * float(u[i])) * C_shared + chiral_term
        G = _sym(Lp + (eps_global * Dp))
        out[int(nlab)] = _expm_via_eig_sym(G)
    return out


def _build_Tn_geom_warp_dirac_v9_phasetransport(nums, d: int, seed: int, cfg: dict) -> dict:
    """Direct-label generator for v9 (used only by closure channels in full mode)."""
    nums = list(map(int, nums))
    d = int(d)
    if d < 4 or (d % 2) != 0:
        raise ValueError("geom_warp_dirac_v9_phasetransport requires even d >= 4 for chiral split")

    u, _, theta_tr = _v9_transport_theta(nums, cfg)
    base = _build_Tn_synth(nums, d=d, seed=int(seed))

    rng = np.random.default_rng(int(seed) * 99991 + 461)
    A0 = rng.standard_normal((d, d))
    H0 = _sym(A0) / np.sqrt(max(1, d))

    rank = int(cfg.get("geom_v9_rank", cfg.get("geom_v4_rank", cfg.get("geom_v3_rank", 1))))
    rank = max(1, min(rank, d))
    U = rng.standard_normal((d, rank)) / np.sqrt(max(1, d))
    C_shared = _sym(U @ U.T)

    n = d // 2
    I = np.eye(n, dtype=np.float64)
    Z = np.zeros((n, n), dtype=np.float64)
    X_chiral = np.block([[Z, I], [I, Z]])

    u_scale = float(cfg.get("geom_v9_u_scale", 1.0))
    theta_scale = float(cfg.get("geom_v9_theta_scale", 0.125))
    eps = float(cfg.get("geom_v9_eps", cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01))))
    eps_global = float(cfg.get("geom_v9_eps_global", 0.02))

    out = {}
    for i, nlab in enumerate(nums):
        Lp = _logm_via_eig_sym(np.asarray(base[int(nlab)], dtype=np.float64))
        chiral_amp = (theta_scale * float(theta_tr[i]))
        Dp = (u_scale * float(u[i])) * H0 + (eps * u_scale * float(u[i])) * C_shared + chiral_amp * X_chiral
        G = _sym(Lp + (eps_global * Dp))
        out[int(nlab)] = _expm_via_eig_sym(G)
    return out

def _tp_relative_delta_summary(Tp_ref: dict, Tp_new: dict, primes_used: list) -> dict:
    ps = list(map(int, primes_used))
    if not ps:
        return {"tp_delta_mean": 0.0, "tp_delta_max": 0.0}
    deltas = []
    for p in ps:
        A = np.asarray(Tp_ref[p], dtype=np.float64)
        B = np.asarray(Tp_new[p], dtype=np.float64)
        denom = float(np.linalg.norm(A, ord="fro") + 1e-12)
        deltas.append(float(np.linalg.norm(B - A, ord="fro")) / denom)
    return {"tp_delta_mean": float(np.mean(deltas)), "tp_delta_max": float(np.max(deltas))}

def build_Tp_geom_v3_shared_axis_graded(primes_used, d: int, seed: int, cfg: dict):
    """
    Phase-3B legal v3.1 backend (graded / Dirac-like chirality):
      Tp[p] = exp( u(p) H0 + eps*u(p) C_shared + theta_scale*theta(p) X_chiral )
    where X_chiral is a fixed symmetric block-mixing template on a chiral split.

    This makes theta(p) act as a structural (non-commuting, graded) degree of freedom,
    rather than a post-hoc label.
    """
    d = int(d)
    if d < 4 or (d % 2) != 0:
        raise ValueError("geom_v3_shared_axis_graded requires even d >= 4 for chiral block split")

    eps = float(cfg.get("geom_v3_eps", 0.01))
    theta_scale_raw = cfg.get("geom_v3_theta_scale", None)
    if theta_scale_raw is None:
        theta_scale_raw = cfg.get("geom_v3_rot_scale", 0.25)
    theta_scale = float(theta_scale_raw)

    u, theta = _warp_u_theta_from_complex_r(primes_used, cfg)

    rng = np.random.default_rng(int(seed) * 99991 + 123)
    # Base symmetric template
    A0 = rng.standard_normal((d, d))
    H0 = _sym(A0) / np.sqrt(max(1, d))

    # Shared low-rank symmetric coupling
    rank = int(cfg.get("geom_v3_rank", 1))
    U = rng.standard_normal((d, rank)) / np.sqrt(max(1, d))
    C_shared = _sym(U @ U.T)

    # Chiral block-mixing template (Dirac-ish grading)
    n = d // 2
    I = np.eye(n, dtype=np.float64)
    Z = np.zeros((n, n), dtype=np.float64)
    X_chiral = np.block([[Z, I], [I, Z]])  # symmetric, swaps/mixes chiral blocks

    Tp = {}
    ps = list(map(int, primes_used))
    for i, p in enumerate(ps):
        H = (u[i] * H0) + (eps * u[i] * C_shared) + (theta_scale * theta[i] * X_chiral)
        Tp[p] = _expm_via_eig_sym(_sym(H))
    return Tp, u, theta

# ----------------------------
# 5) Backend dispatcher alias (optional patch)
# ----------------------------
if "_build_Tp_backend_dispatch" not in globals():
    def _build_Tp_backend_dispatch(tp_backend: str, primes_used: list, d: int, seed: int, cfg: dict):
        tp_backend = str(tp_backend or "legacy")
        alias = {
            "geom_v3": "geom_v3_shared_axis",
            "geom_v3_shared": "geom_v3_shared_axis",
            "geom_v3_graded": "geom_v3_shared_axis_graded",
            "geom_v3_shared_axis_graded": "geom_v3_shared_axis_graded",
            "geom_v4": "geom_warp_dirac_v4",
            "geom_warp_dirac_v4": "geom_warp_dirac_v4",
            "geom_v5": "geom_warp_dirac_v5",
            "geom_warp_dirac_v5": "geom_warp_dirac_v5",
            "geom_v6": "geom_warp_dirac_v6",
            "geom_warp_dirac_v6": "geom_warp_dirac_v6",
            "geom_v7": "geom_warp_dirac_v7_localdefect",
            "geom_warp_dirac_v7_localdefect": "geom_warp_dirac_v7_localdefect",

            "geom_v8": "geom_warp_dirac_v8_boundarydefect",
            "geom_warp_dirac_v8_boundarydefect": "geom_warp_dirac_v8_boundarydefect",

            "geom_v9": "geom_warp_dirac_v9_phasetransport",
            "geom_warp_dirac_v9_phasetransport": "geom_warp_dirac_v9_phasetransport",

            "geom_v10": "geom_warp_dirac_v10_phasefieldsolve",
            "geom_warp_dirac_v10_phasefieldsolve": "geom_warp_dirac_v10_phasefieldsolve",
        }
        tp_backend = alias.get(tp_backend, tp_backend)

        if tp_backend == "legacy":
            Tp = build_Tp_synth(primes_used, d=int(d), seed=int(seed))
            return Tp, None, None
        if tp_backend == "geom_v3_shared_axis":
            Tp, u_vals, theta_vals = build_Tp_geom_v3_shared_axis(primes_used, d=int(d), seed=int(seed), cfg=cfg)
            return Tp, u_vals, theta_vals

        if tp_backend == "geom_v3_shared_axis_graded":
            Tp, u_vals, theta_vals = build_Tp_geom_v3_shared_axis_graded(primes_used, d=int(d), seed=int(seed), cfg=cfg)
            return Tp, u_vals, theta_vals

        if tp_backend == "geom_warp_dirac_v4":
            Tp, u_vals, theta_vals = build_Tp_geom_warp_dirac_v4(primes_used, d=int(d), seed=int(seed), cfg=cfg)
            return Tp, u_vals, theta_vals

        if tp_backend == "geom_warp_dirac_v5":
            Tp, u_vals, theta_vals = build_Tp_geom_warp_dirac_v5(primes_used, d=int(d), seed=int(seed), cfg=cfg)
            return Tp, u_vals, theta_vals

        if tp_backend == "geom_warp_dirac_v6":
            Tp, u_vals, theta_vals = build_Tp_geom_warp_dirac_v6(primes_used, d=int(d), seed=int(seed), cfg=cfg)
            return Tp, u_vals, theta_vals

        if tp_backend == "geom_warp_dirac_v7_localdefect":
            Tp, u_vals, theta_vals = build_Tp_geom_warp_dirac_v7_localdefect(primes_used, d=int(d), seed=int(seed), cfg=cfg)
            return Tp, u_vals, theta_vals

        if tp_backend == "geom_warp_dirac_v8_boundarydefect":
            Tp, u_vals, theta_vals = build_Tp_geom_warp_dirac_v8_boundarydefect(primes_used, d=int(d), seed=int(seed), cfg=cfg)
            return Tp, u_vals, theta_vals

        if tp_backend == "geom_warp_dirac_v9_phasetransport":
            Tp, u_vals, theta_vals = build_Tp_geom_warp_dirac_v9_phasetransport(primes_used, d=int(d), seed=int(seed), cfg=cfg)
            return Tp, u_vals, theta_vals

        if tp_backend == "geom_warp_dirac_v10_phasefieldsolve":
            Tp, u_vals, theta_vals = build_Tp_geom_warp_dirac_v10_phasefieldsolve(primes_used, d=int(d), seed=int(seed), cfg=cfg)
            return Tp, u_vals, theta_vals

        # if you already have v2a/v2b defined elsewhere, let them work
        if tp_backend == "geom_v2a" and "build_Tp_geom_v2a" in globals():
            Tp, u_vals, theta_vals = build_Tp_geom_v2a(primes_used, d=int(d), seed=int(seed), cfg=cfg)
            return Tp, u_vals, theta_vals
        if tp_backend == "geom_v2b" and "build_Tp_geom_v2b" in globals():
            Tp, u_vals, theta_vals = build_Tp_geom_v2b(primes_used, d=int(d), seed=int(seed), cfg=cfg)
            return Tp, u_vals, theta_vals

        raise ValueError(
            f"Unknown tp_backend={tp_backend!r}. Expected: legacy, geom_v3_shared_axis, geom_v3_shared_axis_graded, geom_warp_dirac_v4/v5/v6/v7/v8/v9/v10 (and any v2 you defined)."
        )

print("CELL 2.1 loaded: legacy Tp fallback + geom_v3_shared_axis (complex r & complex rs) + fingerprints.")

# ============================================
# CELL 3/5: Frozen measurement spine (DN-map / Schur complement logdet)
# Phase 3B discipline: do not change these once calibrated.
# ============================================

if "compute_measurement_operator" not in globals():
    def compute_measurement_operator(Tp_true: dict, primes_used: list, E: float, amp: float, cfg: dict) -> np.ndarray:
        """Builds a deterministic symmetric operator A(E) from Tp and energy E."""
        d = int(cfg.get("d", 16))
        sigma = float(cfg.get("meas_sigma", 0.75))

        # weights use log p against energy (fixed; backend-independent except through Tp)
        ps = np.asarray(list(map(int, primes_used)), dtype=np.float64)
        lp = np.log(ps)
        w = (np.cos(float(E) * lp) + 0.35 * np.sin(float(E) * lp)) / np.power(ps, sigma)

        A = np.zeros((d, d), dtype=np.float64)
        for i, p in enumerate(map(int, primes_used)):
            X = np.asarray(Tp_true[p], dtype=np.float64)
            A += float(w[i]) * _sym(X)

        # fixed diagonal stabilizer + controlled amp
        A = _sym(A)
        A += (float(cfg.get("meas_diag", 0.8)) + float(amp)) * np.eye(d, dtype=np.float64)
        return A

if "dnmap_schur_complement" not in globals():
    def dnmap_schur_complement(A: np.ndarray, cfg: dict) -> np.ndarray:
        """Returns Schur complement (DN-map proxy) for a fixed boundary split."""
        A = np.asarray(A, dtype=np.float64)
        d = A.shape[0]

        bidx = get_measurement_boundary_idx(cfg, d)
        b = int(bidx.size)
        iidx = np.asarray([i for i in range(d) if i not in set(map(int, bidx.tolist()))], dtype=np.int64)

        Abb = A[np.ix_(bidx, bidx)]
        Abi = A[np.ix_(bidx, iidx)]
        Aib = A[np.ix_(iidx, bidx)]
        Aii = A[np.ix_(iidx, iidx)]

        jitter = float(cfg.get("meas_jitter", 1e-6))
        Aii_reg = Aii + jitter * np.eye(Aii.shape[0], dtype=np.float64)
        X = np.linalg.solve(Aii_reg, Aib)
        S = Abb - Abi @ X
        return _sym(S)


if "dnmap_schur_complement_raw" not in globals():
    def dnmap_schur_complement_raw(A: np.ndarray, cfg: dict) -> np.ndarray:
        """Returns the *unsymmetrized* Schur complement (diagnostics only).

        NOTE: The measurement spine uses the symmetrized proxy. This helper is
        only to quantify non-Hermitian/nonsymmetric numerical residue introduced
        by finite precision solves.
        """
        A = np.asarray(A, dtype=np.float64)
        d = A.shape[0]

        bidx = get_measurement_boundary_idx(cfg, d)
        iidx = np.asarray([i for i in range(d) if i not in set(map(int, bidx.tolist()))], dtype=np.int64)

        Abb = A[np.ix_(bidx, bidx)]
        Abi = A[np.ix_(bidx, iidx)]
        Aib = A[np.ix_(iidx, bidx)]
        Aii = A[np.ix_(iidx, iidx)]

        jitter = float(cfg.get("meas_jitter", 1e-6))
        Aii_reg = Aii + jitter * np.eye(Aii.shape[0], dtype=np.float64)
        X = np.linalg.solve(Aii_reg, Aib)
        return Abb - Abi @ X

if "dnmap_logdet_score" not in globals():
    def dnmap_logdet_score(Tp_true: dict, primes_used: list, energies: np.ndarray, amp: float, cfg: dict) -> float:
        """Aggregates logdet(DN-map proxy) over energies inside a window."""
        energies = np.asarray(energies, dtype=np.float64)
        if energies.size == 0:
            return 0.0

        d = int(cfg.get("d", 16))
        sigma = float(cfg.get("meas_sigma", 0.75))
        diag = float(cfg.get("meas_diag", 0.8)) + float(amp)
        jitter = float(cfg.get("meas_jitter", 1e-6))
        bidx = get_measurement_boundary_idx(cfg, d)
        b = int(bidx.size)
        iidx = np.asarray([i for i in range(d) if i not in set(map(int, bidx.tolist()))], dtype=np.int64)

        ps = np.asarray(list(map(int, primes_used)), dtype=np.float64)
        lp = np.log(ps)
        # w: (nE, nP)
        Ecol = energies.reshape((-1, 1))
        w = (np.cos(Ecol * lp.reshape((1, -1))) + 0.35 * np.sin(Ecol * lp.reshape((1, -1)))) / np.power(ps.reshape((1, -1)), sigma)

        # stack symmetric prime matrices: (nP, d, d)
        M = np.stack([_sym(np.asarray(Tp_true[int(p)], dtype=np.float64)) for p in map(int, primes_used)], axis=0)

        # A: (nE, d, d)
        A = np.einsum("ep,pij->eij", w, M, optimize=True)
        A = 0.5 * (A + np.swapaxes(A, 1, 2))
        A += diag * np.eye(d, dtype=np.float64)[None, :, :]

        # Schur complement batched
        Abb = A[:, bidx[:, None], bidx[None, :]]
        Abi = A[:, bidx[:, None], iidx[None, :]]
        Aib = A[:, iidx[:, None], bidx[None, :]]
        Aii = A[:, iidx[:, None], iidx[None, :]]

        Aii_reg = Aii + jitter * np.eye(int(d - b), dtype=np.float64)[None, :, :]
        # Solve Aii_reg * X = Aib  => X shape (nE, d-b, b)
        X = np.linalg.solve(Aii_reg, Aib)
        S = Abb - np.matmul(Abi, X)
        S = 0.5 * (S + np.swapaxes(S, 1, 2))
        S_reg = S + jitter * np.eye(b, dtype=np.float64)[None, :, :]

        sign, ld = np.linalg.slogdet(S_reg)
        ld = np.where(sign > 0, ld, ld - 100.0)
        return float(np.mean(ld))


# ============================================
# Phase-3D scaffolding: channel diagnostics pack
# - cache boundary operator snapshots (Λ via Schur complement)
# - define a fixed Cayley transform S from Λ
# - compute trace/logdet orbit consistency and basic channel invariance proxies
# ============================================

if "pick_energy_snapshots" not in globals():
    def pick_energy_snapshots(energies: np.ndarray, n_points: int = 5) -> np.ndarray:
        """Pick a small, deterministic subset of energies from a window grid."""
        energies = np.asarray(energies, dtype=np.float64)
        if energies.size == 0:
            return energies
        n_points = int(max(1, n_points))
        if energies.size <= n_points:
            return np.array(energies, copy=True)
        qs = np.linspace(0.0, 1.0, n_points)
        idx = np.unique(np.clip(np.round(qs * (energies.size - 1)).astype(np.int64), 0, energies.size - 1))
        return np.asarray(energies[idx], dtype=np.float64)


if "cayley_from_lambda" not in globals():
    def cayley_from_lambda(Lam: np.ndarray, eta: float = 1.0) -> np.ndarray:
        """Fixed Cayley transform: S = (Λ - i η I)(Λ + i η I)^{-1}."""
        Lam = np.asarray(Lam)
        if Lam.ndim != 2 or Lam.shape[0] != Lam.shape[1]:
            raise ValueError("cayley_from_lambda: Lam must be square")
        n = int(Lam.shape[0])
        I = np.eye(n, dtype=np.complex128)
        eta_c = np.complex128(float(eta))
        A = Lam.astype(np.complex128) - 1j * eta_c * I
        B = Lam.astype(np.complex128) + 1j * eta_c * I
        # Right-multiply by inv(B) via solve.
        return A @ np.linalg.solve(B, I)


if "_parse_window_key" not in globals():
    def _parse_window_key(k):
        """Parse window keys used in config dicts.

        Accepts:
          - tuple/list: (wlo, whi)
          - string: "wlo,whi" or "wlo_whi" (as produced by folder names)
        """
        if isinstance(k, (tuple, list)) and len(k) == 2:
            return (float(k[0]), float(k[1]))
        if isinstance(k, str):
            s = k.strip()
            if "," in s:
                a, b = s.split(",", 1)
                return (float(a.strip()), float(b.strip()))
            if "_" in s:
                a, b = s.split("_", 1)
                return (float(a.strip()), float(b.strip()))
        raise ValueError(f"Unrecognized window key: {k!r}")


if "_window_key" not in globals():
    def _window_key(wlo: float, whi: float) -> str:
        # Keep consistent with window folder naming in run_one_channel_diag.
        return f"{float(wlo)}_{float(whi)}"


if "_unitarity_and_normality_defects" not in globals():
    def _unitarity_and_normality_defects(S: np.ndarray):
        """Return (unitarity_defect, normality_defect) using Frobenius norms."""
        S = np.asarray(S, dtype=np.complex128)
        n = int(S.shape[0])
        I = np.eye(n, dtype=np.complex128)
        Sf = np.conjugate(S.T)
        SS = Sf @ S
        u = float(np.linalg.norm(SS - I, ord="fro") / (np.linalg.norm(I, ord="fro") + 1e-12))
        # Normality: ||S* S - S S*|| / ||S||^2
        SSt = S @ Sf
        denom = float(np.linalg.norm(S, ord="fro") ** 2) + 1e-12
        ndef = float(np.linalg.norm(SS - SSt, ord="fro") / denom)
        return u, ndef


if "select_cayley_eta_per_window" not in globals():
    def select_cayley_eta_per_window(
        *,
        cfg: dict,
        seed_ref: int,
        anchor_ref: int,
        amp: float,
        k_used: int,
        windows: list,
        energies_snap_by_window: dict,
        tau_grid=(0.25, 0.5, 1.0, 2.0, 4.0),
        lambda_normality: float = 0.25,
    ) -> dict:
        """Deterministically pick one Cayley eta per window using a fixed reference run.

        This is prereg-friendly:
          - uses ONLY (backend, seed_ref, anchor_ref) reference
          - produces eta_win reused for all seeds/anchors
          - no per-anchor tuning

        Returns: dict mapping (wlo,whi) -> eta
        """
        cfg0 = dict(cfg)
        cfg0["anchor_seed"] = int(anchor_ref)
        cfg0["tp_backend"] = str(cfg0.get("tp_backend", "legacy"))

        primes_used = list(map(int, first_primes(int(k_used))))
        Tp_true, u_vals, theta_vals = _build_Tp_backend_dispatch(str(cfg0["tp_backend"]), primes_used, d=int(cfg0.get("d", 16)), seed=int(seed_ref), cfg=cfg0)

        # Normalize energies_snap_by_window keys
        snap_map = {}
        for kk, vv in dict(energies_snap_by_window).items():
            w = _parse_window_key(kk)
            snap_map[w] = np.asarray(list(map(float, vv)), dtype=np.float64)

        eta_by_win = {}
        details_by_win = {}
        for (wlo, whi) in windows:
            w = (float(wlo), float(whi))
            if w not in snap_map:
                raise ValueError(f"Missing energies_snap for window {w}")
            E_snap = np.asarray(snap_map[w], dtype=np.float64)
            if E_snap.size == 0:
                raise ValueError(f"Empty energies_snap for window {w}")

            # Compute Lambda snapshots once.
            Lam_list = []
            for E in E_snap.tolist():
                A = compute_measurement_operator(Tp_true=Tp_true, primes_used=primes_used, E=float(E), amp=float(amp), cfg=cfg0)
                Lam = dnmap_schur_complement(A, cfg=cfg0)
                Lam_list.append(np.asarray(Lam, dtype=np.float64))
            Lam_stack = np.stack(Lam_list, axis=0)

            # Robust Hermitian scale using singular values of Hermitian part.
            sigmas = []
            for Lam in Lam_stack:
                LamH = 0.5 * (Lam + Lam.T)
                try:
                    svals = np.linalg.svd(LamH, compute_uv=False)
                    sigmas.append(float(np.median(np.abs(svals))))
                except Exception:
                    sigmas.append(float(np.median(np.abs(LamH))))
            sigma_win = float(np.median(np.asarray(sigmas, dtype=np.float64)))
            sigma_win = float(max(0.0, sigma_win))

            # Clamp away from 0 (prevents ill-conditioned solve when sigma_win tiny).
            eta_floor = float(1e-6 * max(1.0, sigma_win))

            best = None
            for tau in list(tau_grid):
                eta = float(max(float(tau) * sigma_win, eta_floor))
                # If sigma_win is huge, avoid S ~ I degeneracy by keeping eta modest.
                if sigma_win > 0:
                    eta = float(min(eta, 1e3 * sigma_win))

                u_list = []
                n_list = []
                for Lam in Lam_stack:
                    S = cayley_from_lambda(Lam, eta=float(eta))
                    udef, ndef = _unitarity_and_normality_defects(S)
                    u_list.append(udef)
                    n_list.append(ndef)
                u_med = float(np.median(np.asarray(u_list, dtype=np.float64)))
                n_med = float(np.median(np.asarray(n_list, dtype=np.float64)))
                J = float(u_med + float(lambda_normality) * n_med)

                cand = {
                    "tau": float(tau),
                    "eta": float(eta),
                    "sigma_win": float(sigma_win),
                    "u_median": float(u_med),
                    "n_median": float(n_med),
                    "J": float(J),
                }
                if (best is None) or (cand["J"] < best["J"]):
                    best = cand

            assert best is not None
            eta_by_win[w] = float(best["eta"])
            details_by_win[w] = best

        # Return both eta and selection details for audit.
        return {
            "eta_by_window": eta_by_win,
            "selection_details": details_by_win,
            "selection_spec": {
                "seed_ref": int(seed_ref),
                "anchor_ref": int(anchor_ref),
                "tp_backend": str(cfg0.get("tp_backend", "legacy")),
                "k_used": int(k_used),
                "tau_grid": list(map(float, tau_grid)),
                "lambda_normality": float(lambda_normality),
            },
        }


if "channel_diag_pack" not in globals():
    def channel_diag_pack(
        cfg: dict,
        Tp_true: dict,
        primes_used: list,
        energies: np.ndarray,
        amp: float,
        out_dir: str,
        *,
        n_snap: int = 5,
        trace_K: int = 8,
        cayley_eta: float = 1.0,
        energies_snap_override: np.ndarray | None = None,
        orbit_z: float = 0.5,
        tag: str = "",
    ) -> dict:
        """Compute and write channel diagnostics for a single window.

        Writes a compressed npz snapshot with Λ(E) and S(E) for a small set of energies,
        plus trace/logdet orbit-consistency diagnostics.
        """
        os.makedirs(out_dir, exist_ok=True)

        energies = np.asarray(energies, dtype=np.float64)
        if energies_snap_override is not None:
            E_snap = np.asarray(energies_snap_override, dtype=np.float64)
        else:
            E_snap = pick_energy_snapshots(energies, n_points=int(n_snap))
        if E_snap.size == 0:
            raise ValueError("channel_diag_pack: empty energy grid")

        trace_K = int(max(1, trace_K))
        cayley_eta = float(cayley_eta)
        orbit_z = float(orbit_z)
        if not (0.0 < orbit_z < 1.0):
            raise ValueError("channel_diag_pack: orbit_z must be in (0,1) for convergent series")

        # Optional: apply a deterministic boundary gauge transform (basis change) controlled by anchor_seed.
        # This makes anchor-dependence nontrivial while still being "up to gauge".
        gauge_mode = str(cfg.get("chan_boundary_gauge_mode", "none"))
        gauge_salt = int(cfg.get("chan_boundary_gauge_salt", 12345))
        anchor_seed = int(cfg.get("anchor_seed", 0))
        Qg = None
        if gauge_mode in {"orthogonal_qr", "ortho_qr"}:
            # Real orthogonal gauge (Λ is real symmetric). Q is constant across energies/windows.
            rng = _rng_from_seed(int(anchor_seed) + int(gauge_salt))
            # infer boundary dimension from measurement boundary split
            bidx = get_measurement_boundary_idx(cfg, int(cfg.get("d", 16)))
            b = int(bidx.size)
            G = rng.standard_normal((b, b)).astype(np.float64)
            Q, _ = np.linalg.qr(G)
            # enforce det +1 for determinism (avoid sign flips)
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1.0
            Qg = np.asarray(Q, dtype=np.float64)

        Lam_list = []
        Lam_raw_list = []
        S_list = []
        logdet_IminusS = []
        logdet_IminuszS = []
        trace_powers = []        # shape (nE_snap, K)
        series_partial = []      # shape (nE_snap, K)
        resid_partial = []       # shape (nE_snap, K)

        eigvals_E0 = None
        eigvecs_E0 = None
        comm_norms = None

        for ii, E in enumerate(E_snap.tolist()):
            A = compute_measurement_operator(Tp_true=Tp_true, primes_used=primes_used, E=float(E), amp=float(amp), cfg=cfg)
            Lam_raw = dnmap_schur_complement_raw(A, cfg=cfg)
            Lam = _sym(Lam_raw)

            if Qg is not None:
                Lam = Qg.T @ Lam @ Qg

            S = cayley_from_lambda(Lam, eta=float(cayley_eta))

            Lam_raw_list.append(np.asarray(Lam_raw, dtype=np.float64))
            Lam_list.append(np.asarray(Lam, dtype=np.float64))
            S_list.append(np.asarray(S, dtype=np.complex128))

            I = np.eye(S.shape[0], dtype=np.complex128)
            sign, ld = np.linalg.slogdet(I - S)
            ld_val = np.complex128(ld)
            logdet_IminusS.append(ld_val)

            # Convergent orbit identity: logdet(I - z S) vs -sum_{k<=K} z^k Tr(S^k)/k for |z|<1.
            # IMPORTANT: use eigenvalue-sum log to avoid branch artifacts from slogdet.
            z = np.complex128(float(orbit_z))
            try:
                ev = np.linalg.eigvals(S)
                ldz_val = np.sum(np.log(1.0 - z * ev))
                ldz_val = np.complex128(ldz_val)
            except Exception:
                signz, ldz = np.linalg.slogdet(I - z * S)
                ldz_val = np.complex128(ldz)
            logdet_IminuszS.append(ldz_val)

            # Trace orbit series (damped): -sum_{k<=K} z^k Tr(S^k)/k
            tks = np.empty(trace_K, dtype=np.complex128)
            Sp = np.eye(S.shape[0], dtype=np.complex128)
            series = np.complex128(0.0 + 0.0j)
            series_k = np.empty(trace_K, dtype=np.complex128)
            resid_k = np.empty(trace_K, dtype=np.complex128)
            for k in range(1, trace_K + 1):
                Sp = Sp @ S
                tr = np.trace(Sp)
                tks[k - 1] = tr
                series += -((z ** np.complex128(float(k))) * tr / np.complex128(float(k)))
                series_k[k - 1] = series
                resid_k[k - 1] = np.complex128(ldz_val - series)

            trace_powers.append(tks)
            series_partial.append(series_k)
            resid_partial.append(resid_k)

            if ii == 0:
                try:
                    eigvals, eigvecs = np.linalg.eig(S)
                    eigvals_E0 = np.asarray(eigvals, dtype=np.complex128)
                    eigvecs_E0 = np.asarray(eigvecs, dtype=np.complex128)

                    # Simple per-eigenvector projector commutator proxy.
                    denom = float(np.linalg.norm(S, ord="fro")) + 1e-12
                    comms = []
                    for j in range(eigvecs_E0.shape[1]):
                        v = eigvecs_E0[:, j]
                        vn = np.vdot(v, v)
                        if abs(vn) < 1e-15:
                            continue
                        P = np.outer(v, np.conjugate(v)) / vn
                        C = (S @ P) - (P @ S)
                        comms.append(float(np.linalg.norm(C, ord="fro") / denom))
                    comm_norms = np.asarray(comms, dtype=np.float64)
                except Exception:
                    eigvals_E0 = None
                    eigvecs_E0 = None
                    comm_norms = None

        Lam_stack = np.stack(Lam_list, axis=0)
        Lam_raw_stack = np.stack(Lam_raw_list, axis=0)
        S_stack = np.stack(S_list, axis=0)
        logdet_arr = np.asarray(logdet_IminusS, dtype=np.complex128)
        logdetz_arr = np.asarray(logdet_IminuszS, dtype=np.complex128)
        trace_arr = np.stack(trace_powers, axis=0)
        series_arr = np.stack(series_partial, axis=0)
        resid_arr = np.stack(resid_partial, axis=0)

        # Hermitian/symmetry defect of Λ before symmetrization.
        Lam_def = Lam_raw_stack - np.swapaxes(Lam_raw_stack, 1, 2)
        Lam_def_num = np.linalg.norm(Lam_def.reshape((Lam_def.shape[0], -1)), ord=2, axis=1)
        Lam_def_den = np.linalg.norm(Lam_raw_stack.reshape((Lam_raw_stack.shape[0], -1)), ord=2, axis=1) + 1e-12
        lam_herm_defect = Lam_def_num / Lam_def_den

        snap_digest = _hash_array16(E_snap)
        out_name = f"channel_diag{('__' + tag) if tag else ''}.npz"
        np.savez_compressed(
            os.path.join(out_dir, out_name),
            energies_full=np.asarray(energies, dtype=np.float64),
            energies_snap=np.asarray(E_snap, dtype=np.float64),
            energies_snap_digest=str(snap_digest),
            lambda_raw_snap=Lam_raw_stack,
            lambda_snap=Lam_stack,
            lambda_hermitian_defect=lam_herm_defect,
            S_snap=S_stack,
            logdet_IminusS=logdet_arr,
            orbit_z=float(orbit_z),
            logdet_IminuszS=logdetz_arr,
            trace_powers=trace_arr,
            series_partial=series_arr,
            resid_partial=resid_arr,
            trace_K=int(trace_K),
            cayley_eta=float(cayley_eta),
            eigvals_E0=(np.asarray(eigvals_E0, dtype=np.complex128) if eigvals_E0 is not None else np.asarray([], dtype=np.complex128)),
            comm_norms_E0=(np.asarray(comm_norms, dtype=np.float64) if comm_norms is not None else np.asarray([], dtype=np.float64)),
        )

        summary = {
            "energies_snap_digest": str(snap_digest),
            "nE_full": int(energies.size),
            "nE_snap": int(E_snap.size),
            "trace_K": int(trace_K),
            "cayley_eta": float(cayley_eta),
            "orbit_z": float(orbit_z),
            "resid_partial_abs_mean": float(np.mean(np.abs(resid_arr))) if resid_arr.size else float("nan"),
            "resid_partial_abs_max": float(np.max(np.abs(resid_arr))) if resid_arr.size else float("nan"),
            "logdet_IminusS_abs_mean": float(np.mean(np.abs(logdet_arr))) if logdet_arr.size else float("nan"),
            "logdet_IminuszS_abs_mean": float(np.mean(np.abs(logdetz_arr))) if logdetz_arr.size else float("nan"),
            "lambda_hermitian_defect_median": float(np.median(lam_herm_defect)) if lam_herm_defect.size else float("nan"),
            "lambda_hermitian_defect_max": float(np.max(lam_herm_defect)) if lam_herm_defect.size else float("nan"),
        }
        if comm_norms is not None and comm_norms.size:
            summary.update(
                {
                    "chan_comm_norm_mean": float(np.mean(comm_norms)),
                    "chan_comm_norm_median": float(np.median(comm_norms)),
                    "chan_comm_norm_max": float(np.max(comm_norms)),
                    "chan_comm_norm_n": int(comm_norms.size),
                }
            )
        else:
            summary.update(
                {
                    "chan_comm_norm_mean": float("nan"),
                    "chan_comm_norm_median": float("nan"),
                    "chan_comm_norm_max": float("nan"),
                    "chan_comm_norm_n": int(0),
                }
            )
        return summary


if "run_one_channel_diag" not in globals():
    def run_one_channel_diag(cfg: dict, seed: int, amp: float, out_dir: str) -> pd.DataFrame:
        """Phase-3D style run: no p-values; writes per-window channel diagnostics artifacts."""
        t0 = time.time()
        cfg = dict(cfg)

        os.makedirs(out_dir, exist_ok=True)

        d = int(cfg.get("d", 16))
        dn_stride = int(cfg.get("dnmap_stride", 2))
        windows = list(cfg.get("windows", [(2.0, 5.0)]))
        k_used = int(cfg.get("k", cfg.get("primes_small_k", 13)))
        tp_backend = str(cfg.get("tp_backend", "legacy"))

        # Measurement digests
        meas_boundary_idx = get_measurement_boundary_idx(cfg, d)
        meas_boundary_digest = _hash_array16(meas_boundary_idx)
        meas_params_digest = sha16(
            repr(
                {
                    "d": int(d),
                    "meas_sigma": float(cfg.get("meas_sigma", 0.75)),
                    "meas_diag": float(cfg.get("meas_diag", 0.8)),
                    "meas_boundary_dim": int(cfg.get("meas_boundary_dim", max(1, d // 4))),
                    "meas_jitter": float(cfg.get("meas_jitter", 1e-6)),
                    "meas_boundary_seed": cfg.get("meas_boundary_seed", None),
                    "dnmap_stride": int(dn_stride),
                }
            ).encode("utf-8")
        )

        primes_used = list(map(int, first_primes(int(k_used))))
        Tp_true, u_vals, theta_vals = _build_Tp_backend_dispatch(tp_backend, primes_used, d=d, seed=int(seed), cfg=cfg)

        fp = tp_fingerprint(Tp_true, primes_used, tp_backend=tp_backend, u_vals=u_vals, theta_vals=theta_vals)

        # Pack knobs
        n_snap = int(cfg.get("chan_n_snap", 5))
        trace_K = int(cfg.get("chan_trace_K", 8))
        cayley_eta_default = float(cfg.get("chan_cayley_eta", 1.0))
        orbit_z = float(cfg.get("chan_orbit_z", 0.5))
        # Optional overrides for deterministic pilots:
        # - chan_energy_snap_by_window: dict[(wlo,whi) or "wlo_whi"] -> list of energies
        # - chan_cayley_eta_by_window: dict[(wlo,whi) or "wlo_whi"] -> float eta
        energy_snap_by_win = cfg.get("chan_energy_snap_by_window", None)
        eta_by_win = cfg.get("chan_cayley_eta_by_window", None)
        energy_snap_map = None
        eta_map = None
        if isinstance(energy_snap_by_win, dict):
            energy_snap_map = { _parse_window_key(k): np.asarray(list(map(float, v)), dtype=np.float64) for k, v in energy_snap_by_win.items() }
        if isinstance(eta_by_win, dict):
            eta_map = { _parse_window_key(k): float(v) for k, v in eta_by_win.items() }

        rows = []
        for (wlo, whi) in windows:
            energies = dnmap_energy_grid(cfg, wlo=float(wlo), whi=float(whi), dn_stride=int(dn_stride))
            meas_E_digest = _hash_array16(energies)

            wkey = (float(wlo), float(whi))
            E_override = None
            if energy_snap_map is not None and wkey in energy_snap_map:
                E_override = np.asarray(energy_snap_map[wkey], dtype=np.float64)
            eta_here = float(eta_map.get(wkey, cayley_eta_default)) if eta_map is not None else float(cayley_eta_default)

            tag_E = int(E_override.size) if (E_override is not None) else int(n_snap)

            win_dir = os.path.join(out_dir, f"window_{float(wlo)}_{float(whi)}")
            summary = channel_diag_pack(
                cfg=cfg,
                Tp_true=Tp_true,
                primes_used=primes_used,
                energies=np.asarray(energies, dtype=np.float64),
                amp=float(amp),
                out_dir=win_dir,
                n_snap=int(n_snap),
                trace_K=int(trace_K),
                cayley_eta=float(eta_here),
                energies_snap_override=E_override,
                orbit_z=float(orbit_z),
                tag=f"E{int(tag_E)}_K{int(trace_K)}",
            )

            row = {
                "tp_backend": str(tp_backend),
                "seed": int(seed),
                "anchor_seed": int(cfg.get("anchor_seed", 0)),
                "k": int(k_used),
                "wlo": float(wlo),
                "whi": float(whi),
                "amp": float(amp),
                "dnmap_stride": int(dn_stride),
                "meas_params_digest": str(meas_params_digest),
                "meas_boundary_digest": str(meas_boundary_digest),
                "meas_E_digest": str(meas_E_digest),
                "tp_sig_head_hash": str(fp.get("tp_hash", "")),
                "tp_sig_trace_sum": float(fp.get("tp_sig_trace_sum", float("nan"))),
                "tp_sig_norm_sum": float(fp.get("tp_sig_norm_sum", float("nan"))),
                "elapsed_s": float(time.time() - t0),
            }
            row.update(summary)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "rows.csv"), index=False)
        with open(os.path.join(out_dir, "cfg.json"), "w") as f:
            def _json_safe(x):
                if isinstance(x, dict):
                    out = {}
                    for kk, vv in x.items():
                        try:
                            k2 = kk if isinstance(kk, (str, int, float, bool)) or kk is None else str(kk)
                        except Exception:
                            k2 = str(kk)
                        out[k2] = _json_safe(vv)
                    return out
                if isinstance(x, (list, tuple)):
                    return [_json_safe(v) for v in list(x)]
                return x

            json.dump(_json_safe(cfg), f, indent=2, default=str)
        return df

if "dnmap_p_channel_label_scramble" not in globals():
    def dnmap_p_channel_label_scramble(
        cfg: dict,
        seed: int,
        Tp_true: dict,
        primes_used: list,
        anchor_seed: int,
        wlo: float,
        whi: float,
        dn_stride: int,
        N_null: int,
        amp: float,
    ):
        """Computes p_dnmap via label-scramble null on the frozen DN-map/logdet spine."""
        energies = dnmap_energy_grid(cfg, wlo=float(wlo), whi=float(whi), dn_stride=int(dn_stride))

        # Precompute weight matrix and true prime matrix stack once (nulls are permutations of it).
        d = int(cfg.get("d", 16))
        sigma = float(cfg.get("meas_sigma", 0.75))
        diag = float(cfg.get("meas_diag", 0.8)) + float(amp)
        jitter = float(cfg.get("meas_jitter", 1e-6))
        bidx = get_measurement_boundary_idx(cfg, d)
        b = int(bidx.size)
        iidx = np.asarray([i for i in range(d) if i not in set(map(int, bidx.tolist()))], dtype=np.int64)

        ps = list(map(int, primes_used))
        ps_arr = np.asarray(ps, dtype=np.float64)
        lp = np.log(ps_arr)
        Ecol = energies.reshape((-1, 1))
        w = (np.cos(Ecol * lp.reshape((1, -1))) + 0.35 * np.sin(Ecol * lp.reshape((1, -1)))) / np.power(ps_arr.reshape((1, -1)), sigma)

        M_true = np.stack([_sym(np.asarray(Tp_true[int(p)], dtype=np.float64)) for p in ps], axis=0)
        idx_map = {p: i for i, p in enumerate(ps)}

        eye_d = np.eye(d, dtype=np.float64)
        eye_b = np.eye(b, dtype=np.float64)
        eye_db = np.eye(int(d - b), dtype=np.float64)

        # Precompute einsum contraction paths once per (shape, equation).
        # This avoids repeatedly calling einsum_path inside tight null loops.
        # NOTE: optimize=True in numpy corresponds to the greedy path search.
        _einsum_path_ep = np.einsum_path(
            "ep,pij->eij",
            np.empty((w.shape[0], w.shape[1]), dtype=np.float64),
            M_true,
            optimize="greedy",
        )[0]
        _einsum_path_kep = np.einsum_path(
            "kep,pij->keij",
            np.empty((2, w.shape[0], w.shape[1]), dtype=np.float64),
            M_true,
            optimize="greedy",
        )[0]

        def _score_single(w_one: np.ndarray) -> float:
            """Compute DN-map/logdet score for a single weight matrix.

            w_one: (nE, nP)
            returns scalar
            """
            A = np.einsum("ep,pij->eij", w_one, M_true, optimize=_einsum_path_ep)
            A = 0.5 * (A + np.swapaxes(A, 1, 2))
            A += diag * eye_d[None, :, :]

            Abb = A[:, bidx[:, None], bidx[None, :]]
            Abi = A[:, bidx[:, None], iidx[None, :]]
            Aib = A[:, iidx[:, None], bidx[None, :]]
            Aii = A[:, iidx[:, None], iidx[None, :]]

            Aii_reg = Aii + jitter * eye_db[None, :, :]
            X = np.linalg.solve(Aii_reg, Aib)
            S = Abb - np.matmul(Abi, X)
            S = 0.5 * (S + np.swapaxes(S, 1, 2))
            S_reg = S + jitter * eye_b[None, :, :]

            sign, ld = np.linalg.slogdet(S_reg)
            ld = np.where(sign > 0, ld, ld - 100.0)
            return float(np.mean(ld))

        def _score_from_wbatch(w_batch: np.ndarray) -> np.ndarray:
            """Compute DN-map/logdet scores for a batch of weight matrices.

            w_batch: (K, nE, nP)
            returns: (K,)
            """
            A = np.einsum("kep,pij->keij", w_batch, M_true, optimize=_einsum_path_kep)
            A = 0.5 * (A + np.swapaxes(A, 2, 3))
            A += diag * eye_d[None, None, :, :]

            Abb = A[:, :, bidx[:, None], bidx[None, :]]
            Abi = A[:, :, bidx[:, None], iidx[None, :]]
            Aib = A[:, :, iidx[:, None], bidx[None, :]]
            Aii = A[:, :, iidx[:, None], iidx[None, :]]

            Aii_reg = Aii + jitter * eye_db[None, None, :, :]
            X = np.linalg.solve(Aii_reg, Aib)
            S = Abb - np.matmul(Abi, X)
            S = 0.5 * (S + np.swapaxes(S, 2, 3))
            S_reg = S + jitter * eye_b[None, None, :, :]

            sign, ld = np.linalg.slogdet(S_reg)
            ld = np.where(sign > 0, ld, ld - 100.0)
            return np.mean(ld, axis=1)

        # Observed score (K=1)
        dn_obs = float(_score_from_wbatch(w[None, :, :])[0])

        # Nulls: permutations of the weight columns (cheaper than permuting M_true).
        # Use batching when possible, but fall back to per-null scoring on MemoryError.
        batch = int(cfg.get("dnmap_null_batch", 128))
        batch = max(1, min(batch, int(N_null)))
        null_arr = np.empty(int(N_null), dtype=np.float64)
        sequential = bool(cfg.get("sequential", True))
        ps_list = list(map(int, ps))

        def _fill_one(jj: int) -> float:
            phi = make_label_scramble_phi(
                primes_used,
                scramble_seed=int(anchor_seed) + (100000 * jj if sequential else jj)
            )
            perm = np.asarray([idx_map[int(phi[p])] for p in ps_list], dtype=np.int64)
            inv_perm = np.empty_like(perm)
            inv_perm[perm] = np.arange(perm.size, dtype=np.int64)
            return _score_single(w[:, inv_perm])

        start = 0
        while start < int(N_null):
            bs = min(batch, int(N_null) - start)
            if bs <= 1:
                null_arr[start] = _fill_one(start)
                start += 1
                continue

            try:
                w_batch = np.empty((bs, w.shape[0], w.shape[1]), dtype=np.float64)
                for j in range(bs):
                    jj = start + j
                    phi = make_label_scramble_phi(
                        primes_used,
                        scramble_seed=int(anchor_seed) + (100000 * jj if sequential else jj)
                    )
                    perm = np.asarray([idx_map[int(phi[p])] for p in ps_list], dtype=np.int64)
                    inv_perm = np.empty_like(perm)
                    inv_perm[perm] = np.arange(perm.size, dtype=np.int64)
                    w_batch[j] = w[:, inv_perm]

                null_arr[start:start + bs] = _score_from_wbatch(w_batch)
                start += bs
            except MemoryError:
                # Fallback: per-null scoring with minimal allocations.
                batch = 1
                continue

        p_dnmap = empirical_p_lower(null_arr, float(dn_obs), tie_le=True)

        extras = {}
        if bool(cfg.get("dnmap_uniqueness", False)):
            tol = float(cfg.get("dn_collision_tol", 0.0))
            null_min = float(np.min(null_arr)) if null_arr.size else float("nan")
            null_argmin = int(np.argmin(null_arr)) if null_arr.size else -1
            null_std = float(np.std(null_arr)) if null_arr.size else float("nan")
            null_med = float(np.median(null_arr)) if null_arr.size else float("nan")
            # identity rank among {identity} U {nulls}, lower is better (lower-tail statistic)
            identity_rank = 1 + int(np.sum(null_arr < float(dn_obs)))
            collision_count = int(np.sum(null_arr <= (float(dn_obs) + tol)))
            collision_rate = float(collision_count) / float(null_arr.size) if null_arr.size else 0.0
            extras = dict(
                dn_null_min=float(null_min),
                dn_null_argmin=int(null_argmin),
                dn_identity_rank=int(identity_rank),
                dn_collision_tol=float(tol),
                dn_collision_count=int(collision_count),
                dn_collision_rate=float(collision_rate),
                dn_gap_min=float(null_min - float(dn_obs)) if np.isfinite(null_min) else float("nan"),
                # variance inflation diagnostics (derived only; does not affect p-values)
                dn_null_std=float(null_std),
                dn_null_median=float(null_med),
                dn_null_cv=(float(null_std) / (abs(float(null_med)) + 1e-12)) if np.isfinite(null_std) and np.isfinite(null_med) else float("nan"),
            )

        return float(dn_obs), null_arr, float(p_dnmap), extras


if "dnmap_p_channel_label_scramble_paired_fe" not in globals():
    def dnmap_p_channel_label_scramble_paired_fe(
        cfg: dict,
        seed: int,
        Tp_true: dict,
        primes_used: list,
        anchor_seed: int,
        w1: tuple,
        w2: tuple,
        dn_stride: int,
        N_null: int,
        amp: float,
    ):
        """Phase-3C FE residual using the canonical involution E -> -E.

        For each window W, define the FE residual (scalar) as an aggregation over the *same* DN energy grid:
            FE(W) = agg_{E in grid(W)} |(logXi+G)(E) - (logXi+G)(-E)|

        where logXi(E) is the frozen DN-map/logdet per-energy output and G(E) is an optional
        deterministic analytic counterterm (null-free). Null-calibration is optional and uses
        per-energy null statistics (label-scramble family).

        Returns:
          - dnmap scores + nulls for W1 and W2 (for rigidity p-values)
          - p_fe (global, AND across windows via max)
          - extras containing per-window FE stats, digests, and raw residuals.
        """

        (w1lo, w1hi) = (float(w1[0]), float(w1[1]))
        (w2lo, w2hi) = (float(w2[0]), float(w2[1]))

        # Backward-compat alias: if callers set fe_completion_mode to a calibration mode, treat it as calibration.
        calib = str(cfg.get("fe_calibration_mode", "none")).strip().lower()
        legacy_mode = str(cfg.get("fe_completion_mode", "none")).strip().lower()
        if legacy_mode in {"center_null_mean", "center_null_median", "zscore_null"} and calib in {"none", ""}:
            calib = legacy_mode

        comp_mode = str(cfg.get("fe_completion_mode", "none")).strip().lower()
        alpha_E = float(cfg.get("fe_alpha_E", 1.0))
        beta = float(cfg.get("fe_beta", 1.0))
        G_clip = cfg.get("fe_G_clip", None)
        agg = str(cfg.get("fe_agg", "mean")).strip().lower()
        calib_eps = float(cfg.get("fe_calibration_eps", 1e-12))

        if agg not in {"mean", "median"}:
            raise ValueError(f"Unknown fe_agg={agg!r}")
        if calib not in {"none", "center_null_mean", "center_null_median", "zscore_null"}:
            raise ValueError(f"Unknown fe_calibration_mode={calib!r}")
        if comp_mode not in {"none", "analytic_harmonic"}:
            # allow legacy callers using fe_completion_mode as calibration
            if comp_mode in {"center_null_mean", "center_null_median", "zscore_null"}:
                comp_mode = "none"
            else:
                raise ValueError(f"Unknown fe_completion_mode={comp_mode!r}")

        def _G(E: np.ndarray) -> np.ndarray:
            E = np.asarray(E, dtype=np.float64)
            if comp_mode == "none":
                return np.zeros_like(E)
            # analytic_harmonic: beta * log(1/4 + (alpha_E * E)^2)
            val = beta * np.log(0.25 + (alpha_E * E) ** 2)
            if G_clip is not None:
                gc = float(G_clip)
                val = np.clip(val, -abs(gc), abs(gc))
            return val

        def _agg_abs(x: np.ndarray) -> float:
            x = np.asarray(x, dtype=np.float64)
            if x.size == 0:
                return 0.0
            if agg == "mean":
                return float(np.mean(np.abs(x)))
            return float(np.median(np.abs(x)))

        # Common DN-map setup
        d = int(cfg.get("d", 16))
        sigma = float(cfg.get("meas_sigma", 0.75))
        diag = float(cfg.get("meas_diag", 0.8)) + float(amp)
        jitter = float(cfg.get("meas_jitter", 1e-6))
        bidx = get_measurement_boundary_idx(cfg, d)
        b = int(bidx.size)
        iidx = np.asarray([i for i in range(d) if i not in set(map(int, bidx.tolist()))], dtype=np.int64)

        ps = list(map(int, primes_used))
        ps_arr = np.asarray(ps, dtype=np.float64)
        lp = np.log(ps_arr)
        M_true = np.stack([_sym(np.asarray(Tp_true[int(p)], dtype=np.float64)) for p in ps], axis=0)
        idx_map = {p: i for i, p in enumerate(ps)}

        eye_d = np.eye(d, dtype=np.float64)
        eye_b = np.eye(b, dtype=np.float64)
        eye_db = np.eye(int(d - b), dtype=np.float64)

        # Precompute contraction path once per (nE,nP)
        def _paths(nE: int, nP: int):
            return np.einsum_path(
                "kep,pij->keij",
                np.empty((2, nE, nP), dtype=np.float64),
                M_true,
                optimize="greedy",
            )[0]

        def _dn_score_from_wbatch(w_batch: np.ndarray, einsum_path) -> np.ndarray:
            """Returns scalar DN score per batch entry: mean_E logdet(S(E))."""
            A = np.einsum("kep,pij->keij", w_batch, M_true, optimize=einsum_path)
            A = 0.5 * (A + np.swapaxes(A, 2, 3))
            A += diag * eye_d[None, None, :, :]

            Abb = A[:, :, bidx[:, None], bidx[None, :]]
            Abi = A[:, :, bidx[:, None], iidx[None, :]]
            Aib = A[:, :, iidx[:, None], bidx[None, :]]
            Aii = A[:, :, iidx[:, None], iidx[None, :]]

            Aii_reg = Aii + jitter * eye_db[None, None, :, :]
            X = np.linalg.solve(Aii_reg, Aib)
            S = Abb - np.matmul(Abi, X)
            S = 0.5 * (S + np.swapaxes(S, 2, 3))
            S_reg = S + jitter * eye_b[None, None, :, :]
            sign, ld = np.linalg.slogdet(S_reg)
            ld = np.where(sign > 0, ld, ld - 100.0)
            return np.mean(ld, axis=1)

        def _ld_per_energy_from_wbatch(w_batch: np.ndarray, einsum_path) -> np.ndarray:
            """Returns per-energy logdet(S(E)) for each batch entry.

            w_batch: (K,nE,nP)
            returns: (K,nE)
            """
            A = np.einsum("kep,pij->keij", w_batch, M_true, optimize=einsum_path)
            A = 0.5 * (A + np.swapaxes(A, 2, 3))
            A += diag * eye_d[None, None, :, :]

            Abb = A[:, :, bidx[:, None], bidx[None, :]]
            Abi = A[:, :, bidx[:, None], iidx[None, :]]
            Aib = A[:, :, iidx[:, None], bidx[None, :]]
            Aii = A[:, :, iidx[:, None], iidx[None, :]]

            Aii_reg = Aii + jitter * eye_db[None, None, :, :]
            X = np.linalg.solve(Aii_reg, Aib)
            S = Abb - np.matmul(Abi, X)
            S = 0.5 * (S + np.swapaxes(S, 2, 3))
            S_reg = S + jitter * eye_b[None, None, :, :]
            sign, ld = np.linalg.slogdet(S_reg)
            ld = np.where(sign > 0, ld, ld - 100.0)
            return ld

        def _window_bundle(wlo: float, whi: float):
            energies = dnmap_energy_grid(cfg, wlo=float(wlo), whi=float(whi), dn_stride=int(dn_stride))
            energies = np.asarray(energies, dtype=np.float64)
            nE = int(energies.size)
            if nE <= 0:
                raise ValueError("Empty energy grid in dnmap_fe")

            # Mirror pairing is exact by construction: use same grid with negative sign.
            Epos = energies
            Eneg = -energies
            mirror_E_digest = _hash_array16(np.stack([Epos, Eneg], axis=0))

            Epos_col = Epos.reshape((-1, 1))
            Eneg_col = Eneg.reshape((-1, 1))
            # w(+E) and w(-E)
            wpos = (np.cos(Epos_col * lp.reshape((1, -1))) + 0.35 * np.sin(Epos_col * lp.reshape((1, -1)))) / np.power(ps_arr.reshape((1, -1)), sigma)
            wneg = (np.cos(Eneg_col * lp.reshape((1, -1))) + 0.35 * np.sin(Eneg_col * lp.reshape((1, -1)))) / np.power(ps_arr.reshape((1, -1)), sigma)

            einsum_path = _paths(nE, int(ps_arr.size))

            # Observed
            ld_pos_obs = _ld_per_energy_from_wbatch(wpos[None, :, :], einsum_path)[0]
            ld_neg_obs = _ld_per_energy_from_wbatch(wneg[None, :, :], einsum_path)[0]
            Gpos = _G(Epos)
            Gneg = _G(Eneg)
            # Raw residual per energy
            delta_raw = (ld_pos_obs + Gpos) - (ld_neg_obs + Gneg)
            fe_obs_raw = _agg_abs(delta_raw)

            # Null batches: we store per-energy ld arrays to support per-energy calibration modes.
            Nn = int(N_null)
            batch = int(cfg.get("dnmap_null_batch", 128))
            batch = max(1, min(batch, Nn))
            sequential = bool(cfg.get("sequential", True))
            ps_list = list(map(int, ps))

            ld_pos_null = np.empty((Nn, nE), dtype=np.float64)
            ld_neg_null = np.empty((Nn, nE), dtype=np.float64)
            dn_null_scores = np.empty(Nn, dtype=np.float64)

            start = 0
            while start < Nn:
                bs = min(batch, Nn - start)
                try:
                    wpos_batch = np.empty((bs, nE, ps_arr.size), dtype=np.float64)
                    wneg_batch = np.empty((bs, nE, ps_arr.size), dtype=np.float64)
                    for j in range(bs):
                        jj = start + j
                        phi = make_label_scramble_phi(
                            primes_used,
                            scramble_seed=int(anchor_seed) + (100000 * jj if sequential else jj)
                        )
                        perm = np.asarray([idx_map[int(phi[p])] for p in ps_list], dtype=np.int64)
                        inv_perm = np.empty_like(perm)
                        inv_perm[perm] = np.arange(perm.size, dtype=np.int64)
                        wpos_batch[j] = wpos[:, inv_perm]
                        wneg_batch[j] = wneg[:, inv_perm]

                    ld_pos = _ld_per_energy_from_wbatch(wpos_batch, einsum_path)
                    ld_neg = _ld_per_energy_from_wbatch(wneg_batch, einsum_path)
                    ld_pos_null[start:start + bs] = ld_pos
                    ld_neg_null[start:start + bs] = ld_neg
                    dn_null_scores[start:start + bs] = _dn_score_from_wbatch(wpos_batch, einsum_path)
                    start += bs
                except MemoryError:
                    batch = 1
                    continue

            # DN-map rigidity p-value for this window (unchanged statistic)
            dn_obs = float(np.mean(ld_pos_obs))
            p_dnmap = empirical_p_lower(dn_null_scores, float(dn_obs), tie_le=True)

            # Build calibrated FE residuals
            # Start from completed series
            pos_series_obs = ld_pos_obs + Gpos
            neg_series_obs = ld_neg_obs + Gneg
            pos_series_null = ld_pos_null + Gpos[None, :]
            neg_series_null = ld_neg_null + Gneg[None, :]

            fe_obs = float(fe_obs_raw)
            fe_null = np.empty(Nn, dtype=np.float64)

            if calib == "none":
                delta_obs = pos_series_obs - neg_series_obs
                for j in range(Nn):
                    fe_null[j] = _agg_abs(pos_series_null[j] - neg_series_null[j])
                fe_obs = _agg_abs(delta_obs)
            else:
                if calib == "center_null_mean":
                    mu_pos = np.mean(pos_series_null, axis=0)
                    mu_neg = np.mean(neg_series_null, axis=0)
                    pos_obs_c = pos_series_obs - mu_pos
                    neg_obs_c = neg_series_obs - mu_neg
                    pos_null_c = pos_series_null - mu_pos[None, :]
                    neg_null_c = neg_series_null - mu_neg[None, :]
                elif calib == "center_null_median":
                    med_pos = np.median(pos_series_null, axis=0)
                    med_neg = np.median(neg_series_null, axis=0)
                    pos_obs_c = pos_series_obs - med_pos
                    neg_obs_c = neg_series_obs - med_neg
                    pos_null_c = pos_series_null - med_pos[None, :]
                    neg_null_c = neg_series_null - med_neg[None, :]
                else:
                    # zscore_null
                    mu_pos = np.mean(pos_series_null, axis=0)
                    mu_neg = np.mean(neg_series_null, axis=0)
                    sd_pos = np.std(pos_series_null, axis=0)
                    sd_neg = np.std(neg_series_null, axis=0)
                    pos_obs_c = (pos_series_obs - mu_pos) / (sd_pos + calib_eps)
                    neg_obs_c = (neg_series_obs - mu_neg) / (sd_neg + calib_eps)
                    pos_null_c = (pos_series_null - mu_pos[None, :]) / (sd_pos[None, :] + calib_eps)
                    neg_null_c = (neg_series_null - mu_neg[None, :]) / (sd_neg[None, :] + calib_eps)

                fe_obs = _agg_abs(pos_obs_c - neg_obs_c)
                for j in range(Nn):
                    fe_null[j] = _agg_abs(pos_null_c[j] - neg_null_c[j])

            p_fe_w = empirical_p_lower(fe_null, float(fe_obs), tie_le=True)

            fe_extras = dict(
                fe_obs=float(fe_obs),
                fe_obs_raw=float(fe_obs_raw),
                fe_null_mean=float(np.mean(fe_null)) if fe_null.size else float("nan"),
                fe_null_std=float(np.std(fe_null)) if fe_null.size else float("nan"),
                fe_null_median=float(np.median(fe_null)) if fe_null.size else float("nan"),
                fe_null_min=float(np.min(fe_null)) if fe_null.size else float("nan"),
                fe_null_argmin=int(np.argmin(fe_null)) if fe_null.size else -1,
                fe_identity_rank=1 + int(np.sum(fe_null < float(fe_obs))) if fe_null.size else 1,
                fe_mirror_E_digest=str(mirror_E_digest),
                fe_w=(float(wlo), float(whi)),
                p_fe_window=float(p_fe_w),
            )
            return float(dn_obs), dn_null_scores, float(p_dnmap), float(p_fe_w), fe_extras

        # Compute both windows
        dn1_obs, null1, p_dn1, p_fe1, ex1 = _window_bundle(w1lo, w1hi)
        dn2_obs, null2, p_dn2, p_fe2, ex2 = _window_bundle(w2lo, w2hi)

        # Global FE (AND across windows): require FE-consistency in both windows
        p_fe = float(max(float(p_fe1), float(p_fe2)))

        # Policy digest (includes involution rule and completion/calibration knobs)
        fe_policy_digest = sha16(
            repr(
                {
                    "fe_involution": "E_to_minusE",
                    "fe_pair_w1": (w1lo, w1hi),
                    "fe_pair_w2": (w2lo, w2hi),
                    "fe_calibration_mode": calib,
                    "fe_calibration_eps": float(calib_eps),
                    "fe_completion_mode": comp_mode,
                    "fe_alpha_E": float(alpha_E),
                    "fe_beta": float(beta),
                    "fe_G_clip": (float(G_clip) if G_clip is not None else None),
                    "fe_agg": agg,
                    "fe_mirror_E_digest_w1": ex1.get("fe_mirror_E_digest", ""),
                    "fe_mirror_E_digest_w2": ex2.get("fe_mirror_E_digest", ""),
                }
            ).encode("utf-8")
        )

        extras = dict(
            fe_policy_digest=str(fe_policy_digest),
            fe_involution=str("E_to_minusE"),
            fe_agg=str(agg),
            fe_calibration_mode=str(calib),
            fe_calibration_eps=float(calib_eps),
            fe_completion_mode=str(comp_mode),
            fe_alpha_E=float(alpha_E),
            fe_beta=float(beta),
            fe_G_clip=(float(G_clip) if G_clip is not None else None),
            # Per-window FE diagnostics
            fe_w1=(w1lo, w1hi),
            fe_w2=(w2lo, w2hi),
            p_fe_w1=float(p_fe1),
            p_fe_w2=float(p_fe2),
            fe_obs_w1=float(ex1.get("fe_obs", float("nan"))),
            fe_obs_w2=float(ex2.get("fe_obs", float("nan"))),
            fe_obs_raw_w1=float(ex1.get("fe_obs_raw", float("nan"))),
            fe_obs_raw_w2=float(ex2.get("fe_obs_raw", float("nan"))),
            fe_null_mean_w1=float(ex1.get("fe_null_mean", float("nan"))),
            fe_null_mean_w2=float(ex2.get("fe_null_mean", float("nan"))),
            fe_null_std_w1=float(ex1.get("fe_null_std", float("nan"))),
            fe_null_std_w2=float(ex2.get("fe_null_std", float("nan"))),
            fe_null_min_w1=float(ex1.get("fe_null_min", float("nan"))),
            fe_null_min_w2=float(ex2.get("fe_null_min", float("nan"))),
            fe_identity_rank_w1=int(ex1.get("fe_identity_rank", -1)),
            fe_identity_rank_w2=int(ex2.get("fe_identity_rank", -1)),
            fe_mirror_E_digest_w1=str(ex1.get("fe_mirror_E_digest", "")),
            fe_mirror_E_digest_w2=str(ex2.get("fe_mirror_E_digest", "")),
        )

        return (float(dn1_obs), float(dn2_obs)), (null1, null2), float(p_dn1), float(p_dn2), float(p_fe), extras

# ============================================
# CELL 4/5: Core runner primitives (self-contained)
# - run_one
# - decide_table / add_ptarget_fields
# - pack_run
# - export_sweep_artifacts_v2 (zips the output dir + downloads)
# ============================================

def decide_table(df: pd.DataFrame, alpha: float = 0.01, audit: bool = True) -> pd.DataFrame:
    df = df.copy()
    if "p_family" not in df.columns:
        raise RuntimeError("decide_table: missing p_family")
    df["reject"] = df["p_family"] <= float(alpha)
    return df

def add_ptarget_fields(df: pd.DataFrame) -> pd.DataFrame:
    # Canonical reporting: p_target uses N_target+1 smoothing if N_null_used present
    out = df.copy()
    if "N_null_used" in out.columns and "p_family" in out.columns:
        # Here p_family already computed with add-one smoothing; we keep it as p_family_target too.
        out["p_family_target"] = out["p_family"]
    return out

def pack_run(out_dir: str, df: pd.DataFrame, cfg: dict, alpha: float = 0.01, audit: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "rows.csv"), index=False)
    with open(os.path.join(out_dir, "cfg.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    # quick aggregate
    dec = decide_table(df, alpha=float(alpha), audit=audit)
    agg = (dec.groupby([
                "tp_backend",
                "wlo",
                "whi",
                "pq_anchor_M",
                "pqr_anchor_M",
                "p_pow_kmax",
                "pqr_two_path",
            ], dropna=False)
           .agg(rows=("reject","size"),
                reject_rate=("reject","mean"),
                worst_p=("p_family","max"),
                best_p=("p_family","min"))
           .reset_index())
    agg.to_csv(os.path.join(out_dir, "aggregate.csv"), index=False)
    return agg

def export_sweep_artifacts_v2(df: pd.DataFrame, name: str, do_download: bool = True, out_root: str = "."):
    # writes CSVs + zips the folder "name_artifacts"
    out_dir = os.path.join(out_root, f"{name}_artifacts")
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

    # convenience aggregates
    agg = (df.groupby(["backend","wlo","whi","pqM_target","pqrM_target","kmax_target"], dropna=False)
           .agg(rows=("reject","size"),
                reject_rate=("reject","mean"),
                worst_p=("p_family","max"),
                best_p=("p_family","min"))
           .reset_index()
           .sort_values(["backend","wlo","whi"]))
    agg.to_csv(os.path.join(out_dir, f"{name}__aggregate.csv"), index=False)

    byrun_cols = ["backend","seed","anchor_seed","wlo","whi","p_family","reject","tp_sig_head_hash"]
    byrun_cols = [c for c in byrun_cols if c in df.columns]
    df[byrun_cols].to_csv(os.path.join(out_dir, f"{name}__byrun.csv"), index=False)

    # zip
    zip_path = shutil.make_archive(out_dir, "zip", out_dir)

    if do_download:
        try:
            from google.colab import files
            files.download(zip_path)
        except Exception as e:
            print("Download skipped (not in Colab or blocked):", repr(e))

    return zip_path

def run_one(cfg: dict, seed: int, amp: float, out_dir: str) -> pd.DataFrame:
    t0 = time.time()
    cfg = dict(cfg)

    # Phase 3B discipline: measurement spine must stay frozen across backend swaps.
    if bool(cfg.get("freeze_measurement", False)) and ("CFG_BASE" in globals()):
        for k in ("meas_sigma", "meas_diag", "meas_boundary_dim", "meas_jitter", "meas_boundary_seed", "meas_boundary_idx"):
            if k in cfg and k in CFG_BASE and (str(cfg[k]) != str(CFG_BASE[k])):
                raise ValueError(
                    f"Measurement spine is frozen (freeze_measurement=True) but cfg[{k!r}]={cfg[k]!r} "
                    f"differs from CFG_BASE[{k!r}]={CFG_BASE[k]!r}."
                )

    d = int(cfg.get("d", 16))
    nE = int(cfg.get("nE", 256))
    windows = list(cfg.get("windows", [(0.60,7.50)]))
    dn_stride = int(cfg.get("dnmap_stride", 2))
    primes_small_k = int(cfg.get("primes_small_k", max(cfg.get("n_ops_list",[13]))))
    n_ops_list = list(cfg.get("n_ops_list", [13]))

    N_null = int(cfg.get("N_null", 256))
    alpha = float(cfg.get("alpha", 0.01))
    tp_backend = str(cfg.get("tp_backend", "legacy"))

    # Optional: enforce fixed branch/gauge policy (reviewer-proof, independent of measurement spine).
    if bool(cfg.get("freeze_geometry_policy", False)) and ("CFG_BASE" in globals()):
        for k in ("warp_mode", "warp_unwrap", "warp_y_scale", "gauge_policy"):
            if k in cfg and k in CFG_BASE and (str(cfg[k]) != str(CFG_BASE[k])):
                raise ValueError(
                    f"Geometry policy is frozen (freeze_geometry_policy=True) but cfg[{k!r}]={cfg[k]!r} "
                    f"differs from CFG_BASE[{k!r}]={CFG_BASE[k]!r}."
                )

    geom_policy_digest = sha16(
        repr(
            {
                "tp_backend": str(tp_backend),
                "warp_mode": str(cfg.get("warp_mode", "plus")),
                "warp_unwrap": bool(cfg.get("warp_unwrap", True)),
                "warp_y_scale": float(cfg.get("warp_y_scale", 1.0)),
                "gauge_policy": str(cfg.get("gauge_policy", "none")),
                "geom_v3_eps": float(cfg.get("geom_v3_eps", 0.01)),
                "geom_v3_theta_scale": (float(cfg.get("geom_v3_theta_scale")) if cfg.get("geom_v3_theta_scale") is not None else None),
                "geom_v3_rank": int(cfg.get("geom_v3_rank", 1)),
                "geom_v4_u_scale": float(cfg.get("geom_v4_u_scale", 1.0)),
                "geom_v4_theta_scale": float(cfg.get("geom_v4_theta_scale", 0.125)),
                "geom_v4_rank": int(cfg.get("geom_v4_rank", 1)),
                "geom_v4_eps": float(cfg.get("geom_v4_eps", 0.01)),
                "geom_v4_eps_global": float(cfg.get("geom_v4_eps_global", 0.02)),
                "geom_v4_u_clip": (float(cfg.get("geom_v4_u_clip")) if cfg.get("geom_v4_u_clip") is not None else None),
                "geom_v4_theta_clip": (float(cfg.get("geom_v4_theta_clip")) if cfg.get("geom_v4_theta_clip") is not None else None),
                "geom_v5_u_scale": float(cfg.get("geom_v5_u_scale", 1.0)),
                "geom_v5_theta_scale": float(cfg.get("geom_v5_theta_scale", 0.125)),
                "geom_v5_rank": int(cfg.get("geom_v5_rank", 1)),
                "geom_v5_eps": float(cfg.get("geom_v5_eps", 0.01)),
                "geom_v5_eps_global": float(cfg.get("geom_v5_eps_global", 0.02)),
                "geom_v5_u_clip": (float(cfg.get("geom_v5_u_clip")) if cfg.get("geom_v5_u_clip") is not None else None),
                "geom_v5_theta_clip": (float(cfg.get("geom_v5_theta_clip")) if cfg.get("geom_v5_theta_clip") is not None else None),
                "geom_v5_harmonic_gamma": float(cfg.get("geom_v5_harmonic_gamma", 1.0)),
                "geom_v5_harmonic_strength": float(cfg.get("geom_v5_harmonic_strength", 1.0)),

                "geom_v6_u_scale": float(cfg.get("geom_v6_u_scale", 1.0)),
                "geom_v6_theta_scale": float(cfg.get("geom_v6_theta_scale", 0.125)),
                "geom_v6_rank": int(cfg.get("geom_v6_rank", 1)),
                "geom_v6_eps": float(cfg.get("geom_v6_eps", 0.01)),
                "geom_v6_eps_global": float(cfg.get("geom_v6_eps_global", 0.02)),
                "geom_v6_u_clip": (float(cfg.get("geom_v6_u_clip")) if cfg.get("geom_v6_u_clip") is not None else None),
                "geom_v6_theta_clip": (float(cfg.get("geom_v6_theta_clip")) if cfg.get("geom_v6_theta_clip") is not None else None),
                "geom_v6_phase_lambda": float(cfg.get("geom_v6_phase_lambda", 0.05)),
                "geom_v6_phase_clip": float(cfg.get("geom_v6_phase_clip", 2.0)),

                "geom_v7_u_scale": float(cfg.get("geom_v7_u_scale", 1.0)),
                "geom_v7_theta_scale": float(cfg.get("geom_v7_theta_scale", 0.125)),
                "geom_v7_rank": int(cfg.get("geom_v7_rank", 1)),
                "geom_v7_eps": float(cfg.get("geom_v7_eps", 0.01)),
                "geom_v7_eps_global": float(cfg.get("geom_v7_eps_global", 0.02)),
                "geom_v7_u_clip": (float(cfg.get("geom_v7_u_clip")) if cfg.get("geom_v7_u_clip") is not None else None),
                "geom_v7_theta_clip": (float(cfg.get("geom_v7_theta_clip")) if cfg.get("geom_v7_theta_clip") is not None else None),
                "geom_v7_defect_mode": str(cfg.get("geom_v7_defect_mode", "block")),
                "geom_v7_defect_rho": float(cfg.get("geom_v7_defect_rho", 0.25)),
                "geom_v7_defect_strength": float(cfg.get("geom_v7_defect_strength", 1.0)),
                "geom_v7_defect_clip": float(cfg.get("geom_v7_defect_clip", 0.75)),
                "geom_v7_defect_salt": int(cfg.get("geom_v7_defect_salt", 1)),
                "geom_v7_defect_seed": int(cfg.get("geom_v7_defect_seed", 0)),

                "geom_v8_u_scale": float(cfg.get("geom_v8_u_scale", 1.0)),
                "geom_v8_theta_scale": float(cfg.get("geom_v8_theta_scale", 0.125)),
                "geom_v8_rank": int(cfg.get("geom_v8_rank", 1)),
                "geom_v8_eps": float(cfg.get("geom_v8_eps", 0.01)),
                "geom_v8_eps_global": float(cfg.get("geom_v8_eps_global", 0.02)),
                "geom_v8_u_clip": (float(cfg.get("geom_v8_u_clip")) if cfg.get("geom_v8_u_clip") is not None else None),
                "geom_v8_theta_clip": (float(cfg.get("geom_v8_theta_clip")) if cfg.get("geom_v8_theta_clip") is not None else None),
                "geom_v8_prime_defect_mode": str(cfg.get("geom_v8_prime_defect_mode", "block")),
                "geom_v8_prime_defect_rho": float(cfg.get("geom_v8_prime_defect_rho", 0.25)),
                "geom_v8_defect_strength": float(cfg.get("geom_v8_defect_strength", 1.0)),
                "geom_v8_defect_clip": float(cfg.get("geom_v8_defect_clip", 0.75)),
                "geom_v8_defect_salt": int(cfg.get("geom_v8_defect_salt", 1)),
                "geom_v8_defect_seed": int(cfg.get("geom_v8_defect_seed", 0)),
                "geom_v8_pair_chiral": bool(cfg.get("geom_v8_pair_chiral", True)),
                "geom_v8_defect_norm_mode": str(cfg.get("geom_v8_defect_norm_mode", "fro_match")),

                "geom_v9_u_scale": float(cfg.get("geom_v9_u_scale", 1.0)),
                "geom_v9_theta_scale": float(cfg.get("geom_v9_theta_scale", 0.125)),
                "geom_v9_rank": int(cfg.get("geom_v9_rank", 1)),
                "geom_v9_eps": float(cfg.get("geom_v9_eps", 0.01)),
                "geom_v9_eps_global": float(cfg.get("geom_v9_eps_global", 0.02)),
                "geom_v9_u_clip": (float(cfg.get("geom_v9_u_clip")) if cfg.get("geom_v9_u_clip") is not None else None),
                "geom_v9_theta_clip": (float(cfg.get("geom_v9_theta_clip")) if cfg.get("geom_v9_theta_clip") is not None else None),
                "geom_v9_transport_rule": str(cfg.get("geom_v9_transport_rule", "delta_raw_wrapped")),
                "geom_v9_transport_lambda": float(cfg.get("geom_v9_transport_lambda", 1.0)),
                "geom_v9_transport_clip": float(cfg.get("geom_v9_transport_clip", 0.25)),
                "geom_v9_transport_seed": int(cfg.get("geom_v9_transport_seed", 0)),

                "geom_v10_u_scale": float(cfg.get("geom_v10_u_scale", 1.0)),
                "geom_v10_theta_scale": float(cfg.get("geom_v10_theta_scale", 0.125)),
                "geom_v10_rank": int(cfg.get("geom_v10_rank", 1)),
                "geom_v10_eps": float(cfg.get("geom_v10_eps", 0.01)),
                "geom_v10_eps_global": float(cfg.get("geom_v10_eps_global", 0.02)),
                "geom_v10_u_clip": (float(cfg.get("geom_v10_u_clip")) if cfg.get("geom_v10_u_clip") is not None else None),
                "geom_v10_theta_clip": (float(cfg.get("geom_v10_theta_clip")) if cfg.get("geom_v10_theta_clip") is not None else None),
                "geom_v10_phasefield_graph": str(cfg.get("geom_v10_phasefield_graph", "path")),
                "geom_v10_phasefield_w": float(cfg.get("geom_v10_phasefield_w", 1.0)),
                "geom_v10_phasefield_eta": float(cfg.get("geom_v10_phasefield_eta", 0.25)),
                "geom_v10_phasefield_seed": int(cfg.get("geom_v10_phasefield_seed", 0)),

                "eps_auto_tune": bool(cfg.get("eps_auto_tune", False)),
                "eps_target_delta": float(cfg.get("eps_target_delta", 0.02)),
            }
        ).encode("utf-8")
    )

    pqM = int(cfg.get("pq_anchor_M", 0))
    pqrM = int(cfg.get("pqr_anchor_M", 0))
    kmax = int(cfg.get("p_pow_kmax", 0))
    two_path = bool(cfg.get("pqr_two_path", True))

    phase3_mode = str(cfg.get("phase3_mode", "dnmap_only")).strip().lower()
    closure_semantics = str(cfg.get("closure_semantics", "against_true_v5")).strip().lower()

    # Guardrail to prevent silently running "full" closure mode with semantics
    # that are not comparable to the documented Phase-2C/3A implementation.
    # Policies: "warn" (default), "error"/"raise" (hard fail), "allow" (no-op).
    closure_semantics_policy = str(cfg.get("closure_semantics_policy", "error")).strip().lower()
    doc_aligned_semantics = {"against_true_v5", "against_true", "v5"}
    if phase3_mode not in {"dnmap_only", "dnmap_fe", "dnmap_fe_both"} and closure_semantics not in doc_aligned_semantics:
        msg = (
            f"Full closure mode is using closure_semantics={closure_semantics!r}. "
            f"Doc-aligned semantics are {sorted(doc_aligned_semantics)}. "
            f"Set closure_semantics='against_true_v5' for comparability, or set "
            f"closure_semantics_policy='allow' to bypass."
        )
        if closure_semantics_policy in {"error", "raise", "strict"}:
            raise ValueError(msg)
        if closure_semantics_policy in {"warn", "warning"}:
            print("WARNING:", msg)

    use_pq = bool(cfg.get("use_pq", True)) and (pqM > 0)
    use_pqr = bool(cfg.get("use_pqr", True)) and (pqrM > 0) and bool(two_path)

    # optional gating on dnmap channel (default OFF) unless in dnmap_only mode
    use_dnmap_gate = bool(cfg.get("use_dnmap_gate", False))

    gate_keys = []
    if phase3_mode == "dnmap_only":
        use_dnmap_gate = True
        gate_keys = ["p_dnmap"]
    elif phase3_mode == "dnmap_fe":
        use_dnmap_gate = True
        gate_keys = ["p_fe"]
    elif phase3_mode == "dnmap_fe_both":
        use_dnmap_gate = True
        gate_keys = ["p_fe", "p_rigid"]
    else:
        if use_pq: gate_keys.append("p_pq")
        if use_pqr: gate_keys.append("p_pqr")
        if bool(cfg.get("use_pow", False)) and int(kmax) >= 2:
            gate_keys.append("p_pow")
        if use_dnmap_gate:
            gate_keys.append("p_dnmap")
    if not gate_keys:
        raise ValueError("No gate channels enabled (pqM/pqrM are both 0).")

    enabled = set(gate_keys)
    assert_gate_contract(gate_keys, enabled)

    primes_small = first_primes(primes_small_k)

    # Measurement spine fingerprint inputs (do NOT include Tp; we want to detect accidental seed dependence
    # from boundary/basis/energies/jitter changes, not Tp-induced changes).
    meas_boundary_idx = get_measurement_boundary_idx(cfg, d)
    meas_boundary_digest = _hash_array16(meas_boundary_idx)
    meas_params_digest = sha16(
        repr(
            {
                "d": int(d),
                "meas_sigma": float(cfg.get("meas_sigma", 0.75)),
                "meas_diag": float(cfg.get("meas_diag", 0.8)),
                "meas_boundary_dim": int(cfg.get("meas_boundary_dim", max(1, d // 4))),
                "meas_jitter": float(cfg.get("meas_jitter", 1e-6)),
                "meas_boundary_seed": cfg.get("meas_boundary_seed", None),
                "dnmap_stride": int(dn_stride),
            }
        ).encode("utf-8")
    )

    rows = []
    for n_ops in n_ops_list:
        k_used = min(int(n_ops), primes_small_k)
        primes_used = list(map(int, primes_small[:k_used]))

        # TRUE Tp backend (geometry only enters here)
        Tp_true, u_vals, theta_vals = _build_Tp_backend_dispatch(tp_backend, primes_used, d=d, seed=int(seed), cfg=cfg)

        # Derived-only defect metadata (v7): computed deterministically from primes_used + cfg.
        v7_meta = {}
        if str(tp_backend) == "geom_warp_dirac_v7_localdefect":
            v7_meta = _v7_defect_metadata(primes_used, cfg)

        v8_meta = {}
        if str(tp_backend) == "geom_warp_dirac_v8_boundarydefect":
            v8_meta = _v8_boundarydefect_metadata(primes_used, cfg, d=int(d))

        v9_meta = {}
        if str(tp_backend) == "geom_warp_dirac_v9_phasetransport":
            v9_meta = _v9_phasetransport_metadata(primes_used, cfg)

        v10_meta = {}
        if str(tp_backend) == "geom_warp_dirac_v10_phasefieldsolve":
            v10_meta = _v10_phasefieldsolve_metadata(primes_used, cfg)

        # Optional eps auto-tune for v4/v5/v6: tune *eps_global to hit tp_delta_mean ~ eps_target_delta.
        # This stays Phase-3B legal because only Tp construction changes.
        eps_global_key = (
            "geom_v4_eps_global"
            if str(tp_backend) == "geom_warp_dirac_v4"
            else (
                "geom_v5_eps_global"
                if str(tp_backend) == "geom_warp_dirac_v5"
                else (
                    "geom_v6_eps_global"
                    if str(tp_backend) == "geom_warp_dirac_v6"
                    else (
                        "geom_v7_eps_global"
                        if str(tp_backend) == "geom_warp_dirac_v7_localdefect"
                        else (
                            "geom_v8_eps_global"
                            if str(tp_backend) == "geom_warp_dirac_v8_boundarydefect"
                            else (
                                "geom_v9_eps_global"
                                if str(tp_backend) == "geom_warp_dirac_v9_phasetransport"
                                else ("geom_v10_eps_global" if str(tp_backend) == "geom_warp_dirac_v10_phasefieldsolve" else None)
                            )
                        )
                    )
                )
            )
        )
        eps_global_val = float(cfg.get(eps_global_key, 1.0)) if eps_global_key else float("nan")
        eps_global_before_autotune = float(eps_global_val)
        eps_global_auto = False

        if eps_global_key and bool(cfg.get("eps_auto_tune", False)):
            Tp_legacy, _, _ = _build_Tp_backend_dispatch("legacy", primes_used, d=d, seed=int(seed), cfg=cfg)
            target = float(cfg.get("eps_target_delta", 0.02))
            max_iter = int(cfg.get("eps_tune_max_iter", 20))
            eps_max = float(cfg.get("eps_tune_eps_max", 2.0))
            tol = float(cfg.get("eps_tune_tol", 0.001))

            lo, hi = 0.0, float(max(1e-6, eps_max))

            def _delta_for(scale: float) -> float:
                cfg2 = dict(cfg)
                cfg2[eps_global_key] = float(scale)
                Tp2, _, _ = _build_Tp_backend_dispatch(str(tp_backend), primes_used, d=d, seed=int(seed), cfg=cfg2)
                return float(_tp_relative_delta_summary(Tp_legacy, Tp2, primes_used)["tp_delta_mean"])

            d_lo = _delta_for(lo)
            d_hi = _delta_for(hi)
            # Expand hi if needed.
            expand = 0
            while (d_hi < target) and (hi < eps_max) and (expand < 8):
                hi = min(eps_max, hi * 2.0)
                d_hi = _delta_for(hi)
                expand += 1

            best_scale = hi
            best_err = abs(d_hi - target)
            if abs(d_lo - target) < best_err:
                best_scale, best_err = lo, abs(d_lo - target)

            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                d_mid = _delta_for(mid)
                err = abs(d_mid - target)
                if err < best_err:
                    best_scale, best_err = mid, err
                if err <= tol:
                    break
                if d_mid < target:
                    lo = mid
                else:
                    hi = mid

            cfg[eps_global_key] = float(best_scale)
            eps_global_val = float(best_scale)
            eps_global_auto = True
            # Rebuild Tp_true with tuned scale
            Tp_true, u_vals, theta_vals = _build_Tp_backend_dispatch(tp_backend, primes_used, d=d, seed=int(seed), cfg=cfg)

        # Normalize/tune visibility: log relative delta vs legacy (same seed/primes/d)
        tp_delta = {"tp_delta_mean": 0.0, "tp_delta_max": 0.0}
        if str(tp_backend) != "legacy":
            Tp_legacy, _, _ = _build_Tp_backend_dispatch("legacy", primes_used, d=d, seed=int(seed), cfg=cfg)
            tp_delta = _tp_relative_delta_summary(Tp_legacy, Tp_true, primes_used)

        fp = tp_fingerprint(Tp_true, primes_used, tp_backend=tp_backend, u_vals=u_vals, theta_vals=theta_vals)
        comm = commutator_summary(Tp_true, primes_used)

        # Phase-3C paired-window observable is computed once per (seed,n_ops) and reused across rows.
        fe_payload = None
        if phase3_mode in {"dnmap_fe", "dnmap_fe_both"}:
            if len(windows) < 2:
                raise ValueError("phase3_mode='dnmap_fe' requires at least two windows in cfg['windows']")
            anchor_seed_fe = int(cfg.get("anchor_seed", 0))
            fe_payload = dnmap_p_channel_label_scramble_paired_fe(
                cfg=cfg,
                seed=int(seed),
                Tp_true=Tp_true,
                primes_used=primes_used,
                anchor_seed=int(anchor_seed_fe),
                w1=tuple(windows[0]),
                w2=tuple(windows[1]),
                dn_stride=int(dn_stride),
                N_null=int(N_null),
                amp=float(amp),
            )

        for (wlo, whi) in windows:
            E = dnmap_energy_grid(cfg, wlo=float(wlo), whi=float(whi), dn_stride=int(dn_stride))
            meas_E_digest = _hash_array16(E)

            # anchors
            anchor_seed = int(cfg.get("anchor_seed", 0))
            pq_pairs = select_pq_pairs(primes_used, pqM if use_pq else 0, anchor_seed=anchor_seed)
            pqr_triples = select_pqr_triples(primes_used, pqrM if use_pqr else 0, anchor_seed=anchor_seed)
            anchor_h, pq_d, pqr_d = anchor_hash_and_digest(pq_pairs, pqr_triples)

            # ------------------------------
            # Closure residual channels (real)
            # ------------------------------
            p = {k: float("inf") for k in P_CHANNELS_ALL}
            z = {k: 0.0 for k in P_CHANNELS_ALL}

            # Raw closure residual diagnostics (for semantic/scale audits)
            obs_res_pq = float("inf")
            obs_res_pqr = float("inf")
            obs_res_pow = float("inf")
            null_pq_median = float("nan")
            null_pq_mean = float("nan")
            null_pq_std = float("nan")
            null_pqr_median = float("nan")
            null_pqr_mean = float("nan")
            null_pqr_std = float("nan")
            null_pow_median = float("nan")
            null_pow_mean = float("nan")
            null_pow_std = float("nan")

            null_pq = []
            null_pqr = []
            null_pow = []

            if phase3_mode not in {"dnmap_only", "dnmap_fe", "dnmap_fe_both"}:
                use_pow = bool(cfg.get("use_pow", False)) and int(kmax) >= 2

                if closure_semantics in {"direct_composite", "direct"}:
                    # Build the needed direct composite labels once per run/window.
                    direct_labels = set()
                    if use_pq:
                        for (pp, qq) in pq_pairs:
                            direct_labels.add(int(pp) * int(qq))
                    if use_pqr:
                        for (pp, qq, rr) in pqr_triples:
                            direct_labels.add(int(pp) * int(qq) * int(rr))
                    if use_pow:
                        for pp in primes_used:
                            for kk in range(2, int(kmax) + 1):
                                direct_labels.add(int(pp) ** int(kk))

                    direct_Tn = _build_Tn_backend_dispatch(tp_backend, sorted(direct_labels), d=int(d), seed=int(seed), cfg=cfg)

                    # Observed residual summaries
                    obs_res = closure_residuals_from_Tp(
                        Tp=Tp_true,
                        direct_Tn=direct_Tn,
                        pq_pairs=pq_pairs,
                        pqr_triples=pqr_triples,
                        kmax=int(kmax),
                        two_path=bool(two_path),
                        cfg=cfg,
                    )
                    obs_pq = float(obs_res.get("res_pq", float("inf")))
                    obs_pqr = float(obs_res.get("res_pqr", float("inf")))
                    obs_pow = float(obs_res.get("res_pow", float("inf")))
                elif closure_semantics in {"against_true_v5", "against_true", "v5"}:
                    composites_true = build_composites_true_products(
                        Tp_true=Tp_true,
                        primes_used=primes_used,
                        kmax=int(kmax),
                        two_path=bool(two_path),
                    )
                    # Observed baseline = identity mapping (doc-aligned)
                    phi_id = {int(pp): int(pp) for pp in primes_used}
                    Tp_id = build_Tp_from_phi(Tp_true, primes_used, phi_id)
                    obs_res = closure_residuals_against_true_products(
                        Tp=Tp_id,
                        composites_true=composites_true,
                        pq_pairs=pq_pairs,
                        pqr_triples=pqr_triples,
                        kmax=int(kmax),
                        two_path=bool(two_path),
                        use_pq=bool(use_pq),
                        use_pqr=bool(use_pqr),
                        use_pow=bool(use_pow),
                        cfg=cfg,
                    )
                    obs_pq = float(obs_res.get("res_pq", float("inf")))
                    obs_pqr = float(obs_res.get("res_pqr", float("inf")))
                    obs_pow = float(obs_res.get("res_pow", float("inf")))
                else:
                    raise ValueError(
                        f"Unknown closure_semantics={closure_semantics!r}. Expected: direct_composite or against_true_v5."
                    )

                obs_res_pq = obs_pq
                obs_res_pqr = obs_pqr
                obs_res_pow = obs_pow

                sequential = bool(cfg.get("sequential", True))
                early_stop = bool(cfg.get("early_stop", True))

                for j in range(N_null):
                    phi = make_label_scramble_phi(primes_used, scramble_seed=anchor_seed + (100000 * j if sequential else j))
                    Tp_null = build_Tp_from_phi(Tp_true, primes_used, phi)

                    if closure_semantics in {"direct_composite", "direct"}:
                        null_res = closure_residuals_from_Tp(
                            Tp=Tp_null,
                            direct_Tn=direct_Tn,
                            pq_pairs=pq_pairs,
                            pqr_triples=pqr_triples,
                            kmax=int(kmax),
                            two_path=bool(two_path),
                            cfg=cfg,
                        )
                    else:
                        null_res = closure_residuals_against_true_products(
                            Tp=Tp_null,
                            composites_true=composites_true,
                            pq_pairs=pq_pairs,
                            pqr_triples=pqr_triples,
                            kmax=int(kmax),
                            two_path=bool(two_path),
                            use_pq=bool(use_pq),
                            use_pqr=bool(use_pqr),
                            use_pow=bool(use_pow),
                            cfg=cfg,
                        )

                    if use_pq:
                        null_pq.append(float(null_res.get("res_pq", float("inf"))))
                    if use_pqr:
                        null_pqr.append(float(null_res.get("res_pqr", float("inf"))))
                    if use_pow:
                        null_pow.append(float(null_res.get("res_pow", float("inf"))))

                    if early_stop and use_pq and (j >= 64):
                        rej, _ = can_stop_now_lower_tail(j + 1, int(np.sum(np.asarray(null_pq) <= obs_pq)), N_null, alpha)
                        if rej:
                            break

                # empirical p-values (lower tail: unusually SMALL residual => reject)
                if use_pq and len(null_pq):
                    arr = np.asarray(null_pq, dtype=np.float64)
                    null_pq_median = float(np.median(arr))
                    null_pq_mean = float(np.mean(arr))
                    null_pq_std = float(np.std(arr))
                    p["p_pq"] = empirical_p_lower(arr, obs_pq, tie_le=True)
                    z["p_pq"] = float(-z_effect(arr, obs_pq))

                if use_pqr and len(null_pqr):
                    arr2 = np.asarray(null_pqr, dtype=np.float64)
                    null_pqr_median = float(np.median(arr2))
                    null_pqr_mean = float(np.mean(arr2))
                    null_pqr_std = float(np.std(arr2))
                    p["p_pqr"] = empirical_p_lower(arr2, obs_pqr, tie_le=True)
                    z["p_pqr"] = float(-z_effect(arr2, obs_pqr))

                if bool(cfg.get("use_pow", False)) and int(kmax) >= 2 and len(null_pow):
                    arr3 = np.asarray(null_pow, dtype=np.float64)
                    null_pow_median = float(np.median(arr3))
                    null_pow_mean = float(np.mean(arr3))
                    null_pow_std = float(np.std(arr3))
                    p["p_pow"] = empirical_p_lower(arr3, obs_pow, tie_le=True)
                    z["p_pow"] = float(-z_effect(arr3, obs_pow))

            # ---- DN-map/logdet channel (frozen measurement spine) ----
            p_fe = float("nan")
            p_rigid = float("nan")
            fe_extras = {}
            dn_extras = {}
            if phase3_mode in {"dnmap_fe", "dnmap_fe_both"}:
                (s1_obs, s2_obs), (null1, null2), p_dn1, p_dn2, p_fe, fe_extras = fe_payload
                p_rigid = float(min(float(p_dn1), float(p_dn2)))
                if (float(wlo), float(whi)) == (float(windows[0][0]), float(windows[0][1])):
                    dn_obs, dn_null_scores, p_dnmap = float(s1_obs), np.asarray(null1, dtype=np.float64), float(p_dn1)
                elif (float(wlo), float(whi)) == (float(windows[1][0]), float(windows[1][1])):
                    dn_obs, dn_null_scores, p_dnmap = float(s2_obs), np.asarray(null2, dtype=np.float64), float(p_dn2)
                else:
                    # Fallback: if extra windows were provided, compute p_dnmap normally.
                    dn_obs, dn_null_scores, p_dnmap, dn_extras = dnmap_p_channel_label_scramble(
                        cfg=cfg,
                        seed=int(seed),
                        Tp_true=Tp_true,
                        primes_used=primes_used,
                        anchor_seed=int(anchor_seed),
                        wlo=float(wlo),
                        whi=float(whi),
                        dn_stride=int(dn_stride),
                        N_null=int(N_null),
                        amp=float(amp),
                    )
            else:
                dn_obs, dn_null_scores, p_dnmap, dn_extras = dnmap_p_channel_label_scramble(
                    cfg=cfg,
                    seed=int(seed),
                    Tp_true=Tp_true,
                    primes_used=primes_used,
                    anchor_seed=int(anchor_seed),
                    wlo=float(wlo),
                    whi=float(whi),
                    dn_stride=int(dn_stride),
                    N_null=int(N_null),
                    amp=float(amp),
                )

            # OFF means OFF: p_family = min over gate_keys only
            def _gate_value(pk: str) -> float:
                if pk == "p_fe":
                    return float(p_fe)
                if pk == "p_rigid":
                    return float(p_rigid)
                if pk == "p_dnmap":
                    return float(p_dnmap)
                return float(p.get(pk, float("inf")))

            if phase3_mode == "dnmap_fe_both":
                # AND-style: require both channels <= alpha by using max aggregator.
                p_family = float(max(_gate_value(k) for k in gate_keys))
            else:
                p_family = float(min(_gate_value(k) for k in gate_keys))

            geom_policy_summary = (
                f"backend={tp_backend}"
                f"|warp={str(cfg.get('warp_mode','plus'))}"
                f"|unwrap={bool(cfg.get('warp_unwrap', True))}"
                f"|y_scale={float(cfg.get('warp_y_scale', 1.0))}"
                f"|eps_auto={bool(cfg.get('eps_auto_tune', False))}"
                f"|eps_target={float(cfg.get('eps_target_delta', 0.02))}"
                f"|tp_delta_mean={float(tp_delta.get('tp_delta_mean', 0.0)):.6g}"
                f"|tp_delta_max={float(tp_delta.get('tp_delta_max', 0.0)):.6g}"
            )
            if str(tp_backend) == "geom_warp_dirac_v7_localdefect" and v7_meta:
                geom_policy_summary += (
                    f"|v7_rule={v7_meta.get('defect_support_rule','')}"
                    f"|v7_support={int(v7_meta.get('defect_support_size', 0))}"
                    f"|v7_digest={str(v7_meta.get('defect_primes_marked_digest',''))}"
                )
            if str(tp_backend) == "geom_warp_dirac_v8_boundarydefect" and v8_meta:
                geom_policy_summary += (
                    f"|v8_rule={v8_meta.get('v8_support_rule','')}"
                    f"|v8_support={int(v8_meta.get('v8_support_size', 0))}"
                    f"|v8_bsupport={int(v8_meta.get('v8_boundary_support_size', 0))}"
                    f"|v8_digest={str(v8_meta.get('v8_primes_marked_digest',''))}"
                )
            if str(tp_backend) == "geom_warp_dirac_v9_phasetransport" and v9_meta:
                geom_policy_summary += (
                    f"|v9_rule={str(v9_meta.get('v9_transport_rule',''))}"
                    f"|v9_lam={float(v9_meta.get('v9_transport_lambda', 0.0)):.6g}"
                    f"|v9_clip={float(v9_meta.get('v9_transport_clip', 0.0)):.6g}"
                    f"|v9_td={str(v9_meta.get('v9_theta_tr_digest',''))}"
                )
            if str(tp_backend) == "geom_warp_dirac_v10_phasefieldsolve" and v10_meta:
                geom_policy_summary += (
                    f"|v10_graph={str(v10_meta.get('v10_phasefield_graph',''))}"
                    f"|v10_w={float(v10_meta.get('v10_phasefield_w', 0.0)):.6g}"
                    f"|v10_eta={float(v10_meta.get('v10_phasefield_eta', 0.0)):.6g}"
                    f"|v10_td={str(v10_meta.get('v10_theta_sol_digest',''))}"
                )

            row = dict(
                backend=str(tp_backend),
                tp_backend=str(tp_backend),
                seed=int(seed),
                amp=float(amp),
                n_ops=int(n_ops),
                k_used=int(k_used),
                wlo=float(wlo),
                whi=float(whi),

                # explicit closure targets (stable grouping keys)
                pqM_target=int(pqM),
                pqrM_target=int(pqrM),
                kmax_target=int(kmax),

                phase3_mode=str(phase3_mode),

                # Phase-3C paired-window observable (present when phase3_mode='dnmap_fe')
                p_fe=float(p_fe) if np.isfinite(p_fe) else float("nan"),
                p_rigid=float(p_rigid) if np.isfinite(p_rigid) else float("nan"),
                fe_policy_digest=str(fe_extras.get("fe_policy_digest", "")) if fe_extras else "",
                fe_involution=str(fe_extras.get("fe_involution", "")) if fe_extras else "",
                fe_agg=str(fe_extras.get("fe_agg", "")) if fe_extras else "",
                fe_completion_mode=str(fe_extras.get("fe_completion_mode", "")) if fe_extras else "",
                fe_calibration_mode=str(fe_extras.get("fe_calibration_mode", "")) if fe_extras else "",
                fe_alpha_E=float(fe_extras.get("fe_alpha_E", float("nan"))) if fe_extras else float("nan"),
                fe_beta=float(fe_extras.get("fe_beta", float("nan"))) if fe_extras else float("nan"),
                fe_G_clip=(float(fe_extras.get("fe_G_clip")) if fe_extras and fe_extras.get("fe_G_clip") is not None else None),

                # Per-window FE diagnostics (compare E -> -E on the same grid)
                p_fe_window=float(
                    fe_extras.get("p_fe_w1", float("nan"))
                    if fe_extras and (float(wlo), float(whi)) == (float(fe_extras.get("fe_w1", (0.0, 0.0))[0]), float(fe_extras.get("fe_w1", (0.0, 0.0))[1]))
                    else (
                        fe_extras.get("p_fe_w2", float("nan"))
                        if fe_extras and (float(wlo), float(whi)) == (float(fe_extras.get("fe_w2", (0.0, 0.0))[0]), float(fe_extras.get("fe_w2", (0.0, 0.0))[1]))
                        else float("nan")
                    )
                ) if fe_extras else float("nan"),
                fe_obs=float(
                    fe_extras.get("fe_obs_w1", float("nan"))
                    if fe_extras and (float(wlo), float(whi)) == (float(fe_extras.get("fe_w1", (0.0, 0.0))[0]), float(fe_extras.get("fe_w1", (0.0, 0.0))[1]))
                    else (
                        fe_extras.get("fe_obs_w2", float("nan"))
                        if fe_extras and (float(wlo), float(whi)) == (float(fe_extras.get("fe_w2", (0.0, 0.0))[0]), float(fe_extras.get("fe_w2", (0.0, 0.0))[1]))
                        else float("nan")
                    )
                ) if fe_extras else float("nan"),
                fe_obs_raw=float(
                    fe_extras.get("fe_obs_raw_w1", float("nan"))
                    if fe_extras and (float(wlo), float(whi)) == (float(fe_extras.get("fe_w1", (0.0, 0.0))[0]), float(fe_extras.get("fe_w1", (0.0, 0.0))[1]))
                    else (
                        fe_extras.get("fe_obs_raw_w2", float("nan"))
                        if fe_extras and (float(wlo), float(whi)) == (float(fe_extras.get("fe_w2", (0.0, 0.0))[0]), float(fe_extras.get("fe_w2", (0.0, 0.0))[1]))
                        else float("nan")
                    )
                ) if fe_extras else float("nan"),
                fe_null_mean=float(
                    fe_extras.get("fe_null_mean_w1", float("nan"))
                    if fe_extras and (float(wlo), float(whi)) == (float(fe_extras.get("fe_w1", (0.0, 0.0))[0]), float(fe_extras.get("fe_w1", (0.0, 0.0))[1]))
                    else (
                        fe_extras.get("fe_null_mean_w2", float("nan"))
                        if fe_extras and (float(wlo), float(whi)) == (float(fe_extras.get("fe_w2", (0.0, 0.0))[0]), float(fe_extras.get("fe_w2", (0.0, 0.0))[1]))
                        else float("nan")
                    )
                ) if fe_extras else float("nan"),
                fe_null_std=float(
                    fe_extras.get("fe_null_std_w1", float("nan"))
                    if fe_extras and (float(wlo), float(whi)) == (float(fe_extras.get("fe_w1", (0.0, 0.0))[0]), float(fe_extras.get("fe_w1", (0.0, 0.0))[1]))
                    else (
                        fe_extras.get("fe_null_std_w2", float("nan"))
                        if fe_extras and (float(wlo), float(whi)) == (float(fe_extras.get("fe_w2", (0.0, 0.0))[0]), float(fe_extras.get("fe_w2", (0.0, 0.0))[1]))
                        else float("nan")
                    )
                ) if fe_extras else float("nan"),
                fe_null_min=float(
                    fe_extras.get("fe_null_min_w1", float("nan"))
                    if fe_extras and (float(wlo), float(whi)) == (float(fe_extras.get("fe_w1", (0.0, 0.0))[0]), float(fe_extras.get("fe_w1", (0.0, 0.0))[1]))
                    else (
                        fe_extras.get("fe_null_min_w2", float("nan"))
                        if fe_extras and (float(wlo), float(whi)) == (float(fe_extras.get("fe_w2", (0.0, 0.0))[0]), float(fe_extras.get("fe_w2", (0.0, 0.0))[1]))
                        else float("nan")
                    )
                ) if fe_extras else float("nan"),
                fe_identity_rank=int(
                    fe_extras.get("fe_identity_rank_w1", -1)
                    if fe_extras and (float(wlo), float(whi)) == (float(fe_extras.get("fe_w1", (0.0, 0.0))[0]), float(fe_extras.get("fe_w1", (0.0, 0.0))[1]))
                    else (
                        fe_extras.get("fe_identity_rank_w2", -1)
                        if fe_extras and (float(wlo), float(whi)) == (float(fe_extras.get("fe_w2", (0.0, 0.0))[0]), float(fe_extras.get("fe_w2", (0.0, 0.0))[1]))
                        else -1
                    )
                ) if fe_extras else -1,
                fe_mirror_E_digest=str(
                    fe_extras.get("fe_mirror_E_digest_w1", "")
                    if fe_extras and (float(wlo), float(whi)) == (float(fe_extras.get("fe_w1", (0.0, 0.0))[0]), float(fe_extras.get("fe_w1", (0.0, 0.0))[1]))
                    else (
                        fe_extras.get("fe_mirror_E_digest_w2", "")
                        if fe_extras and (float(wlo), float(whi)) == (float(fe_extras.get("fe_w2", (0.0, 0.0))[0]), float(fe_extras.get("fe_w2", (0.0, 0.0))[1]))
                        else ""
                    )
                ) if fe_extras else "",

                gauge_policy=str(cfg.get("gauge_policy", "none")),
                geom_policy_digest=str(geom_policy_digest),
                geom_policy_summary=str(geom_policy_summary),

                # closure semantics provenance
                closure_semantics=str(closure_semantics),
                closure_semantics_policy=str(closure_semantics_policy),
                closure_residual_mode=str(cfg.get("closure_residual_mode", "relative_ref")),

                # geometry knobs (Tp-only provenance; helps Guidance 28 audits)
                warp_mode=str(cfg.get("warp_mode", "plus")),
                warp_unwrap=bool(cfg.get("warp_unwrap", True)),
                warp_y_scale=float(cfg.get("warp_y_scale", 1.0)),
                warp_rs_re=float(cfg.get("warp_rs_re", 1.0)),
                warp_rs_im=float(cfg.get("warp_rs_im", 1.0)),
                geom_v2_depth_scale=float(cfg.get("geom_v2_depth_scale", 1.0)),
                geom_v2_rot_scale=float(cfg.get("geom_v2_rot_scale", 0.25)),
                geom_v3_eps=float(cfg.get("geom_v3_eps", 0.01)),
                geom_v3_rot_scale=float(cfg.get("geom_v3_rot_scale", 0.25)),
                geom_v3_rank=int(cfg.get("geom_v3_rank", 1)),
                geom_v3_theta_scale=float(cfg.get("geom_v3_theta_scale")) if cfg.get("geom_v3_theta_scale") is not None else float(cfg.get("geom_v3_rot_scale", 0.25)),
                geom_v3_theta_scale_is_none=bool(cfg.get("geom_v3_theta_scale") is None),

                geom_v4_u_scale=float(cfg.get("geom_v4_u_scale", 1.0)),
                geom_v4_theta_scale=float(cfg.get("geom_v4_theta_scale", 0.125)),
                geom_v4_rank=int(cfg.get("geom_v4_rank", 1)),
                geom_v4_eps=float(cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01))),
                geom_v4_eps_global=float(cfg.get("geom_v4_eps_global", 0.02)),
                geom_v4_u_clip=(float(cfg.get("geom_v4_u_clip")) if cfg.get("geom_v4_u_clip") is not None else None),
                geom_v4_theta_clip=(float(cfg.get("geom_v4_theta_clip")) if cfg.get("geom_v4_theta_clip") is not None else None),
                geom_v4_eps_global_auto=bool(eps_global_auto if eps_global_key == "geom_v4_eps_global" else False),

                # v5 knobs (filled for audit even if backend != v5)
                geom_v5_u_scale=float(cfg.get("geom_v5_u_scale", 1.0)),
                geom_v5_theta_scale=float(cfg.get("geom_v5_theta_scale", 0.125)),
                geom_v5_rank=int(cfg.get("geom_v5_rank", 1)),
                geom_v5_eps=float(cfg.get("geom_v5_eps", cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01)))),
                geom_v5_eps_global=float(cfg.get("geom_v5_eps_global", 0.02)),
                geom_v5_u_clip=(float(cfg.get("geom_v5_u_clip")) if cfg.get("geom_v5_u_clip") is not None else None),
                geom_v5_theta_clip=(float(cfg.get("geom_v5_theta_clip")) if cfg.get("geom_v5_theta_clip") is not None else None),
                geom_v5_harmonic_gamma=float(cfg.get("geom_v5_harmonic_gamma", 1.0)),
                geom_v5_harmonic_strength=float(cfg.get("geom_v5_harmonic_strength", 1.0)),
                geom_v5_eps_global_auto=bool(eps_global_auto if eps_global_key == "geom_v5_eps_global" else False),

                # v6 knobs (filled for audit even if backend != v6)
                geom_v6_u_scale=float(cfg.get("geom_v6_u_scale", 1.0)),
                geom_v6_theta_scale=float(cfg.get("geom_v6_theta_scale", 0.125)),
                geom_v6_rank=int(cfg.get("geom_v6_rank", 1)),
                geom_v6_eps=float(cfg.get("geom_v6_eps", cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01)))),
                geom_v6_eps_global=float(cfg.get("geom_v6_eps_global", 0.02)),
                geom_v6_u_clip=(float(cfg.get("geom_v6_u_clip")) if cfg.get("geom_v6_u_clip") is not None else None),
                geom_v6_theta_clip=(float(cfg.get("geom_v6_theta_clip")) if cfg.get("geom_v6_theta_clip") is not None else None),
                geom_v6_phase_lambda=float(cfg.get("geom_v6_phase_lambda", 0.05)),
                geom_v6_phase_clip=float(cfg.get("geom_v6_phase_clip", 2.0)),
                geom_v6_eps_global_auto=bool(eps_global_auto if eps_global_key == "geom_v6_eps_global" else False),

                # v7 knobs (filled for audit even if backend != v7)
                geom_v7_u_scale=float(cfg.get("geom_v7_u_scale", 1.0)),
                geom_v7_theta_scale=float(cfg.get("geom_v7_theta_scale", 0.125)),
                geom_v7_rank=int(cfg.get("geom_v7_rank", 1)),
                geom_v7_eps=float(cfg.get("geom_v7_eps", cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01)))),
                geom_v7_eps_global=float(cfg.get("geom_v7_eps_global", 0.02)),
                geom_v7_u_clip=(float(cfg.get("geom_v7_u_clip")) if cfg.get("geom_v7_u_clip") is not None else None),
                geom_v7_theta_clip=(float(cfg.get("geom_v7_theta_clip")) if cfg.get("geom_v7_theta_clip") is not None else None),
                geom_v7_defect_mode=str(cfg.get("geom_v7_defect_mode", "block")),
                geom_v7_defect_rho=float(cfg.get("geom_v7_defect_rho", 0.25)),
                geom_v7_defect_strength=float(cfg.get("geom_v7_defect_strength", 1.0)),
                geom_v7_defect_clip=float(cfg.get("geom_v7_defect_clip", 0.75)),
                geom_v7_defect_salt=int(cfg.get("geom_v7_defect_salt", 1)),
                geom_v7_eps_global_auto=bool(eps_global_auto if eps_global_key == "geom_v7_eps_global" else False),

                # v7 defect diagnostics (derived-only)
                defect_support_size=int(v7_meta.get("defect_support_size", 0)) if v7_meta else 0,
                defect_support_rule=str(v7_meta.get("defect_support_rule", "")) if v7_meta else "",
                defect_primes_marked_digest=str(v7_meta.get("defect_primes_marked_digest", "")) if v7_meta else "",
                # Simple audit scalar: eps_global * defect_strength before/after eps autotune.
                defect_amp_before_autotune=(
                    float(eps_global_before_autotune) * float(v7_meta.get("defect_strength", 0.0))
                    if v7_meta and np.isfinite(eps_global_before_autotune)
                    else 0.0
                ),
                defect_amp_after_autotune=(
                    float(eps_global_val) * float(v7_meta.get("defect_strength", 0.0))
                    if v7_meta and np.isfinite(eps_global_val)
                    else 0.0
                ),

                # v8 knobs (filled for audit even if backend != v8)
                geom_v8_u_scale=float(cfg.get("geom_v8_u_scale", 1.0)),
                geom_v8_theta_scale=float(cfg.get("geom_v8_theta_scale", 0.125)),
                geom_v8_rank=int(cfg.get("geom_v8_rank", 1)),
                geom_v8_eps=float(cfg.get("geom_v8_eps", cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01)))),
                geom_v8_eps_global=float(cfg.get("geom_v8_eps_global", 0.02)),
                geom_v8_u_clip=(float(cfg.get("geom_v8_u_clip")) if cfg.get("geom_v8_u_clip") is not None else None),
                geom_v8_theta_clip=(float(cfg.get("geom_v8_theta_clip")) if cfg.get("geom_v8_theta_clip") is not None else None),
                geom_v8_prime_defect_mode=str(cfg.get("geom_v8_prime_defect_mode", "block")),
                geom_v8_prime_defect_rho=float(cfg.get("geom_v8_prime_defect_rho", 0.25)),
                geom_v8_defect_strength=float(cfg.get("geom_v8_defect_strength", 1.0)),
                geom_v8_defect_clip=float(cfg.get("geom_v8_defect_clip", 0.75)),
                geom_v8_defect_salt=int(cfg.get("geom_v8_defect_salt", 1)),
                geom_v8_defect_seed=int(cfg.get("geom_v8_defect_seed", 0)),
                geom_v8_pair_chiral=bool(cfg.get("geom_v8_pair_chiral", True)),
                geom_v8_defect_norm_mode=str(cfg.get("geom_v8_defect_norm_mode", "fro_match")),
                geom_v8_eps_global_auto=bool(eps_global_auto if eps_global_key == "geom_v8_eps_global" else False),

                # v8 defect diagnostics (derived-only)
                v8_support_size=int(v8_meta.get("v8_support_size", 0)) if v8_meta else 0,
                v8_support_rule=str(v8_meta.get("v8_support_rule", "")) if v8_meta else "",
                v8_primes_marked_digest=str(v8_meta.get("v8_primes_marked_digest", "")) if v8_meta else "",
                v8_boundary_support_size=int(v8_meta.get("v8_boundary_support_size", 0)) if v8_meta else 0,
                v8_boundary_digest=str(v8_meta.get("v8_boundary_digest", "")) if v8_meta else "",
                v8_defect_amp_before_autotune=(
                    float(eps_global_before_autotune) * float(v8_meta.get("v8_strength", 0.0))
                    if v8_meta and np.isfinite(eps_global_before_autotune)
                    else 0.0
                ),
                v8_defect_amp_after_autotune=(
                    float(eps_global_val) * float(v8_meta.get("v8_strength", 0.0))
                    if v8_meta and np.isfinite(eps_global_val)
                    else 0.0
                ),

                # v9 knobs (filled for audit even if backend != v9)
                geom_v9_u_scale=float(cfg.get("geom_v9_u_scale", 1.0)),
                geom_v9_theta_scale=float(cfg.get("geom_v9_theta_scale", 0.125)),
                geom_v9_rank=int(cfg.get("geom_v9_rank", 1)),
                geom_v9_eps=float(cfg.get("geom_v9_eps", cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01)))),
                geom_v9_eps_global=float(cfg.get("geom_v9_eps_global", 0.02)),
                geom_v9_u_clip=(float(cfg.get("geom_v9_u_clip")) if cfg.get("geom_v9_u_clip") is not None else None),
                geom_v9_theta_clip=(float(cfg.get("geom_v9_theta_clip")) if cfg.get("geom_v9_theta_clip") is not None else None),
                geom_v9_transport_rule=str(cfg.get("geom_v9_transport_rule", "delta_raw_wrapped")),
                geom_v9_transport_lambda=float(cfg.get("geom_v9_transport_lambda", 1.0)),
                geom_v9_transport_clip=float(cfg.get("geom_v9_transport_clip", 0.25)),
                geom_v9_transport_seed=int(cfg.get("geom_v9_transport_seed", 0)),
                geom_v9_eps_global_auto=bool(eps_global_auto if eps_global_key == "geom_v9_eps_global" else False),

                # v9 transport diagnostics (derived-only)
                v9_theta_tr_digest=str(v9_meta.get("v9_theta_tr_digest", "")) if v9_meta else "",

                # v10 knobs (filled for audit even if backend != v10)
                geom_v10_u_scale=float(cfg.get("geom_v10_u_scale", 1.0)),
                geom_v10_theta_scale=float(cfg.get("geom_v10_theta_scale", 0.125)),
                geom_v10_rank=int(cfg.get("geom_v10_rank", 1)),
                geom_v10_eps=float(cfg.get("geom_v10_eps", cfg.get("geom_v4_eps", cfg.get("geom_v3_eps", 0.01)))),
                geom_v10_eps_global=float(cfg.get("geom_v10_eps_global", 0.02)),
                geom_v10_u_clip=(float(cfg.get("geom_v10_u_clip")) if cfg.get("geom_v10_u_clip") is not None else None),
                geom_v10_theta_clip=(float(cfg.get("geom_v10_theta_clip")) if cfg.get("geom_v10_theta_clip") is not None else None),
                geom_v10_phasefield_graph=str(cfg.get("geom_v10_phasefield_graph", "path")),
                geom_v10_phasefield_w=float(cfg.get("geom_v10_phasefield_w", 1.0)),
                geom_v10_phasefield_eta=float(cfg.get("geom_v10_phasefield_eta", 0.25)),
                geom_v10_phasefield_seed=int(cfg.get("geom_v10_phasefield_seed", 0)),
                geom_v10_eps_global_auto=bool(eps_global_auto if eps_global_key == "geom_v10_eps_global" else False),

                # v10 phase-field diagnostics (derived-only)
                v10_theta_sol_digest=str(v10_meta.get("v10_theta_sol_digest", "")) if v10_meta else "",

                eps_auto_tune=bool(cfg.get("eps_auto_tune", False)),
                eps_target_delta=float(cfg.get("eps_target_delta", 0.02)),

                null_family="label_scramble",
                # NOTE: for full mode, closure null uses <= N_null iterations; for dnmap_only it's dnmap's count.
                N_null_used=int(len(null_pq)) if phase3_mode not in {"dnmap_only", "dnmap_fe", "dnmap_fe_both"} and use_pq else int(len(dn_null_scores)),
                N_null_target=int(N_null),
                dnmap_stride=int(dn_stride),

                # measurement spine fingerprints (seed-audit)
                meas_params_digest=str(meas_params_digest),
                meas_boundary_digest=str(meas_boundary_digest),
                meas_E_digest=str(meas_E_digest),
                meas_boundary_seed=(int(cfg.get("meas_boundary_seed")) if cfg.get("meas_boundary_seed") is not None else None),

                pq_anchor_M=int(len(pq_pairs)),
                pqr_anchor_M=int(len(pqr_triples)),
                p_pow_kmax=int(kmax),
                pqr_two_path=bool(two_path),

                use_pq=bool(use_pq),
                use_pqr=bool(use_pqr),
                use_dnmap_gate=bool(use_dnmap_gate),
                gate_channels="|".join(gate_keys),

                anchor_seed=int(anchor_seed),
                anchor_hash=str(anchor_h),
                pq_digest=str(pq_d),
                pqr_digest=str(pqr_d),

                p_family=float(p_family),
                p_pq=float(p.get("p_pq", float("inf"))),
                p_pqr=float(p.get("p_pqr", float("inf"))),
                p_pow=float(p.get("p_pow", float("inf"))),

                z_pq=float(z.get("p_pq", 0.0)),
                z_pqr=float(z.get("p_pqr", 0.0)),
                z_pow=float(z.get("p_pow", 0.0)),

                # Closure residual raw + null summaries (only meaningful in phase3_mode='full')
                obs_res_pq=float(obs_res_pq),
                obs_res_pqr=float(obs_res_pqr),
                obs_res_pow=float(obs_res_pow),
                null_pq_median=float(null_pq_median),
                null_pq_mean=float(null_pq_mean),
                null_pq_std=float(null_pq_std),
                null_pq_cv=float(null_pq_std) / (float(null_pq_median) + float(cfg.get("closure_eps", 1e-12))) if np.isfinite(null_pq_std) and np.isfinite(null_pq_median) else float("nan"),
                delta_pq=(float(null_pq_median) - float(obs_res_pq)) if np.isfinite(null_pq_median) and np.isfinite(obs_res_pq) else float("nan"),
                snr_pq=((float(null_pq_median) - float(obs_res_pq)) / (float(null_pq_std) + float(cfg.get("closure_eps", 1e-12)))) if np.isfinite(null_pq_std) and np.isfinite(null_pq_median) and np.isfinite(obs_res_pq) else float("nan"),
                null_pqr_median=float(null_pqr_median),
                null_pqr_mean=float(null_pqr_mean),
                null_pqr_std=float(null_pqr_std),
                null_pqr_cv=float(null_pqr_std) / (float(null_pqr_median) + float(cfg.get("closure_eps", 1e-12))) if np.isfinite(null_pqr_std) and np.isfinite(null_pqr_median) else float("nan"),
                delta_pqr=(float(null_pqr_median) - float(obs_res_pqr)) if np.isfinite(null_pqr_median) and np.isfinite(obs_res_pqr) else float("nan"),
                snr_pqr=((float(null_pqr_median) - float(obs_res_pqr)) / (float(null_pqr_std) + float(cfg.get("closure_eps", 1e-12)))) if np.isfinite(null_pqr_std) and np.isfinite(null_pqr_median) and np.isfinite(obs_res_pqr) else float("nan"),
                null_pow_median=float(null_pow_median),
                null_pow_mean=float(null_pow_mean),
                null_pow_std=float(null_pow_std),
                null_pow_cv=float(null_pow_std) / (float(null_pow_median) + float(cfg.get("closure_eps", 1e-12))) if np.isfinite(null_pow_std) and np.isfinite(null_pow_median) else float("nan"),
                delta_pow=(float(null_pow_median) - float(obs_res_pow)) if np.isfinite(null_pow_median) and np.isfinite(obs_res_pow) else float("nan"),
                snr_pow=((float(null_pow_median) - float(obs_res_pow)) / (float(null_pow_std) + float(cfg.get("closure_eps", 1e-12)))) if np.isfinite(null_pow_std) and np.isfinite(null_pow_median) and np.isfinite(obs_res_pow) else float("nan"),

                # DN-map channel logging (always computed; optionally gated)
                p_dnmap=float(p_dnmap),
                dn_obs=float(dn_obs),
                dn_null_median=float(np.median(dn_null_scores)) if len(dn_null_scores) else 0.0,
                dn_null_mean=float(np.mean(dn_null_scores)) if len(dn_null_scores) else 0.0,
                dn_null_std=float(np.std(dn_null_scores)) if len(dn_null_scores) else 0.0,
                dn_null_cv=(float(np.std(dn_null_scores)) / (abs(float(np.median(dn_null_scores))) + 1e-12)) if len(dn_null_scores) else 0.0,
                dn_delta=(float(dn_obs) - float(np.mean(dn_null_scores))) if len(dn_null_scores) else 0.0,
                dn_snr=((float(dn_obs) - float(np.median(dn_null_scores))) / (float(np.std(dn_null_scores)) + 1e-12)) if len(dn_null_scores) else 0.0,

                # DN-map uniqueness KPIs (if enabled)
                dnmap_uniqueness=bool(cfg.get("dnmap_uniqueness", False)),
                dn_null_min=float(dn_extras.get("dn_null_min", float("nan"))) if isinstance(dn_extras, dict) else float("nan"),
                dn_null_argmin=int(dn_extras.get("dn_null_argmin", -1)) if isinstance(dn_extras, dict) else -1,
                dn_identity_rank=int(dn_extras.get("dn_identity_rank", -1)) if isinstance(dn_extras, dict) else -1,
                dn_collision_tol=float(dn_extras.get("dn_collision_tol", float(cfg.get("dn_collision_tol", 0.0)))) if isinstance(dn_extras, dict) else float(cfg.get("dn_collision_tol", 0.0)),
                dn_collision_count=int(dn_extras.get("dn_collision_count", -1)) if isinstance(dn_extras, dict) else -1,
                dn_collision_rate=float(dn_extras.get("dn_collision_rate", float("nan"))) if isinstance(dn_extras, dict) else float("nan"),
                dn_gap_min=float(dn_extras.get("dn_gap_min", float("nan"))) if isinstance(dn_extras, dict) else float("nan"),

                **fp,
                **comm,
                **tp_delta,
            )
            row["reject"] = bool(row["p_family"] <= alpha)
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "rows.csv"), index=False)
    print(f"[run_one] DONE backend={tp_backend} rows={len(df)} elapsed={time.time()-t0:.1f}s out_dir={out_dir}")
    return df

# ============================================
# CELL 4 (REPLACE): Phase 3B v3 boundary probe runner (backend-keyed)
# - Guarantees pqM_target/pqrM_target/kmax_target exist (fixes KeyError)
# - Aggregates by backend
# - Uses your existing run_one/pack_run/decide_table/add_ptarget_fields if present
#   (otherwise, it will still run, but skip the missing post-processing)
# ============================================

import os, time
import pandas as pd
import numpy as np

def _maybe(fn_name: str):
    return globals().get(fn_name, None)

def phase3B_v3_boundary_probe(
    backends=("legacy", "geom_v3_shared_axis"),
    seeds=(0, 1),
    anchors=(2, 9, 14, 44, 46, 51, 60),
    k=13,
    pqM=1,
    pqrM=0,
    kmax=0,
    N_target=512,
    windows=((0.60, 7.50), (2.0, 5.0)),
    dnmap_stride=2,
    amp=0.03,
    phase3_mode=None,
    closure_semantics=None,
    closure_semantics_policy=None,
    out_root="out_phase3B_v3_k13_boundary",
):
    t0 = time.time()
    rows = []
    windows = [tuple(w) for w in windows]

    for backend in list(backends):
        for seed in list(seeds):
            for aseed in list(anchors):
                cfg = dict(CFG_BASE) if "CFG_BASE" in globals() else {}
                cfg.update({
                    "tp_backend": str(backend),
                    "n_ops_list": [int(k)],
                    "windows": windows,
                    "N_null": int(N_target),
                    "dnmap_stride": int(dnmap_stride),
                    "anchor_seed": int(aseed),

                    # closure knobs (OFF means OFF)
                    "pq_anchor_M": int(pqM),
                    "pqr_anchor_M": int(pqrM),
                    "p_pow_kmax": int(kmax),

                    # enforce pq-only boundary probe behavior
                    "use_pq": True,
                    "use_pqr": False,
                    "use_pow": False,
                    "use_tower": False,
                })

                # Allow caller to force Phase-3B gating mode.
                # - None: inherit from CFG_BASE (default behavior)
                # - 'dnmap_only' or 'full': overrides per-run.
                if phase3_mode is not None:
                    cfg["phase3_mode"] = str(phase3_mode)

                # Allow caller to force closure semantics + policy (only matters for full mode).
                if closure_semantics is not None:
                    cfg["closure_semantics"] = str(closure_semantics)
                if closure_semantics_policy is not None:
                    cfg["closure_semantics_policy"] = str(closure_semantics_policy)

                # optional: if you have apply_closure_knobs, let it enforce internal flags
                if _maybe("apply_closure_knobs"):
                    cfg = apply_closure_knobs(cfg, pqM=int(pqM), pqrM=int(pqrM), kmax=int(kmax),
                                              use_pow=False, use_tower=False)

                out_dir = os.path.join(
                    out_root,
                    f"backend_{backend}",
                    f"N{int(N_target)}",
                    f"seed{int(seed)}_anchor{int(aseed)}"
                )
                print(f"\n=== 3B v3 boundary backend={backend} seed={seed} anchor={aseed} ===")
                df = run_one(cfg, seed=int(seed), amp=float(amp), out_dir=out_dir)

                # --- required meta cols for aggregation ---
                df["backend"] = str(backend)
                df["pqM_target"] = int(pqM)
                df["pqrM_target"] = int(pqrM)
                df["kmax_target"] = int(kmax)
                df["k_used"] = int(k)
                df["N_target"] = int(N_target)
                df["anchor_seed"] = int(aseed)
                df["seed"] = int(seed)
                df["out_dir"] = out_dir

                # optional per-run packaging if present
                if _maybe("pack_run"):
                    pack_run(out_dir, df, cfg, alpha=float(cfg.get("alpha", 0.01)), audit=True)

                rows.append(df)

    big = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    alpha_stats = float(big.get("alpha", pd.Series([0.01])).iloc[0]) if len(big) else 0.01

    dec = _maybe("decide_table")(big, alpha=float(alpha_stats), audit=True) if (_maybe("decide_table") and len(big)) else big.copy()
    dec2 = _maybe("add_ptarget_fields")(dec) if _maybe("add_ptarget_fields") else dec

    # backend-keyed aggregate (this is what you wanted in 3B)
    group_cols = ["backend", "k_used", "pqM_target", "pqrM_target", "kmax_target", "wlo", "whi"]
    agg = (dec2.groupby(group_cols, dropna=False)
           .agg(
               rows=("reject", "size") if "reject" in dec2.columns else ("backend", "size"),
               reject_rate=("reject", "mean") if "reject" in dec2.columns else ("backend", lambda x: np.nan),
               worst_p=("p_family_target", "max") if "p_family_target" in dec2.columns else ("p_family", "max"),
               best_p=("p_family_target", "min") if "p_family_target" in dec2.columns else ("p_family", "min"),
           )
           .reset_index()
           .sort_values(group_cols))

    os.makedirs(out_root, exist_ok=True)
    dec2.to_csv(os.path.join(out_root, "phase3B_v3_boundary_k13.csv"), index=False)
    agg.to_csv(os.path.join(out_root, "phase3B_v3_boundary_k13__aggregate.csv"), index=False)

    print("\n[Phase3B v3] aggregate:")
    print(agg)
    print(f"[Phase3B v3] wrote: {out_root}/phase3B_v3_boundary_k13.csv")
    print(f"[Phase3B v3] runtime: {time.time()-t0:.1f}s")

    return dec2, agg

if __name__ == "__main__":
    # --- Recommended call (exactly your minimal protocol) ---
    dec3B_v3, agg3B_v3 = phase3B_v3_boundary_probe(
        backends=("legacy", "geom_v3_shared_axis"),
        seeds=(0, 1),
        anchors=(2, 9, 14, 44, 46, 51, 60),
        k=13,
        pqM=1, pqrM=0, kmax=0,
        N_target=512,
        windows=((0.60, 7.50), (2.0, 5.0)),
        dnmap_stride=2,
        amp=0.03,
        out_root="out_phase3B_v3_k13_boundary"
    )


