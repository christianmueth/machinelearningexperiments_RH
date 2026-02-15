from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hecke import HeckeParams, hecke_Tn
from .metrics import fro_norm


@dataclass(frozen=True)
class BulkParams:
    N: int
    weight_k: int = 0


def eps_prime(p: int, s: complex) -> complex:
    """Default epsilon(p; s) used in the doc’s ladder signatures.

    Uses p^{-(2s-1)} so that powers generate p^{-m(2s-1)}.
    """

    p = int(p)
    return p ** (-(2.0 * s - 1.0))


def eps_n(n: int, s: complex) -> complex:
    n = int(n)
    return n ** (-(2.0 * s - 1.0))


def _is_prime_power(n: int) -> bool:
    n = int(n)
    if n <= 1:
        return False
    # trial division is fine for our tiny composite lists
    for p in range(2, int(np.sqrt(n)) + 1):
        if n % p != 0:
            continue
        m = n
        while m % p == 0:
            m //= p
        return m == 1
    return True  # n is prime


def _normalize_operator(T: np.ndarray, *, method: str | None, target: float) -> np.ndarray:
    if method is None:
        return T
    method = str(method).lower().strip()
    target = float(target)
    if target <= 0:
        raise ValueError("target must be positive")

    T = np.asarray(T, dtype=np.complex128)

    if method in {"fro", "frob", "frobenius"}:
        nrm = fro_norm(T)
    elif method in {"op", "spectral"}:
        # largest singular value
        s = np.linalg.svd(T, compute_uv=False)
        nrm = float(s[0]) if s.size else 0.0
    else:
        raise ValueError(f"unknown normalization method: {method}")

    if nrm == 0.0:
        return T
    return (target / nrm) * T


def _dual_involution_J(N: int) -> np.ndarray:
    """Default involution used for 'dual_*_J' completion modes.

    We use the anti-diagonal identity (basis reversal), which is unitary and
    satisfies J^{-1}=J. This is a cheap way to ensure the dual/adjoint term is
    not trivially identical to the primal term when generators are symmetric.
    """

    N = int(N)
    J = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        J[i, N - 1 - i] = 1.0
    return J


def build_A(
    *,
    s: complex,
    primes: list[int],
    comps: list[int] | None,
    params: BulkParams,
    prime_scale: float = 1.0,
    comp_scale: float = 1.0,
    comps_mode: str = "all",
    generator_norm: str | None = None,
    generator_norm_target: float = 1.0,
    match_comp_to_prime: bool = False,
    bulk_mode: str = "one_channel",
    completion_mode: str = "none",
    dual_scale: float = 1.0,
    return_components: bool = False,
) -> np.ndarray:
    N = int(params.N)
    hecke_params = HeckeParams(N=N, weight_k=int(params.weight_k))

    bulk_mode = str(bulk_mode).lower().strip()
    if bulk_mode not in {"one_channel", "two_channel_symmetric"}:
        raise ValueError("bulk_mode must be 'one_channel' or 'two_channel_symmetric'")

    completion_mode = str(completion_mode).lower().strip()
    dual_scale_c = complex(float(dual_scale))
    if completion_mode not in {"none", "dual_1_minus_s", "dual_1_minus_s_j"}:
        raise ValueError("completion_mode must be 'none', 'dual_1_minus_s', or 'dual_1_minus_s_J'")

    J = None
    if completion_mode == "dual_1_minus_s_j":
        J = _dual_involution_J(N)

    def _sym(T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=np.complex128)
        return 0.5 * (T + T.T)

    if bulk_mode == "two_channel_symmetric":
        # Two-channel assembly: A is 2N x 2N with off-diagonal s / (1-s) blocks.
        # This is a cheap, explicit way to realize the doc's bulk duality under channel-swap.
        K_s = np.zeros((N, N), dtype=np.complex128)
        K_1ms = np.zeros((N, N), dtype=np.complex128)
        for p in primes:
            Tp = _sym(hecke_Tn(int(p), hecke_params))
            Tp = _normalize_operator(Tp, method=generator_norm, target=generator_norm_target)
            K_s = K_s + eps_prime(int(p), s) * Tp
            K_1ms = K_1ms + eps_prime(int(p), (1.0 - s)) * Tp

        comps_list: list[int] = []
        if comps:
            comps_list = [int(x) for x in comps]

        comps_mode = str(comps_mode).lower().strip()
        if comps_list and comps_mode != "all":
            if comps_mode in {"prime_powers", "primepower", "pp"}:
                comps_list = [n for n in comps_list if _is_prime_power(n)]
            elif comps_mode in {"non_prime_powers", "non_primepower", "non_pp"}:
                comps_list = [n for n in comps_list if (n > 1 and (not _is_prime_power(n)))]
            else:
                raise ValueError(f"unknown comps_mode: {comps_mode}")

        C_s = np.zeros((N, N), dtype=np.complex128)
        C_1ms = np.zeros((N, N), dtype=np.complex128)
        if comps_list:
            for n in comps_list:
                Tn = _sym(hecke_Tn(int(n), hecke_params))
                Tn = _normalize_operator(Tn, method=generator_norm, target=generator_norm_target)
                C_s = C_s + eps_n(int(n), s) * Tn
                C_1ms = C_1ms + eps_n(int(n), (1.0 - s)) * Tn

        prime_scale_c = complex(prime_scale)
        comp_scale_c = complex(comp_scale)
        if match_comp_to_prime and comps_list:
            target = abs(prime_scale_c) * fro_norm(K_s)
            current = abs(comp_scale_c) * fro_norm(C_s)
            if current > 0.0:
                comp_scale_c = comp_scale_c * (target / current)

        U = prime_scale_c * K_s + comp_scale_c * C_s
        V = prime_scale_c * K_1ms + comp_scale_c * C_1ms
        I = np.eye(N, dtype=np.complex128)
        Z = np.zeros((N, N), dtype=np.complex128)
        top = np.concatenate([I, U], axis=1)
        bot = np.concatenate([dual_scale_c * V, I], axis=1)
        A2 = np.concatenate([top, bot], axis=0)
        if return_components:
            return (np.asarray(A2, dtype=np.complex128), U, dual_scale_c * V)  # type: ignore[return-value]
        return np.asarray(A2, dtype=np.complex128)

    # Default: existing one-channel assembly
    M_prime = np.zeros((N, N), dtype=np.complex128)
    for p in primes:
        Tp = hecke_Tn(int(p), hecke_params)
        Tp = _normalize_operator(Tp, method=generator_norm, target=generator_norm_target)
        M_prime = M_prime + eps_prime(int(p), s) * Tp
        if completion_mode in {"dual_1_minus_s", "dual_1_minus_s_j"}:
            # On the critical line, 1-s = conj(s), so this supplies the conjugate weight
            # without introducing an explicit conjugation, keeping holomorphic dependence.
            dual_op = Tp.conj().T
            if J is not None:
                dual_op = J @ dual_op @ J
            M_prime = M_prime + dual_scale_c * eps_prime(int(p), (1.0 - s)) * dual_op

    comps_list: list[int] = []
    if comps:
        comps_list = [int(x) for x in comps]

    comps_mode = str(comps_mode).lower().strip()
    if comps_list and comps_mode != "all":
        if comps_mode in {"prime_powers", "primepower", "pp"}:
            comps_list = [n for n in comps_list if _is_prime_power(n)]
        elif comps_mode in {"non_prime_powers", "non_primepower", "non_pp"}:
            comps_list = [n for n in comps_list if (n > 1 and (not _is_prime_power(n)))]
        else:
            raise ValueError(f"unknown comps_mode: {comps_mode}")

    M_comp = np.zeros((N, N), dtype=np.complex128)
    if comps_list:
        for n in comps_list:
            Tn = hecke_Tn(int(n), hecke_params)
            Tn = _normalize_operator(Tn, method=generator_norm, target=generator_norm_target)
            M_comp = M_comp + eps_n(int(n), s) * Tn
            if completion_mode in {"dual_1_minus_s", "dual_1_minus_s_j"}:
                dual_op = Tn.conj().T
                if J is not None:
                    dual_op = J @ dual_op @ J
                M_comp = M_comp + dual_scale_c * eps_n(int(n), (1.0 - s)) * dual_op

    prime_scale_c = complex(prime_scale)
    comp_scale_c = complex(comp_scale)
    if match_comp_to_prime and comps_list:
        target = abs(prime_scale_c) * fro_norm(M_prime)
        current = abs(comp_scale_c) * fro_norm(M_comp)
        if current > 0.0:
            comp_scale_c = comp_scale_c * (target / current)

    A = np.eye(N, dtype=np.complex128) + prime_scale_c * M_prime + comp_scale_c * M_comp

    if return_components:
        # caller expects: (A, M_prime_scaled, M_comp_scaled)
        return (np.asarray(A, dtype=np.complex128), prime_scale_c * M_prime, comp_scale_c * M_comp)  # type: ignore[return-value]

    return np.asarray(A, dtype=np.complex128)
