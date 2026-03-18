"""
Standalone reference implementation for the paper
"Numeraire-Based Distributional Sensitivity Engine for European Options
under Stochastic Interest Rates".

The module implements the NDSE pricing and sensitivity representation

    V_t = a1 * w1 + a2 * w2,

with primitives

    a1 = S_t * exp(-q * tau),
    a2 = -K * P(t, T),
    m  = log(S_t / (K * exp(q * tau) * P(t, T))).

The stochastic-discounting extension uses two probability measures:

    * Q^+ : spot-leg Doleans tilt measure,
    * Q^T : T-forward / bond-numeraire measure.

The implementation contains:

    * NDSEPrimitives                 : F_t-measurable primitives and derivatives,
    * NDSEBlocks / NDSESimulator     : block and simulator interfaces,
    * ndse_price                     : two-weight price,
    * ndse_first_derivative          : four-input first-derivative engine,
    * ndse_mixed_partial_components  : mixed-partial decomposition,
    * OneFactorHJMEquitySimulator    : minimal one-factor HJM-style backend,
    * NDSEMonteCarloBlocks           : Monte Carlo CDF/PDF block estimators.

The Monte Carlo layer simulates under Q and reweights with

    L^+ = exp(-E_{t,T}),
    L^T = DF_{t,T} / P(t, T).

NumPy is the only external dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
from typing import Callable, Dict, Protocol, Tuple, runtime_checkable

import numpy as np

__all__ = [
    "NDSEPrimitives",
    "NDSEBlocks",
    "NDSESimulator",
    "OneFactorHJMEquitySimulator",
    "NDSEMonteCarloBlocks",
    "norm_pdf",
    "norm_cdf",
    "ndse_weights",
    "ndse_price",
    "ndse_first_derivative",
    "ndse_mixed_partial_components",
    "ndse_mixed_partial",
]

ArrayDict = Dict[str, np.ndarray]


# =========================
# Normal PDF/CDF utilities
# =========================

_SQRT_2 = sqrt(2.0)
_SQRT_2PI = sqrt(2.0 * pi)


def norm_pdf(x):
    """Standard normal density evaluated elementwise."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def norm_cdf(x):
    """Standard normal distribution function evaluated elementwise."""
    x = np.asarray(x, dtype=float)
    vec_erf = np.vectorize(erf, otypes=[float])
    return 0.5 * (1.0 + vec_erf(x / _SQRT_2))


# ==========================================
# NDSE primitives (standalone)
# ==========================================

@dataclass(frozen=True)
class NDSEPrimitives:
    """
    NDSE primitive tuple (a1, a2, eta, mtilde).

    Parameters
    ----------
    S
        Current spot S_t.
    K
        Strike.
    tau
        Time to maturity T - t.
    q
        Deterministic dividend yield on [t, T].
    bond
        Current zero-coupon bond price P(t, T).
    eta
        Generic scalar distributional parameter.
    D
        +1 for call, -1 for put.
    """

    S: float
    K: float
    tau: float
    q: float
    bond: float
    eta: float
    D: int = +1

    def primitives(self) -> Tuple[float, float, float, float]:
        a1 = self.S * exp(-self.q * self.tau)
        a2 = -self.K * self.bond
        m = log(self.S / (self.K * exp(self.q * self.tau) * self.bond))
        return a1, a2, self.eta, m

    def d_primitives(self, var: str) -> Tuple[float, float, float, float]:
        a1, _, _, _ = self.primitives()

        if var == "S":
            return exp(-self.q * self.tau), 0.0, 0.0, 1.0 / self.S
        if var == "K":
            return 0.0, -self.bond, 0.0, -1.0 / self.K
        if var == "bond":
            return 0.0, -self.K, 0.0, -1.0 / self.bond
        if var == "q":
            return -self.tau * a1, 0.0, 0.0, -self.tau
        if var == "tau":
            return -self.q * a1, 0.0, 0.0, -self.q
        if var == "eta":
            return 0.0, 0.0, 1.0, 0.0

        raise ValueError("Unsupported var. Use one of: S, K, bond, q, tau, eta")

    def d2_primitives(self, var1: str, var2: str) -> Tuple[float, float, float, float]:
        a1, _, _, _ = self.primitives()
        key = tuple(sorted((var1, var2)))

        d2a1 = 0.0
        d2a2 = 0.0
        d2eta = 0.0
        d2m = 0.0

        # a1 = S exp(-q tau)
        if key == ("q", "q"):
            d2a1 = (self.tau ** 2) * a1
        elif key == ("q", "tau"):
            d2a1 = -a1 + self.q * self.tau * a1
        elif key == ("S", "q"):
            d2a1 = -self.tau * exp(-self.q * self.tau)
        elif key == ("S", "tau"):
            d2a1 = -self.q * exp(-self.q * self.tau)
        elif key == ("tau", "tau"):
            d2a1 = (self.q ** 2) * a1

        # a2 = -K * bond
        if key == ("K", "bond"):
            d2a2 = -1.0

        # m = log S - log K - q tau - log bond
        if key == ("S", "S"):
            d2m = -1.0 / (self.S * self.S)
        elif key == ("K", "K"):
            d2m = 1.0 / (self.K * self.K)
        elif key == ("bond", "bond"):
            d2m = 1.0 / (self.bond * self.bond)
        elif key == ("q", "tau"):
            d2m = -1.0

        return d2a1, d2a2, d2eta, d2m


# ==========================================
# Distribution-block and simulator protocols
# ==========================================

class NDSEBlocks:
    """Interface for stochastic-discounting distribution blocks."""

    def F1(self, m: float, eta: float, D: int) -> float:
        raise NotImplementedError

    def F2(self, m: float, eta: float, D: int) -> float:
        raise NotImplementedError

    def dF1_dm(self, m: float, eta: float) -> float:
        raise NotImplementedError

    def dF2_dm(self, m: float, eta: float) -> float:
        raise NotImplementedError

    def dF1_deta(self, m: float, eta: float) -> float:
        raise NotImplementedError

    def dF2_deta(self, m: float, eta: float) -> float:
        raise NotImplementedError

    def d2F1_dm2(self, m: float, eta: float) -> float:
        raise NotImplementedError

    def d2F2_dm2(self, m: float, eta: float) -> float:
        raise NotImplementedError

    def d2F1_deta2(self, m: float, eta: float) -> float:
        raise NotImplementedError

    def d2F2_deta2(self, m: float, eta: float) -> float:
        raise NotImplementedError

    def d2F1_deta_dm(self, m: float, eta: float) -> float:
        raise NotImplementedError

    def d2F2_deta_dm(self, m: float, eta: float) -> float:
        raise NotImplementedError


@runtime_checkable
class NDSESimulator(Protocol):
    """Protocol for simulation backends used by NDSEMonteCarloBlocks."""

    def simulate(self, eta: float) -> ArrayDict:
        """Return E, E_tilde, L_plus, and L_T arrays."""


# ==========================================
# Weights and engine formulas
# ==========================================


def _signed_from_call_prob(prob_call: float, D: int) -> float:
    if D == +1:
        return prob_call
    if D == -1:
        return prob_call - 1.0
    raise ValueError("D must be +1 or -1")


def ndse_weights(a1: float, a2: float, eta: float, m: float, D: int, blocks: NDSEBlocks) -> Dict[str, float]:
    """Return the price-level and first-derivative weight blocks."""
    w1 = blocks.F1(m, eta, D)
    w2 = blocks.F2(m, eta, D)

    F1_eta = blocks.dF1_deta(m, eta)
    F2_eta = blocks.dF2_deta(m, eta)
    F1_m = blocks.dF1_dm(m, eta)
    F2_m = blocks.dF2_dm(m, eta)

    w3 = a1 * F1_eta + a2 * F2_eta
    w4 = a1 * F1_m + a2 * F2_m

    return {
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "w4": w4,
        "F1_eta": F1_eta,
        "F2_eta": F2_eta,
        "F1_m": F1_m,
        "F2_m": F2_m,
    }


def ndse_price(prims: NDSEPrimitives, blocks: NDSEBlocks) -> float:
    """Two-weight NDSE price V_t = a1 * w1 + a2 * w2."""
    a1, a2, eta, m = prims.primitives()
    w = ndse_weights(a1, a2, eta, m, prims.D, blocks)
    return w["w1"] * a1 + w["w2"] * a2


def ndse_first_derivative(prims: NDSEPrimitives, var: str, blocks: NDSEBlocks) -> float:
    """Four-input first-derivative contraction."""
    a1, a2, eta, m = prims.primitives()
    da1, da2, deta, dm = prims.d_primitives(var)
    w = ndse_weights(a1, a2, eta, m, prims.D, blocks)
    return w["w1"] * da1 + w["w2"] * da2 + w["w3"] * deta + w["w4"] * dm


def _maybe_mixed(blocks: NDSEBlocks, which: str, m: float, eta: float) -> float:
    if which == "F1_meta":
        fn = getattr(blocks, "d2F1_dm_deta", None)
        if callable(fn):
            return fn(m, eta)
        return blocks.d2F1_deta_dm(m, eta)
    if which == "F2_meta":
        fn = getattr(blocks, "d2F2_dm_deta", None)
        if callable(fn):
            return fn(m, eta)
        return blocks.d2F2_deta_dm(m, eta)
    raise ValueError("Unknown selector")


def V_matrix(m: float, eta: float, blocks: NDSEBlocks) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return (
        (blocks.dF1_deta(m, eta), blocks.dF1_dm(m, eta)),
        (blocks.dF2_deta(m, eta), blocks.dF2_dm(m, eta)),
    )


def V_eta_matrix(m: float, eta: float, blocks: NDSEBlocks) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return (
        (blocks.d2F1_deta2(m, eta), _maybe_mixed(blocks, "F1_meta", m, eta)),
        (blocks.d2F2_deta2(m, eta), _maybe_mixed(blocks, "F2_meta", m, eta)),
    )


def V_m_matrix(m: float, eta: float, blocks: NDSEBlocks) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return (
        (blocks.d2F1_deta_dm(m, eta), blocks.d2F1_dm2(m, eta)),
        (blocks.d2F2_deta_dm(m, eta), blocks.d2F2_dm2(m, eta)),
    )


def _matvec2(M: Tuple[Tuple[float, float], Tuple[float, float]], v: Tuple[float, float]) -> Tuple[float, float]:
    return (
        M[0][0] * v[0] + M[0][1] * v[1],
        M[1][0] * v[0] + M[1][1] * v[1],
    )


def _dot2(u: Tuple[float, float], v: Tuple[float, float]) -> float:
    return u[0] * v[0] + u[1] * v[1]


def P2_matrix_form(
    *,
    a12: Tuple[float, float],
    a12_x1: Tuple[float, float],
    a12_x2: Tuple[float, float],
    a34_x1: Tuple[float, float],
    a34_x2: Tuple[float, float],
    m: float,
    eta: float,
    eta_x2: float,
    m_x2: float,
    blocks: NDSEBlocks,
) -> Dict[str, float]:
    """Return the correction channels for mixed partials."""
    V = V_matrix(m, eta, blocks)

    cross_1 = _dot2(a12_x1, _matvec2(V, a34_x2))
    cross_2 = _dot2(a12_x2, _matvec2(V, a34_x1))
    p2_cross = cross_1 + cross_2

    V_eta = V_eta_matrix(m, eta, blocks)
    V_m = V_m_matrix(m, eta, blocks)
    V_x2 = (
        (V_eta[0][0] * eta_x2 + V_m[0][0] * m_x2, V_eta[0][1] * eta_x2 + V_m[0][1] * m_x2),
        (V_eta[1][0] * eta_x2 + V_m[1][0] * m_x2, V_eta[1][1] * eta_x2 + V_m[1][1] * m_x2),
    )
    p2_non = _dot2(a12, _matvec2(V_x2, a34_x1))

    return {"cross": p2_cross, "non": p2_non, "total": p2_cross + p2_non}


def ndse_mixed_partial_components(prims: NDSEPrimitives, var1: str, var2: str, blocks: NDSEBlocks) -> Dict[str, float]:
    """Return dot, P2_cross, P2_non, P2, and total for a mixed partial."""
    a1, a2, eta, m = prims.primitives()

    da1_1, da2_1, deta_1, dm_1 = prims.d_primitives(var1)
    da1_2, da2_2, deta_2, dm_2 = prims.d_primitives(var2)
    d2a1, d2a2, d2eta, d2m = prims.d2_primitives(var1, var2)

    w = ndse_weights(a1, a2, eta, m, prims.D, blocks)
    dot_term = w["w1"] * d2a1 + w["w2"] * d2a2 + w["w3"] * d2eta + w["w4"] * d2m

    p2_parts = P2_matrix_form(
        a12=(a1, a2),
        a12_x1=(da1_1, da2_1),
        a12_x2=(da1_2, da2_2),
        a34_x1=(deta_1, dm_1),
        a34_x2=(deta_2, dm_2),
        m=m,
        eta=eta,
        eta_x2=deta_2,
        m_x2=dm_2,
        blocks=blocks,
    )

    total = dot_term + p2_parts["total"]
    return {
        "dot": dot_term,
        "P2_cross": p2_parts["cross"],
        "P2_non": p2_parts["non"],
        "P2": p2_parts["total"],
        "total": total,
    }


def ndse_mixed_partial(prims: NDSEPrimitives, var1: str, var2: str, blocks: NDSEBlocks) -> float:
    """Convenience wrapper for the total mixed partial."""
    return ndse_mixed_partial_components(prims, var1, var2, blocks)["total"]


# ==========================================
# Minimal HJM-style Monte Carlo simulator
# ==========================================

@dataclass
class OneFactorHJMEquitySimulator:
    r"""
    Simulate the NDSE random objects under Q on [t, T].

    The rate side is represented through the bond-volatility loading Sigma(u, T),
    which enters

        log(DF / P) = -∫ Sigma dW^r - 0.5 ∫ Sigma^2 du.

    The equity side is represented through sigma_S(u). Correlation between the
    equity and rate Brownian drivers is allowed.

    eta_mode controls how the scalar NDSE parameter eta enters:

        * "joint_scale"  : eta scales both sigma_S and Sigma,
        * "equity_scale" : eta scales sigma_S only,
        * "bond_scale"   : eta scales Sigma only,
        * "none"         : eta is inactive inside the simulator.
    """

    tau: float
    sigma_s: float | Callable[[float], float]
    sigma_bond: float | Callable[[float], float]
    rho: float = 0.0
    n_paths: int = 50000
    n_steps: int = 250
    seed: int = 12345
    eta_mode: str = "joint_scale"

    def __post_init__(self):
        if not (-1.0 < self.rho < 1.0):
            raise ValueError("rho must lie in (-1, 1)")
        rng = np.random.default_rng(self.seed)
        self._Zr = rng.standard_normal((self.n_paths, self.n_steps))
        self._Zu = rng.standard_normal((self.n_paths, self.n_steps))

    def _base_sigma_s(self, t: float) -> float:
        return float(self.sigma_s(t) if callable(self.sigma_s) else self.sigma_s)

    def _base_sigma_bond(self, t: float) -> float:
        return float(self.sigma_bond(t) if callable(self.sigma_bond) else self.sigma_bond)

    def _sigma_s_eta(self, t: float, eta: float) -> float:
        base = self._base_sigma_s(t)
        if self.eta_mode in ("joint_scale", "equity_scale"):
            return float(eta) * base
        return base

    def _sigma_bond_eta(self, t: float, eta: float) -> float:
        base = self._base_sigma_bond(t)
        if self.eta_mode in ("joint_scale", "bond_scale"):
            return float(eta) * base
        return base

    def simulate(self, eta: float) -> ArrayDict:
        dt = self.tau / self.n_steps
        sqdt = sqrt(dt)
        rho_perp = sqrt(1.0 - self.rho * self.rho)

        dW_r = sqdt * self._Zr
        dW_s = sqdt * (self.rho * self._Zr + rho_perp * self._Zu)

        I_s = np.zeros(self.n_paths, dtype=float)
        V_s = np.zeros(self.n_paths, dtype=float)
        I_r = np.zeros(self.n_paths, dtype=float)
        V_r = np.zeros(self.n_paths, dtype=float)

        for i in range(self.n_steps):
            t_mid = (i + 0.5) * dt
            sig_s = self._sigma_s_eta(t_mid, eta)
            sig_b = self._sigma_bond_eta(t_mid, eta)

            I_s += sig_s * dW_s[:, i]
            V_s += (sig_s * sig_s) * dt
            I_r += sig_b * dW_r[:, i]
            V_r += (sig_b * sig_b) * dt

        E = -I_s + 0.5 * V_s
        log_LT = -I_r - 0.5 * V_r
        E_tilde = E + log_LT

        return {
            "E": E,
            "E_tilde": E_tilde,
            "L_plus": np.exp(-E),
            "L_T": np.exp(log_LT),
        }


# ==========================================
# Monte Carlo distribution blocks for NDSE
# ==========================================

@dataclass
class NDSEMonteCarloBlocks(NDSEBlocks):
    """
    Estimate NDSE CDF/PDF blocks by simulation under Q with reweighting.

    The m-derivatives use a Gaussian kernel. The eta-derivatives use common-
    random-number finite differences. The simulator may be any backend that
    satisfies the NDSESimulator protocol.
    """

    simulator: NDSESimulator
    h_m: float = 0.03
    fd_eta: float = 1e-3
    smooth_cdf: bool = False
    cache: bool = True

    def __post_init__(self):
        self._cache_store: Dict[float, ArrayDict] = {}

    def _run(self, eta: float) -> ArrayDict:
        key = float(eta)
        if self.cache and key in self._cache_store:
            return self._cache_store[key]
        out = self.simulator.simulate(eta)
        required = {"E", "E_tilde", "L_plus", "L_T"}
        missing = required.difference(out.keys())
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Simulator output is missing required keys: {missing_str}")
        if self.cache:
            self._cache_store[key] = out
        return out

    def _call_prob_pdf_dpdf(self, weight_key: str, m: float, eta: float) -> Tuple[float, float, float]:
        out = self._run(eta)
        E_tilde = out["E_tilde"]
        weights = out[weight_key]

        if self.smooth_cdf:
            cdf_arg = (m - E_tilde) / self.h_m
            prob_call = float(np.mean(weights * norm_cdf(cdf_arg)))
        else:
            prob_call = float(np.mean(weights * (E_tilde < m)))

        z = (m - E_tilde) / self.h_m
        pdf_kernel = norm_pdf(z) / self.h_m
        dF_dm = float(np.mean(weights * pdf_kernel))
        d2F_dm2 = float(np.mean(weights * (-z * norm_pdf(z)) / (self.h_m * self.h_m)))
        return prob_call, dF_dm, d2F_dm2

    def _eta_bump_pair(self, eta: float) -> Tuple[float, float]:
        h = float(self.fd_eta)
        eta_plus = eta + h
        eta_minus = eta - h
        if eta_minus <= 0.0:
            eta_minus = max(1e-8, eta * 0.5)
        return eta_plus, eta_minus

    def _eta_derivatives(self, weight_key: str, m: float, eta: float) -> Tuple[float, float, float]:
        prob0, pdf0, _ = self._call_prob_pdf_dpdf(weight_key, m, eta)
        eta_plus, eta_minus = self._eta_bump_pair(eta)
        h_eff = eta_plus - eta_minus

        prob_p, pdf_p, _ = self._call_prob_pdf_dpdf(weight_key, m, eta_plus)
        prob_m, pdf_m, _ = self._call_prob_pdf_dpdf(weight_key, m, eta_minus)

        d_prob = (prob_p - prob_m) / h_eff
        d2_prob = 2.0 * ((prob_p - prob0) / (eta_plus - eta) - (prob0 - prob_m) / (eta - eta_minus)) / h_eff
        d_pdf = (pdf_p - pdf_m) / h_eff
        return d_prob, d2_prob, d_pdf

    def F1(self, m: float, eta: float, D: int) -> float:
        prob_call, _, _ = self._call_prob_pdf_dpdf("L_plus", m, eta)
        return _signed_from_call_prob(prob_call, D)

    def F2(self, m: float, eta: float, D: int) -> float:
        prob_call, _, _ = self._call_prob_pdf_dpdf("L_T", m, eta)
        return _signed_from_call_prob(prob_call, D)

    def dF1_dm(self, m: float, eta: float) -> float:
        _, pdf, _ = self._call_prob_pdf_dpdf("L_plus", m, eta)
        return pdf

    def dF2_dm(self, m: float, eta: float) -> float:
        _, pdf, _ = self._call_prob_pdf_dpdf("L_T", m, eta)
        return pdf

    def dF1_deta(self, m: float, eta: float) -> float:
        d_prob, _, _ = self._eta_derivatives("L_plus", m, eta)
        return d_prob

    def dF2_deta(self, m: float, eta: float) -> float:
        d_prob, _, _ = self._eta_derivatives("L_T", m, eta)
        return d_prob

    def d2F1_dm2(self, m: float, eta: float) -> float:
        _, _, d2pdf = self._call_prob_pdf_dpdf("L_plus", m, eta)
        return d2pdf

    def d2F2_dm2(self, m: float, eta: float) -> float:
        _, _, d2pdf = self._call_prob_pdf_dpdf("L_T", m, eta)
        return d2pdf

    def d2F1_deta2(self, m: float, eta: float) -> float:
        _, d2_prob, _ = self._eta_derivatives("L_plus", m, eta)
        return d2_prob

    def d2F2_deta2(self, m: float, eta: float) -> float:
        _, d2_prob, _ = self._eta_derivatives("L_T", m, eta)
        return d2_prob

    def d2F1_deta_dm(self, m: float, eta: float) -> float:
        _, _, d_pdf = self._eta_derivatives("L_plus", m, eta)
        return d_pdf

    def d2F2_deta_dm(self, m: float, eta: float) -> float:
        _, _, d_pdf = self._eta_derivatives("L_T", m, eta)
        return d_pdf

    def d2F1_dm_deta(self, m: float, eta: float) -> float:
        return self.d2F1_deta_dm(m, eta)

    def d2F2_dm_deta(self, m: float, eta: float) -> float:
        return self.d2F2_deta_dm(m, eta)


# ==========================================
# Small analytic BSM helper for regression
# ==========================================


def _bsm_call_price(S: float, K: float, r: float, q: float, tau: float, sigma: float) -> float:
    if sigma <= 0.0 or tau <= 0.0:
        return max(S * exp(-q * tau) - K * exp(-r * tau), 0.0)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)
    return S * exp(-q * tau) * float(norm_cdf(d1)) - K * exp(-r * tau) * float(norm_cdf(d2))


# ==========================================
# Example usage / sanity checks
# ==========================================


def _print_metric(label: str, value: float) -> None:
    print(f"  {label:<16}: {value}")


def _run_bsm_collapse_example() -> Dict[str, float]:
    S = 100.0
    K = 100.0
    tau = 0.50
    r = 0.03
    q = 0.01
    sigma = 0.20
    bond = exp(-r * tau)

    prims = NDSEPrimitives(S=S, K=K, tau=tau, q=q, bond=bond, eta=1.0, D=+1)
    sim = OneFactorHJMEquitySimulator(
        tau=tau,
        sigma_s=sigma,
        sigma_bond=0.0,
        rho=0.0,
        n_paths=120000,
        n_steps=1,
        seed=2026,
        eta_mode="none",
    )
    blocks = NDSEMonteCarloBlocks(simulator=sim, h_m=0.02, fd_eta=1e-3, smooth_cdf=False)

    return {
        "price_mc": ndse_price(prims, blocks),
        "price_bsm": _bsm_call_price(S, K, r, q, tau, sigma),
        "delta": ndse_first_derivative(prims, "S", blocks),
        "gamma": ndse_mixed_partial_components(prims, "S", "S", blocks)["total"],
        "sigma_equity": sigma,
        "rho": 0.0,
    }


def _run_hjm_example() -> Dict[str, float]:
    S = 100.0
    K = 100.0
    tau = 0.50
    r = 0.03
    q = 0.01
    sigma = 0.20
    bond = exp(-r * tau)

    def sigma_bond_fn(t: float) -> float:
        return 0.10 * (tau - t)

    prims = NDSEPrimitives(S=S, K=K, tau=tau, q=q, bond=bond, eta=1.0, D=+1)
    sim = OneFactorHJMEquitySimulator(
        tau=tau,
        sigma_s=sigma,
        sigma_bond=sigma_bond_fn,
        rho=-0.35,
        n_paths=120000,
        n_steps=120,
        seed=2027,
        eta_mode="joint_scale",
    )
    blocks = NDSEMonteCarloBlocks(simulator=sim, h_m=0.025, fd_eta=5e-3, smooth_cdf=False)
    decomp = ndse_mixed_partial_components(prims, "S", "eta", blocks)

    return {
        "price": ndse_price(prims, blocks),
        "delta": ndse_first_derivative(prims, "S", blocks),
        "dV_dbond": ndse_first_derivative(prims, "bond", blocks),
        "dV_deta": ndse_first_derivative(prims, "eta", blocks),
        "d2V_dS_deta": decomp["total"],
        "dot": decomp["dot"],
        "P2_cross": decomp["P2_cross"],
        "P2_non": decomp["P2_non"],
        "sigma_equity": sigma,
        "rho": -0.35,
    }


def main() -> None:
    example_1 = _run_bsm_collapse_example()
    print("Example 1: zero-bond-vol BSM-collapse check")
    print("  Interpretation   : deterministic-discounting limit of NDSE")
    print("  Rate input       : Sigma_bond(t, T) ≡ 0")
    _print_metric("NDSE price (MC)", example_1["price_mc"])
    _print_metric("BSM price", example_1["price_bsm"])
    _print_metric("delta", example_1["delta"])
    _print_metric("gamma", example_1["gamma"])

    example_2 = _run_hjm_example()
    print("\nExample 2: one-factor HJM-style stochastic-rate NDSE run")
    print("  Interpretation   : stochastic discounting with active bond-volatility loading")
    print("  Bond loading     : Sigma_bond(t, T) = 0.10 * (T - t)")
    _print_metric("price", example_2["price"])
    _print_metric("delta", example_2["delta"])
    _print_metric("dV/dbond", example_2["dV_dbond"])
    _print_metric("dV/deta", example_2["dV_deta"])
    _print_metric("d2V/(dS deta)", example_2["d2V_dS_deta"])
    _print_metric("dot", example_2["dot"])
    _print_metric("P2_cross", example_2["P2_cross"])
    _print_metric("P2_non", example_2["P2_non"])


if __name__ == "__main__":
    main()
