"""Generic non-HJM example for the Numeraire-Based Distributional Sensitivity Engine."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Dict

import numpy as np

from Numeraire_Based_DSE import (
    NDSEMonteCarloBlocks,
    NDSEPrimitives,
    ndse_first_derivative,
    ndse_mixed_partial_components,
    ndse_price,
)


@dataclass
class GenericNDSEFactorSimulator:
    r"""
    Non-HJM NDSE simulator based on correlated Gaussian factors.

    The backend builds the NDSE random objects directly from

        E        = -X_s + 0.5 Var(X_s),
        log L^T  = Y_r - 0.5 Var(Y_r),
        E_tilde  = E + log L^T,

    where X_s and Y_r are correlated Gaussian factors on [t, T].
    The class satisfies the simulator protocol used by NDSEMonteCarloBlocks.
    """

    tau: float
    sigma_equity: float
    sigma_discount: float
    rho: float = 0.0
    n_paths: int = 100000
    seed: int = 314159
    eta_mode: str = "joint_scale"

    def __post_init__(self):
        if not (-1.0 < self.rho < 1.0):
            raise ValueError("rho must lie in (-1, 1)")
        self._rng = np.random.default_rng(self.seed)

    def _equity_scale(self, eta: float) -> float:
        if self.eta_mode in ("joint_scale", "equity_scale"):
            return float(eta) * self.sigma_equity
        return float(self.sigma_equity)

    def _discount_scale(self, eta: float) -> float:
        if self.eta_mode in ("joint_scale", "discount_scale"):
            return float(eta) * self.sigma_discount
        return float(self.sigma_discount)

    def simulate(self, eta: float) -> Dict[str, np.ndarray]:
        z1 = self._rng.standard_normal(self.n_paths)
        z2 = self._rng.standard_normal(self.n_paths)
        z_discount = self.rho * z1 + sqrt(1.0 - self.rho * self.rho) * z2

        sig_s = self._equity_scale(eta)
        sig_d = self._discount_scale(eta)

        xs = sig_s * sqrt(self.tau) * z1
        yr = sig_d * sqrt(self.tau) * z_discount

        var_xs = (sig_s * sig_s) * self.tau
        var_yr = (sig_d * sig_d) * self.tau

        E = -xs + 0.5 * var_xs
        log_LT = yr - 0.5 * var_yr
        E_tilde = E + log_LT

        return {
            "E": E,
            "E_tilde": E_tilde,
            "L_plus": np.exp(-E),
            "L_T": np.exp(log_LT),
        }


def _print_metric(label: str, value: float) -> None:
    print(f"  {label:<16}: {value}")


def run_example() -> Dict[str, float]:
    """Return a generic non-HJM NDSE example output dictionary."""
    S = 100.0
    K = 95.0
    tau = 0.75
    q = 0.01
    r_flat = 0.03
    bond = exp(-r_flat * tau)

    sim = GenericNDSEFactorSimulator(
        tau=tau,
        sigma_equity=0.22,
        sigma_discount=0.14,
        rho=0.25,
        n_paths=120000,
        seed=2028,
        eta_mode="joint_scale",
    )
    blocks = NDSEMonteCarloBlocks(
        simulator=sim,
        h_m=0.025,
        fd_eta=5e-3,
        smooth_cdf=False,
    )
    prims = NDSEPrimitives(
        S=S,
        K=K,
        tau=tau,
        q=q,
        bond=bond,
        eta=1.0,
        D=+1,
    )

    cross = ndse_mixed_partial_components(prims, "S", "eta", blocks)
    a1, a2, eta, m = prims.primitives()

    return {
        "a1": a1,
        "a2": a2,
        "eta": eta,
        "m": m,
        "price": ndse_price(prims, blocks),
        "delta": ndse_first_derivative(prims, "S", blocks),
        "dV_dbond": ndse_first_derivative(prims, "bond", blocks),
        "dV_deta": ndse_first_derivative(prims, "eta", blocks),
        "d2V_dS_deta": cross["total"],
        "dot": cross["dot"],
        "P2_cross": cross["P2_cross"],
        "P2_non": cross["P2_non"],
    }


def main() -> None:
    results = run_example()
    print("Example type        : generic non-HJM NDSE example")
    print("Rate specification  : correlated Gaussian-factor backend")
    print("Measure block input : backend supplies E, E_tilde, L_plus, and L_T directly")
    _print_metric("a1", results["a1"])
    _print_metric("a2", results["a2"])
    _print_metric("eta", results["eta"])
    _print_metric("m", results["m"])
    _print_metric("price", results["price"])
    _print_metric("delta", results["delta"])
    _print_metric("dV/dbond", results["dV_dbond"])
    _print_metric("dV/deta", results["dV_deta"])
    _print_metric("d2V/(dS deta)", results["d2V_dS_deta"])
    _print_metric("dot", results["dot"])
    _print_metric("P2_cross", results["P2_cross"])
    _print_metric("P2_non", results["P2_non"])


if __name__ == "__main__":
    main()
