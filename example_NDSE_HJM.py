"""HJM-style example for the Numeraire-Based Distributional Sensitivity Engine."""

from math import exp
from typing import Dict

from Numeraire_Based_DSE import (
    NDSEMonteCarloBlocks,
    NDSEPrimitives,
    OneFactorHJMEquitySimulator,
    ndse_first_derivative,
    ndse_mixed_partial_components,
    ndse_price,
)


def _print_metric(label: str, value: float) -> None:
    print(f"  {label:<16}: {value}")


def run_example() -> Dict[str, float]:
    """Return a one-factor HJM-style NDSE example output dictionary."""
    S = 100.0
    K = 100.0
    tau = 0.50
    q = 0.01
    r_flat = 0.03
    bond = exp(-r_flat * tau)

    def sigma_bond_fn(t: float) -> float:
        return 0.10 * (tau - t)

    sim = OneFactorHJMEquitySimulator(
        tau=tau,
        sigma_s=0.20,
        sigma_bond=sigma_bond_fn,
        rho=-0.35,
        n_paths=100000,
        n_steps=120,
        seed=2027,
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

    decomp = ndse_mixed_partial_components(prims, "S", "eta", blocks)
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
        "d2V_dS_deta": decomp["total"],
        "dot": decomp["dot"],
        "P2_cross": decomp["P2_cross"],
        "P2_non": decomp["P2_non"],
    }


def main() -> None:
    results = run_example()
    print("Example type        : HJM-style NDSE example")
    print("Rate specification  : one-factor bond-volatility loading")
    print("Bond loading        : Sigma_bond(t, T) = 0.10 * (T - t)")
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
