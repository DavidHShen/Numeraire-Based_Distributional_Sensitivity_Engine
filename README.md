# Numeraire-Based Distributional Sensitivity Engine

Reference implementation accompanying the paper **"Numeraire-Based Distributional Sensitivity Engine for European Options under Stochastic Interest Rates"**.

## Overview

This repository implements a standalone **NDSE** framework for European option pricing and sensitivities under stochastic discounting. The construction extends the deterministic-discounting DSE representation by normalizing the stochastic discount factor with the bond price $P(t,T)$, which restores an $\mathcal{F}_t$-measurable exercise boundary for a modified Doleans-Dade log variable.

At the representation level, the option value keeps the same two-weight shell,

$$
V_t = a_1 w_1 + a_2 w_2,
$$

but the second measure changes from the deterministic-discounting setting to a stochastic-discounting numeraire measure. In the bond-numeraire case, the strike leg is evaluated under the $T$-forward measure $\mathbb{Q}^T$.

The repository is organized so that the **engine shell** and the **distributional blocks** remain separate:

- the core engine handles the primitives, Jacobians, and contraction formulas;
- the backend supplies simulated or otherwise computed CDF/PDF blocks under the relevant measures.

## Repository contents

```text
Numeraire-Based_Distributional_Sensitivity_Engine/
├── Numeraire_Based_DSE.py
├── example_NDSE_HJM.py
├── example_NDSE_generic.py
├── README.md
└── requirements.txt
```

### `Numeraire_Based_DSE.py`

Core standalone NDSE module. It provides:

- the primitive layer `NDSEPrimitives`;
- the block/simulator interfaces `NDSEBlocks` and `NDSESimulator`;
- the two-weight pricing function `ndse_price`;
- the first-derivative engine `ndse_first_derivative`;
- the mixed-partial decomposition `ndse_mixed_partial_components`;
- a one-factor HJM-style backend `OneFactorHJMEquitySimulator`;
- Monte Carlo block estimators `NDSEMonteCarloBlocks`.

### `example_NDSE_HJM.py`

Standalone **HJM-style** example script using the built-in one-factor bond-volatility backend.

### `example_NDSE_generic.py`

Standalone **non-HJM** example script using a generic correlated Gaussian-factor backend that supplies the required block inputs directly.

## Mathematical structure implemented in code

### 1. Primitive layer

For current time $t$, maturity $T$, and $\tau = T-t$, the code uses the primitive tuple

$$
(a_1,a_2,\eta,\widetilde m).
$$

The paper-consistent definitions are

$$
a_1 = \frac{S_t}{Q_{t,T}}, \qquad a_2 = -K P(t,T), \qquad \widetilde m = \log\ \left(\frac{S_t}{K Q_{t,T} P(t,T)}\right).
$$

Under deterministic dividend yield $q$ on $[t,T]$,

$$
Q_{t,T} = e^{q\tau}, \qquad a_1 = S_t e^{-q\tau}.
$$

This is the specialization implemented in the current code.

### 2. Two-weight price representation

The option value is written as

$$
V_t = a_1 w_1 + a_2 w_2,
$$

where

- $w_1$ is the signed digital block under the spot-leg tilt measure $\mathbb{Q}^+$;
- $w_2$ is the signed digital block under the bond-numeraire / $T$-forward measure $\mathbb{Q}^T$.

### 3. First-order sensitivity template

The code implements the four-input contraction

$$
\frac{\partial V}{\partial x}
= w_1 \frac{\partial a_1}{\partial x}
\+ w_2 \frac{\partial a_2}{\partial x}
\+ w_3 \frac{\partial \eta}{\partial x}
\+ w_4 \frac{\partial \widetilde m}{\partial x},
$$

where $w_3$ and $w_4$ are built from parameter derivatives of the CDF blocks and from the PDF blocks evaluated at the boundary.

### 4. Mixed partials

Mixed partials are organized into three pieces:

- a **dot term** carrying second derivatives of the primitives;
- a **P2\_cross** term collecting primitive-motion products;
- a **P2\_non** term collecting the remaining correction channel.

The decomposition is returned explicitly by `ndse_mixed_partial_components`.

### 5. Monte Carlo block layer

The Monte Carlo implementation uses the likelihood-ratio identities

$$
L^+ = e^{-E_{t,T}}, \qquad L^T = \frac{DF_{t,T}}{P(t,T)},
$$

and estimates:

- CDF blocks from weighted indicators;
- PDF blocks from a mollified Dirac delta / Gaussian kernel;
- $\eta$-derivative blocks from common-random-number finite differences.

The engine therefore consumes distributional blocks, while the backend is responsible for producing them.

## Usage

Install the minimal dependency set from the repository root:

```bash
pip install -r requirements.txt
```

Run the main demonstration in the core module from the repository root:

```bash
python Numeraire_Based_DSE.py
```

Run the HJM-style example from the repository root:

```bash
python example_NDSE_HJM.py
```

Run the generic non-HJM example from the repository root:

```bash
python example_NDSE_generic.py
```

Import the core module in another script:

```python
from Numeraire_Based_DSE import (
    NDSEPrimitives,
    NDSEMonteCarloBlocks,
    ndse_price,
    ndse_first_derivative,
    ndse_mixed_partial_components,
)
```

## Output interpretation


### Running `python Numeraire_Based_DSE.py`

The core module prints two blocks in sequence. The first block is a **validation case**; the second block is the actual **stochastic-discounting illustration**.

#### Core module: zero-bond-vol BSM-collapse check

This output block is the **deterministic-discounting limit** of NDSE. The bond-volatility loading is set to zero, so the stochastic-rate layer is inactive. This block is included as a collapse-to-BSM consistency check rather than as the main NDSE stochastic-rate example.

Typical printed entries in this block are interpreted as follows:

- `NDSE price (MC)`: Monte Carlo NDSE price in the zero-bond-vol limit.
- `BSM price`: Black-Scholes-Merton benchmark used for comparison in the same limit.
- `delta`: first derivative with respect to spot, evaluated by the NDSE engine.
- `gamma`: second derivative with respect to spot, reported for the collapse check.

A small difference between `NDSE price (MC)` and `BSM price` is expected because the NDSE value is estimated by Monte Carlo. The relevant interpretation is therefore **numerical agreement up to simulation error**, not literal identity digit by digit.

#### Core module: one-factor HJM-style stochastic-rate NDSE run

This output block is the actual **stochastic-discounting** run. Rate randomness is active through the nonzero bond-volatility loading, so this is the main NDSE demonstration within the core module.

Typical printed entries in this block are interpreted as follows:

- `price`: NDSE option value under stochastic discounting.
- `delta`: first derivative with respect to spot.
- `dV/dbond`: sensitivity with respect to the bond-price primitive $P(t,T)$, that is, the strike-leg discounting channel.
- `dV/deta`: sensitivity with respect to the boundary-scale variable $\eta$.
- `dot`: second-derivative contribution coming directly from primitive Hessian terms.
- `P2_cross`: mixed correction channel built from cross-motion of the weights and primitives.
- `P2_non`: remaining non-cross correction channel in the mixed-partial decomposition.

The mixed-partial output is therefore intended to show the **decomposition structure** derived in the paper, not merely a single aggregated second derivative.

### Running `example_NDSE_HJM.py`

This file is a **standalone HJM-style example script** separated from the core module. It isolates the stochastic-rate backend example without also printing the BSM-collapse validation block.

### Running `example_NDSE_generic.py`

This file is a **standalone generic non-HJM example script** separated from the core module. It illustrates that the NDSE shell is not restricted to the built-in HJM-style backend.

## Design notes

### Separation between engine and backend

A central design point of the repository is that the engine layer does not assume a single term-structure model. Any backend is admissible provided that it supplies the four arrays required by the block layer:

$$
E, \qquad \widetilde E, \qquad L^+, \qquad L^T.
$$

This mirrors the paper's viewpoint that the NDSE shell is model-agnostic once the relevant conditional distribution blocks have been identified.

### HJM role in the repository

The built-in HJM component is intentionally compact. It is included as a transparent bond-volatility-driven backend rather than as a full production HJM library. Its role is to provide a direct implementation of the paper's stochastic-discounting construction in a clean and inspectable form.

### Current derivative channel for $\eta$

The present Monte Carlo block class computes $\eta$-derivative blocks by common-random-number finite differences. This matches the block-based implementation strategy of the paper, but it is not the only possible route. Likelihood-ratio or Malliavin extensions can be added later without changing the engine shell.

## Scope and limitations

- European options only.
- Deterministic dividend yield in the current primitive specialization.
- NumPy-only implementation.
- The HJM simulator is a compact demonstration backend, not a full calibration library.
- Kernel smoothing is used for the $\widetilde m$-derivative / PDF blocks.

## Suggested citation

Repository title:

```text
Numeraire-Based Distributional Sensitivity Engine
```

Paper title:

```text
Numeraire-Based Distributional Sensitivity Engine for European Options under Stochastic Interest Rates
```

## File-level map to the paper

The repository follows the paper's structure closely:

- the two-weight price representation is implemented in `ndse_price`;
- the four-input first-derivative template is implemented in `ndse_first_derivative`;
- the mixed-partial organization is implemented in `ndse_mixed_partial_components`;
- the Monte Carlo distribution blocks are implemented in `NDSEMonteCarloBlocks`;
- the one-factor stochastic-rate illustration is implemented in `OneFactorHJMEquitySimulator` and `example_NDSE_HJM.py`.

## Dependency

- `numpy`
