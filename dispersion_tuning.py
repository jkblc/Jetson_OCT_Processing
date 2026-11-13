"""Automatic dispersion tuning: find the (C2, C3) that make the image sharpest.

Residual dispersion smears every reflector's point-spread function in depth,
and a smeared PSF spreads intensity over more pixels. So the sharpest image is
the one whose intensity is most concentrated, which the Rényi-2 entropy
measures (Wojtkowski 2004):

    H₂(I) = −log Σ pᵢ²,   pᵢ = Iᵢ / ΣI

Lower H₂ ↔ intensity more concentrated ↔ sharper PSF.

The search only ever sees a scalar cost, so it scores the image the
reconstruction in recon.py actually produces.

NumPy and SciPy only — importable without CuPy or a GPU.
"""

import numpy as np
from scipy.optimize import minimize

# Search box. The tuner sliders use the same limits, so a tuned value can
# always be shown on — and nudged from — the slider it initialises.
C2_RANGE = (-200.0, 200.0)
C3_RANGE = (-100.0, 100.0)


def renyi2_entropy(intensity):
    """Rényi-2 entropy of a non-negative intensity image (NumPy).

    Returns a Python float; lower is sharper. An empty or non-finite image
    returns +inf so the search steps away from it rather than stopping on it.
    """
    I = np.asarray(intensity, dtype=np.float64)
    total = I.sum()
    if not np.isfinite(total) or total <= 0:
        return float('inf')
    return float(-np.log(np.sum((I / total) ** 2) + 1e-30))


def two_stage_search(cost_fn, c2_range=C2_RANGE, c3_range=C3_RANGE,
                     grid_shape=(9, 9)):
    """Two-stage dispersion search: coarse grid → Nelder-Mead refinement.

    `cost_fn(c2, c3)` returns a float to minimise. Stage 1 sweeps a
    grid_shape grid over c2_range × c3_range to escape local minima. Stage 2
    polishes the grid winner with Nelder-Mead.

    Returns (c2_opt, c3_opt) as Python floats, clipped to the input ranges.
    """
    print("Auto-tuning dispersion (Rényi-2 entropy minimization)...")

    # ── Stage 1: coarse grid ────────────────────────────────────────────
    c2_grid = np.linspace(c2_range[0], c2_range[1], grid_shape[0])
    c3_grid = np.linspace(c3_range[0], c3_range[1], grid_shape[1])

    n_evals = grid_shape[0] * grid_shape[1]
    best_cost, best_c2, best_c3 = float('inf'), 0.0, 0.0

    for c2 in c2_grid:
        for c3 in c3_grid:
            cost = cost_fn(float(c2), float(c3))
            if cost < best_cost:
                best_cost, best_c2, best_c3 = cost, float(c2), float(c3)

    print(f"  Stage 1 ({n_evals}-point grid): "
          f"C2={best_c2:+.2f}, C3={best_c3:+.2f} → H₂={best_cost:.4f}")

    # ── Stage 2: Nelder-Mead polish ─────────────────────────────────────
    c2_step = (c2_range[1] - c2_range[0]) / (grid_shape[0] * 4)
    c3_step = (c3_range[1] - c3_range[0]) / (grid_shape[1] * 4)

    def clipped_cost(params):
        c2 = float(np.clip(params[0], *c2_range))
        c3 = float(np.clip(params[1], *c3_range))
        return cost_fn(c2, c3)

    init_simplex = np.array([
        [best_c2,           best_c3          ],
        [best_c2 + c2_step, best_c3          ],
        [best_c2,           best_c3 + c3_step],
    ])

    res = minimize(clipped_cost, x0=np.array([best_c2, best_c3]),
                   method='Nelder-Mead',
                   options={'initial_simplex': init_simplex,
                            'xatol': 0.1, 'fatol': 1e-5, 'maxiter': 80})

    c2_opt = float(np.clip(res.x[0], *c2_range))
    c3_opt = float(np.clip(res.x[1], *c3_range))
    delta = res.fun - best_cost

    print(f"  Stage 2 ({res.nit} Nelder-Mead iters): "
          f"C2={c2_opt:+.2f}, C3={c3_opt:+.2f} → H₂={res.fun:.4f} "
          f"(Δ={delta:+.4f} vs grid)")
    return c2_opt, c3_opt
