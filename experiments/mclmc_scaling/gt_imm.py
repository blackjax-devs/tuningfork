"""Ground-truth inverse mass matrix utilities for MCLMC scaling-law study (S1/S3).

All three IMM structures (diagonal, low-rank+diagonal, dense) are represented
as a single ``LowRankInverseMassMatrix(sigma, U, lam)`` with variable rank k:

  k = 0        -> diagonal only (U is (d,0), lam is (0,))
  0 < k < d    -> low-rank + diagonal correction
  k = d        -> exact dense (all eigenvectors included)

The extraction mirrors ``blackjax.adaptation.mclmc_lrd_adaptation._extract_lrd_from_samples``
analytically (from the true Sigma rather than pilot draws):

  1. sigma = sqrt(diag Sigma)
  2. R = D^{-1/2} Sigma D^{-1/2}   (correlation matrix, D = diag(sigma^2))
  3. Eigendecompose R: R V = V Lambda
  4. Select top-k by |lambda - 1| (directions deviating most from isotropic)
  5. Build LowRankInverseMassMatrix(sigma=sigma, U=V[:,top_k], lam=Lambda[top_k])

kappa_eff formula:
  kappa_eff = kappa(M^{-1} Sigma^{-1})

where Sigma^{-1} is the Hessian of -log pi and M^{-1} is the LRD IMM.
For optimal preconditioning (M^{-1}=Sigma), this gives I, so kappa=1.

Verified equivalences for k=0 (diagonal):
  M^{-1} = diag(sigma^2), M^{-1} Sigma^{-1} = diag(sigma^2) Sigma^{-1}
  eigenvalues same as correlation matrix R = D^{-1} Sigma D^{-1}
  kappa_eff = kappa(R) approx 863 for ill_cond_50.

For k=d (dense, M^{-1}=Sigma):
  M^{-1} Sigma^{-1} = Sigma Sigma^{-1} = I  =>  kappa_eff = 1.0.

Correctness gates (asserted in __main__):
  - ill_cond_50 k=0  -> kappa_eff approx 863  (deep-dive: 14% reduction from kappa=1000)
  - ill_cond_50 k=d  -> kappa_eff = 1.0
  - mvn_10     all k -> kappa_eff approx 1.0

Real-model extension (S3 + funnel exploration):
  gt_from_draws(model_name, k=None) -> (imm, gt_var, gt_mean, d)
    Loads catalog/<model>/groundtruth_samples/blackjax/draws.npz (UNCONSTRAINED
    space — confirmed: draws are NUTS states.position which are always unconstrained,
    e.g. eight_schools_ncp tau shows negative values from HalfCauchy softplus
    transform).  Ravels multi-site draws to (n_samples, d) via ravel_pytree and
    applies _extract_lrd_from_samples with k=d (dense) by default.

    Supported real models (GT draws verified present):
      - german_credit, logistic_synthetic, eight_schools_ncp (S3 real panel)
      - irt_1pl, irt_2pl, stoch_vol, horseshoe        (S3 real panel)
      - neals_funnel  (d=10; v ~ N(0,9), theta_i|v ~ N(0,exp(v)); funnel panel)
        GT draws: catalog/neals_funnel/groundtruth_samples/blackjax/draws.npz
        (v: shape=(40000,), theta: shape=(40000, 9); both in unconstrained space
         since neals_funnel has no constrained transforms — v and theta are native
         unconstrained parameters).

Usage:
  python gt_imm.py        # runs self-check with correctness gates
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)

from blackjax.adaptation.mclmc_lrd_adaptation import _extract_lrd_from_samples
from blackjax.mcmc.metrics import LowRankInverseMassMatrix

# ---------------------------------------------------------------------------
# Ground-truth covariance helpers
# ---------------------------------------------------------------------------


def gt_cov(model_name: str) -> tuple:
    """Return (Sigma, Sigma_inv) for a named model.

    Parameters
    ----------
    model_name : str
        One of "ill_cond_50" or "mvn_10".

    Returns
    -------
    Sigma : np.ndarray, shape (d, d), float64
    Sigma_inv : np.ndarray, shape (d, d), float64
    """
    if model_name == "ill_cond_50":
        from tuningfork.model.ill_cond_50 import COV_NP

        Sigma = COV_NP.astype(np.float64)
        Sigma_inv = np.linalg.inv(Sigma)
        return Sigma, Sigma_inv

    elif model_name == "mvn_10":
        from tuningfork.model.mvn_10 import DIM

        Sigma = np.eye(DIM, dtype=np.float64)
        Sigma_inv = np.eye(DIM, dtype=np.float64)
        return Sigma, Sigma_inv

    else:
        raise ValueError(
            f"Unknown model: {model_name!r}. Expected 'ill_cond_50' or 'mvn_10'."
        )


# ---------------------------------------------------------------------------
# Real-model GT IMM from catalog draws (S3 extension)
# ---------------------------------------------------------------------------

# Default catalog root: tuningfork/tuningfork/catalog/
_CATALOG_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # experiments/mclmc_scaling/
    "..",  # experiments/
    "..",  # tuningfork/ (repo root)
    "tuningfork",
    "catalog",
)


def gt_from_draws(
    model_name: str,
    k: int | None = None,
) -> tuple:
    """Build GT dense (or LRD-k) IMM from committed catalog GT draws.

    Loads ``tuningfork/catalog/<model_name>/groundtruth_samples/blackjax/draws.npz``
    (unconstrained space — draws are NUTS states.position; confirmed unconstrained
    by e.g. eight_schools_ncp tau showing negative values from HalfCauchy transform).

    Ravels multi-site draws to ``(n_samples, d)`` using ``ravel_pytree`` with the
    ordering established by ``build_logdensity_fn``'s ``init_position`` dict.  Uses
    ``_extract_lrd_from_samples`` (the shipped LRD extractor) to build the IMM.

    Parameters
    ----------
    model_name : str
        Name of a registered real model, e.g. ``"german_credit"``,
        ``"eight_schools_ncp"``, ``"irt_2pl"``, etc.
        **Synthetic models** (``"ill_cond_50"``, ``"mvn_10"``) must use
        ``gt_cov`` + ``gt_lrd_imm`` instead (analytic Sigma available).
    k : int | None
        Rank of the LRD approximation.  Default ``None`` → dense (k = d).
        Use ``k = 0`` for diagonal-only (sigma only, U=(d,0), lam=(0,)).

    Returns
    -------
    imm : LowRankInverseMassMatrix
        GT IMM at the requested rank.
    gt_var : np.ndarray, shape (d,)
        Per-dimension variance of the unconstrained draws (used for bias metric
        in ``run_mclmc_fixed_imm`` for real models: |Var_mcmc - gt_var| / gt_var).
    gt_mean : np.ndarray, shape (d,)
        Per-dimension mean of the unconstrained draws (chain center for the
        real-model bias metric).
    d : int
        Unconstrained dimensionality of the model.

    Raises
    ------
    FileNotFoundError
        If the draws.npz file does not exist at the expected catalog path.
    ValueError
        If ``model_name`` is one of the synthetic models (use ``gt_cov`` instead).

    Notes
    -----
    Space confirmation:
        The draws are in UNCONSTRAINED space.  Evidence:
        1. ``certify_reference.py`` line 747: ``draws = states.position`` — NUTS
           positions are always in unconstrained space.
        2. Line 873-876 in certify_reference.py: ``postprocess_fn`` is applied to
           convert FROM unconstrained to constrained for posteriordb comparison.
        3. Empirical: eight_schools_ncp ``tau`` (HalfCauchy, positive-constrained)
           shows min=-7.74 in draws.npz — only possible in unconstrained (log) space.

    Ravel ordering:
        Multi-site draws (e.g. ``mu``, ``tau``, ``theta_raw`` for eight_schools_ncp)
        are ravelled using ``ravel_pytree`` applied to the init_position dict from
        ``build_logdensity_fn``.  The ordering is deterministic and consistent with
        the logdensity_fn input format used by the MCLMC sampler.
    """
    _SYNTHETIC = {"ill_cond_50", "mvn_10"}
    if model_name in _SYNTHETIC:
        raise ValueError(
            f"gt_from_draws: {model_name!r} is a synthetic model with an analytic "
            "covariance. Use gt_cov() + gt_lrd_imm() instead."
        )

    # Locate and load the committed GT draws
    draws_path = os.path.join(
        _CATALOG_ROOT, model_name, "groundtruth_samples", "blackjax", "draws.npz"
    )
    if not os.path.exists(draws_path):
        raise FileNotFoundError(
            f"GT draws not found for {model_name!r}. Expected: {draws_path}"
        )

    data = np.load(draws_path)
    draw_keys = list(data.files)

    # Get the unconstrained dimensionality and ravel ordering from the model
    # registry so the flat representation is consistent with logdensity_fn.
    from tuningfork.model._numpyro import build_logdensity_fn as _build_logdensity_fn
    from tuningfork.model._registry import MODELS

    entry = MODELS[model_name]
    # Use a deterministic seed for init (only needed for ravel ordering)
    _init_key = jax.random.key(0)
    init_pos, _logdensity_fn, _postprocess_fn = _build_logdensity_fn(_init_key, entry)

    d = entry.dim

    # Ravel all draws to (n_samples, d) using the init_pos key ordering
    # Handle float32 → float64 cast (GT draws are stored float32)
    n_samples_total = data[draw_keys[0]].shape[0]

    # Build a single sample to get the ravel function
    sample_0 = {}
    for site in draw_keys:
        arr = data[site]
        if arr.ndim == 1:
            sample_0[site] = jnp.array(arr[0], dtype=jnp.float64)
        else:
            sample_0[site] = jnp.array(arr[0], dtype=jnp.float64)

    flat_0, unravel_fn = ravel_pytree(sample_0)
    assert flat_0.shape[0] == d, (
        f"gt_from_draws: ravel dim mismatch for {model_name}: "
        f"got {flat_0.shape[0]}, expected {d} from entry.dim"
    )

    # Build batch position dict for vmap-ravel
    # Use float64 for numerical stability (GT draws are float32; cast here)
    batch_pos = {}
    for site in draw_keys:
        arr = data[site].astype(np.float64)
        batch_pos[site] = jnp.array(arr)

    # vmap ravel_pytree over n_samples
    flat_draws = jax.vmap(lambda pos: ravel_pytree(pos)[0])(batch_pos)
    # flat_draws: (n_samples_total, d)
    flat_draws_np = np.array(flat_draws, dtype=np.float64)

    # Compute GT statistics
    gt_mean = flat_draws_np.mean(axis=0)  # (d,)
    gt_var = flat_draws_np.var(axis=0)  # (d,) = E[(x - mean)^2]

    # Default k = d (dense)
    rank = k if k is not None else d

    # Use _extract_lrd_from_samples (the shipped extractor) to build the IMM
    flat_draws_jax = jnp.array(flat_draws_np)
    # NOTE: _extract_lrd_from_samples returns 4 values on the fixed blackjax branch
    # (fix/adjusted-mclmc-fix-L-order-bug): sigma, U_k, lam_k, lam_all_sorted.
    # The pinned blackjax (359205da) returned 3. Unpack 4, discard the 4th.
    sigma_arr, U_arr, lam_arr, _ = _extract_lrd_from_samples(flat_draws_jax, k=rank)

    imm = LowRankInverseMassMatrix(
        sigma=sigma_arr,
        U=U_arr,
        lam=lam_arr,
    )

    return imm, gt_var, gt_mean, d


# ---------------------------------------------------------------------------
# LRD extraction from exact Sigma (analytic mirror of _extract_lrd_from_samples)
# ---------------------------------------------------------------------------


def gt_lrd_imm(Sigma: np.ndarray, k: int) -> LowRankInverseMassMatrix:
    """Build GT LowRankInverseMassMatrix from the true covariance Sigma.

    Mirrors ``_extract_lrd_from_samples`` analytically:
      sigma = sqrt(diag Sigma)
      R = D^{-1/2} Sigma D^{-1/2}   (correlation matrix)
      Eigendecompose R (real symmetric, full eig)
      Select top-k by |lambda - 1|

    Parameters
    ----------
    Sigma : np.ndarray, shape (d, d), float64
        True covariance matrix.
    k : int
        Rank of the LRD approximation.
        k=0 -> diagonal only; k=d -> exact dense.

    Returns
    -------
    LowRankInverseMassMatrix(sigma, U, lam)
        sigma : (d,), positive diagonal scaling
        U     : (d, k), orthonormal columns
        lam   : (k,), eigenvalues
    """
    d = Sigma.shape[0]
    assert 0 <= k <= d, f"k must be in [0, d={d}], got {k}"

    # Step 1: marginal standard deviations
    sigma = np.sqrt(np.diag(Sigma))  # (d,)

    if k == 0:
        # Diagonal-only: U is (d, 0), lam is (0,)
        U = np.zeros((d, 0), dtype=np.float64)
        lam = np.zeros((0,), dtype=np.float64)
        return LowRankInverseMassMatrix(
            sigma=jnp.array(sigma),
            U=jnp.array(U),
            lam=jnp.array(lam),
        )

    # Step 2: correlation matrix R = D^{-1/2} Sigma D^{-1/2}
    inv_sigma = 1.0 / sigma  # (d,)
    R = inv_sigma[:, None] * Sigma * inv_sigma[None, :]  # (d, d)
    # Symmetrise to eliminate floating-point asymmetry
    R = (R + R.T) / 2.0

    # Step 3: eigendecomposition of the real symmetric correlation matrix
    # np.linalg.eigh returns eigenvalues in ascending order
    lam_all, V_all = np.linalg.eigh(R)  # lam_all: (d,), V_all: (d, d)

    # Step 4: select top-k by |lambda - 1| (deviations from isotropic)
    deviation = np.abs(lam_all - 1.0)
    sort_idx = np.argsort(deviation)[::-1]  # descending
    top_idx = sort_idx[:k]

    lam_k = lam_all[top_idx]  # (k,)
    U_k = V_all[:, top_idx]  # (d, k)

    return LowRankInverseMassMatrix(
        sigma=jnp.array(sigma),
        U=jnp.array(U_k),
        lam=jnp.array(lam_k),
    )


# ---------------------------------------------------------------------------
# Effective condition number
# ---------------------------------------------------------------------------


def kappa_eff(Sigma: np.ndarray, imm: LowRankInverseMassMatrix) -> float:
    """Effective condition number of the IMM-preconditioned curvature.

    For HMC/MCLMC with inverse mass matrix M^{-1}, convergence is governed
    by the eigenvalues of M^{-1} H where H = Sigma^{-1} is the Hessian of
    -log pi (for a Gaussian target with covariance Sigma).

    kappa_eff = kappa(M^{-1} Sigma^{-1})

    When M^{-1} = Sigma (perfect preconditioning): M^{-1} Sigma^{-1} = I,
    so kappa_eff = 1.0.

    For k=0 diagonal (M^{-1} = diag(sigma^2)):
      eigenvalues of diag(sigma^2) Sigma^{-1} are the eigenvalues of
      the correlation matrix R = D^{-1} Sigma D^{-1}.
      kappa_eff = kappa(R) approx 863 for ill_cond_50.

    Implementation note: M^{-1} Sigma^{-1} is not symmetric in general.
    We use np.linalg.eigvals and take real parts (eigenvalues are real and
    positive for PD M^{-1} and PD Sigma since the product is similar to a
    symmetric PD matrix via M^{-1} Sigma^{-1} ~ M^{-1/2} Sigma^{-1} M^{-1/2}).

    Parameters
    ----------
    Sigma : np.ndarray, shape (d, d), float64
    imm : LowRankInverseMassMatrix

    Returns
    -------
    float : kappa_eff = kappa(M^{-1} Sigma^{-1})
    """
    d = Sigma.shape[0]
    sigma = np.array(imm.sigma, dtype=np.float64)  # (d,)
    U = np.array(imm.U, dtype=np.float64)  # (d, k)
    lam = np.array(imm.lam, dtype=np.float64)  # (k,)

    Sigma_inv = np.linalg.inv(Sigma)

    # Build M^{-1} = D (I + U(Lambda-I)U^T) D  where D = diag(sigma)
    D = np.diag(sigma)  # (d, d)
    if U.shape[1] == 0:
        # k=0: M^{-1} = diag(sigma^2)
        M_inv = D @ D
    else:
        correction = U @ np.diag(lam - 1.0) @ U.T  # (d, d)
        M_inv = D @ (np.eye(d) + correction) @ D

    # kappa(M^{-1} Sigma^{-1})
    product = M_inv @ Sigma_inv
    eigvals = np.linalg.eigvals(product)
    eigvals_real = np.real(eigvals)  # all real since M^{-1} Sigma^{-1} ~ sym PD
    kappa = float(eigvals_real.max() / eigvals_real.min())
    return kappa


# ---------------------------------------------------------------------------
# Gradient check helper
# ---------------------------------------------------------------------------


def gradient_check(
    logdensity_fn,
    test_positions: list,
    *,
    tol: float = 1e-5,
    desc: str = "",
) -> None:
    """Assert that the log-density gradient matches -x (standard normal in whitened space).

    For a correctly whitened Gaussian N(0, I), the log-density is
    logpi(z) = -1/2 ||z||^2 + const, so d logpi/dz = -z, i.e. grad + z approx 0.

    Parameters
    ----------
    logdensity_fn : callable
        Log-density in the whitened (z) space.
    test_positions : list of arrays
        Test positions in whitened space.
    tol : float
        Tolerance for max|grad + z|.
    desc : str
        Description for error messages.
    """
    for z in test_positions:
        z = jnp.array(z, dtype=jnp.float64)
        grad = jax.grad(logdensity_fn)(z)
        residual = jnp.max(jnp.abs(grad + z))
        if float(residual) > tol:
            raise AssertionError(
                f"gradient_check FAILED ({desc}): max|d logpi/dz + z| = "
                f"{float(residual):.2e} > tol={tol}"
            )
    print(f"  gradient_check OK ({desc}): max residual < {tol}")


# ---------------------------------------------------------------------------
# kappa_eff table helper
# ---------------------------------------------------------------------------


def kappa_eff_table(model_name: str, k_values=None) -> list:
    """Compute kappa_eff(k) for a range of ranks.

    Parameters
    ----------
    model_name : str
    k_values : list[int] | None
        If None, uses a default grid of ~10 points from 0 to d.

    Returns
    -------
    list of dicts with keys 'k', 'kappa_eff', 'structure'
    """
    Sigma, _ = gt_cov(model_name)
    d = Sigma.shape[0]

    if k_values is None:
        import math

        if d <= 10:
            k_values = list(range(d + 1))
        else:
            log_pts = np.logspace(0, math.log10(d), 8).astype(int)
            k_values = sorted(set([0] + list(log_pts) + [d]))

    rows = []
    for k in k_values:
        imm = gt_lrd_imm(Sigma, k)
        keff = kappa_eff(Sigma, imm)
        if k == 0:
            structure = "diagonal"
        elif k == d:
            structure = "dense"
        else:
            structure = f"low-rank(k={k})"
        rows.append({"k": k, "kappa_eff": keff, "structure": structure})
    return rows


# ---------------------------------------------------------------------------
# Self-check (correctness gates)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("gt_imm.py self-check -- correctness gates")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Gate 1: ill_cond_50
    # ----------------------------------------------------------------
    print("\n--- ill_cond_50 (d=50, kappa=1000) ---")
    Sigma_ic, Sigma_inv_ic = gt_cov("ill_cond_50")
    d_ic = Sigma_ic.shape[0]
    print(f"  d = {d_ic}, true kappa(Sigma) = {float(np.linalg.cond(Sigma_ic)):.1f}")

    # k=0 -> diagonal-only
    imm_diag = gt_lrd_imm(Sigma_ic, k=0)
    keff_diag = kappa_eff(Sigma_ic, imm_diag)
    print(f"  k=0  (diagonal): kappa_eff = {keff_diag:.3f}  (expected approx 863)")
    assert (
        abs(keff_diag - 863) < 20
    ), f"GATE FAILED: k=0 ill_cond_50 kappa_eff={keff_diag:.1f}, expected approx 863 (+-20)"
    print("  GATE PASS: k=0 kappa_eff approx 863")

    # k=d -> dense (should be exactly 1)
    imm_dense = gt_lrd_imm(Sigma_ic, k=d_ic)
    keff_dense = kappa_eff(Sigma_ic, imm_dense)
    print(f"  k=d  (dense):    kappa_eff = {keff_dense:.6f}  (expected = 1.0)")
    assert (
        abs(keff_dense - 1.0) < 1e-4
    ), f"GATE FAILED: k=d ill_cond_50 kappa_eff={keff_dense:.6f}, expected = 1.0 (+-1e-4)"
    print("  GATE PASS: k=d kappa_eff = 1.0")

    # ----------------------------------------------------------------
    # Gate 2: mvn_10 (isotropic, all k should give kappa_eff approx 1)
    # ----------------------------------------------------------------
    print("\n--- mvn_10 (d=10, isotropic Sigma=I) ---")
    Sigma_mv, Sigma_inv_mv = gt_cov("mvn_10")
    d_mv = Sigma_mv.shape[0]

    for k in [0, 5, d_mv]:
        imm_mv = gt_lrd_imm(Sigma_mv, k)
        keff_mv = kappa_eff(Sigma_mv, imm_mv)
        print(f"  k={k:2d}: kappa_eff = {keff_mv:.6f}  (expected approx 1.0)")
        assert (
            abs(keff_mv - 1.0) < 1e-6
        ), f"GATE FAILED: mvn_10 k={k} kappa_eff={keff_mv:.8f}, expected = 1.0 (+-1e-6)"
    print("  GATE PASS: mvn_10 all k -> kappa_eff approx 1.0")

    # ----------------------------------------------------------------
    # Gate 3: gradient check
    # ----------------------------------------------------------------
    print("\n--- gradient check ---")
    rng = np.random.default_rng(0)

    # Standard normal log-density: grad = -z (sanity check)
    def std_normal_logdens(x):
        return -0.5 * jnp.dot(x, x)

    test_zs_mv = [
        jnp.array(rng.standard_normal(d_mv), dtype=jnp.float64) for _ in range(3)
    ]
    gradient_check(std_normal_logdens, test_zs_mv, desc="std normal (sanity)")

    # Cholesky whitening of ill_cond_50: x = L z  =>  logpi_w(z) = -1/2 z^T z
    L_ic = np.linalg.cholesky(Sigma_ic)
    L_ic_jax = jnp.array(L_ic, dtype=jnp.float64)

    def logpi_chol_whitened(z):
        # logpi(Lz) = -1/2 (Lz)^T Sigma^{-1} (Lz) = -1/2 z^T z
        return -0.5 * jnp.dot(z, z)

    test_zs_ic = [
        jnp.array(rng.standard_normal(d_ic), dtype=jnp.float64) for _ in range(3)
    ]
    gradient_check(
        logpi_chol_whitened, test_zs_ic, desc="Cholesky-whitened ill_cond_50"
    )

    print("\n" + "=" * 60)
    print("ALL CORRECTNESS GATES PASSED")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Print kappa_eff tables
    # ----------------------------------------------------------------
    print("\n--- kappa_eff(k) table: ill_cond_50 ---")
    rows_ic = kappa_eff_table("ill_cond_50")
    print(f"  {'k':>4}  {'structure':<20}  {'kappa_eff':>10}")
    print(f"  {'-'*4}  {'-'*20}  {'-'*10}")
    for r in rows_ic:
        print(f"  {r['k']:>4}  {r['structure']:<20}  {r['kappa_eff']:>10.4f}")

    print("\n--- kappa_eff(k) table: mvn_10 ---")
    rows_mv = kappa_eff_table("mvn_10")
    print(f"  {'k':>4}  {'structure':<20}  {'kappa_eff':>10}")
    print(f"  {'-'*4}  {'-'*20}  {'-'*10}")
    for r in rows_mv:
        print(f"  {r['k']:>4}  {r['structure']:<20}  {r['kappa_eff']:>10.4f}")

    sys.exit(0)
