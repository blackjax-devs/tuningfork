---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# bjx-bench Phase 1 Demo: Tier-A Reference Generation

`bjx-bench` is a BlackJAX-native benchmark library for MCMC/VI/SMC samplers. Its
goal is a reproducible, certified reference suite for 14 posterior models — from
simple Gaussians to Lotka-Volterra ODEs — together with a fair comparison harness
using `min-bulk-ESS / total_grad_evals` as the headline metric. The full design lives
in [`../PLAN_bjx_bench.md`](../PLAN_bjx_bench.md); the Phase 1 API specification
is in [`../PLAN_bjx_bench_API.md`](../PLAN_bjx_bench_API.md).

This notebook walks through Tier-A reference generation for the three Phase 1 starter
models: MVN-10 (analytic), Neal's Funnel (analytic), and 8-Schools NCP (long-NUTS).
It also demonstrates the cache-hit semantics that make repeated calls instant, and
closes with a NOTES section answering six API-design open questions from §9 of the
API plan.

```{code-cell} ipython3
:tags: [hide-cell]

import os
import time
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Point the cache at a fresh temp dir so this notebook is self-contained
# and does not pollute the committed reference directory.
_DEMO_CACHE = Path(tempfile.mkdtemp(prefix="bjx_bench_demo_"))
os.environ["BJX_BENCH_REFERENCE_DIR"] = str(_DEMO_CACHE)

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["figure.figsize"] = (7, 4)
```

```{code-cell} ipython3
from bjx_bench.registry import REGISTRY
from bjx_bench.reference._io import get_reference_draws, get_reference_summaries

print("Registry models:", list(REGISTRY.keys()))
```

---

## 1. MVN-10 (analytic Path A)

The 10-dimensional standard Gaussian is the sanity baseline.  Because it has a
closed-form sampler, `get_reference_draws` follows Path A: it calls
`ENTRY.analytic_sampler(rng_key, n)` and returns i.i.d. exact draws instantly,
with no MCMC required.

```{code-cell} ipython3
entry_mvn = REGISTRY["mvn_10"]
print(f"Model: {entry_mvn.name}")
print(f"  dim={entry_mvn.dim}, class={entry_mvn.class_}")
print(f"  reference_method={entry_mvn.reference_method.value}")
```

```{code-cell} ipython3
key_mvn = jax.random.key(0)

t0 = time.perf_counter()
draws_mvn = get_reference_draws(entry_mvn, n=10_000, rng_key=key_mvn,
                                cache_dir=_DEMO_CACHE)
t1 = time.perf_counter()
print(f"Generated {draws_mvn['x'].shape[0]:,} draws in {t1-t0:.2f}s")
print(f"  site 'x' shape: {draws_mvn['x'].shape}")  # (10000, 10)
```

We overlay the empirical density of `x[0]` (the first component) against the
analytic `N(0, 1)`:

```{code-cell} ipython3
x0 = np.asarray(draws_mvn["x"][:, 0])
x_grid = np.linspace(-4, 4, 300)
analytic_pdf = np.exp(-0.5 * x_grid**2) / np.sqrt(2 * np.pi)

fig, ax = plt.subplots()
ax.hist(x0, bins=60, density=True, alpha=0.6, label="empirical x[0]", color="steelblue")
ax.plot(x_grid, analytic_pdf, "r-", lw=2, label="N(0,1) analytic")
ax.set_xlabel("x[0]")
ax.set_ylabel("density")
ax.set_title("MVN-10: empirical vs analytic marginal")
ax.legend()
plt.tight_layout()
plt.show()

summaries_mvn = get_reference_summaries(entry_mvn, cache_dir=_DEMO_CACHE)
print(f"Summary  mean={float(summaries_mvn.mean['x'][0]):.4f}  "
      f"std={float(summaries_mvn.std['x'][0]):.4f}  "
      f"(analytic: mean=0, std=1)")
```

---

## 2. Neal's Funnel (analytic Path A)

Neal's Funnel is a 10-dimensional hierarchical distribution notorious for its
geometry: the marginal variance of the `theta` variables varies by orders of
magnitude depending on `v`.  Here it is the reference sampler, not the sampler
under test, so we can afford to sample it exactly.

```{code-cell} ipython3
entry_funnel = REGISTRY["neals_funnel"]
print(f"Model: {entry_funnel.name}")
print(f"  dim={entry_funnel.dim}, class={entry_funnel.class_}")
print(f"  reference_method={entry_funnel.reference_method.value}")
```

```{code-cell} ipython3
key_funnel = jax.random.key(1)

draws_funnel = get_reference_draws(entry_funnel, n=10_000, rng_key=key_funnel,
                                   cache_dir=_DEMO_CACHE)
print("Sites:", {k: v.shape for k, v in draws_funnel.items()})
```

The scatter plot of `(v, theta[0])` reveals the funnel geometry: when `v` is
large and positive, `theta` values fan out widely; when `v` is small, they
cluster near zero.

```{code-cell} ipython3
v = np.asarray(draws_funnel["v"])
theta0 = np.asarray(draws_funnel["theta"][:, 0])

fig, ax = plt.subplots()
ax.scatter(v, theta0, s=2, alpha=0.3, color="steelblue")
ax.set_xlabel("v  (log-scale factor)")
ax.set_ylabel("theta[0]")
ax.set_title("Neal's Funnel: (v, theta[0]) scatter")
plt.tight_layout()
plt.show()

summaries_funnel = get_reference_summaries(entry_funnel, cache_dir=_DEMO_CACHE)
print(f"v:       mean={float(summaries_funnel.mean['v']):.3f}  "
      f"std={float(summaries_funnel.std['v']):.3f}  (analytic: 0, 3)")
print(f"theta[0]: mean={float(summaries_funnel.mean['theta'][0]):.3f}  "
      f"std={float(summaries_funnel.std['theta'][0]):.3f}")
```

---

## 3. 8-Schools NCP (long-NUTS Path B)

The 8-Schools model has no closed-form marginals, so Tier-A uses Path B: a long
single-chain NUTS run with Stan window adaptation, followed by a certification gate
(split-R̂ ≤ 1.01, min per-chunk bulk-ESS ≥ 400, 0 divergences, E-BFMI ≥ 0.3).

We use small parameters here so the notebook finishes in under 2 minutes on CPU:
`n_warmup=500`, `n_samples=4000`, `n_chunks=4`.  The full production run uses
`n_warmup=5000`, `n_samples=100000`, `n_chunks=10`.

```{code-cell} ipython3
entry_schools = REGISTRY["eight_schools_ncp"]
print(f"Model: {entry_schools.name}")
print(f"  dim={entry_schools.dim}, class={entry_schools.class_}")
print(f"  reference_method={entry_schools.reference_method.value}")
```

```{code-cell} ipython3
from bjx_bench.calibration.tier_a import certify_reference_nuts

key_schools = jax.random.key(42)

t0 = time.perf_counter()
draws_schools, summaries_schools, adaptation, cert = certify_reference_nuts(
    entry_schools,
    key_schools,
    n_warmup=500,
    n_samples=4000,
    n_chunks=4,
)
t1 = time.perf_counter()

print(f"NUTS run finished in {t1-t0:.1f}s")
print(f"\nCertification result:")
print(f"  passed          = {cert.passed}")
print(f"  split_rhat_max  = {cert.split_rhat_max:.4f}  (gate: <= 1.01)")
print(f"  min_chunk_ess   = {cert.min_chunk_bulk_ess:.1f}  (gate: >= 400)")
print(f"  num_divergences = {cert.num_divergences}         (gate: == 0)")
print(f"  e_bfmi          = {cert.e_bfmi:.4f}  (gate: >= 0.3)")
print(f"\nAdaptation: step_size={adaptation.step_size:.4f}, "
      f"num_leapfrog_median={adaptation.num_leapfrog_median}")
```

The draws are in **unconstrained** space. To recover the constrained `mu` and
`tau` we use the `postprocess_fn` from `build_logdensity_fn`:

```{code-cell} ipython3
from bjx_bench.registry._numpyro import build_logdensity_fn

key_pp = jax.random.key(99)
_, _, postprocess_fn = build_logdensity_fn(key_pp, entry_schools)

# Apply postprocess_fn to one draw to inspect the constrained sites
sample_constrained = postprocess_fn(
    {k: v[0] for k, v in draws_schools.items()}
)
print("Constrained sample sites:", list(sample_constrained.keys()))
print("  mu =", float(sample_constrained["mu"]))
print("  tau=", float(sample_constrained["tau"]))
```

```{code-cell} ipython3
# Plot mu and tau marginals from unconstrained draws
# tau is stored in log space (HalfCauchy uses softplus internally);
# apply postprocess_fn to all draws via vmap
postprocess_batch = jax.vmap(postprocess_fn)
constrained_draws = postprocess_batch(draws_schools)

mu_arr = np.asarray(constrained_draws["mu"])
tau_arr = np.asarray(constrained_draws["tau"])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(mu_arr, bins=50, density=True, color="steelblue", alpha=0.7)
axes[0].set_xlabel("mu (constrained)")
axes[0].set_ylabel("density")
axes[0].set_title("8-Schools NCP: mu marginal")

axes[1].hist(tau_arr, bins=50, density=True, color="coral", alpha=0.7)
axes[1].set_xlabel("tau (constrained)")
axes[1].set_ylabel("density")
axes[1].set_title("8-Schools NCP: tau marginal")

plt.tight_layout()
plt.show()

print(f"mu:  mean={mu_arr.mean():.2f}, std={mu_arr.std():.2f}")
print(f"tau: mean={tau_arr.mean():.2f}, std={tau_arr.std():.2f}")
```

---

## 4. Cache-hit demo

The load-or-generate semantics mean that a second call to `get_reference_draws`
with a valid cache is dramatically faster than the first (no sampling needed).

```{code-cell} ipython3
# First call: populates the cache (already done above, but let's re-time it)
key_cache = jax.random.key(0)

t0 = time.perf_counter()
_ = get_reference_draws(entry_mvn, n=10_000, rng_key=key_cache,
                        cache_dir=_DEMO_CACHE)
t1 = time.perf_counter()
time_cold = t1 - t0

# Second call: cache hit — just loads the npz
t2 = time.perf_counter()
_ = get_reference_draws(entry_mvn, n=10_000, rng_key=key_cache,
                        cache_dir=_DEMO_CACHE)
t3 = time.perf_counter()
time_warm = t3 - t2

print(f"Cold (generate + write): {time_cold*1000:.1f} ms")
print(f"Warm (cache hit):        {time_warm*1000:.1f} ms")
if time_warm > 0:
    print(f"Speedup:                 ~{time_cold/time_warm:.0f}x")
```

The stamp written to `metadata/<name>.json` records the `bjx_bench` version, the
repo git SHA, the generator path, and the certification result. Any version bump
invalidates all caches (conservative Phase 1 policy; Phase 2 may add per-model SHA
tracking).

---

## 5. NOTES — Six empirical open questions (API plan §9)

These were deferred to the SWE agent for verification by reading source code,
not guessing.

### Q1. MALA grad cost per step

Source: `/blackjax/blackjax/mcmc/mala.py:build_kernel`.

Inside `kernel(rng_key, state, logdensity_fn, step_size)`:
1. `grad_fn = jax.value_and_grad(logdensity_fn)` — defines the function, no call.
2. `integrator = diffusions.overdamped_langevin(grad_fn)` — builds the one-step function.
3. `new_state = integrator(key_integrator, state, step_size)` — inside
   `overdamped_langevin.one_step`, `logdensity_grad_fn(position)` is called exactly
   **once** (at the proposed position).
4. `compute_acceptance_ratio(state, new_state, step_size=step_size)` — uses
   `new_state.logdensity_grad` (cached from step 3) and `state.logdensity` (cached
   in `MALAState`); **no additional grad call**.

**Conclusion: 1 grad eval per MALA step.**

The previous-position grad stored in `MALAState.logdensity_grad` is used in the
reverse-energy term of the MH ratio; it is not recomputed each step.

### Q2. MCLMC grad cost per step

Source: `/blackjax/blackjax/mcmc/mclmc.py:build_kernel` and
`/blackjax/blackjax/mcmc/integrators.py`.

The MCLMC kernel calls `with_isokinetic_maruyama(integrator(...))`, which performs
one partial momentum refresh, then **one call** to the integrator, then another
partial refresh. No additional grad evals in the refreshment steps (they operate
on momentum only).

The default integrator is `isokinetic_mclachlan` with coefficients
`[b1, a1, b2, a1, b1]` (5 terms). The `generalized_two_stage_integrator` alternates
between the momentum update (operator1) and the position update (operator2). With
5 alternating coefficients starting at momentum, the sequence is:

    momentum(b1) → position(a1) → momentum(b2) → position(a1) → momentum(b1)

That is **2 position updates**, each invoking `euclidean_position_update_fn` which
calls `jax.value_and_grad(logdensity_fn)` once.

**Conclusion: 2 grad evals per MCLMC step (isokinetic_mclachlan default).**

For other integrators: `isokinetic_velocity_verlet` uses `[0.5, 1.0, 0.5]` → 1
position update → 1 grad; `isokinetic_yoshida` uses 7 coefficients → 3 position
updates → 3 grads; `isokinetic_omelyan` uses 11 coefficients → 5 position updates
→ 5 grads. The formula is: **number of odd-index (1-based) coefficients** in the
palindromic sequence.

### Q3. BarkerInfo content

Source: `/blackjax/blackjax/mcmc/barker.py`.

```python
class BarkerInfo(NamedTuple):
    acceptance_rate: float
    is_accepted: bool
    proposal: BarkerState
```

Fields:
- `acceptance_rate` — MH acceptance probability for the proposed move.
- `is_accepted` — boolean flag; True when the proposal was accepted.
- `proposal` — the full `BarkerState` (position, logdensity, logdensity_grad) of
  the proposed point, regardless of acceptance.

There is **no per-step counter** (e.g. no `num_integration_steps`). For Phase 2
grad accounting, Barker always costs exactly **1 grad per step** (called at the
proposed position via `grad_fn(proposed_pos)` inside `build_kernel`), similar to
MALA. The grad cost is fixed and does not require reading `info`.

### Q4. PyTree shape after `run_inference_algorithm`

Confirmed empirically in `tests/test_tier_a_nuts.py` and by reading
`bjx_bench/calibration/tier_a.py`.

After:
```python
final_state, (states, infos) = run_inference_algorithm(
    rng_key=rng_key_sample,
    inference_algorithm=nuts,
    num_steps=n_samples,
    initial_state=adapted_state,
)
```

`states.position` is a `dict[site_name, jax.Array]` where each array has shape
`(n_samples, *site_shape)`. For the 8-Schools NCP model with NumPyro sites
`{mu, tau, theta_raw}`:

- `states.position["mu"]` has shape `(n_samples,)` — scalar site.
- `states.position["tau"]` has shape `(n_samples,)` — scalar site, softplus-transformed.
- `states.position["theta_raw"]` has shape `(n_samples, 8)` — vector site.

The leading `n_samples` axis comes from `jax.lax.scan`'s accumulation of per-step
outputs. Dict keys are preserved through scan because JAX treats dicts as PyTrees.

This shape is verified in `tests/test_tier_a_nuts.py:TestCertifyNutsInterface.test_draws_sample_axis`.

### Q5. Cache invalidation granularity

Phase 1 takes the **conservative approach**: any `bjx_bench` version bump
invalidates ALL cached artifacts, not just those whose model code changed.

The check is in `bjx_bench/reference/_io.py:_cache_is_valid`:

```python
if meta.get("bjx_bench_version") != current_version:
    return False
if meta.get("code_sha") != current_sha:
    return False
```

Both conditions must match (version string AND git HEAD SHA). This means:

- A pip reinstall at the same version is fine (SHA unchanged).
- A version bump in `pyproject.toml` immediately invalidates all caches.
- An uncommitted code change also invalidates caches (SHA changes after commit).

This is intentionally conservative. Phase 2 may add per-model content-hash tracking
(hash the model module source) to allow unrelated model changes to preserve each
other's caches.

### Q6. Concurrent-writer safety

Phase 1 **assumes no concurrent `bjx-bench tier-a <model>` invocations on the same
model**. The assumption is documented in the module docstring of
`bjx_bench/reference/_io.py`.

`_atomic_write_npz` and `_atomic_write_json` both write to a temporary file and
then call `tmp.replace(path)` (a POSIX atomic rename). A single writer is therefore
safe: readers always see either the old complete file or the new complete file.

However, two concurrent writers on the same model could both generate, then race on
the rename. The last writer wins (POSIX rename is atomic but not serialized between
two processes). The metadata and draws files could end up from different writers,
creating a mismatch — e.g. draws from writer A with metadata (including cert) from
writer B.

For Phase 1 (single-user, local-disk benchmarks) this is acceptable. Phase 2
should add a `.lock` file (e.g. via `fcntl.flock`) around the generate-and-write
block if parallel `bjx-bench` invocations become part of the workflow.
