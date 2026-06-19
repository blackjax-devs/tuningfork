# vmap multichain parity gate — patch spec

**Status:** USER-ACCEPTED (policy). For SWE to land in the multichain harness.
**Author:** statistician. **Provenance:** #25 banana diagnosis, git_head `8937e088`.
**Reference implementation:** `experiments/mclmc_scaling/cert25_banana_vmap.py` (gate block).
**Evidence:** `diag_vmap_banana_p1.py` + `_p1_results.json`, `diag_vmap_banana_p1b.py` +
`_p1b_results.json`, `cert25_banana_vmap_results.json` (all float64, on disk).

> NOTE — this file was rewritten 2026-06-19 to replace an earlier draft that carried
> placeholder/wrong numbers (λ≈0.17, KS p≥0.167, "acc Δ≤9.5e-4", and an **absolute 5e-3
> acceptance bar**). Those figures are NOT from any run; the absolute acc bar is
> mis-calibrated (see B.3). Use the numbers below — they are the measured values.

## Motivation

Bit-exact position parity (loop vs vmap) is the **wrong acceptance bar for a chaotic
sampler**. Measured on banana × adjusted_mclmc_dynamic:

- The fp seed is pure rounding, **not** algorithmic/key: vmap-batched grad & logdensity are
  **bit-identical** to scalar even over distinct lane inputs (|Δ|=0, all lanes); per-chain
  keys are bit-identical (loop keys == stacked vmap keys, all 24 chain-keys).
- That ~1e-15 seed **amplifies exponentially** along multi-step trajectories (positive
  Lyapunov). Measured real-sampler chain-0 divergence curve, avg=18, step 0.21126:
  |Δ| = 1.78e-15 (sample 0) → 4.9e-14 (s3) → 6.1e-12 (s10) → 5.2e-7 (s20) → 9.36 (s50,
  saturated). **log-slope λ ≈ 0.106/sample**, half-saturates by **sample ~41**, saturates
  ~√maxvar ≈ 3.0. **avg=1 control: no amplification** (stays ~1e-16, λ≈0.009). So bitwise
  divergence is expected, benign, and trajectory-length-driven.
- The **sampled distribution is identical**: KS not rejected (real min p = 0.0325, max D =
  0.0185 across 12 seed×dim tests; 2 cells dip <0.05 as expected under the null with 12
  comparisons); num_integration_steps exact; divergences equal; mean acceptance agrees to
  ≤8.1e-3 (within MC noise — see B.3).

A loop fallback on bit-parity failure would fire on every real run and **mask genuine bugs**
behind "fell back to loop". Remove the fallback; gate on statistical equivalence.

## The two-part gate

Run once at harness startup on a reference cell (loop vs vmap, identical per-chain keys).

### Gate A — structural (catch real codegen/key bugs before chaos masks them)
1. **Key identity:** `jax.random.key_data(loop_keys)` == `jax.random.key_data(stacked_vmap_keys)`, bitwise. Else **BLOCK_BUG**.
2. **Sample-1 micro-parity:** run **1** sample loop vs vmap; require `max|Δposition| < 1e-10`.
   (At n=1 chaos has not amplified — banana measured 3.72e-15. A structural bug shows here.)
   Else **BLOCK_BUG**.

### Gate B — statistical equivalence (tolerate chaotic fp amplification)
On a reference cell of `n_ref` samples (recommend n_ref ≥ 2000), all chains:
1. `num_integration_steps` identical (Δ == 0). *RNG-driven; must match exactly.* Else **BLOCK_BUG**.
2. divergence counts equal (Δ == 0). Else **BLOCK_BUG**.
3. **acceptance — STATISTICAL bar (NOT a fixed absolute):** `|Δ mean_acc| < K_SE · SE`,
   `SE = sqrt(acc·(1−acc)/n_ref)`, `K_SE = 4`.
   - *Why:* at n_ref=2000, acc≈0.95 → SE≈4.9e-3, so the loop-vs-vmap acc difference scale is
     ~√2·SE≈6.9e-3 (measured 8.1e-3). A fixed **5e-3** bar sits **below the estimator's own
     noise floor** → trips on Monte Carlo noise, not drift. (This exact mistake aborted the
     first re-cert; recalibrating to 4·SE≈1.95e-2 passed, KS confirming same distribution.)
     K_SE=4 tolerates MC spread while still blocking real bugs (which shift acc by O(0.1)).
   Else **BLOCK_BUG**.
4. **marginals:** per dim, `KS p > 0.05` **OR** `KS D < 0.05`.
   Moment fallback (no scipy): `|Δmean|/sd < 0.05` AND `|Δvar|/var < 0.10`. Else **BLOCK_BUG**.

### Decision rule (NO loop fallback)
- **A pass AND B pass** → `VMAP_OK`: vmap is the canonical multichain path. Proceed.
- **A pass, B fail** → `BLOCK_BUG`: distributional drift is a real defect. Investigate; do not ship, do not fall back.
- **A fail** → `BLOCK_BUG`: structural bug (keys / codegen). Investigate.

A fallback would convert a real bug into a silent slow-path; we want it surfaced.

## Reference implementation (pseudocode)

```python
def multichain_parity_gate(sample_loop, sample_vmap, keys_loop, keys_stacked, *,
                           n_ref=2000, ks_p=0.05, ks_D=0.05, K_SE=4.0,
                           micro_tol=1e-10):
    # (A.1) key identity
    if not array_equal(key_data(stack(keys_loop)), key_data(keys_stacked)):
        return "BLOCK_BUG"
    # (A.2) sample-1 micro-parity
    a1L, *_ = sample_loop(n=1); a1V, *_ = sample_vmap(n=1)
    if max_abs(a1L - a1V) >= micro_tol:
        return "BLOCK_BUG"
    # reference cell
    aL, dL, acL, nsL = sample_loop(n=n_ref)   # (ch, n, d), div, acc, nsteps
    aV, dV, acV, nsV = sample_vmap(n=n_ref)
    if abs(mean(nsL) - mean(nsV)) != 0.0:  return "BLOCK_BUG"   # nsteps exact
    if abs(mean(dL)  - mean(dV))  != 0.0:  return "BLOCK_BUG"   # div equal
    acc = 0.5*(mean(acL) + mean(acV)); se = sqrt(acc*(1-acc)/n_ref)
    if abs(mean(acL) - mean(acV)) >= K_SE*se:  return "BLOCK_BUG"   # statistical acc bar
    for j in range(d):                                              # marginals
        D, p = ks_2samp(aL[...,j].ravel(), aV[...,j].ravel())
        if not (p > ks_p or D < ks_D):  return "BLOCK_BUG"
    return "VMAP_OK"
```

## Injected-bug test (must return `BLOCK_BUG`)

Inject each into the **vmap path only**, run the gate, assert it returns `BLOCK_BUG`. Also
assert the clean vmap path returns `VMAP_OK` (so the gate can't pass by always-blocking):

| Injected bug                              | Gate part that catches it             |
|-------------------------------------------|---------------------------------------|
| step_size × 1.05 on vmap path             | B.acc (≫4·SE) and/or B.KS             |
| inverse_mass_matrix × 1.5                  | B.KS (marginal var drifts) + B.acc    |
| wrong RNG key (seed+1) on vmap chains      | A.key-identity → BLOCK_BUG            |
| off-by-one / dropped half-step integrator  | B.nsteps (Δ≠0) or A.micro → BLOCK_BUG |
| sign flip on a gradient component          | A.micro (1-sample) → BLOCK_BUG        |

Verified by `inject_bug_gate_check.py` (sandbox): clean→VMAP_OK, every injected bug→BLOCK_BUG.

## Notes for SWE
- Thresholds (`K_SE=4`, micro `1e-10`, KS `p>0.05`/`D<0.05`, n_ref≥2000) calibrated on banana
  (acc~0.95, d=2). The acc bar `K·SE` is model-agnostic by construction (SE scales with acc,n);
  for very-low-acc or tiny-n_ref targets it auto-widens correctly. Ping @statistician (via
  @tl) for per-model threshold questions.
- **The acc bar MUST be the statistical `K·SE` form, not a hard-coded constant** — that was
  the single calibration error in this work and is the most important detail to get right.
