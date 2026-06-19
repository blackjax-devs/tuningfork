# #22 SCOPING — Adaptive / position-dependent L for adjusted MCLMC on stiff geometry

**Author:** @statistician · **Status:** scoping writeup only (no recipe/code changes) · **Date:** 2026-06-19
**Grounding data:** `sweep_dynl_variety_results.json` + `sweep_stiff_k_results.json`
(origin/evidence/mclmc-cert25-statistician @ 9da88e1), recomputed per-seed.

---

## 1. The question

The dynamic-L ladder shows three *geometry-opposite* optima for a single global `avg`:

| model | optimal global L | trend with longer L | acceptance vs L |
|---|---|---|---|
| ill_cond_50 (anisotropic Gaussian) | **SHORT** (avg=2) | bias ↑ 0.18→1.0, ESS ↓ 1928→382 | **flat ~0.93–0.95** |
| banana (mild curvature) | **INTERMEDIATE** (avg=18–54) | U-shaped bias min, then overshoot | smooth decline 0.99→0.72 |
| horseshoe (funnel) | **never passes** (REVIEW plateau) | bias ↓ 0.95→0.33 monotone, never <0.29 | **erratic 0.64–0.88** |

Scope: is there a *tractable warmup signal* that could drive a position- or phase-dependent L
instead of one global `avg` — and is it worth a sweep?

## 2. The discriminating signal is already free: acceptance variability

The per-seed acceptance column is the cleanest free discriminator of *which geometries can
even benefit* from adaptive L:

- **ill_cond_50** — acceptance is flat (~0.94) across *all* L and seeds. Flat acceptance ⇒
  curvature is **position-independent** (it's a Gaussian; the Hessian is constant). A single
  global L is information-theoretically sufficient; the only question is cost, and short L wins.
  The rising 2nd-moment bias at long L is **integration drift along the stiff direction**
  (ε too large for the smallest length-scale, accumulated over many steps), not an L-scheduling
  problem. **Adaptive L cannot help here.**
- **horseshoe** — acceptance swings 0.64–0.88 across seeds (and, by the funnel's nature, across
  draws *within* a run). Erratic acceptance is the signature of **position-dependent curvature**:
  a single (step, L) is simultaneously too long for the neck and too short for the mouth. This is
  the **only archetype where adaptive/position-dependent L has theoretical headroom.**
- **banana** — smooth monotone acceptance decline ⇒ mild, slowly-varying curvature; a global
  intermediate L is an adequate compromise. Marginal headroom at best.

**Takeaway:** the value of adaptive L is *funnel-only*. ill_cond and banana are adequately served
by a (geometry-appropriate) global L. Scoping should target horseshoe and nothing else.

## 3. Candidate signals (cheapest first)

All computable during the existing `mclmc_find_L_and_step_size` warmup with **zero extra
gradient evaluations** unless noted:

1. **Within-run per-trajectory acceptance variance / per-step ΔH variance** — already emitted
   (`info.acceptance_rate`, energy error). High intra-run variance ⇒ position-dependent stiffness
   ⇒ adaptive-L headroom. *Free.* Best first-pass screen.
2. **Leapfrog gradient-difference curvature proxy** — `‖∇logπ(x_{k+1}) − ∇logπ(x_k)‖ / ‖Δx‖`
   along the trajectory approximates the local Hessian operator norm. The gradients are *already
   evaluated* by leapfrog; the proxy is a free by-product. Sets a principled local L ∝ 1/√λ_max.
3. **EEVPD trend along/across the trajectory** — the EEVPD tuner already targets a fixed value to
   set step_size; EEVPD *spiking* mid-trajectory flags entry into a high-curvature region and
   could trigger early momentum refresh (= locally shorter L). *Free.*
4. **No-U-turn / position-momentum dot-product criterion** — a per-draw stopping rule that adapts
   trajectory length to local geometry. The most principled position-dependent L — but see §4: it
   is essentially NUTS.

## 4. The central feasibility risk: reversibility

adjusted_mclmc is a **Metropolis** sampler. Today the trajectory length is
`ceil(uniform(key)·rescale(avg))` — a function of the RNG key and `avg` **only, independent of
position** (`adjusted_mclmc_dynamic.py:234`). *That independence is exactly what makes the
randomized-length proposal reversible* (it also explains why realized_avg is step-invariant —
the step *count* is RNG-driven, not dynamics-driven).

Making L depend on the **current position** breaks detailed balance unless the reverse-proposal
probability enters the MH ratio. There are only two correct routes:

- **(a) Position-dependent L with a reversible construction** — this is, in effect, rebuilding
  NUTS' careful reversible tree/slice machinery. If we go here, the honest baseline to beat is
  *just running NUTS on the funnel*.
- **(b) Phase-dependent L only** — L is a function of warmup *iteration* (a schedule:
  coarse→fine), frozen at sampling. Trivially correct (no position dependence), but strictly
  weaker — it cannot adapt within a single funnel traversal, which is where horseshoe fails.

This fork is the key scoping verdict: **you are either reinventing NUTS (a) or settling for a
schedule that can't fix the funnel (b).**

## 5. Minimal experiment (instrumentation-only, no kernel change)

**Goal:** decide whether a *free* local signal predicts the locally-optimal L on the funnel,
*before* any kernel work.

1. Instrument horseshoe warmup to log, per draw: position, per-trajectory acceptance, ΔH, and the
   §3.2 gradient-difference curvature proxy. (Pure logging of already-computed quantities.)
2. Bin draws by funnel depth (the local-scale parameter) and/or by the curvature proxy.
3. Per bin, estimate the **local decorrelation length** (within-bin position autocorrelation).
4. **Decision gate:**
   - If decorrelation length varies **>~2–3×** across curvature bins *and* the curvature proxy is
     cleanly measurable at warmup resolution ⇒ position-dependent L has real headroom → proceed to
     a sweep, but **only** via a reversible (NUTS-style) construction, benchmarked head-to-head
     against plain NUTS on horseshoe.
   - If decorrelation length is roughly flat across bins, or the signal is too noisy at the EEVPD
     step scale ⇒ the horseshoe REVIEW plateau is a **reversibility-limited geometry wall**;
     adaptive L won't move it. Recommendation becomes "route funnels to NUTS; keep MCLMC global-L
     for constant/mild curvature."

Cost: ~1–2 h, CPU, banana-class budget. No production code touched.

## 6. Verdict

- **Worth a *cheap instrumentation probe* (§5): yes.** It is decisive and nearly free, and it
  cleanly separates "real adaptive-L headroom" from "geometry wall."
- **Worth a *kernel sweep* right now: no — gated on §5.** A position-dependent-L kernel is only
  justified if §5 shows headroom *and* we accept that the correct version is NUTS-shaped (route a),
  at which point the comparison is against NUTS itself.
- **Out of scope for adaptive L:** ill_cond_50 (constant curvature; short global L + the real fix
  is step/ε control for integration drift) and banana (mild; global intermediate L suffices).

**One-line recommendation to @tl:** run the §5 horseshoe instrumentation probe; hold all kernel
work behind its decision gate; if it shows headroom, the honest framing is "MCLMC-flavoured NUTS,"
so scope a direct NUTS-on-funnel baseline alongside.

---

## §5 RESULTS — horseshoe instrumentation probe (executed 2026-06-19)

**Method:** 1200-draw NUTS reference cloud on the 204-D Finnish (NCP) horseshoe (div=0,
tuned ε≈0.0095, tau_tilde span [−3.22, +2.73] ≈ 400× in global scale). 150 positions
stratified across 6 funnel-depth (tau) bins; at each, full 204×204 Hessian eigenspectrum +
cheap warmup-ε proxies. Scripts: `/tmp/issue22_scoping/{smoke,phase2,phase2b}.py`.

**Findings:**
1. **Global λ_max (~6e4) is likelihood-dominated and position-stable** — the funnel does not
   appear in it. The relevant adaptive-L quantity is the slow scale / local condition number, not λ_max.
2. **The neglog-Hessian is indefinite everywhere the sampler lives** — ~15–16 negative eigenvalues
   per position (λ_min ≈ −80 to −100). Saddle-rich, strongly ill-conditioned, but *uniformly so*.
3. **The slow length-scale is position-INDEPENDENT across funnel depth.** Robust p05 slow-scale
   1/√λ = 3.3–3.6 across all tau bins (**ratio 1.09×**); the noisier smallest-positive eigenvalue
   varies 2.8× but is **uncorrelated with depth** (Spearman ρ(|tau|, slow-eig) ≈ 0.00).

**Decision gate: NO.** Within-region decorrelation length does NOT vary >2–3× with funnel depth.
There is no depth-structured spatial signal for a position- or phase-dependent L to adapt to.

**Why (reverses the §1 premise):** the catalog horseshoe is the **non-centered (NCP)** Finnish
horseshoe — NCP deliberately *deflates* the funnel into a deterministic transform, leaving a
posterior bulk whose local geometry is roughly homogeneous (just uniformly stiff + indefinite).
The REVIEW-plateau at avg≤108 is therefore **not** a "single global L can't serve varying
geometry" problem; it is a **globally-too-short-trajectory + global-conditioning** problem:
the uniform condition number is enormous (√κ implied optimal avg ≈ 10³, vs the 108 tested), and
~15 indefinite directions signal the diagonal/LRD IMM isn't capturing the correlation structure.

**Recommendation (HOLD past gate — for the user, no work started):**
- **Drop adaptive/position-dependent L for the NCP horseshoe** — there's nothing depth-varying to
  adapt to; it would not move the plateau.
- The motivated levers instead are **global**: (a) a much longer global trajectory (avg → 10²–10³
  toward √κ) and (b) richer preconditioning (full/low-rank IMM that captures the ~15 indefinite
  correlated directions). Both are global-config changes, not adaptive-L.
- Adaptive-L *might* still matter for a **centered** funnel (where curvature genuinely varies with
  depth) — but the catalog uses NCP, so it's out of scope here. If a centered variant is ever
  benchmarked, re-run this exact probe on it before reconsidering.
- ill_cond_50 / banana unchanged from §2: global L suffices.

**Net:** adaptive-L is not worth a sweep for any current catalog model. Verdict downgraded from
"funnel-only, gated" (pre-probe) to "not motivated for the NCP catalog; the horseshoe plateau is a
global conditioning/length problem."
