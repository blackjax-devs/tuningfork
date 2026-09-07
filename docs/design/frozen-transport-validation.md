# Frozen coordinate transport — supplied-chart validation artifact

Status: **validation artifact only.** This package supplies a frozen coordinate
chart and the algebra needed to check it. It contains no fitter, no selection
rule, no warmup descriptor, no execution recipe and no registry entry, and it is
not wired into any sampling path.

## The chart

Work in preconditioned coordinates `z = L^{-1}(q - center)`, with `L` a diagonal
scale optionally composed with a symmetric low-rank factor. Fix a constant
affine vector field

```
V(z) = alpha * z + a * (h . z) + c
```

subject to three **structural constraints**:

```
||h|| = 1        h . c = 1        h . a = -alpha
```

Under those constraints the clock rate is identically one:

```
d/ds (h . z) = alpha (h.z) + (h.a)(h.z) + (h.c) = 1
```

Three exact consequences follow — exact in real arithmetic, not fitted
approximations:

1. the last chart coordinate (the **clock**) advances at unit rate, so it can be
   read off directly as `t = h . z`;
2. the inverse is closed-form and exact: flowing back by `t` lands on the section
   `h . z = 0`, with no iteration and no root find;
3. `log|det J| = alpha (d-1) t + log|det L|` is the exact log-Jacobian, because
   the section columns of the Jacobian are `exp(alpha t)` times an orthonormal
   basis of `h`-perp while the clock column has unit `h`-component.

The structural constraints are consequently **gates, not documentation**: a chart
that violates them has a wrong log-Jacobian and a broken inverse.

## Why a non-zero Stein residual is the useful case

For a chart field `V`, the quantity `grad(log pi) . V + div V` is exactly the
clock derivative of the transformed density, `d/dt log pi_chart`.

It is tempting to want that residual to be zero — an exact symmetry. That is not
the useful target. The useful structure is a factorisation, and it follows from a
premise that must be stated in full:

> **Analytic statement.** Suppose that on the **full real line** in `t`, and on a
> **global Cartesian chart with global support**, the transformed density
> satisfies
>
> ```
> d/dt log pi_chart(t, s) = -kappa * t + beta
> ```
>
> identically, with `kappa` a **constant** independent of `t` and `s`, and
> `kappa > 0`. Then integrating in `t` gives
>
> ```
> log pi_chart(t, s) = -kappa t^2 / 2 + beta t + C(s)
> ```
>
> so the clock is Gaussian with precision `kappa`, independent of the section.

Three conditions carry the conclusion and none is cosmetic: the identity must hold
**globally in `t`**, `kappa` must be **constant**, and the chart must be a global
Cartesian one with global support. Under those conditions — and only under them —
`kappa <= 0` fails to normalise, which is the sense in which a zero residual
corresponds to an improper (flat) clock. On a **bounded or otherwise restricted
clock domain** that conclusion does not apply: a non-positive `kappa` can be
perfectly normalisable there. Nothing here is a general prohibition on other
domains; it is a statement about this premise.

**Analytic premise versus numerical corroboration.** Evaluating the identity at
finitely many points does **not** establish it. For a *supplied* chart whose
parameters are known — the case tested here — the identity can be verified
symbolically and the finite evaluations merely corroborate the algebra. For a
*learned* chart it is an open empirical question: a fit that matches on its
training support says nothing about the full line, says nothing about whether
`kappa` is genuinely constant off that support, and does not guarantee
`kappa > 0`. A learned chart with `kappa <= 0`, or with `kappa` drifting in `s`,
would not factorise at all. The tests below corroborate the supplied case; they
do not and cannot discharge the premise for a fitted one.

Neal's funnel is the worked example, and `test_chart_refuters.py` pins all three
of the quantities that are easy to conflate:

| quantity | funnel value |
|---|---|
| raw-field Stein residual `grad(log pi).V + div V` | `-t/9` (**not** zero) |
| regression residual of a fitter carrying `kappa t - beta` slack | `~0` |
| resulting structure | Gaussian clock `N(0, 9)`, section-independent |

The tests check the decisive discriminator directly: the mixed partial
`d_s d_t log pi_chart` is zero to `1e-10`, the clock curvature is constant at
`-1/9`, and the section score is unchanged across the clock.

**Limits, stated plainly.** A small residual measured on a fitter's training
support proves none of the following: that the identity holds off that support;
that a fitted `kappa` is constant; that a fitted `kappa` is positive; that the
chart is numerically valid; or that any of it improves mixing. Those are separate
questions and this artifact tests none of them.

## Validation tiers

| file | what it establishes |
|---|---|
| `test_chart_algebra.py` | structural identities, exact log-Jacobian against `slogdet(jacfwd)`, two-sided inverse, supplied score against a non-hooked reference, score pullback invertibility — diagonal and low-rank |
| `test_chart_mutants.py` | finite-aware gates against a valid chart, a wrong log-determinant, a wrong score, non-finite providers, and an omitted-normalisation defect |
| `test_chart_refuters.py` | identity, pure-linear, rotated, non-normal generator, and the exact-funnel structure above |
| `test_phi_precision.py` | checks values AND derivatives against an independent 60-digit oracle at named points: zero, small arguments, both sides of each crossover, and a large-argument cancellation |
| `test_chart_conditioning.py` | records the clock residual as a runtime diagnostic; asserts availability, not a degradation threshold |

Two methodological points that the tests encode rather than assume:

* **The score reference must not be hooked.** Testing a supplied score by
  differentiating a `custom_jvp` density that supplies that same score is
  circular — AD returns the rule under test. The reference is the plain
  `native_logdensity(forward(y)) + log_det(y)`. For the same reason the inverse
  is tested as a supplied map: AD cannot manufacture a global inverse from
  `forward` alone.
* **Mutants must be executed.** Asserting an algebraic offset without running a
  wrong provider proves nothing about the gates. Running them is also what
  corrected the expected table: rescaling `h` turned out to be invisible to the
  determinant gate, because the reflector is built from the normalised direction
  and so `forward` is unchanged to rounding.

The two rows worth reading twice are `M4` (drop the Jacobian-derivative term from
the score) and `M5` (drop the rank-one term from the field). Both pass the
log-determinant and round-trip gates and are caught **only** by the independent
score gate — which is why that gate is mandatory rather than a convenience.

## Numerical range, and what is not claimed

The flow carries a factor `exp(alpha * clock)`, and `h . z` is recovered from a
difference of two such terms. So the **clock coordinate** loses accuracy with the
clock while the rest of the flow stays at dtype precision. Two consequences, both
established by independent review rather than asserted here:

**The residual is a runtime diagnostic.** The unit-clock-rate identity says
`h . z` equals `t` exactly, so `|h . z - t|` measures the damage directly. It is
one dot product over quantities `forward` already computes. An earlier version of
this document said a useful guard "has to be derived from the chart parameters
rather than inferred from the returned values". That was wrong: it cannot be a
*finiteness* check, but it need not come from the parameters either.

**A projection repairs it, and is not shipped.** `z <- z + h (t - h . z)`
displaces by exactly the residual, so it is a no-op on a correct value and exact
in real arithmetic. It is documented, not adopted: it changes the implemented
forward map, so adopting it needs its own map / inverse / Jacobian / score
verification across rotated `h`, non-normal generators and low-rank
preconditioners. A good round trip alone cannot certify it.

**Severity belongs to the (chart, target) pair, not the chart.** An isotropic
Gaussian's score is `-q`, with no exponential clock dependence, so its score stays
accurate even when the clock coordinate has been destroyed. The funnel is
therefore load-bearing as the probe target in
`test_chart_conditioning.py`, and swapping it for a tamer target would certify a
chart whose clock coordinate is annihilated.

**No conditioning bound is claimed**, and no "intrinsic" degradation: a removable
cancellation is not a barrier. Nothing here is a certified bound, which would need
a derivation covering the chart parameters, coordinates, dtype, the conditioning
of the target's own score, and the error scale.

## Supported inputs

`make_chart` **projects** `h`, `c` and `a` onto the structural constraints, and
**refuses** what it cannot repair: zero `h`; zero or mis-shaped `scale`; a
low-rank basis supplied without eigenvalues or with mismatched rank;
non-positive eigenvalues; and non-orthonormal **spectrally active** columns.

Orthonormality is scoped to active columns (`lam != 1`) deliberately. Neutral
columns contribute exactly zero to `_lowrank` at every power and zero to the
log-determinant, so requiring them to be orthonormal would refuse legitimate
inputs. Active columns need orthonormality, not merely orthogonality.

`log_det` is a log-**absolute** determinant, so a negative `scale` entry is
supported: it flips orientation without changing the volume element.

`phi` serves **float32 and float64 only**. Half precisions raise rather than fall
back, because the shipped crossover is measurably the *worst* available choice for
them — a silent fallback would be actively harmful, not merely unsupported.

## Cost

A supplied chart pays no fitting charge in a run that uses it, but it is not
free: `forward`, `pullback_score` and the log-Jacobian are evaluated per step,
and the chart action is `O(d)` plus whatever the preconditioner costs. Any future
comparison must keep transform evaluation, score, Jacobian, validation and
fresh-cache costs visible.

## Conventions

`CONVENTION_VERSION = "frozen-transport-chart/v1"` fixes the clock-last
coordinate order, the reflector orientation keyed on `h[-1]`, and the per-dtype
`phi` crossover.

**v1 still names the shipped behaviour.** The repairs in this revision — the
log-absolute determinant, the clamped series branch, the refused half precisions,
the enforced input contract — change results only where the previous code
returned NaN or served an unsupported dtype. None changes the map for a valid
input. Adopting the clock projection *would* be a convention change, which is one
reason it is documented rather than shipped. Earlier exploratory implementations used a clock-first order and a
different reflector; **no bitwise equivalence with it is claimed or intended.**
