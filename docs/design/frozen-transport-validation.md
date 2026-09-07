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

It is tempting to want that residual to be zero — an exact symmetry. That is the
wrong target. If instead the transformed density satisfies

```
d/dt log pi_chart(t, s) = -kappa * t + beta
```

globally, with constant `kappa > 0` and no dependence on the section `s`, then
integrating gives

```
log pi_chart(t, s) = -kappa t^2 / 2 + beta t + C(s)
```

— an exactly Gaussian clock, independent of the section. That factorisation is
the useful structure. A residual of exactly zero would instead correspond to a
flat, improper clock.

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
that a fitted `kappa` is positive (a non-positive `kappa` is an improper clock);
that the chart is numerically valid; or that any of it improves mixing. Those
are separate questions and this artifact tests none of them.

## Validation tiers

| file | tier | what it establishes |
|---|---|---|
| `test_chart_algebra.py` | 0 | structural identities, exact log-Jacobian against `slogdet(jacfwd)`, two-sided inverse, supplied score against a non-hooked reference, score pullback invertibility — diagonal and low-rank |
| `test_chart_mutants.py` | 1 | seven wrong-chart mutants, each executed, with the exact gate pattern each triggers |
| `test_chart_refuters.py` | 2 | identity, pure-linear, rotated, non-normal generator, and the exact-funnel structure above |
| `test_phi_precision.py` | — | selects the `phi` crossover per dtype by measured value and derivative error |
| `test_chart_conditioning.py` | — | empirical conditioning indicator and the silent-failure band |

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

## Numerical range

The flow carries a factor `exp(alpha * clock)`, so far along the clock the score
contracts exponentially large quantities whose true value is small. Accuracy is
lost well before anything overflows. Measured with a float32-vs-float64 ladder on
the funnel chart, the score is accurate to float32 rounding up to
`alpha * clock ~ 15`, is wrong by `O(1)` relative at `alpha * clock ~ 20` while
every component is still **finite**, and only becomes non-finite around
`alpha * clock ~ 40`.

The practical consequence is that a reject-on-NaN guard cannot detect this
failure: across a wide band the chart returns finite, ordinary-looking numbers
that are wrong by orders of magnitude. A useful guard has to be derived from the
chart parameters rather than inferred from the returned values.

`exp(alpha * clock)` is offered as an **empirical indicator** that orders the
accurate band against the corrupted one. It is not a certified bound: that would
need a derivation covering the chart parameters, the coordinates, the dtype, the
conditioning of the target's own score, and the error scale. The overflow point
quoted above is a property of these parameters and this dtype pair, not a
universal domain limit.

## Cost

A supplied chart pays no fitting charge in a run that uses it, but it is not
free: `forward`, `pullback_score` and the log-Jacobian are evaluated per step,
and the chart action is `O(d)` plus whatever the preconditioner costs. Any future
comparison must keep transform evaluation, score, Jacobian, validation and
fresh-cache costs visible.

## Conventions

`CONVENTION_VERSION = "frozen-transport-chart/v1"` fixes the clock-last
coordinate order, the reflector orientation keyed on `h[-1]`, and the per-dtype
`phi` crossover. Earlier exploratory work used a clock-first order and a
different reflector; **no bitwise equivalence with it is claimed or intended.**
