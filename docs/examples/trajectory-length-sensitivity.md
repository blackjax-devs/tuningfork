# Trajectory-length sensitivity, with every expectand and every cost on the table

**Status:** example
**Runs in:** ~25 s on one CPU core

```bash
JAX_PLATFORM_NAME=cpu uv run pytest \
  tests/e2e/test_trajectory_length_sensitivity.py -s -n 0
```

The example lives in `tests/e2e/test_trajectory_length_sensitivity.py`, so it is
executed by CI and cannot silently rot. `-s` prints the two reports and the
comparison table.

## What it does

One warmup is paid. Its adapted `step_size` and `inverse_mass_matrix` are frozen
and handed to two sampling arms that differ in exactly one thing — the
distribution of the trajectory length:

| arm | sampler | trajectory length |
|---|---|---|
| `fixed_L4` | `hmc` | fixed `num_integration_steps = 4` |
| `uniform_L2_6` | `dynamic_hmc` | `step_policy = {"kind": "uniform_int", "low": 2, "high": 7}` |

Both arms then report the **same three expectands** of their own draws:

- `x_0` — a first moment,
- `x_0_sq` — an even (squared) function,
- `x_0x1` — a cross product.

Everything runs through codegen. `Recipe.step_policy` and `no_warmup` with a
pinned geometry already express "randomised L on geometry frozen after warmup",
so no hand-written sampling script is involved and no codegen capability was
added for this example.

## Why the same draws need more than one number

On the `fixed_L4` arm above (`mvn_10`, 4 chains × 300 draws = 1200 draws):

| expectand | raw-mean ESS | bulk ESS | tail ESS | rank R-hat |
|---|---|---|---|---|
| `x_0` | 3645.6 | 3610.4 | 1076.1 | 1.0100 |
| `x_0_sq` | 583.9 | 577.9 | 753.8 | 1.0095 |
| `x_0x1` | 571.1 | 568.7 | 787.8 | 1.0057 |

The first moment's raw-mean ESS is **three times the number of draws** — HMC is
antithetic for `x_0`, and an ESS above the draw count is the expected
consequence, not an error. The squared and cross functions of those *same
draws* sit at about one sixth of it. A comparison that quoted only the first
moment would report an excellent number for a run whose second moments are six
times slower.

That is the whole reason the report puts `raw_mean_ess` next to the
rank-normalised statistics instead of choosing one.

## What happened when the trajectory length was randomised

Same frozen geometry, same seed, same expectands, near-identical paid gradient
work (4800 vs 4770 sampling transition gradients in the arms themselves):

| expectand | statistic | `fixed_L4` | `uniform_L2_6` | ratio |
|---|---|---|---|---|
| `x_0` | raw-mean ESS | 3645.6 | 2887.2 | 0.792 |
| `x_0` | bulk ESS | 3610.4 | 3189.2 | 0.883 |
| `x_0` | tail ESS | 1076.1 | 787.7 | 0.732 |
| `x_0_sq` | raw-mean ESS | 583.9 | 458.6 | 0.785 |
| `x_0_sq` | bulk ESS | 577.9 | 430.7 | 0.745 |
| `x_0_sq` | tail ESS | 753.8 | 584.0 | 0.775 |
| `x_0x1` | raw-mean ESS | 571.1 | 503.5 | 0.882 |
| `x_0x1` | bulk ESS | 568.7 | 477.5 | 0.840 |
| `x_0x1` | tail ESS | 787.8 | 671.1 | 0.852 |

**On this target, randomising the trajectory length was worse on every
expectand.**

### What this does not show

This is one model, one frozen geometry, one seed, and short chains. It is a
demonstration that trajectory length distribution is a variable worth measuring
per expectand, and a counterexample to "randomising helps"; it is **not**
evidence that randomising hurts in general. Nothing here supports a change to
any recipe, gate, or headline-metric policy, and the example's assertions are
deliberately structural — they check that the comparison is composed and costed
correctly, never that one arm wins.

`rank_rhat` appears in the reports but not in the ratio table: R-hat is a
convergence ratio read against its own threshold, so a ratio of two R-hats and
"R-hat per second" are both meaningless. The comparison withholds them.

## How the costs are accounted

Both arms run `no_warmup`, so each arm's own telemetry records
`warmup_grad_evals = 0`. That is a **measured zero**, not a gap. But both arms
*inherit* a warmup neither of them paid for, which appears in neither arm's
telemetry, so two different questions have two different answers:

- **standalone** — what this alternative would cost on its own. The shared
  warmup is charged to each arm. This is the view the example reports, and it is
  the right one for "which of these would I run?".
- **combined** — what the experiment actually spent. The shared warmup is
  charged once across both arms.

`CostAccounting.combine` builds either view explicitly; it never infers which
one you meant, and it refuses to combine an accounting with a contributor it
already contains, so a shared warmup cannot be charged twice by accident.

Costs that were not measured stay unknown with a reason and are never written as
zero. In this example that is `compile_seconds`: the wall clocks include JIT
compilation, compilation is not separately measured, and no estimate of it is
subtracted from either side. Separating compilation is a distinct design
question, not something to approximate here.

The gradient denominator is warmup plus **sampling transition** gradients,
recovered from the persisted per-step statistics via each sampler's own declared
`grad_count_per_step` contract, rejected transitions included. It is a subtotal:
it cannot see initialization or controller internals, so every comparison
carries that exclusion list, and ESS divided by it is an upper bound on
efficiency rather than a measurement of it.

Both arms here use exact counting conventions, so the per-gradient comparison is
produced. That is not always the case. Some samplers declare a
`grad_count_convention` that is explicitly inexact — `orbital_hmc` counts one
gradient per step where the kernel evaluates a whole orbit of `period ∈ [2, 20]`
positions, and the four `laplace_*` methods exclude line-search gradients. Such
a count is **not the same unit** as an exact one, so when either side of a
comparison uses one, `compare_reports` withholds the per-gradient figure
entirely rather than publishing a ratio with a caveat attached. Without that
rule, comparing `orbital_hmc` against `nuts` on identical draws reports the
orbital arm as 7.0× more efficient per gradient, which is an artifact of the
undercount and nothing else.

What is *not* withheld: wall-clock normalisation, which is measured rather than
counted and so does not depend on any convention; the uncosted ESS ratios; and
the declared convention and basis strings, which stay with the report so the run
remains reproducible and a reader can judge the count themselves.

This detection reads each sampler's `grad_count_convention` and nothing else.
Notes are free text and use words like "approximation" for unrelated reasons —
`irmh` describes a proposal fitted from a Laplace approximation, `mgrad_gaussian`
a first-order approximation to the log-likelihood — and both count exactly, so
scanning notes would withhold legitimate comparisons.

The VI family is handled separately, because its inexactness is declared outside
that field. `meanfield_vi` and `fullrank_vi` declare the convention `"1"` and
explain in their notes that it describes the *optimisation* phase — at sample
time no gradient is evaluated at all. A sampling-gradient subtotal is therefore
not a meaningful quantity for them, and the derivation is refused outright
rather than reported, discriminated by the descriptor's existing `family` field
rather than by a duplicated cost table. The declared convention is preserved in
the refusal reason.

**The absence of a flag is still not a certificate that a count is complete.**
The marker sees what a convention explicitly declares; a convention that
undercounts without saying so in those words is not detected, and at least one
does. `rmhmc` declares `info.num_integration_steps` while its integrator runs a
fixed-point iteration whose true cost is several gradients per step — reported
by review, not independently confirmed here, and left unflagged. A structured
contract on the sampler descriptor is what would make this sound; the mechanism
described above is disclosure of declared limitations, not a completeness
guarantee.

## A note on ties, and why the severity is keyed on cardinality

Rank-normalised statistics depend on how a backend ranks tied values:
`blackjax.diagnostics` assigns ordinal ranks, ArviZ averages them. Every report
row discloses ties whenever any are present, and names the backend's behaviour.

The *severity* of that disclosure — wording only — is keyed on how few distinct
values a trace takes rather than on what share of it is tied. A small
exploratory probe on a single 4 × 500 fixture, one seed, one arrangement and the
pinned backend versions:

| distinct values | tie fraction | blackjax bulk ESS | ArviZ bulk ESS | ratio |
|---|---|---|---|---|
| 2000 (continuous) | 0.000 | 1825.1 | 1825.1 | 1.00 |
| 200 | 0.900 | 1812.4 | 1823.9 | 1.01 |
| 50 | 0.975 | 1806.8 | 1838.9 | 1.02 |
| 10 | 0.995 | 725.2 | 1883.4 | 2.60 |
| 5 | 0.998 | 60.7 | 1914.3 | 31.6 |
| 2 | 0.999 | 11.8 | 1850.6 | 156.2 |

A continuous chain with 89% of its values repeated by rejection holds still has
essentially every value distinct, and the backends agree to 0.4%. A 2-valued
indicator has a comparable tie fraction and the backends differ by two orders of
magnitude. That is enough to show tie fraction alone is a poor severity signal,
so `DEFAULT_TIE_BLOCK_THRESHOLD` keys on the mean tie block (`n_total /
n_distinct`) instead.

**This is not a calibration.** These are a handful of fixtures at one chain
length, one temporal arrangement and one pair of backend versions. They do not
establish where the effect becomes material in general, and a backend release
that changes its tie handling changes the picture entirely — which is why every
report records the backend *and its version* alongside the numbers.

The threshold changes wording and nothing else. It is never a validity boundary
and never a reliability certificate: a trace with a single repeated value is
still disclosed, the affected backend is still named, and the tie fraction and
mean tie block are reported numerically on every row regardless of severity.
