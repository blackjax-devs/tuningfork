# Copyright 2026- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Base types for the tuningfork model registry.

Every benchmark posterior is described by a single ``Posterior`` dataclass.
Whether a model is a one-line analytic Gaussian or a full hierarchical NumPyro
program, the runner, cache, and CLI always see the same surface — no subclassing.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from jax import Array

__all__ = ["ReferenceMethod", "Posterior"]


class ReferenceMethod(str, Enum):
    """How the reference draws are produced for a given posterior."""

    ANALYTIC = "analytic"
    NUTS = "nuts"


@dataclass(frozen=True)
class Posterior:
    """Registry entry for a single benchmark posterior.

    Every model in the suite — analytic Gaussians, GLMs, ODE inverses — exposes
    this exact surface so the runner is model-agnostic.

    Parameters
    ----------
    name
        Unique identifier, e.g. ``"mvn_10"`` or ``"eight_schools_ncp"``.
    dim
        Latent dimensionality in *unconstrained* space.
    class_
        Broad category string: ``"gaussian"`` | ``"hierarchical"`` |
        ``"funnel"`` | ``"glm"`` | ...
    numpyro_model
        A NumPyro model callable.  For analytic models this is a thin wrapper
        that calls ``numpyro.sample``; for hierarchical models it contains the
        full generative story.
    model_args
        Positional arguments forwarded to ``numpyro_model``.
    model_kwargs
        Keyword arguments forwarded to ``numpyro_model``.
    analytic_sampler
        Optional callable ``(rng_key: jax.Array, n: int) -> dict[site_name, Array]``
        that draws *n* i.i.d. samples from the exact posterior in unconstrained
        space.  When set, ``reference_method`` is ``ANALYTIC``; otherwise it
        is ``NUTS``.
    posteriordb_id
        PosteriorDB identifier for cross-checking (e.g.
        ``"8_schools-eight_schools_noncentered"``).
    citations
        Tuple of citation strings.
    description
        Human-readable description of the posterior.

    Notes
    -----
    The ``reference_method`` property is the single dispatch point for the
    cache and runner; do not branch on model ``class_``.

    Why no inheritance / ``Hierarchical(Posterior)`` subclass: registry
    consumers (cache, runner, CLI) should not branch on type.
    """

    # ---- identity ----
    name: str
    dim: int
    class_: str

    # ---- model definition (NumPyro by default) ----
    numpyro_model: Callable[..., None]
    model_args: tuple[Any, ...] = ()
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    # ---- reference path ----
    # Exactly one of (analytic_sampler, needs_long_nuts) determines the path.
    analytic_sampler: Callable[[Array, int], dict[str, Array]] | None = None

    # ---- metadata ----
    posteriordb_id: str | None = None
    citations: tuple[str, ...] = ()
    description: str = ""
    tags: tuple[
        str, ...
    ] = ()  # for recommendation queries; populated as more models land

    # ---- per-model cert overrides ----
    # ``None`` means "use the global default" (``_DIVERGENCE_RATE_TOLERANCE``
    # in ``tuningfork.calibration.certify_reference``; currently 0.001 = 0.1%
    # of n_samples). A non-None value overrides only this model's gate and
    # MUST cite the diagnostic justification in the model file. Used so a
    # single model's structural geometry (e.g. an AR(1) unit-root excursion
    # tail visited at low probability) doesn't force the global gate to
    # loosen for every model.
    divergence_rate_tolerance: float | None = None

    # ---- diagnostic rendering hints (interactive catalog notebook) ----
    headline_params: tuple[str, ...] | None = None
    """Subset of free `numpyro.sample` site names to render with
    az.plot_trace + az.plot_pair in the catalog's interactive notebook.
    Remaining sites go to az.plot_forest. None = no filtering (show all,
    appropriate for small-dim models). Every name MUST be a numpyro.sample
    site (NOT numpyro.deterministic) — deterministics aren't materialised
    into draws.npz / data_vars."""

    headline_coords: dict[str, list[int]] | None = None
    """Per-model 'interesting indices' within multi-dim sites. E.g.,
    {'beta': [0, 1, 2, 3, 4, 5, 6, 7]} on german_credit selects the
    intercept + 7 numerical features for trace/pair; the 18 categorical
    dummies still render in the forest plot. None = show all coords within
    each headline param (the default). See the 2026-05-18 decision doc
    appendix for ratified per-model values."""

    # ---- per-model NUTS cert target acceptance ----
    # NUTS target acceptance rate used when generating the reference draws for
    # this model.  Default 0.80 matches Stan's default and is appropriate for
    # smooth posteriors.  Models with sharper curvature (e.g. high-dimensional
    # latent-GP priors) may benefit from a higher value (e.g. 0.90) to reduce
    # step-size and ensure the leapfrog integrator stays stable.
    #
    # Setting this here (rather than hardcoding ``entry.name == "lgcp"`` in
    # the CLI) keeps model-specific cert parameters co-located with the model
    # definition, making it trivially grep-able and audit-able.
    reference_target_acceptance: float = 0.80

    # When True, this model REQUIRES ``JAX_ENABLE_X64=1`` at cert time —
    # float32 cannot stably evaluate the model's log-density (e.g., dense
    # Cholesky on a high-d kernel matrix produces NaN at float32 precision).
    # ``certify_reference_nuts`` asserts ``jax.config.read("jax_enable_x64")``
    # is True for any entry with this flag set, raising a clear error
    # otherwise. Default ``False`` for backwards-compatibility — all existing
    # models cert cleanly at the JAX default float32. The flag MUST cite the
    # specific numerical issue in the model file's docstring.
    requires_x64: bool = False

    # ---- elliptical-slice wiring (prior for gradient-free latent-Gaussian samplers) ----
    # Both fields must be set together for ``elliptical_slice`` wiring.
    # Values are per numpyro.sample site; each entry is a flat list of floats
    # whose length equals the site's unconstrained dimensionality.
    #
    # ``prior_mean``: prior mean µ for each site.
    #   E.g. {"log_noise_scale": [-2.0], "f_raw": [0.0, ..., 0.0]}
    # ``prior_cov_diag``: diagonal of the prior covariance Σ for each site.
    #   E.g. {"log_noise_scale": [1.0], "f_raw": [1.0, ..., 1.0]}
    #
    # The generated ``no_warmup`` protocol uses these to call
    # ``blackjax.elliptical_slice`` with the correct Gaussian prior structure.
    # Generated execution builds the likelihood-only logdensity by subtracting
    # the Gaussian prior from the joint log-posterior:
    #   loglik(x) = logposterior(x) − logprior_gaussian(x, µ, diag(Σ))
    #   logprior_gaussian = −0.5 · Σ_site Σ_dim (x − µ)² / σ²
    #
    # Only ``latent_gaussian`` class models need these fields.  Leaving them
    # ``None`` for other models is correct; the generated no_warmup protocol
    # raises a clear
    # error if ``elliptical_slice`` is attempted without them.
    prior_mean: dict[str, list[float]] | None = None
    prior_cov_diag: dict[str, list[float]] | None = None

    # ---- per-model GT precision override ----
    # Controls the floating-point precision used by ``generate_nuts_multichain``
    # when regenerating this model's ground-truth draws.  ``"float32"`` (the
    # default) matches JAX's default precision mode.  Set to ``"float64"`` for
    # models where f32 rounding destabilizes per-chain warmup adaptation
    # (e.g. horseshoe on aarch64: a 204-D funnel posterior where f32 rounding
    # over 2000 warmup steps tips one chain into a micro step-size trap, causing
    # R̂=1.25 on fresh regen; x64=True at the same committed seed mixes cleanly).
    # When set to ``"float64"``, ``generate_nuts_multichain`` calls
    # ``jax.config.update("jax_enable_x64", True)`` before any JAX computation.
    # The setting is self-describing: it is recorded in the emitted
    # ``sampler_config.precision`` field of ``summary_v2.json``.
    # Only ``"float32"`` and ``"float64"`` are accepted; ``"float64"`` MUST cite
    # the specific numerical issue in the model file's docstring.
    groundtruth_precision: str = "float32"

    # ---- derived ----
    @property
    def reference_method(self) -> ReferenceMethod:
        """Return the reference generation method for this entry."""
        if self.analytic_sampler is not None:
            return ReferenceMethod.ANALYTIC
        return ReferenceMethod.NUTS

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"{self.name}: dim must be positive, got {self.dim}")
        if not callable(self.numpyro_model):
            raise TypeError(f"{self.name}: numpyro_model must be callable")
        if self.groundtruth_precision not in ("float32", "float64"):
            raise ValueError(
                f"{self.name}: groundtruth_precision must be 'float32' or 'float64', "
                f"got {self.groundtruth_precision!r}"
            )
