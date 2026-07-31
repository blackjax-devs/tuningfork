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
"""Descriptor for Laplace-marginal dynamic multinomial HMC recipe emission.

Generated emission targets ``blackjax.laplace_dmhmc``
(``blackjax/mcmc/laplace_dynamic_hmc.py`` with
``build_proposal=multinomial_hmc_proposal``).

This is the combination of both dynamic trajectory length (quasi-random step
count, as in ``laplace_dhmc``) and multinomial trajectory proposal (as in
``laplace_mhmc``).  The "2×2" variant that eliminates both periodic-orbit
sensitivity and the endpoint-proposal acceptance bias.

Algorithm summary
-----------------
Same as ``laplace_dhmc`` (dynamic Laplace HMC) but with multinomial sampling
over the full quasi-random trajectory.  The state carries both ``theta_star``
(MAP latent warm-start) and ``random_generator_arg`` (Halton step-count seed).

For the full algorithm description, see:
``sampling-book/book/algorithms/laplace_hmc_demo.md``

Grad cost approximation
-----------------------
Same approximation as ``laplace_dhmc`` (step count varies per transition)::

    grad_count_per_step = _laplace_grad_count

``extra_required_kwargs=("log_joint_fn", "theta_init")``: generated emission
splits the configured model into ``phi`` and ``theta`` and supplies a typed
joint log-density plus latent initializer.  It raises ``ValueError`` when no
phi/theta split is registered for the model.

References
----------
- ``blackjax/mcmc/laplace_dynamic_hmc.py`` — upstream implementation
  (``laplace_dmhmc`` is ``laplace_dynamic_hmc`` with
  ``build_proposal=blackjax.mcmc.hmc.multinomial_hmc_proposal``).
- ``sampling-book/book/algorithms/laplace_hmc_demo.md`` — full algorithm description.
- Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. §A.2.
"""

from tuningfork.base_method._base import BaseMethod, HyperparamSpace
from tuningfork.base_method._laplace_common import _laplace_grad_count

__all__ = ["ENTRY"]


ENTRY = BaseMethod(
    name="laplace_dmhmc",
    family="mcmc",
    # Grad cost: (num_integration_steps + 1) × lbfgs_iter_num — measured proxy.
    # num_integration_steps varies per transition (quasi-random schedule).
    grad_count_per_step=_laplace_grad_count,
    grad_count_convention="(info.num_integration_steps + 1) × info.lbfgs_iter_num (lower bound; line-search ≈ 1)",
    default_hp_space=(HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),),
    needs_mass_matrix=True,
    target_acceptance_rate=0.8,
    extra_required_kwargs=("log_joint_fn", "theta_init"),
    # The generated Laplace preamble supplies the phi/theta model split and
    # initializes the sampler-specific state.
    notes=(
        "Dynamic multinomial HMC on the Laplace-approximated marginal log-density. "
        "Combines dynamic trajectory length (quasi-random step count, avoids periodic-orbit "
        "sensitivity) with multinomial trajectory sampling (Betancourt 2017 §A.2, avoids "
        "endpoint-proposal acceptance bias). The '2×2' variant in the Laplace-marginal family. "
        "blackjax.laplace_dmhmc = laplace_dynamic_hmc with "
        "build_proposal=multinomial_hmc_proposal (blackjax.mcmc.hmc.multinomial_hmc_proposal). "
        "State carries theta_star (MAP latent warm-start) and random_generator_arg (Halton index). "
        "Only step_size is recipe-resolved; trajectory length is adapted internally. "
        "Grad cost approximation: num_integration_steps * 5 (coarse; varies per transition). "
        "extra_required_kwargs=('log_joint_fn', 'theta_init'); generated emission "
        "requires a registered model phi/theta split and fails explicitly when "
        "that configuration is absent. "
        "See sampling-book/book/algorithms/laplace_hmc_demo.md for full algorithm description."
    ),
)
