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
"""Descriptor for Laplace-marginal multinomial HMC recipe emission.

Generated emission targets ``blackjax.laplace_mhmc``
(``blackjax/mcmc/laplace_hmc.py`` with
``build_proposal=multinomial_hmc_proposal``).

This is identical to ``laplace_hmc`` except the M-H endpoint proposal is replaced
by multinomial sampling over the full trajectory (Betancourt 2017 §A.2).  This
changes *trajectory selection*, not the hyperparameter surface — the HP space is
identical to ``laplace_hmc``.

Algorithm summary
-----------------
Same as ``laplace_hmc`` (HMC on Laplace-approximated marginal) but proposes
from the full trajectory via multinomial sampling rather than the trajectory
endpoint.  This typically yields better ESS per gradient at the cost of no
rejection step.

For the full algorithm description, see:
``sampling-book/book/algorithms/laplace_hmc_demo.md``

Grad cost approximation
-----------------------
Same approximation as ``laplace_hmc``::

    grad_count_per_step = _laplace_grad_count

``extra_required_kwargs=("log_joint_fn", "theta_init")``: generated emission
splits the configured model into ``phi`` and ``theta`` and supplies a typed
joint log-density plus latent initializer.  It raises ``ValueError`` when no
phi/theta split is registered for the model.

References
----------
- ``blackjax/mcmc/laplace_hmc.py`` — upstream implementation (``laplace_mhmc``
  is ``laplace_hmc`` with ``build_proposal=blackjax.mcmc.hmc.multinomial_hmc_proposal``).
- ``sampling-book/book/algorithms/laplace_hmc_demo.md`` — full algorithm description.
- Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. §A.2.
"""

from tuningfork.base_method._base import BaseMethod, HyperparamSpace
from tuningfork.base_method._laplace_common import _laplace_grad_count

__all__ = ["ENTRY"]


ENTRY = BaseMethod(
    name="laplace_mhmc",
    family="mcmc",
    # Grad cost: (num_integration_steps + 1) × lbfgs_iter_num — measured proxy.
    grad_count_per_step=_laplace_grad_count,
    grad_count_convention="(info.num_integration_steps + 1) × info.lbfgs_iter_num (lower bound; line-search ≈ 1)",
    default_hp_space=(
        HyperparamSpace("step_size", "loguniform", low=1e-3, high=1.0),
        HyperparamSpace("num_integration_steps", "int", low=1, high=20),
    ),
    needs_mass_matrix=True,
    target_acceptance_rate=0.8,
    extra_required_kwargs=("log_joint_fn", "theta_init"),
    # The generated Laplace preamble supplies the phi/theta model split and
    # initializes the sampler-specific state.
    notes=(
        "Multinomial HMC on the Laplace-approximated marginal log-density. "
        "Identical to laplace_hmc but proposes from the full trajectory via multinomial "
        "sampling (Betancourt 2017 §A.2) rather than the endpoint. "
        "Typically yields better ESS per gradient at the cost of no rejection step. "
        "blackjax.laplace_mhmc = laplace_hmc with build_proposal=multinomial_hmc_proposal "
        "(blackjax.mcmc.hmc.multinomial_hmc_proposal — NOT the top-level alias). "
        "Grad cost approximation: num_integration_steps * 5 (coarse). "
        "extra_required_kwargs=('log_joint_fn', 'theta_init'); generated emission "
        "requires a registered model phi/theta split and fails explicitly when "
        "that configuration is absent. "
        "See sampling-book/book/algorithms/laplace_hmc_demo.md for full algorithm description."
    ),
)
