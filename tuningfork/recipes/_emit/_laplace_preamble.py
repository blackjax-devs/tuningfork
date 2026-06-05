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
"""Emit-time Python function for the Laplace preamble section.

Replaces ``_templates/laplace_preamble.py.tmpl`` (40 LOC, string.Template).
All slot resolution is done in Python — no $slot markers in the output.
D8 compliant: emitted string imports only from blackjax (allowed by D8).
"""

from __future__ import annotations

from typing import Any


def emit_laplace_preamble(ctx: dict[str, Any]) -> str:
    """Emit the Laplace preamble section (phi/theta split, log_joint_fn, factories).

    Inserted between the standard preamble and the warmup body for laplace_*
    recipes. D8 compliant: imports from blackjax.mcmc.laplace_marginal only.

    Parameters
    ----------
    ctx : dict
        Substitution context from ``emit_script()``.
        Required keys: model_name, phi_sites_repr, theta_sites_repr,
        laplace_factories_expr.

    Returns
    -------
    str
        Python source for the Laplace preamble block.
    """
    lines: list[str] = []
    a = lines.append

    a("# === LAPLACE: phi/theta split and LaplaceMarginal factories ===")
    a(
        f"# Model: {ctx['model_name']} -- phi sites: {ctx['phi_sites_repr']}, theta sites: {ctx['theta_sites_repr']}"
    )
    a("#")
    a("# D8 compliant: zero `import tuningfork` in the inference path.")
    a(
        "# blackjax.mcmc.laplace_marginal is part of the blackjax package, not tuningfork."
    )
    a("from blackjax.mcmc.laplace_marginal import laplace_marginal_factory as _lmf")
    a("")
    a(f"_phi_sites = {ctx['phi_sites_repr']}")
    a(f"_theta_sites = {ctx['theta_sites_repr']}")
    a("")
    a("# Save the joint logdensity_fn from preamble before overriding.")
    a("_joint_logdensity_fn = logdensity_fn")
    a("")
    a(
        "# Factored joint: log p(theta, phi | y) = _joint_logdensity_fn({**theta, **phi})"
    )
    a("# This is the log_joint_fn expected by blackjax.laplace_hmc / laplace_mhmc etc.")
    a("def log_joint_fn(theta, phi):")
    a("    return _joint_logdensity_fn({**theta, **phi})")
    a("")
    a("")
    a("# Split the full initial position into phi and theta components.")
    a(
        "# build_logdensity_fn() initialises all sites to prior mean (0 for log-scale params)."
    )
    a("_full_init = init_position")
    a("theta_init = {k: _full_init[k] for k in _theta_sites}")
    a("phi_init = {k: _full_init[k] for k in _phi_sites}")
    a("")
    a("# Override: warmup templates and sampler templates see phi-space only.")
    a("init_position = phi_init")
    a("")
    a("# Build one LaplaceMarginal per warmup phase.")
    a("# LaplaceMarginal(phi[, theta_prev=None]) -> (lp: float, theta_star: pytree)")
    a("#")
    a(
        "# window_adaptation(blackjax.laplace_hmc, laplace, ...) uses the LaplaceMarginal as"
    )
    a("# its logdensity_fn.  laplace_hmc.init(phi, laplace) calls:")
    a("#   jax.value_and_grad(laplace, has_aux=True)(phi)")
    a("# which requires the (lp, aux) return contract that LaplaceMarginal satisfies.")
    a(
        "# (A plain phi->float marginal would fail here because has_aux=True expects a tuple.)"
    )
    a(f"_laplace_warmup = [{ctx['laplace_factories_expr']}]")
    a("")
    a(
        "# Set logdensity_fn to phase 1 LaplaceMarginal -- picked up by the first warmup template."
    )
    a("logdensity_fn = _laplace_warmup[0]")

    return "\n".join(lines)
