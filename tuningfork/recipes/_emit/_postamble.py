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
"""Emit-time Python function for the recipe postamble section.

Replaces ``_templates/postamble.py.tmpl`` (50 LOC, string.Template).
All slot resolution is done in Python -- no $slot markers in the output.

The info-diagnostics and draws-stats blocks are passed in via ctx (already
built by ``_build_info_diagnostics_block`` / ``_build_draws_ss_block`` in
``_emit_script.py``).  No new descriptors needed -- the existing
``info_diagnostics_block`` and ``draws_ss_block`` context entries carry them.
"""

from __future__ import annotations

from typing import Any


def emit_postamble(ctx: dict[str, Any]) -> str:
    """Emit the postamble section of the recipe reproduction script.

    Parameters
    ----------
    ctx : dict
        Substitution context from ``emit_script()``.
        Required keys: info_diagnostics_block, draws_ss_block,
        diagnostics_close_body, model_name, base_method_name, warmup_name,
        recipe_id.

    Returns
    -------
    str
        Python source for the postamble block.
    """
    lines: list[str] = []
    a = lines.append

    a("# SYNC: block until sampling scan completes before any host materialisation.")
    a("# Syncs both _samples (positions) and _infos (diagnostics) -- they are outputs")
    a("# of the same jax.lax.scan and must be fully realised before int()/float()")
    a("# calls below or before stamping the wall clock.  Without this, int(jnp.sum(")
    a("# _infos.is_divergent)) would force sync via the buffer protocol under")
    a("# buffer-pool contention (same deadlock pattern as the gp_regression recert")
    a("# hang, 2026-05-28).")
    a("jax.block_until_ready((_samples, _infos))")
    a(ctx["diagnostics_close_body"])
    a("")
    a("# Timing split: warmup vs sampling -- stamped immediately after the sync so the")
    a("# clock captures actual compute wall, not dispatch latency.")
    a("_sampling_wall = _recipe_time.perf_counter() - _warmup_t1")
    a("_recipe_wall = _recipe_time.perf_counter() - _recipe_t0")
    a("")
    a("# Print headline diagnostics for verification.")
    a(
        "# T1.5: info fields resolved per sampler family at emit time (no hasattr probes)."
    )
    a(ctx["info_diagnostics_block"])
    a("")
    a("# ---------------------------------------------------------------------------")
    a("# Persist draws as .npz -- no external library dependency.")
    a("# ---------------------------------------------------------------------------")
    a(
        "# _samples.position is a pytree dict[str, array(num_chains, num_samples, *shape)]"
    )
    a("# for numpyro models.  We save the posterior arrays directly so users can load")
    a("# them with np.load() or pass them to arviz offline:")
    a('#   draws = np.load("...<recipe>.draws.npz")')
    a('#   az.from_dict({"posterior": {k: draws[k] for k in draws.files}})')
    a("try:")
    a("    _pos = _samples.position")
    a(
        "    # T1.5: numpyro models always return dict positions; pytree-ravel fallback removed."
    )
    a("    _draws_dict = {k: np.asarray(v) for k, v in _pos.items()}")
    a("    # Optionally include per-step sample stats as extra keys (prefixed '_ss_').")
    a(ctx["draws_ss_block"])
    model = ctx["model_name"]
    method = ctx["base_method_name"]
    warmup = ctx["warmup_name"]
    a(
        f'    _npz_path = "{model}" + "__" + "{method}" + "__" + "{warmup}" + ".draws.npz"'
    )
    a("    np.savez(_npz_path, **_draws_dict)")
    a('    print(f"[draws written to {_npz_path}]")')
    a("except Exception as _save_exc:")
    a("    print(")
    a(
        '        f"[WARNING] draws persist failed -- inference output is still valid; npz skipped. "'
    )
    a('        f"Reason: {_save_exc}"')
    a("    )")
    a("")
    recipe_id = ctx["recipe_id"]
    a("print(")
    a(
        f"    f\"[recipe='{recipe_id}'] n_divergences={{_n_div}} mean_acceptance={{_acceptance:.3f}}\""
    )
    a('    f" warmup_wall_seconds={_warmup_wall:.1f}"')
    a('    f" sampling_wall_seconds={_sampling_wall:.1f}"')
    a('    f" wall_seconds={_recipe_wall:.1f}"')
    a(")")
    a(
        'print("TUNINGFORK_TIMINGS " + json.dumps({"sampling_seconds": _sampling_wall, "total_seconds": _recipe_wall, "warmup_seconds": _warmup_wall}, sort_keys=True, separators=(",", ":")))'
    )
    a("# Naive scalar verdict signal -- sufficient for the round-trip CI gate.")
    a('print("DONE")')

    return "\n".join(lines)
