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

All slot resolution is done in Python; no slot markers remain in the output.

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
    a("# Info fields are resolved per sampler family at emit time.")
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
    a("    # NumPyro models return dict positions.")
    a("    _draws_dict = {k: np.asarray(v) for k, v in _pos.items()}")
    stat_prefix = ctx["sample_stat_prefix"]
    a(
        f"    _reserved_positions = sorted(k for k in _draws_dict "
        f"if k.startswith({stat_prefix!r}))"
    )
    a("    if _reserved_positions:")
    a(
        f"        raise ValueError('position names use reserved generated-stat prefix "
        f"{stat_prefix}: ' + ', '.join(_reserved_positions))"
    )
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
    a("# Persist compact, strict-JSON generated-run telemetry.")
    a("# Written before success timing/exit; never contains per-step arrays.")
    a("try:")
    a("    def _telemetry_json(value):")
    a("        if value is None or isinstance(value, (str, bool, int)):")
    a("            return value")
    a("        if isinstance(value, float):")
    a("            if not np.isfinite(value):")
    a("                raise ValueError('telemetry value is not finite')")
    a("            return value")
    a("        if isinstance(value, dict):")
    a("            if any(not isinstance(k, str) for k in value):")
    a("                raise TypeError('telemetry mapping keys must be strings')")
    a("            return {k: _telemetry_json(v) for k, v in value.items()}")
    a("        if all(hasattr(value, _name) for _name in ('sigma', 'U', 'lam')):")
    a(
        "            return {'type': 'low_rank_inverse_mass_matrix', 'sigma': _telemetry_json(value.sigma), 'U': _telemetry_json(value.U), 'lam': _telemetry_json(value.lam)}"
    )
    a("        if isinstance(value, (list, tuple)):")
    a("            return [_telemetry_json(v) for v in value]")
    a("        if isinstance(value, np.ndarray) or hasattr(value, 'shape'):")
    a("            return _telemetry_json(np.asarray(value).tolist())")
    a("        if isinstance(value, np.generic):")
    a("            return _telemetry_json(value.item())")
    a("        raise TypeError(f'unsupported telemetry value: {type(value).__name__}')")
    a("    _geometry = {")
    for key, expr in ctx.get("telemetry_geometry_expr", {}).items():
        a(f"        {key!r}: {expr},")
    a("    }")
    a(
        "    _geometry_unavailable_reason = "
        f"{ctx.get('telemetry_geometry_unavailable_reason')!r}"
    )
    a(f"    _geometry_source = {ctx.get('telemetry_geometry_source')!r}")
    a(f"    _geometry_scope = {ctx.get('telemetry_geometry_scope')!r}")
    a(
        "    if _geometry_unavailable_reason is None and any(value is None for value in _geometry.values()):"
    )
    a("        _geometry = {}")
    a("        _geometry_source = 'unavailable'")
    a("        _geometry_scope = None")
    a(
        "        _geometry_unavailable_reason = 'adapted geometry fields were not returned by warmup'"
    )
    a("    _fixed = {}")
    if ctx.get("fixed_num_integration_steps") is not None:
        a(
            f"    _fixed['num_integration_steps'] = {ctx['fixed_num_integration_steps']!r}"
        )
    a("    _manifest = json.loads(EXECUTION_MANIFEST_JSON)")
    a("    _telemetry = {")
    a(
        f"        'schema': {ctx.get('telemetry_schema', 'tuningfork.generated-run-telemetry.v2')!r},"
    )
    a("        'plan_hash': _manifest['plan_hash'],")
    a("        'executable_config_hash': _manifest['executable_config_hash'],")
    a("        'draws_artifact': _manifest['normalized_plan']['artifact_filename'],")
    a("        'geometry': _telemetry_json(_geometry),")
    a("        'geometry_source': _geometry_source,")
    a("        'geometry_scope': _geometry_scope,")
    a("        'geometry_unavailable_reason': _geometry_unavailable_reason,")
    a("        'fixed': _telemetry_json(_fixed),")
    a(
        "        'timing_seconds': {'warmup': _warmup_wall, 'sampling': _sampling_wall, 'total': _recipe_wall},"
    )
    a("        'warmup_grad_evals': _warmup_grad_evals,")
    a("        'warmup_grad_evals_reason': _warmup_grad_evals_reason,")
    a(
        "        'resolved_step_policy': "
        f"_telemetry_json({ctx.get('telemetry_resolved_step_policy_expr', 'None')}),"
    )
    a("    }")
    a(
        "    _telemetry_path = _manifest['normalized_plan']['telemetry_artifact_filename']"
    )
    a("    with open(_telemetry_path, 'w', encoding='utf-8') as _telemetry_file:")
    a(
        "        json.dump(_telemetry_json(_telemetry), _telemetry_file, sort_keys=True, separators=(',', ':'), allow_nan=False)"
    )
    a("        _telemetry_file.write('\\n')")
    a('    print(f"[telemetry written to {_telemetry_path}]")')
    a("except Exception as _telemetry_exc:")
    a(
        "    raise RuntimeError(f'generated telemetry emission failed closed: {_telemetry_exc}') from _telemetry_exc"
    )
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
