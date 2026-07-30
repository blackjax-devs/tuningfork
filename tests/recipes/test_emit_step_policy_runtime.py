"""JAX execution smoke tests for emitted step-policy functions."""

import jax
import jax.numpy as jnp
import pytest

from tuningfork.recipes._emit._step_policy import emit_step_policy

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    "spec",
    [
        None,
        {"kind": "uniform_int", "low": 2, "high": 10},
        {"kind": "empirical", "values": [2, 4, 8], "weights": [1, 2, 1]},
        {"kind": "poisson", "lam": 8, "low": 1},
        {"kind": "poisson", "lam": 8, "low": 1, "high": 16},
        {"kind": "log_uniform_int", "low": 1, "high": 64},
        {"kind": "pow2_choice", "options": [1, 2, 4, 8]},
    ],
)
def test_emitted_policy_executes_and_jits(spec):
    namespace = {"jax": jax, "jnp": jnp}
    exec(emit_step_policy(spec), namespace)
    fn = jax.jit(namespace["_integration_steps_fn"])
    sample = fn(jax.random.key(0))
    assert sample.shape == ()
    assert int(sample) >= 1
    if spec is None:
        assert int(sample) < 10
    elif spec["kind"] == "uniform_int":
        assert spec["low"] <= int(sample) < spec["high"]
    elif spec["kind"] == "empirical":
        assert int(sample) in spec["values"]
    elif spec["kind"] == "poisson" and spec.get("high") is not None:
        assert int(sample) < spec["high"]
    elif spec["kind"] == "log_uniform_int":
        assert spec["low"] <= int(sample) <= spec["high"]
    elif spec["kind"] == "pow2_choice":
        assert int(sample) in spec["options"]
