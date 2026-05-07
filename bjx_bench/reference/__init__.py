"""Reference cache layer for bjx-bench.

The only module callers (CLI, runner, tests) should import is
``bjx_bench.reference._io``:

    from bjx_bench.reference._io import get_reference_draws

Everything in ``_io`` goes through a load-or-generate resolution path that
handles analytic models (Path A) and long-NUTS models (Path B).
"""
