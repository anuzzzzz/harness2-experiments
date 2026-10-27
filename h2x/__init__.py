"""h2x: pre-registered experiments measuring whether Harness2 improves agent benchmark results.

The package has three layers:

* ``h2x.suites``: Harness2 suite adapters (``harness2.suites`` entry points) that turn a benchmark
  into tasks, rollouts and hidden grades. The improver never sees a grade.
* ``h2x.tau_runner`` / ``h2x.harbor_runner`` / ``h2x.lab``: subprocess runners that execute one
  rollout inside the benchmark's own framework and write the trajectory plus a hidden grade.
* ``h2x.cli``: ``plan``, ``train``, ``eval``, ``report`` used by the GitHub Actions workflow.
"""

__version__ = "0.1.0"
