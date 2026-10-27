"""Run one Harbor trial (Terminal-Bench 2.0, SWE-bench Pro) with Claude Code and an h2x harness.

``agent`` imports harbor and is only loaded inside ``harbor run`` (via --agent module:Class);
``streams`` is pure and is what the Harness2 process uses to render trajectories.
"""
