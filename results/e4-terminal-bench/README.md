# Terminal-Bench 2.0 — terminal tasks in containers

**Question.** Does a Claude Code harness (CLAUDE.md, skills, scripts, hooks) that Harness² refines on 3 Terminal-Bench 2.0 tasks raise Claude Haiku 5.5's pass rate on 12 different held-out tasks?

**Verdict:** not run yet  
Held-out tasks: 0 (×1 repeats) · base – → final – (Δ – pts, 95% CI – to –; tasks better/worse/same 0/0/0, sign test p=–)  
In-task (paper protocol, train tasks): base – → adopted – over 0 task(s)  
Spend: $0.00 of $22

Metric: Harbor verifier reward (1 = the task's tests pass after the agent finishes, else 0). Models: executor `anthropic/claude-haiku-5-5`, improver `anthropic/claude-sonnet-5-5`

Pre-registered tasks ([`tasks.lock.json`](../../experiments/e4-terminal-bench/tasks.lock.json)): 3 train (`db-wal-recovery`, `git-multibranch`, `sanitize-git-repo`), 12 held-out test. Budget caps: train $13, eval $8.5.

## Per task

| task | base | final |
|---|---|---|
