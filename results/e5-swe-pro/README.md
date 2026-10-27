# SWE-bench Pro (openlibrary) — repository issues

**Question.** Does a Claude Code harness that Harness² refines on 3 openlibrary issues raise Claude Haiku 5.5's resolve rate on 10 other openlibrary issues, i.e. does repository knowledge learned by the harness transfer?

**Verdict:** not run yet  
Held-out tasks: 0 (×1 repeats) · base – → final – (Δ – pts, 95% CI – to –; tasks better/worse/same 0/0/0, sign test p=–)  
In-task (paper protocol, train tasks): base – → adopted – over 0 task(s)  
Spend: $0.01 of $23.5

Metric: Harbor verifier reward (1 = the instance's fail-to-pass and pass-to-pass tests pass, else 0). Models: executor `anthropic/claude-haiku-5-5`, improver `anthropic/claude-sonnet-5-5`

Pre-registered tasks ([`tasks.lock.json`](../../experiments/e5-swe-pro/tasks.lock.json)): 3 train (`instance_internetarchive__openlibrary-6afdb09df692223c3a31df65cfa92f15e5614c01-v08d8e8889ec945ab821fb156c04c7d2e2810debb`, `instance_internetarchive__openlibrary-c05ccf2cd8baa81609434e0e35c4a63bc0da5a25-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4`, `instance_internetarchive__openlibrary-f3b26c2c0721f8713353fe4b341230332e30008d-v0f5aece3601a5b4419f7ccec1dbda2071be28ee4`), 10 held-out test. Budget caps: train $14, eval $9.

## Per task

| task | base | final |
|---|---|---|
