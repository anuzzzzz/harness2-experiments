# τ²-bench airline — customer service

**Question.** Does a harness that Harness² refines on 3 airline conversations raise Claude Haiku 5.5's pass rate on the 20 held-out conversations of tau2's official airline test split?

**Verdict:** not run yet  
Held-out tasks: 0 (×3 repeats) · base – → final – (Δ – pts, 95% CI – to –; tasks better/worse/same 0/0/0, sign test p=–)  
In-task (paper protocol, train tasks): base – → adopted – over 0 task(s)  
Spend: $0.02 of $15.5

Metric: tau2 reward per conversation (1 = the database ends exactly as the policy-correct resolution leaves it and required information was communicated, else 0). Models: executor `anthropic/claude-haiku-5-5`, user `anthropic/claude-haiku-5-5`, improver `anthropic/claude-sonnet-5-5`

Pre-registered tasks ([`tasks.lock.json`](../../experiments/e3-tau-airline/tasks.lock.json)): 3 train (`11`, `28`, `33`), 20 held-out test. Budget caps: train $10, eval $5.

| arm | pass^1 | pass^2 | pass^3 |
|---|---|---|---|
| base | – | – | – |
| final | – | – | – |

## Per task

| task | base | final |
|---|---|---|
