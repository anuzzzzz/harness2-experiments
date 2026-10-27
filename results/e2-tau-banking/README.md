# τ³-bench banking_knowledge — customer service over a knowledge base

**Question.** Does a harness that Harness² refines on 3 banking conversations raise Claude Haiku 5.5's pass rate on 24 held-out conversations, where answers depend on a 698-document knowledge base searched with BM25?

**Verdict:** not run yet  
Held-out tasks: 0 (×2 repeats) · base – → final – (Δ – pts, 95% CI – to –; tasks better/worse/same 0/0/0, sign test p=–)  
In-task (paper protocol, train tasks): base – → adopted – over 0 task(s)  
Spend: $0.00 of $16.5

Metric: tau2 reward per conversation (1 = the database ends in the expected state, else 0). Models: executor `anthropic/claude-haiku-5-5`, user `anthropic/claude-haiku-5-5`, improver `anthropic/claude-sonnet-5-5`

Pre-registered tasks ([`tasks.lock.json`](../../experiments/e2-tau-banking/tasks.lock.json)): 3 train (`task_017`, `task_050`, `task_093`), 24 held-out test. Budget caps: train $10, eval $6.

| arm | pass^1 | pass^2 |
|---|---|---|
| base | – | – |
| final | – | – |

## Per task

| task | base | final |
|---|---|---|
