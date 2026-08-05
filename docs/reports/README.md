# Performance report history

Sequential log of performance investigations and reports for this solver. Each report is self-contained (states what changed since the previous one and what was measured); read them in order for the full history, or just the latest for current state.

- [001_p1_p4_filo2_baseline.md](001_p1_p4_filo2_baseline.md) — Initial investigation into the suspicious P=4 speedup claim. Root-caused the iteration-budget mismatch between P=1/P=4, fixed the stale `Num Routes` header and Stage 3 cost-bookkeeping drift, ran a real FILO2 baseline. Flagged (not fixed) a capacity-violation bug found under a stress-test iteration count.
- [002_capacity_fix_and_rebalance.md](002_capacity_fix_and_rebalance.md) — Root-caused and fixed the capacity-violation bug (missing route-info rescan in Stage 5). Rebalanced the P=4 iteration budget to give every thread the same absolute search budget as P=1. Re-measured against FILO2, found diminishing returns from more iterations alone, and lays out the next concrete steps (extend Stage 5/Stage 3 budgets) to close the remaining ~7% cost gap.
