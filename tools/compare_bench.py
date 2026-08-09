"""
compare_bench.py -- paired comparison between two tools/bench.py CSVs (e.g. before/after a
single fix), matched by (instance, seed) so the comparison is apples-to-apples rather than
comparing different seed sets. Used throughout docs/reports/005_cost_optimization.md's
per-change attribution table.

Usage:
    python tools/compare_bench.py results/bench/000_baseline_tier1.csv results/bench/004_phase1_1_ruin_seed_fix.csv
"""
import csv
import sys


def load(path):
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["feasible"] != "True":
                continue
            try:
                cost = float(row["cost"])
            except (ValueError, TypeError):
                continue
            rows[(row["instance"], row["seed"])] = cost
    return rows


def main():
    if len(sys.argv) != 3:
        print("Usage: python tools/compare_bench.py <before.csv> <after.csv>")
        sys.exit(1)
    before, after = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(before) & set(after))
    missing_before = set(after) - set(before)
    missing_after = set(before) - set(after)

    if not keys:
        print("No overlapping (instance, seed) pairs with feasible costs in both files.")
        sys.exit(1)

    per_instance = {}
    for inst, seed in keys:
        per_instance.setdefault(inst, []).append((before[(inst, seed)], after[(inst, seed)]))

    print(f"{'instance':<16} {'n':>3} {'before_mean':>14} {'after_mean':>14} {'delta_pct':>10}")
    total_before = total_after = 0.0
    total_n = 0
    for inst in sorted(per_instance):
        pairs = per_instance[inst]
        b_mean = sum(p[0] for p in pairs) / len(pairs)
        a_mean = sum(p[1] for p in pairs) / len(pairs)
        delta = 100.0 * (a_mean - b_mean) / b_mean
        print(f"{inst:<16} {len(pairs):>3} {b_mean:>14.1f} {a_mean:>14.1f} {delta:>9.3f}%")
        total_before += sum(p[0] for p in pairs)
        total_after += sum(p[1] for p in pairs)
        total_n += len(pairs)

    overall_delta = 100.0 * (total_after - total_before) / total_before
    print(f"\nOverall (n={total_n} paired runs): mean cost delta = {overall_delta:+.3f}%")
    if missing_before:
        print(f"Note: {len(missing_before)} (instance,seed) pairs only in 'after' file (ignored)")
    if missing_after:
        print(f"Note: {len(missing_after)} (instance,seed) pairs only in 'before' file (ignored)")


if __name__ == "__main__":
    main()
