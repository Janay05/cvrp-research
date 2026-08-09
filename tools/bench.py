"""
bench.py -- measurement harness for docs/reports/005_cost_optimization.md.

Runs the compiled solver over (instance x seed) combinations, independently verifies
every single run with src/verifier.py (never sampled -- an infeasible result is a
stop-the-line bug, not a data point), scores each result against a reference cost via
tools/score_sol.py where a reference .sol is available, and writes one CSV row per run
plus a mean-gap-to-reference summary.

This exists because report 004 could not separate a real ~0.3% gain from run-to-run noise
(the solver was fully deterministic prior to this phase's --seed flag). Every fix in
Phase 1+ is expected to move cost by roughly that order of magnitude, so gains must be
measured over multiple seeds, not eyeballed from one run.

Runs are sequential by design (not parallelized against each other): the solver always
writes to the same results/final_solution.txt and results/run_log.txt, so concurrent runs
would race on those files. Each run's solution is copied out to a unique path immediately
after the run completes, before the next run starts.

Usage:
    python tools/bench.py --instances tools/tier1_instances.txt --seeds 1,2,3,4,5 \
        --extra-args "-p 4" --tag baseline

    python tools/bench.py --instances data/instances/I/Valle-D-Aosta.vrp --seeds 1,2,3 \
        --extra-args "-p 4 --stage2-ms 40000 --stage3-ms 1000 --stage5-ms 60000" --tag tier2_baseline
"""
import argparse
import csv
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
from score_sol import score_solution  # noqa: E402

DEFAULT_EXE = os.path.join(REPO_ROOT, "src", "build", "Release", "cvrp_parallel.exe")
VERIFIER = os.path.join(REPO_ROOT, "src", "verifier.py")
FINAL_SOL = os.path.join(REPO_ROOT, "results", "final_solution.txt")


def find_reference_sol(vrp_path):
    """Mirrors the directory layout of data/bks/<Set>/<Set>/<name>.sol and the special-cased
    I-set reference vendored at baselines/filo2/results/i-bks/<name>.sol (see the docstring
    of score_sol.py for the id-remapping convention both use). Returns None if no reference
    is vendored for this instance (e.g. the hgs_partitions/ chunk files)."""
    stem = os.path.splitext(os.path.basename(vrp_path))[0]
    set_name = os.path.basename(os.path.dirname(os.path.abspath(vrp_path)))
    if set_name == "I":
        candidate = os.path.join(REPO_ROOT, "baselines", "filo2", "results", "i-bks", stem + ".sol")
    else:
        candidate = os.path.join(REPO_ROOT, "data", "bks", set_name, set_name, stem + ".sol")
    return candidate if os.path.exists(candidate) else None


def load_instance_list(spec):
    """spec is either a single .vrp path, a comma-separated list of .vrp paths, or a text
    file with one .vrp path per line (blank lines and '#' comments ignored)."""
    if "," in spec:
        return [s.strip() for s in spec.split(",") if s.strip()]
    if spec.endswith(".vrp"):
        return [spec]
    paths = []
    with open(spec) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(line)
    return paths


_COST_RE = re.compile(r"Final cost:\s*(\d+)")
_TIME_RE = re.compile(r"Total time:\s*([\d.]+)\s*ms")


def run_once(exe, vrp_path, seed, extra_args, timeout_s):
    cmd = [exe, "-f", vrp_path, "--seed", str(seed)] + shlex.split(extra_args)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"error": f"TIMEOUT after {timeout_s}s"}
    wall_ms = (time.perf_counter() - t0) * 1000.0
    if proc.returncode != 0:
        return {"error": f"solver exited {proc.returncode}: {proc.stderr[-2000:]}"}

    out = proc.stdout
    m_cost = _COST_RE.search(out)
    m_time = _TIME_RE.search(out)
    solver_cost = int(m_cost.group(1)) if m_cost else None
    solver_ms = float(m_time.group(1)) if m_time else None
    return {"wall_ms": wall_ms, "solver_ms": solver_ms, "solver_cost": solver_cost, "stdout": out}


def verify_solution(vrp_path, sol_path):
    proc = subprocess.run([sys.executable, VERIFIER, vrp_path, sol_path],
                           cwd=REPO_ROOT, capture_output=True, text=True)
    feasible = proc.returncode == 0 and "Verification SUCCESS" in proc.stdout
    return feasible, proc.stdout + proc.stderr


def parse_num_routes(sol_path):
    with open(sol_path) as f:
        for line in f:
            if line.startswith("Num Routes:"):
                return int(line.split(":")[1].strip())
    return None


FIELDNAMES = ["instance", "seed", "cost", "wall_ms", "solver_ms", "num_routes", "feasible",
              "ref_cost", "gap_pct", "retried"]


def load_done_pairs(out_csv):
    """For --resume: (instance, seed) pairs already present in an existing CSV, so a run
    interrupted partway through (e.g. by the intermittent hang noted in
    docs/reports/005_cost_optimization.md, Phase 0) can pick back up without re-running
    everything that already succeeded."""
    done = set()
    if not os.path.exists(out_csv):
        return done
    with open(out_csv, newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["instance"], int(row["seed"])))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", required=True, help="a .vrp path, comma-separated paths, or a list file")
    ap.add_argument("--seeds", default="1", help="comma-separated seed list, e.g. 1,2,3,4,5")
    ap.add_argument("--extra-args", default="-p 4", help="extra CLI args passed to the solver, quoted")
    ap.add_argument("--tag", required=True, help="label for this run, used in output paths")
    ap.add_argument("--exe", default=DEFAULT_EXE)
    ap.add_argument("--out", default=None, help="CSV output path; default results/bench/<tag>.csv")
    ap.add_argument("--timeout", type=float, default=120.0,
                     help="per-run timeout in seconds (default 120; raise for large Tier-2/3 instances)")
    ap.add_argument("--resume", action="store_true",
                     help="skip (instance, seed) pairs already present in --out, append new rows to it")
    args = ap.parse_args()

    instances = load_instance_list(args.instances)
    seeds = [int(s) for s in args.seeds.split(",")]
    out_csv = args.out or os.path.join(REPO_ROOT, "results", "bench", f"{args.tag}.csv")
    sol_dir = os.path.join(REPO_ROOT, "results", "bench", args.tag)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(sol_dir, exist_ok=True)

    done_pairs = load_done_pairs(out_csv) if args.resume else set()
    write_mode = "a" if (args.resume and done_pairs) else "w"
    csv_file = open(out_csv, write_mode, newline="")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if write_mode == "w":
        writer.writeheader()
        csv_file.flush()

    all_rows = []  # kept in memory too, only for the end-of-run summary print
    timeouts = []

    def write_row(row):
        writer.writerow(row)
        csv_file.flush()
        all_rows.append(row)

    for vrp_path in instances:
        vrp_abs = vrp_path if os.path.isabs(vrp_path) else os.path.join(REPO_ROOT, vrp_path)
        stem = os.path.splitext(os.path.basename(vrp_path))[0]
        ref_sol = find_reference_sol(vrp_abs)
        ref_cost = None
        if ref_sol:
            ref_cost, _, _, ref_complete = score_solution(vrp_abs, ref_sol)
            if not ref_complete:
                print(f"WARNING: reference solution for {stem} is incomplete; ignoring it", file=sys.stderr)
                ref_cost = None

        for seed in seeds:
            if (stem, seed) in done_pairs:
                continue
            print(f"[{args.tag}] {stem} seed={seed} ...", file=sys.stderr)
            result = run_once(args.exe, vrp_abs, seed, args.extra_args, args.timeout)
            retried = False
            if "error" in result and result["error"].startswith("TIMEOUT"):
                # The solver has an intermittent hang (see docs/reports/005_cost_optimization.md,
                # Phase 0) not tied to any specific instance -- a fresh process on the same
                # (instance, seed) has run to completion in seconds immediately afterward every
                # time this has been observed. One retry avoids losing an otherwise-good batch to
                # a rare event, but `retried=True` is recorded so how often this actually happens
                # stays visible instead of silently masked.
                print(f"  TIMEOUT after {args.timeout}s -- retrying once", file=sys.stderr)
                result = run_once(args.exe, vrp_abs, seed, args.extra_args, args.timeout)
                retried = True

            if "error" in result:
                print(f"  ERROR: {result['error']}", file=sys.stderr)
                write_row({"instance": stem, "seed": seed, "cost": "", "wall_ms": "", "solver_ms": "",
                           "num_routes": "", "feasible": "ERROR", "ref_cost": ref_cost or "", "gap_pct": "",
                           "retried": retried})
                continue

            dest_sol = os.path.join(sol_dir, f"{stem}_seed{seed}.txt")
            shutil.copyfile(FINAL_SOL, dest_sol)
            feasible, verifier_out = verify_solution(vrp_abs, dest_sol)
            num_routes = parse_num_routes(dest_sol)

            if not feasible:
                print(f"  INFEASIBLE -- verifier output:\n{verifier_out}", file=sys.stderr)

            cost = result["solver_cost"]
            gap_pct = None
            if feasible and ref_cost and cost is not None:
                gap_pct = 100.0 * (cost - ref_cost) / ref_cost

            if retried:
                timeouts.append((stem, seed))

            write_row({
                "instance": stem, "seed": seed, "cost": cost,
                "wall_ms": round(result["wall_ms"], 1),
                "solver_ms": result["solver_ms"],
                "num_routes": num_routes,
                "feasible": feasible,
                "ref_cost": ref_cost or "",
                "gap_pct": round(gap_pct, 4) if gap_pct is not None else "",
                "retried": retried,
            })

    csv_file.close()

    gaps = [r["gap_pct"] for r in all_rows if isinstance(r["gap_pct"], float)]
    infeasible = [r for r in all_rows if r["feasible"] not in (True,)]
    print(f"\nWrote {len(all_rows)} new row(s) to {out_csv} ({len(done_pairs)} pre-existing rows kept)")
    if gaps:
        print(f"Mean gap to reference: {sum(gaps)/len(gaps):.3f}%  (min {min(gaps):.3f}%, max {max(gaps):.3f}%, n={len(gaps)})")
    else:
        print("No reference costs available for any run -- gap summary skipped.")
    if infeasible:
        print(f"*** {len(infeasible)} run(s) were NOT verified feasible -- see stderr above ***")
    if timeouts:
        print(f"*** {len(timeouts)} run(s) needed a retry after a timeout: {timeouts} ***")


if __name__ == "__main__":
    main()
