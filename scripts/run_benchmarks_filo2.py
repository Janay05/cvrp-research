#!/usr/bin/env python3
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
"""
FILO2 Benchmark Runner

Iterates over CVRP instance files, executes the filo2 solver for each one,
monitors peak memory usage and wall-clock time, parses the final route cost,
and appends cleanly-formatted rows to results.csv.

CSV columns (matching Google Sheet):
  Instance Name | Size (N) | Seed | Peak Memory (MB) | Exec Time (s) | Cost | Notes/Hardware

Usage:
    python run_benchmarks.py                         # Run all instances in instances/
    python run_benchmarks.py --folder instances/B    # Run only the B dataset
    python run_benchmarks.py --seed 42               # Use a specific seed
    python run_benchmarks.py --iterations 1000       # Override coreopt iterations
    python run_benchmarks.py --exe build/filo2.exe   # Custom executable path
"""

import argparse
import csv
import os
import platform
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    print("ERROR: psutil is required for memory monitoring.")
    print("Install it with:  pip install psutil")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_EXE = PROJECT_ROOT / "baselines" / "filo2" / "build" / "filo2.exe"
DEFAULT_INSTANCE_DIR = PROJECT_ROOT / "data" / "instances"
DEFAULT_CSV = PROJECT_ROOT / "results" / "results_filo2.csv"
DEFAULT_OUTPATH = PROJECT_ROOT / "results" / "benchmark_outputs_filo2"

INSTANCE_EXTENSIONS = {".vrp", ".txt"}

CSV_COLUMNS = [
    "Instance Name",
    "Size (N)",
    "Seed",
    "Peak Memory (MB)",
    "Exec Time (s)",
    "Cost",
    "Notes/Hardware",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_hardware_tag() -> str:
    """Build a short hardware description string."""
    cpu = platform.processor() or "Unknown CPU"
    try:
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        ram_gb = "?"
    return f"Local {platform.node()} ({cpu}, {ram_gb}GB RAM)"


def parse_dimension_from_file(filepath: Path) -> int:
    """
    Read the DIMENSION field from the instance file header.
    Returns the number of nodes (including depot).
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.upper().startswith("DIMENSION"):
                # Handle formats like "DIMENSION : 4001" or "DIMENSION: 4001"
                parts = line.split(":")
                if len(parts) >= 2:
                    return int(parts[1].strip())
    return -1


def parse_cost_from_stdout(stdout: str) -> float:
    """
    Parse the best objective value from filo2's VERBOSE stdout.
    Looks for the line: "obj = <cost>, n. routes = <routes>"
    Falls back to other patterns if needed.
    """
    # Primary pattern (VERBOSE): "obj = 123456, n. routes = 77"
    # Handles integers, floats, and scientific notation (e.g. 1.9873e+05)
    # We want the LAST occurrence (the "Best solution found" one)
    matches = re.findall(r"obj\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)", stdout)
    if matches:
        return float(matches[-1])

    return -1.0


def parse_cost_from_outfile(outpath: Path, instance_path: Path, seed: int) -> float:
    """
    Parse cost from the .out file that filo2 writes.
    File format: <cost>\t<time>\n

    Note: filo2's get_basename() splits on '/' which doesn't work on Windows,
    so it may use the full path as the filename. We try both patterns.
    """
    # Try the simple basename first  (what it should be)
    candidates = [
        outpath / f"{instance_path.name}_seed-{seed}.out",
    ]
    # Also try the full-path mangled name filo2 actually writes on Windows
    full_path_name = str(instance_path).replace("\\", "/").split("/")[-1]
    if full_path_name != instance_path.name:
        candidates.append(outpath / f"{full_path_name}_seed-{seed}.out")
    # Also scan outpath for any .out files matching *_seed-{seed}.out
    # as a last resort
    for outfile in candidates:
        if outfile.exists():
            with open(outfile, "r") as f:
                content = f.read().strip()
                if content:
                    parts = content.split("\t")
                    if parts:
                        try:
                            return float(parts[0])
                        except ValueError:
                            continue
    # Last resort: scan outpath for any matching .out file
    for f in outpath.glob(f"*_seed-{seed}.out"):
        with open(f, "r") as fh:
            content = fh.read().strip()
            if content:
                parts = content.split("\t")
                if parts:
                    try:
                        return float(parts[0])
                    except ValueError:
                        continue
    return -1.0


class MemoryMonitor:
    """
    Monitors the peak RSS (Resident Set Size) of a running process
    by polling it in a background thread.
    """

    def __init__(self, pid: int, poll_interval: float = 0.1):
        self.pid = pid
        self.poll_interval = poll_interval
        self.peak_memory_bytes = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5)

    def get_peak_mb(self) -> float:
        return round(self.peak_memory_bytes / (1024 * 1024), 2)

    def _monitor(self):
        try:
            proc = psutil.Process(self.pid)
            while not self._stop_event.is_set():
                try:
                    mem_info = proc.memory_info()
                    current = mem_info.rss  # Resident Set Size
                    if current > self.peak_memory_bytes:
                        self.peak_memory_bytes = current
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                self._stop_event.wait(self.poll_interval)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Core Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_instance(
    exe: Path,
    instance_path: Path,
    seed: int,
    iterations: int,
    outpath: Path,
    hardware_tag: str,
) -> dict:
    """
    Run filo2 on a single instance file and return a dict of metrics.
    """
    instance_name = instance_path.name
    size_n = parse_dimension_from_file(instance_path)

    # Build the command
    cmd = [
        str(exe),
        str(instance_path),
        "--seed", str(seed),
        "--coreopt-iterations", str(iterations),
        "--outpath", str(outpath) + os.sep,
    ]

    print(f"\n{'='*70}")
    print(f"  Instance : {instance_name}")
    print(f"  Size (N) : {size_n}")
    print(f"  Seed     : {seed}")
    print(f"  Command  : {' '.join(cmd)}")
    print(f"{'='*70}")

    # Launch the process
    start_time = time.perf_counter()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Start memory monitoring
    mem_monitor = MemoryMonitor(process.pid, poll_interval=0.05)
    mem_monitor.start()

    # Capture output
    stdout, _ = process.communicate()
    elapsed = round(time.perf_counter() - start_time, 2)

    # Stop monitoring
    mem_monitor.stop()
    peak_mem_mb = mem_monitor.get_peak_mb()

    # Print stdout for live visibility
    if stdout:
        # Print last ~15 meaningful lines to avoid flooding the console
        lines = [l for l in stdout.strip().splitlines() if l.strip()]
        tail = lines[-15:] if len(lines) > 15 else lines
        print("  --- solver output (tail) ---")
        for l in tail:
            print(f"  {l}")
        print("  ----------------------------")

    # Parse cost – try the .out file first (most reliable), then stdout
    cost = parse_cost_from_outfile(outpath, instance_path, seed)
    if cost < 0:
        cost = parse_cost_from_stdout(stdout or "")

    # Return code check
    if process.returncode != 0:
        print(f"  [WARN] Solver exited with code {process.returncode}")

    result = {
        "Instance Name": instance_name,
        "Size (N)": size_n,
        "Seed": seed,
        "Peak Memory (MB)": peak_mem_mb,
        "Exec Time (s)": elapsed,
        "Cost": cost if cost >= 0 else "ERROR",
        "Notes/Hardware": hardware_tag,
    }

    print(f"  [OK] Cost={result['Cost']}  Time={elapsed}s  PeakMem={peak_mem_mb}MB")
    return result


def append_to_csv(csv_path: Path, row: dict):
    """Append a single result row to the CSV file, creating headers if needed."""
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def collect_instances(folder: Path) -> list[Path]:
    """Recursively collect all instance files (.vrp, .txt) under the given folder."""
    instances = []
    for root, _, files in os.walk(folder):
        for fname in sorted(files):
            fpath = Path(root) / fname
            if fpath.suffix.lower() in INSTANCE_EXTENSIONS:
                instances.append(fpath)
    return sorted(instances, key=lambda p: (p.parent.name, parse_dimension_from_file(p)))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FILO2 Benchmark Runner — automate instance solving and metric collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmarks.py                                  # Run all instances
  python run_benchmarks.py --folder instances/X             # Only X dataset
  python run_benchmarks.py --seed 42 --iterations 5000      # Custom seed & iterations
  python run_benchmarks.py --csv my_results.csv             # Custom output CSV
  python run_benchmarks.py --instances instances/B/Leuven1.txt instances/B/Leuven2.txt
        """,
    )
    parser.add_argument(
        "--exe",
        type=Path,
        default=DEFAULT_EXE,
        help=f"Path to the filo2 executable (default: {DEFAULT_EXE})",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=None,
        help="Folder containing instance files to run. Searched recursively. "
             "Defaults to the entire instances/ directory.",
    )
    parser.add_argument(
        "--instances",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit list of instance file paths to run (overrides --folder).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed passed to the solver (default: 0).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100000,
        help="Number of core optimization iterations (default: 100000).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Output CSV file path (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default=None,
        help='Hardware tag for the Notes/Hardware column (default: auto-detected).',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List instances that would be run without executing.",
    )

    args = parser.parse_args()

    # Validate executable
    if not args.exe.exists():
        print(f"ERROR: Solver executable not found at: {args.exe}")
        print("       Build it first with:")
        print("         cd filo2 && mkdir build && cd build")
        print("         cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_VERBOSE=1")
        print("         make -j")
        sys.exit(1)

    # Collect instances
    if args.instances:
        instance_files = []
        for p in args.instances:
            if p.exists():
                instance_files.append(p.resolve())
            else:
                print(f"WARNING: Instance file not found, skipping: {p}")
    else:
        folder = args.folder or DEFAULT_INSTANCE_DIR
        if not folder.exists():
            print(f"ERROR: Instance folder not found: {folder}")
            sys.exit(1)
        instance_files = collect_instances(folder)

    if not instance_files:
        print("ERROR: No instance files found.")
        sys.exit(1)

    # Hardware tag
    hardware_tag = args.hardware or get_hardware_tag()

    # Output directory for solver .out files
    outpath = DEFAULT_OUTPATH
    outpath.mkdir(parents=True, exist_ok=True)

    # Summary
    print(f"\n+{'='*66}+")
    print(f"|{'FILO2 Benchmark Runner':^66}|")
    print(f"+{'='*66}+")
    print(f"|  Executable  : {str(args.exe):<49}|")
    print(f"|  Instances   : {len(instance_files):<49}|")
    print(f"|  Seed        : {args.seed:<49}|")
    print(f"|  Iterations  : {args.iterations:<49}|")
    print(f"|  CSV output  : {str(args.csv):<49}|")
    print(f"|  Hardware    : {hardware_tag[:49]:<49}|")
    print(f"+{'='*66}+")

    if args.dry_run:
        print("\n[DRY RUN] Instances that would be executed:")
        for i, inst in enumerate(instance_files, 1):
            dim = parse_dimension_from_file(inst)
            print(f"  {i:>3}. {inst.name:<40}  (N={dim})")
        print(f"\nTotal: {len(instance_files)} instances. Exiting (dry run).")
        sys.exit(0)

    # Run each instance
    print(f"\nStarting benchmark at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    total = len(instance_files)
    successes = 0

    for idx, inst_path in enumerate(instance_files, 1):
        print(f"\n[{idx}/{total}] Processing {inst_path.name} ...")

        try:
            result = run_instance(
                exe=args.exe,
                instance_path=inst_path,
                seed=args.seed,
                iterations=args.iterations,
                outpath=outpath,
                hardware_tag=hardware_tag,
            )
            append_to_csv(args.csv, result)
            successes += 1
        except Exception as e:
            print(f"  [FAIL] FAILED: {e}")
            # Log the failure as a row too
            error_row = {
                "Instance Name": inst_path.name,
                "Size (N)": parse_dimension_from_file(inst_path),
                "Seed": args.seed,
                "Peak Memory (MB)": "ERROR",
                "Exec Time (s)": "ERROR",
                "Cost": "ERROR",
                "Notes/Hardware": f"FAILED: {e}",
            }
            append_to_csv(args.csv, error_row)

    # Final summary
    print(f"\n{'='*70}")
    print(f"  Benchmark complete: {successes}/{total} instances succeeded.")
    print(f"  Results saved to: {args.csv}")
    print(f"  Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
