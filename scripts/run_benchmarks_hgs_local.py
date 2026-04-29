#!/usr/bin/env python3
"""
HGS-CVRP Benchmark Runner — LOCAL VERSION (Windows Laptop)

Defaults tuned for a 16GB RAM / Intel i7 laptop:
  - 2 concurrent workers  (2 × 4GB = 8GB max, leaving headroom for OS)
  - 4096 MB memory limit per instance
  - 45-minute time limit
  - Reads result_hgs.csv and automatically SKIPS already-completed instances
    so you can safely resume without re-running finished work.

CSV columns:
  Instance Name | Size (N) | Seed | Peak Memory (MB) | Exec Time (s) | Cost | Notes/Hardware
"""

import argparse
import csv
import os
import platform
import subprocess
import sys
import threading
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path

# ── stdout/stderr encoding fix for Windows console ────────────────────────────
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# Lock for writing to CSV concurrently
CSV_LOCK = threading.Lock()

try:
    import psutil
except ImportError:
    print("ERROR: psutil is required for memory monitoring.")
    print("Install it with:  pip install psutil")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Constants — LOCAL defaults
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# MSVC on Windows places the binary under build/Release/
DEFAULT_EXE     = PROJECT_ROOT / "baselines" / "HGS-CVRP" / "build" / "Release" / "hgs.exe"
DEFAULT_INSTANCE_DIR = PROJECT_ROOT / "data" / "instances"
DEFAULT_CSV     = PROJECT_ROOT / "results" / "result_hgs.csv"
DEFAULT_OUTPATH = PROJECT_ROOT / "results" / "benchmark_outputs_hgs"

DEFAULT_WORKERS   = 2       # safe for 16 GB laptop
DEFAULT_MEM_MB    = 4096    # 4 GB per instance
DEFAULT_TIME_S    = 2700    # 45 minutes
DEFAULT_ITER      = 20000

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
    cpu = platform.processor() or "Unknown CPU"
    try:
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        ram_gb = "?"
    return f"Local {platform.node()} ({cpu}, {ram_gb}GB RAM)"


def parse_dimension_from_file(filepath: Path) -> int:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.upper().startswith("DIMENSION"):
                parts = line.split(":")
                if len(parts) >= 2:
                    return int(parts[1].strip())
    return -1


def parse_cost_from_solfile(sol_path: Path) -> float:
    if not sol_path.exists():
        return -1.0
    with open(sol_path, "r", encoding="utf-8") as f:
        for line in reversed(f.readlines()):
            line = line.strip()
            if line.lower().startswith("cost"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        pass
    return -1.0


class MemoryMonitor:
    """Monitors peak RSS memory and kills the process if it exceeds mem_limit_mb."""

    def __init__(self, process, poll_interval: float = 0.1, mem_limit_mb: float = None):
        self.process = process
        self.pid = process.pid
        self.poll_interval = poll_interval
        self.mem_limit_mb = mem_limit_mb
        self.peak_memory_bytes = 0
        self.oom_triggered = False
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
                if self.process.poll() is not None:
                    break
                try:
                    current = proc.memory_info().rss
                    if current > self.peak_memory_bytes:
                        self.peak_memory_bytes = current
                    if self.mem_limit_mb and (current / (1024 * 1024)) > self.mem_limit_mb:
                        self.oom_triggered = True
                        self.process.kill()
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                self._stop_event.wait(self.poll_interval)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Core Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_instance(exe, instance_path, seed, time_limit_s, it_limit, mem_limit_mb, outpath, hardware_tag) -> dict:
    instance_name = instance_path.name
    size_n = parse_dimension_from_file(instance_path)
    sol_path = outpath / f"{instance_name}_seed-{seed}.sol"

    cmd = [str(exe), str(instance_path), str(sol_path),
           "-seed", str(seed), "-t", str(time_limit_s), "-it", str(it_limit)]

    print(f"\n{'='*70}")
    print(f"  Instance : {instance_name}")
    print(f"  Size (N) : {size_n}")
    print(f"  Seed     : {seed}")
    print(f"  Command  : {' '.join(cmd)}")
    print(f"{'='*70}")

    start_time = time.perf_counter()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    mem_monitor = MemoryMonitor(process, poll_interval=0.1, mem_limit_mb=mem_limit_mb)
    mem_monitor.start()

    is_tle = False
    stdout = ""
    try:
        stdout, _ = process.communicate(timeout=time_limit_s + 10)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, _ = process.communicate()
        is_tle = True

    elapsed = round(time.perf_counter() - start_time, 2)
    mem_monitor.stop()
    peak_mem_mb = mem_monitor.get_peak_mb()

    if stdout:
        lines = [l for l in stdout.strip().splitlines() if l.strip()]
        for l in (lines[-15:] if len(lines) > 15 else lines):
            print(f"  {l}")

    cost = "ERROR"
    if mem_monitor.oom_triggered:
        cost = "OOM"
        print("  [FAIL] Out of Memory limit exceeded")
    elif is_tle:
        cost = "TLE"
        print("  [FAIL] Time Limit Exceeded")
    else:
        parsed_cost = parse_cost_from_solfile(sol_path)
        if parsed_cost >= 0:
            cost = parsed_cost
        elif process.returncode != 0:
            cost = "ERROR"
            print(f"  [WARN] Solver exited with code {process.returncode}")

    result = {
        "Instance Name": instance_name,
        "Size (N)": size_n,
        "Seed": seed,
        "Peak Memory (MB)": peak_mem_mb,
        "Exec Time (s)": elapsed,
        "Cost": cost,
        "Notes/Hardware": hardware_tag,
    }

    status = "OK" if isinstance(cost, float) else cost
    print(f"  [{status}] Cost={cost}  Time={elapsed}s  PeakMem={peak_mem_mb}MB")
    return result


def append_to_csv(csv_path: Path, row: dict):
    with CSV_LOCK:
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def collect_instances(folder: Path) -> list:
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
        description="HGS-CVRP Benchmark Runner (LOCAL) — safe defaults for a 16GB Windows laptop."
    )
    parser.add_argument("--exe",        type=Path,  default=DEFAULT_EXE,
                        help=f"Path to hgs.exe (default: {DEFAULT_EXE})")
    parser.add_argument("--folder",     type=Path,  default=None,
                        help="Folder containing instance files.")
    parser.add_argument("--instances",  type=Path,  nargs="+", default=None,
                        help="Explicit list of instance file paths.")
    parser.add_argument("--seed",       type=int,   default=1,
                        help="Random seed (default: 1).")
    parser.add_argument("--time-limit", type=int,   default=DEFAULT_TIME_S,
                        help=f"Time limit per instance in seconds (default: {DEFAULT_TIME_S}s / 45 min).")
    parser.add_argument("--it",         type=int,   default=DEFAULT_ITER,
                        help=f"Max iterations without improvement (default: {DEFAULT_ITER}).")
    parser.add_argument("--workers",    type=int,   default=DEFAULT_WORKERS,
                        help=f"Concurrent workers (default: {DEFAULT_WORKERS} — safe for 16GB laptop).")
    parser.add_argument("--mem-limit",  type=float, default=DEFAULT_MEM_MB,
                        help=f"Memory limit per instance in MB (default: {DEFAULT_MEM_MB} MB).")
    parser.add_argument("--csv",        type=Path,  default=DEFAULT_CSV,
                        help=f"Output CSV file (default: {DEFAULT_CSV}).")
    parser.add_argument("--hardware",   type=str,   default=None,
                        help="Override hardware tag in CSV.")
    parser.add_argument("--dry-run",    action="store_true",
                        help="List instances to run without executing.")

    args = parser.parse_args()

    if not args.exe.exists():
        print(f"ERROR: Solver executable not found at: {args.exe}")
        print("Build it first with:  cd baselines\\HGS-CVRP && mkdir build && cd build && cmake .. && cmake --build . --config Release --target bin")
        sys.exit(1)

    if args.instances:
        instance_files = [p.resolve() for p in args.instances if p.exists()]
    else:
        folder = args.folder or DEFAULT_INSTANCE_DIR
        if not folder.exists():
            print(f"ERROR: Instance folder not found: {folder}")
            sys.exit(1)
        instance_files = collect_instances(folder)

    # ── Resume: skip already-completed instances ───────────────────────────
    if args.csv.exists():
        completed = set()
        with open(args.csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("Instance Name"):
                    completed.add(row["Instance Name"])
        before = len(instance_files)
        instance_files = [f for f in instance_files if f.name not in completed]
        print(f"Resume mode: skipping {before - len(instance_files)} already-completed instances.")

    if not instance_files:
        print("All instances already completed! Check results/result_hgs.csv")
        sys.exit(0)

    hardware_tag = args.hardware or get_hardware_tag()
    outpath = Path(DEFAULT_OUTPATH)
    outpath.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n+{'='*66}+")
    print(f"|{'HGS-CVRP Benchmark Runner  [LOCAL]':^66}|")
    print(f"+{'='*66}+")
    print(f"|  Executable  : {str(args.exe):<49}|")
    print(f"|  Instances   : {len(instance_files):<49}|")
    print(f"|  Seed        : {args.seed:<49}|")
    print(f"|  Time Limit  : {args.time_limit}s  (45 min){'':<38}|"[:68] + "|")
    print(f"|  Workers     : {args.workers}  (safe for 16GB RAM){'':<33}|"[:68] + "|")
    print(f"|  Mem Limit   : {args.mem_limit}MB per instance{'':<34}|"[:68] + "|")
    print(f"|  CSV output  : {str(args.csv):<49}|")
    print(f"|  Hardware    : {hardware_tag[:49]:<49}|")
    print(f"+{'='*66}+")

    if args.dry_run:
        print("\n[DRY RUN] Instances that would be executed:")
        for i, inst in enumerate(instance_files, 1):
            print(f"  {i:>3}. {inst.name:<40}  (N={parse_dimension_from_file(inst)})")
        sys.exit(0)

    print(f"\nStarting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with {args.workers} workers\n")
    total = len(instance_files)

    def process_instance(inst_path):
        try:
            result = run_instance(
                exe=args.exe, instance_path=inst_path, seed=args.seed,
                time_limit_s=args.time_limit, it_limit=args.it,
                mem_limit_mb=args.mem_limit, outpath=outpath, hardware_tag=hardware_tag,
            )
            append_to_csv(args.csv, result)
            return isinstance(result["Cost"], float)
        except Exception as e:
            print(f"  [FAIL] {inst_path.name}: {e}")
            append_to_csv(args.csv, {
                "Instance Name": inst_path.name,
                "Size (N)": parse_dimension_from_file(inst_path),
                "Seed": args.seed,
                "Peak Memory (MB)": "ERROR",
                "Exec Time (s)": "ERROR",
                "Cost": "ERROR",
                "Notes/Hardware": f"FAILED: {e}",
            })
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        successes = sum(executor.map(process_instance, instance_files))

    print(f"\n{'='*70}")
    print(f"  Benchmark complete: {successes}/{total} instances succeeded.")
    print(f"  Results saved to: {args.csv}")
    print(f"  Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
