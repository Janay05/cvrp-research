"""
combine_results.py
Merges HGS and FILO2 benchmark results into a single comparison CSV,
and looks up the BKS (Best Known Solution) from data/bks/ sol files.

Output columns:
  Instance Name | Size (N) | BKS | 
  HGS Cost | HGS Time (s) | HGS Peak Memory (MB) | HGS Status |
  FILO2 Cost | FILO2 Time (s) | FILO2 Peak Memory (MB) | FILO2 Status
"""

import csv
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HGS_CSV      = PROJECT_ROOT / "results" / "result_hgs.csv"
FILO2_CSV    = PROJECT_ROOT / "results" / "results._filo2.csv"
BKS_ROOT     = PROJECT_ROOT / "data" / "bks"
OUT_CSV      = PROJECT_ROOT / "results" / "combined_results.csv"

OUT_COLUMNS = [
    "Instance Name", "Size (N)", "BKS",
    "HGS Cost", "HGS Time (s)", "HGS Peak Memory (MB)", "HGS Status",
    "FILO2 Cost", "FILO2 Time (s)", "FILO2 Peak Memory (MB)", "FILO2 Status",
]

# ── BKS Loader ────────────────────────────────────────────────────────────────
def load_bks() -> dict:
    """
    Walk all .sol files under data/bks/ and parse the last 'Cost <value>' line.
    Returns a dict: {instance_name_without_extension -> bks_cost}
    """
    bks = {}
    for sol_file in BKS_ROOT.rglob("*.sol"):
        cost = None
        try:
            lines = sol_file.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            for line in reversed(lines):
                line = line.strip()
                if line.lower().startswith("cost"):
                    parts = line.split()
                    if len(parts) >= 2:
                        cost = float(parts[1])
                        break
        except Exception:
            pass
        if cost is not None:
            # Key is the stem (filename without extension), e.g. "X-n101-k25"
            bks[sol_file.stem] = cost
    return bks


def get_bks(bks_dict: dict, instance_name: str):
    """
    Match instance file name (e.g. 'X-n101-k25.vrp' or 'Antwerp1.txt') 
    to a BKS entry by stripping the extension.
    """
    stem = Path(instance_name).stem  # e.g. 'X-n101-k25'
    return bks_dict.get(stem, "")


# ── CSV Loaders ───────────────────────────────────────────────────────────────
def load_csv_best(csv_path: Path) -> dict:
    """
    Load a benchmark CSV. For instances with multiple rows (multiple seeds/runs),
    keep the row with the lowest numeric Cost. Non-numeric costs (OOM, TLE, ERROR)
    are kept only if no successful run exists.
    Returns: {instance_name -> row_dict}
    """
    best = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("Instance Name", "").strip()
            if not name:
                continue
            cost_str = row.get("Cost", "").strip()
            try:
                cost_val = float(cost_str)
            except ValueError:
                cost_val = None  # OOM / TLE / ERROR

            if name not in best:
                best[name] = row
            else:
                prev_cost_str = best[name].get("Cost", "").strip()
                try:
                    prev_cost_val = float(prev_cost_str)
                except ValueError:
                    prev_cost_val = None

                # Prefer numeric over non-numeric; prefer lower numeric
                if cost_val is not None:
                    if prev_cost_val is None or cost_val < prev_cost_val:
                        best[name] = row
    return best


def get_status(cost_str: str) -> str:
    """Return OK / OOM / TLE / ERROR / N/A based on cost string."""
    if not cost_str or cost_str.strip() == "":
        return "N/A"
    s = cost_str.strip().upper()
    if s in ("OOM", "TLE", "ERROR"):
        return s
    try:
        float(s)
        return "OK"
    except ValueError:
        return "ERROR"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading BKS solutions...")
    bks_dict = load_bks()
    print(f"  Found {len(bks_dict)} BKS entries.")

    print("Loading HGS results...")
    hgs_data = load_csv_best(HGS_CSV)
    print(f"  Found {len(hgs_data)} unique HGS instances.")

    print("Loading FILO2 results...")
    filo2_data = load_csv_best(FILO2_CSV)
    print(f"  Found {len(filo2_data)} unique FILO2 instances.")

    # Union of all instance names
    all_instances = sorted(set(hgs_data.keys()) | set(filo2_data.keys()))
    print(f"\nTotal unique instances across both algorithms: {len(all_instances)}")

    rows = []
    for name in all_instances:
        hgs_row   = hgs_data.get(name, {})
        filo2_row = filo2_data.get(name, {})

        # Size — prefer whichever has it
        size = hgs_row.get("Size (N)") or filo2_row.get("Size (N)") or ""

        bks = get_bks(bks_dict, name)

        hgs_cost   = hgs_row.get("Cost", "")
        hgs_time   = hgs_row.get("Exec Time (s)", "")
        hgs_mem    = hgs_row.get("Peak Memory (MB)", "")
        hgs_status = get_status(hgs_cost) if hgs_row else "N/A"

        filo2_cost   = filo2_row.get("Cost", "")
        filo2_time   = filo2_row.get("Exec Time (s)", "")
        filo2_mem    = filo2_row.get("Peak Memory (MB)", "")
        filo2_status = get_status(filo2_cost) if filo2_row else "N/A"

        rows.append({
            "Instance Name":         name,
            "Size (N)":              size,
            "BKS":                   bks,
            "HGS Cost":              hgs_cost,
            "HGS Time (s)":          hgs_time,
            "HGS Peak Memory (MB)":  hgs_mem,
            "HGS Status":            hgs_status,
            "FILO2 Cost":            filo2_cost,
            "FILO2 Time (s)":        filo2_time,
            "FILO2 Peak Memory (MB)": filo2_mem,
            "FILO2 Status":          filo2_status,
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Summary stats
    hgs_ok     = sum(1 for r in rows if r["HGS Status"] == "OK")
    filo2_ok   = sum(1 for r in rows if r["FILO2 Status"] == "OK")
    bks_filled = sum(1 for r in rows if r["BKS"] != "")

    print(f"\n{'='*55}")
    print(f"  Combined results saved to: {OUT_CSV}")
    print(f"  Total rows           : {len(rows)}")
    print(f"  BKS values found     : {bks_filled}/{len(rows)}")
    print(f"  HGS OK results       : {hgs_ok}/{len(rows)}")
    print(f"  FILO2 OK results     : {filo2_ok}/{len(rows)}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
