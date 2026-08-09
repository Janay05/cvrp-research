"""
score_sol.py -- independently compute the cost of a CVRPLIB-style .sol route file
against its .vrp instance, without trusting any cost the .sol file itself claims.

Used to turn the vendored reference solutions (data/bks/**/*.sol,
baselines/filo2/results/i-bks/*.sol) into real reference costs for benchmarking,
following the same "recompute from scratch, don't trust the reported number" discipline
as src/verifier.py -- see docs/reports/005_cost_optimization.md, Phase 0.

Id convention (verified empirically against data/bks/X/X/X-n101-k25.sol, whose embedded
Cost 27591 matches baselines/filo2/results/csvs/filo2-x.csv's z0 for the same instance):
BKS/reference .sol files do NOT use raw file-native NODE_COORD_SECTION ids. They use the
same "depot removed, survivors renumbered 1..dimension-1 in file order" convention our own
solver's output uses (src/VrpParser.cpp's nextId loop; src/verifier.py's `node + 1` shift
assumes depot file-id 1). So BKS customer id `k` means: skip the depot in file-id order,
then take the k-th surviving node. This is generalized here (not hardcoded to depot id 1)
by building the same nextId->file-id map VrpParser.cpp builds, then reversing it.

.vrp parsing intentionally mirrors src/verifier.py's tokenizing (same NODE_COORD_SECTION
parsing, same round() distance convention) so a gap-to-reference computed by this tool is
apples-to-apples with what verifier.py checks against our own output.

Usage:
    python tools/score_sol.py <instance.vrp> <reference.sol>
Or import:
    from score_sol import score_solution
    cost, num_routes, embedded_cost = score_solution(vrp_path, sol_path)
"""
import math
import re
import sys


def parse_vrp(vrp_file):
    """Returns (coords, depot_id, bks_to_file) where coords/depot_id are file-native ids
    (matching src/verifier.py's convention), and bks_to_file[k] maps a BKS/reference .sol
    customer id `k` (1..dimension-1, depot already removed and survivors renumbered in file
    order -- see module docstring) back to its file-native NODE_COORD_SECTION id, mirroring
    src/VrpParser.cpp's nextId loop exactly (including for depot_id != 1 instances)."""
    coords = {}
    mode = None
    depot_id = None
    with open(vrp_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] in ("NODE_COORD_SECTION",):
                mode = "COORD"
                continue
            if parts[0] in ("DEMAND_SECTION",):
                mode = "DEMAND"
                continue
            if parts[0] in ("DEPOT_SECTION",):
                mode = "DEPOT"
                continue
            if parts[0] == "EOF":
                break
            if mode == "COORD":
                coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
            elif mode == "DEPOT":
                val = int(parts[0])
                if val == -1:
                    mode = None
                elif depot_id is None:
                    depot_id = val
    if depot_id is None:
        depot_id = 1  # every instance in this repo's data/ uses file-id 1 as the depot

    dimension = max(coords.keys())
    bks_to_file = {}
    next_id = 1
    for file_id in range(1, dimension + 1):
        if file_id == depot_id:
            continue
        bks_to_file[next_id] = file_id
        next_id += 1

    return coords, depot_id, bks_to_file


def dist(coords, n1, n2):
    dx = coords[n1][0] - coords[n2][0]
    dy = coords[n1][1] - coords[n2][1]
    return round(math.sqrt(dx * dx + dy * dy))


_ROUTE_RE = re.compile(r"Route\s*#?\d+\s*:\s*(.*)")
_COST_RE = re.compile(r"^\s*Cost\s+([0-9.]+)\s*$", re.IGNORECASE)


def parse_sol(sol_file):
    routes = []
    embedded_cost = None
    with open(sol_file, 'r') as f:
        for line in f:
            m = _ROUTE_RE.match(line.strip())
            if m:
                ids = [int(tok) for tok in m.group(1).split()]
                routes.append(ids)
                continue
            m = _COST_RE.match(line)
            if m:
                embedded_cost = float(m.group(1))
    return routes, embedded_cost


def score_solution(vrp_file, sol_file):
    """Returns (recomputed_cost, num_routes, embedded_cost_or_None, all_customers_visited_once)."""
    coords, depot_id, bks_to_file = parse_vrp(vrp_file)
    routes, embedded_cost = parse_sol(sol_file)

    total_cost = 0
    visited = set()
    duplicate = False
    for route in routes:
        prev = depot_id
        for bks_node in route:
            node = bks_to_file[bks_node]
            if node in visited:
                duplicate = True
            visited.add(node)
            total_cost += dist(coords, prev, node)
            prev = node
        total_cost += dist(coords, prev, depot_id)

    expected = set(coords.keys()) - {depot_id}
    complete = (not duplicate) and (visited == expected)

    return total_cost, len(routes), embedded_cost, complete


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tools/score_sol.py <instance.vrp> <reference.sol>")
        sys.exit(1)
    vrp_arg, sol_arg = sys.argv[1], sys.argv[2]
    cost, num_routes, embedded, complete = score_solution(vrp_arg, sol_arg)
    print(f"Recomputed cost: {cost}")
    print(f"Num routes: {num_routes}")
    print(f"Embedded 'Cost' line: {embedded}")
    if embedded is not None and abs(embedded - cost) > 1e-6:
        print(f"WARNING: recomputed cost {cost} != embedded cost {embedded} "
              f"(diff {embedded - cost}) -- likely a distance-rounding convention difference")
    print(f"All customers visited exactly once: {complete}")
    if not complete:
        print("WARNING: solution is not a complete/feasible tour over this instance")
