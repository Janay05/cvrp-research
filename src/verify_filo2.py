import math
import sys
import os

# Independent verifier for FILO2's native .vrp.sol format (space-separated node IDs, one
# route per "Route #N: id id id ..." line, depot implicit at both ends, no header). Written
# because verifier.py only parses our own solver's "Route X (Load: Y): 0 -> ... -> 0" format --
# FILO2's numbers had been taken from its self-reported .out file with no independent check.
# Reuses the same coord/demand/capacity parsing as verifier.py so both tools trust the same
# .vrp file the same way.
#
# Critical: FILO2's own Parser.cpp reads each line's vertex_index token and DISCARDS it,
# storing coords/demands positionally by 0-indexed read order instead (data.demands[i] for
# the i-th DEMAND_SECTION line, not demands[vertex_index]). Its Solution::store_to_file then
# writes that same 0-indexed internal id straight to the .sol file. So a ".sol" id of X is
# file node id (X+1), not X -- confirmed by cross-referencing against the .vrp file directly
# (a naive same-id lookup silently "succeeded" on individual spot checks purely by coincidence
# at this instance's scale, then produced ~45% spurious capacity violations in aggregate,
# which is what caught the bug).

def verify_filo2_solution(vrp_file, sol_file, reported_cost):
    coords = {}
    demands = {}
    capacity = 0
    dimension = 0
    depot = None

    with open(vrp_file, 'r') as f:
        lines = f.readlines()

    mode = None
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "CAPACITY" and len(parts) >= 3:
            capacity = int(parts[2])
        elif parts[0] == "CAPACITY:" and len(parts) >= 2:
            capacity = int(parts[1])
        elif parts[0] == "DIMENSION" and len(parts) >= 3:
            dimension = int(parts[2])
        elif parts[0] == "DIMENSION:" and len(parts) >= 2:
            dimension = int(parts[1])
        elif parts[0] == "NODE_COORD_SECTION":
            mode = "COORD"
        elif parts[0] == "DEMAND_SECTION":
            mode = "DEMAND"
        elif parts[0] == "DEPOT_SECTION":
            mode = "DEPOT"
        elif parts[0] == "EOF":
            break
        elif mode == "COORD":
            coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
        elif mode == "DEMAND":
            demands[int(parts[0])] = int(parts[1])
        elif mode == "DEPOT":
            if depot is None and parts[0] != "-1":
                depot = int(parts[0])

    def dist(n1, n2):
        dx = coords[n1][0] - coords[n2][0]
        dy = coords[n1][1] - coords[n2][1]
        return round(math.sqrt(dx * dx + dy * dy))

    if not os.path.exists(sol_file):
        print(f"ERROR: Solution file {sol_file} not found!")
        sys.exit(1)

    with open(sol_file, 'r') as f:
        lines = f.readlines()

    visited = set()
    total_cost = 0
    route_count = 0

    for line in lines:
        line = line.strip()
        if not line.startswith("Route"):
            continue
        path_str = line.split(":", 1)[1].strip()
        # +1: FILO2's .sol ids are 0-indexed internal read-order positions, not file node ids.
        nodes = [int(x) + 1 for x in path_str.split()]

        route_load = 0
        for node in nodes:
            if node in visited:
                print(f"ERROR: Node {node} visited multiple times!")
                sys.exit(1)
            visited.add(node)
            route_load += demands.get(node, 0)

        if route_load > capacity:
            print(f"ERROR: Route load {route_load} exceeds capacity {capacity}!")
            sys.exit(1)

        full_route = [depot] + nodes + [depot]
        route_cost = sum(dist(full_route[i], full_route[i + 1]) for i in range(len(full_route) - 1))
        total_cost += route_cost
        route_count += 1

    expected_nodes = set(range(1, dimension + 1)) - {depot}
    if visited != expected_nodes:
        missing = expected_nodes - visited
        extra = visited - expected_nodes
        print(f"ERROR: Nodes mismatch! Missing: {len(missing)} Extra: {len(extra)}")
        sys.exit(1)

    print(f"Verification SUCCESS!")
    print(f"Feasibility: Valid ({route_count} routes, {len(visited)} customers, depot={depot})")
    print(f"Independently computed cost: {total_cost}")
    print(f"FILO2's self-reported cost:  {reported_cost}")
    if total_cost != reported_cost:
        print(f"MISMATCH: independent cost differs from FILO2's self-report by {total_cost - reported_cost}")
        sys.exit(1)
    else:
        print(f"MATCH: FILO2's self-reported cost is independently confirmed correct.")

if __name__ == "__main__":
    vrp_arg = sys.argv[1]
    sol_arg = sys.argv[2]
    reported_arg = int(sys.argv[3])
    verify_filo2_solution(vrp_arg, sol_arg, reported_arg)
