import math
import sys
import os

def verify_solution(vrp_file, sol_file):
    coords = {}
    demands = {}
    capacity = 0
    dimension = 0

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
            
    def dist(n1, n2):
        dx = coords[n1][0] - coords[n2][0]
        dy = coords[n1][1] - coords[n2][1]
        return round(math.sqrt(dx*dx + dy*dy))

    visited = set()
    total_cost = 0
    reported_cost = None
    reported_num_routes = None
    parsed_route_count = 0

    if not os.path.exists(sol_file):
        print(f"ERROR: Solution file {sol_file} not found!")
        sys.exit(1)

    with open(sol_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("Final Cost:"):
            reported_cost = int(line.split(":")[1].strip())
        elif line.startswith("Num Routes:"):
            reported_num_routes = int(line.split(":")[1].strip())
        elif line.startswith("Route"):
            # Route 0 (Load: 100): 0 -> 1711 -> ... -> 0
            path_str = line.split("): ")[1].strip()
            nodes = [int(x) for x in path_str.split(" -> ")]
            
            route_load = 0
            route_cost = 0
            
            for i in range(1, len(nodes) - 1):
                node = nodes[i]
                if node in visited:
                    print(f"ERROR: Node {node} visited multiple times!")
                    sys.exit(1)
                visited.add(node)
                route_load += demands.get(node + 1, 0)
                
            for i in range(len(nodes) - 1):
                n1 = nodes[i] + 1
                n2 = nodes[i+1] + 1
                route_cost += dist(n1, n2)
                
            if route_load > capacity:
                print(f"ERROR: Route load {route_load} exceeds capacity {capacity}!")
                sys.exit(1)
                
            total_cost += route_cost
            parsed_route_count += 1

    expected_nodes = set(range(1, dimension))

    if visited != expected_nodes:
        missing = expected_nodes - visited
        extra = visited - expected_nodes
        print(f"ERROR: Nodes mismatch!")
        print(f"Missing: {len(missing)} nodes")
        print(f"Extra: {len(extra)} nodes")
        sys.exit(1)

    if reported_cost is None or reported_num_routes is None:
        print(f"ERROR: Solution file is missing 'Final Cost:' or 'Num Routes:' header!")
        sys.exit(1)

    if reported_cost != total_cost:
        print(f"ERROR: Reported cost {reported_cost} does not match independently computed cost {total_cost}!")
        sys.exit(1)

    if reported_num_routes != parsed_route_count:
        print(f"ERROR: Reported Num Routes {reported_num_routes} does not match parsed route count {parsed_route_count}!")
        sys.exit(1)

    print(f"Verification SUCCESS!")
    print(f"Feasibility: Valid")
    print(f"Total computed cost: {total_cost} (matches reported Final Cost)")
    print(f"Num Routes: {parsed_route_count} (matches reported Num Routes)")

if __name__ == "__main__":
    verify_solution("test_2000.vrp", "results/final_solution.txt")
