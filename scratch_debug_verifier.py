import sys

vrp_file = sys.argv[1]
sol_file = sys.argv[2]

demands = {}
with open(vrp_file, 'r') as f:
    lines = f.readlines()
    
mode = None
for line in lines:
    parts = line.strip().split()
    if not parts: continue
    if parts[0] == "NODE_COORD_SECTION": mode = "COORD"
    elif parts[0] == "DEMAND_SECTION": mode = "DEMAND"
    elif parts[0] == "DEPOT_SECTION": mode = "DEPOT"
    elif parts[0] == "EOF": break
    elif mode == "DEMAND":
        demands[int(parts[0])] = int(parts[1])

with open(sol_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith("Route"):
            parts = line.split("):")
            route_nodes_str = parts[1].strip()
            if route_nodes_str:
                nodes = [int(x) for x in route_nodes_str.replace("->", " ").split()]
                route_load = 0
                for node in nodes:
                    route_load += demands.get(node + 1, 0)
                if route_load > 50:
                    print(f"ERROR: Route load {route_load} exceeds capacity 50!")
                    print(f"Nodes: {nodes}")
                    print(f"Demands: {[demands.get(n + 1, 0) for n in nodes]}")
                    break
