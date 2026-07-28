import csv
import os
import math
import subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from hilbertcurve.hilbertcurve import HilbertCurve

def read_vrp(filepath):
    metadata, nodes, demands = {}, {}, {}
    with open(filepath, 'r') as f: lines = f.readlines()
    section = "META"
    for line in lines:
        line = line.strip()
        if not line or line == "EOF": continue
        if line.startswith("NODE_COORD_SECTION"): section = "COORD"; continue
        if line.startswith("DEMAND_SECTION"): section = "DEMAND"; continue
        if line.startswith("DEPOT_SECTION"): section = "DEPOT"; continue
        if section == "META":
            if ":" in line:
                key, val = line.split(":", 1)
                metadata[key.strip()] = val.strip()
        elif section == "COORD":
            parts = line.split()
            # NO int() CASTING. Preserve geographical float precision.
            nodes[int(parts[0])] = (float(parts[1]), float(parts[2]))
        elif section == "DEMAND":
            parts = line.split()
            demands[int(parts[0])] = float(parts[1])
            
    depot_coord = nodes.pop(1)
    depot_demand = demands.pop(1, 0.0)
    customer_ids = list(nodes.keys())
    customer_coords = np.array(list(nodes.values()))
    customer_demands = np.array([demands[cid] for cid in customer_ids])
    return metadata, depot_coord, depot_demand, customer_coords, customer_demands

def hilbert_split(cust_coords, cust_dems, capacity):
    """
    Partitions nodes using a 1D Hilbert Space-Filling Curve to preserve 
    2D spatial locality while strictly enforcing routing capacity.
    """
    # 1. Hilbert curves require an integer grid. Scale floats to integers.
    min_x, min_y = np.min(cust_coords[:, 0]), np.min(cust_coords[:, 1])
    scaled_coords = np.round((cust_coords - [min_x, min_y]) * 10000).astype(int)
    
    # 2. Determine grid size (p) based on max coordinate value
    max_val = np.max(scaled_coords)
    p = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 1
    hilbert_curve = HilbertCurve(p, 2)
    
    # 3. Calculate Hilbert index (1D distance) for every node
    hilbert_distances = []
    for coord in scaled_coords:
        dist = hilbert_curve.distance_from_point(coord.tolist())
        hilbert_distances.append(dist)
        
    # 4. Sort customers strictly by Hilbert index
    sort_idx = np.argsort(hilbert_distances)
    sorted_coords = cust_coords[sort_idx]
    sorted_dems = cust_dems[sort_idx]
    
    # 5. Slice linearly along the curve by cumulative demand
    target_demand = float(capacity) * max(1, int(len(cust_coords) / 300))
    splits_coords, splits_dems = [], []
    cur_coords, cur_dems, cur_demand = [], [], 0.0
    
    for coord, dem in zip(sorted_coords, sorted_dems):
        if cur_demand + dem > target_demand and cur_coords:
            splits_coords.append(np.array(cur_coords))
            splits_dems.append(np.array(cur_dems))
            cur_coords, cur_dems, cur_demand = [], [], 0.0
            
        cur_coords.append(coord)
        cur_dems.append(dem)
        cur_demand += dem
        
    if cur_coords:
        splits_coords.append(np.array(cur_coords))
        splits_dems.append(np.array(cur_dems))
        
    return splits_coords, splits_dems

def write_vrp_chunk(output_path, metadata, depot_coord, depot_demand, chunk_coords, chunk_demands):
    total_nodes = len(chunk_coords) + 1
    with open(output_path, 'w') as f:
        f.write(f"NAME : {os.path.basename(output_path)}\n")
        f.write("COMMENT : Partitioned chunk for HGS (Hilbert Sweep)\n")
        f.write("TYPE : CVRP\n")
        f.write(f"DIMENSION : {total_nodes}\n")
        f.write(f"EDGE_WEIGHT_TYPE : {metadata.get('EDGE_WEIGHT_TYPE', 'EUC_2D')}\n")
        f.write(f"CAPACITY : {metadata.get('CAPACITY', '1000')}\nNODE_COORD_SECTION\n")
        
        f.write(f"1 {depot_coord[0]} {depot_coord[1]}\n")
        for idx, coord in enumerate(chunk_coords): 
            f.write(f"{idx + 2} {coord[0]} {coord[1]}\n")
            
        f.write("DEMAND_SECTION\n")
        f.write(f"1 {depot_demand}\n")
        for idx, demand in enumerate(chunk_demands): 
            f.write(f"{idx + 2} {demand}\n")
        f.write("DEPOT_SECTION\n 1\n -1\nEOF\n")

def run_hgs(exe_path, chunk_path, slice_id):
    """Runs HGS with 900s time limit to allow deep search on proper geometries."""
    sol_path = chunk_path.with_suffix('.sol')
    cmd = [str(exe_path), str(chunk_path), str(sol_path), "-t", "300", "-seed", "42"]
    subprocess.run(cmd, capture_output=True, text=True)
    
    cost = None
    if sol_path.exists():
        with open(sol_path, "r", encoding="utf-8") as f:
            for line in reversed(f.readlines()):
                if line.strip().lower().startswith("cost"):
                    cost = float(line.split()[1])
                    break
    return slice_id, cost

def main():
    csv_file = "/home/ubuntu/cvrp/scripts/test_baseline.csv"
    hgs_exe = Path("/home/ubuntu/cvrp/baselines/HGS-CVRP/build/hgs") 
    base_dir = Path("/home/ubuntu/cvrp/data/instances/hgs_partitions")
    
    with open(csv_file, 'r') as f: reader = list(csv.reader(f))
    header, rows = reader[0], reader[1:]
    fixed_count = 0
    
    chunk_tasks = {}
    all_futures = {}
    
    # HARDWARE STARVATION FIX: 2 workers max
    with ThreadPoolExecutor(max_workers=4) as executor:
        print("Preparing Hilbert slices and feeding them to the 2-core pool...")
        for i, row in enumerate(rows):
            if len(row) >= 5 and row[4] != "OK" and "OK" not in row[4] and "[TOTAL AGGREGATE]" not in row[1]:
                instance_name, chunk_name = row[0], row[1]
                chunk_path = base_dir / f"{instance_name}_partitions" / chunk_name
                
                if chunk_path.exists():
                    meta, dep_coord, dep_dem, cust_coords, cust_dems = read_vrp(chunk_path)
                    capacity = meta.get('CAPACITY', 1000)
                    
                    # Generate Hilbert slices
                    split_coords, split_dems = hilbert_split(cust_coords, cust_dems, capacity)
                    num_splits = len(split_coords)
                    
                    chunk_tasks[i] = {
                        'expected_slices': num_splits,
                        'completed_slices': 0,
                        'total_cost': 0,
                        'success': True,
                        'chunk_path': chunk_path,
                        'chunk_name': chunk_name
                    }
                    
                    for k in range(num_splits):
                        split_file_path = chunk_path.parent / f"{chunk_path.stem}_hilbert_{k}.vrp"
                        write_vrp_chunk(split_file_path, meta, dep_coord, dep_dem, split_coords[k], split_dems[k])
                        future = executor.submit(run_hgs, hgs_exe, split_file_path, k)
                        all_futures[future] = (i, k)
                        
        print(f"\n[STARTING] Fired {len(all_futures)} Hilbert tasks into the pool!\n")
        
        for future in as_completed(all_futures):
            i, slice_id = all_futures[future]
            task = chunk_tasks[i]
            _, cost = future.result()
            
            # COST VALIDATION FIX: Strictly reject penalties over 1,000,000
            if cost is not None and cost < 1000000:
                task['total_cost'] += cost
                print(f"  -> {task['chunk_name']} (Slice {slice_id}) finished: {cost:.2f}")
            else:
                task['success'] = False
                print(f"  !! HGS failed or penalized on {task['chunk_name']} (Slice {slice_id}) - Cost: {cost}")
                
            task['completed_slices'] += 1
            
            if task['completed_slices'] == task['expected_slices']:
                if task['success']:
                    print(f"  [SUCCESS] {task['chunk_name']} completely resolved. Combined Cost: {task['total_cost']:.2f}")
                    rows[i][2] = f"{task['total_cost']:.2f}"
                    rows[i][4] = "OK (Hilbert Sweep)"
                    fixed_count += 1
                    task['chunk_path'].rename(str(task['chunk_path']) + ".BAK") 
                else:
                    print(f"  [SKIPPED] {task['chunk_name']} had partial failures. Not updating CSV.")
                    
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(rows)

    print(f"\nDone! Successfully patched {fixed_count} chunks using Hilbert Curves.")

if __name__ == "__main__":
	main()
