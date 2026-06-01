import csv
import os
import math
import subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def write_vrp_chunk(output_path, metadata, depot_coord, depot_demand, chunk_coords, chunk_demands):
    total_nodes = len(chunk_coords) + 1
    with open(output_path, 'w') as f:
        f.write(f"NAME : {os.path.basename(output_path)}\n")
        f.write("COMMENT : Partitioned chunk for HGS\n")
        f.write("TYPE : CVRP\n")
        f.write(f"DIMENSION : {total_nodes}\n")
        f.write(f"EDGE_WEIGHT_TYPE : {metadata.get('EDGE_WEIGHT_TYPE', 'EUC_2D')}\n")
        f.write(f"CAPACITY : {metadata.get('CAPACITY', '1000')}\nNODE_COORD_SECTION\n")
        f.write(f"1 {int(depot_coord[0])} {int(depot_coord[1])}\n")
        for idx, coord in enumerate(chunk_coords): f.write(f"{idx + 2} {int(coord[0])} {int(coord[1])}\n")
        f.write("DEMAND_SECTION\n")
        f.write(f"1 {int(depot_demand)}\n")
        for idx, demand in enumerate(chunk_demands): f.write(f"{idx + 2} {int(demand)}\n")
        f.write("DEPOT_SECTION\n 1\n -1\nEOF\n")

def run_hgs(exe_path, chunk_path, slice_id):
    """Runs HGS and gets the cost, customized for parallel execution."""
    sol_path = chunk_path.with_suffix('.sol')
    cmd = [str(exe_path), str(chunk_path), str(sol_path), "-t", "240", "-seed", "42"]
    subprocess.run(cmd, capture_output=True, text=True)
    
    cost = None
    if sol_path.exists():
        with open(sol_path, "r", encoding="utf-8") as f:
            for line in reversed(f.readlines()):
                if line.strip().lower().startswith("cost"):
                    cost = float(line.split()[1])
                    break
        sol_path.unlink()
    return slice_id, cost

def main():
    csv_file = "HGS_Partition_Costs.csv"
    hgs_exe = Path("/mnt/c/internship/iitm/cvrp/baselines/HGS-CVRP/build_linux/hgs") 
    base_dir = Path("/mnt/c/internship/iitm/cvrp/data/instances/hgs_partitions")
    
    with open(csv_file, 'r') as f: reader = list(csv.reader(f))
    header, rows = reader[0], reader[1:]
    fixed_count = 0
    
    for i, row in enumerate(rows):
        if len(row) >= 5 and row[4] != "OK" and "OK" not in row[4] and "[TOTAL AGGREGATE]" not in row[1]:
            instance_name, chunk_name = row[0], row[1]
            chunk_path = base_dir / f"{instance_name}_partitions" / chunk_name
            
            if chunk_path.exists():
                meta, dep_coord, dep_dem, cust_coords, cust_dems = read_vrp(chunk_path)
                total_customers = len(cust_coords)
                num_splits = math.ceil(total_customers / 1500)
                
                print(f"\n[PARALLEL SWEEP] {chunk_name} -> Firing {num_splits} HGS instances simultaneously...")
                
                sort_indices = np.argsort(cust_coords[:, 0])
                sorted_coords = cust_coords[sort_indices]
                sorted_dems = cust_dems[sort_indices]
                
                split_coords = np.array_split(sorted_coords, num_splits)
                split_dems = np.array_split(sorted_dems, num_splits)
                
                total_cost = 0
                success = True
                
                # --- MULTIPROCESSING MAGIC ---
                # This limits it to 6 parallel tasks so your laptop doesn't crash from RAM overload
                with ThreadPoolExecutor(max_workers=6) as executor:
                    futures = []
                    for k in range(num_splits):
                        split_file_path = chunk_path.parent / f"{chunk_path.stem}_sweep_{k}.vrp"
                        write_vrp_chunk(split_file_path, meta, dep_coord, dep_dem, split_coords[k], split_dems[k])
                        # Submit task to CPU cores
                        futures.append(executor.submit(run_hgs, hgs_exe, split_file_path, k))
                    
                    # Wait for them all to finish and gather results
                    for future in as_completed(futures):
                        slice_id, cost = future.result()
                        if cost is not None:
                            print(f"  -> Slice {slice_id} finished: {cost:.2f}")
                            total_cost += cost
                        else:
                            print(f"  !! HGS failed on slice {slice_id} (Timed out after 4 min)")
                            success = False
                
                if success:
                    print(f"  -> Success! Combined Sweep Cost: {total_cost:.2f}")
                    rows[i][2] = f"{total_cost:.2f}"
                    rows[i][4] = "OK (Parallel Sweep)"
                    fixed_count += 1
                    chunk_path.rename(str(chunk_path) + ".BAK") 
                else:
                    print("  -> Framework skipped updating row due to partial slice failure.")
                    
                # Save dynamically after every success so we don't lose progress!
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(rows)

    print(f"\nDone! Successfully patched {fixed_count} chunks.")

if __name__ == "__main__":
    main()