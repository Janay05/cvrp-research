import pandas as pd
import numpy as np
import os
import sys
import subprocess
import time
import argparse
import concurrent.futures
from pathlib import Path

def parse_vrp(vrp_path):
    nodes = {}
    capacity = 0
    with open(vrp_path, 'r') as f:
        lines = f.readlines()
    
    in_coord = False
    in_demand = False
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        if line.startswith("CAPACITY"):
            capacity = float(line.split(':')[1].strip())
        elif line.startswith("NODE_COORD_SECTION"):
            in_coord = True
            in_demand = False
        elif line.startswith("DEMAND_SECTION"):
            in_coord = False
            in_demand = True
        elif line.startswith("DEPOT_SECTION"):
            break
        elif in_coord:
            parts = line.split()
            nodes[int(parts[0])] = {'x': float(parts[1]), 'y': float(parts[2]), 'demand': 0.0}
        elif in_demand:
            parts = line.split()
            nodes[int(parts[0])]['demand'] = float(parts[1])
            
    return nodes, capacity

def write_cluster_vrp(cluster_df, nodes_dict, capacity, out_vrp_path):
    # Prepare HGS VRP format
    # The depot is always node 1. Other nodes are 2, 3, ... N
    
    # cluster_df contains original_id, x, y
    N = len(cluster_df) + 1
    
    with open(out_vrp_path, 'w') as f:
        f.write(f"NAME : {out_vrp_path.stem}\n")
        f.write(f"COMMENT : Micro-cluster subproblem\n")
        f.write(f"TYPE : CVRP\n")
        f.write(f"DIMENSION : {N}\n")
        f.write(f"EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write(f"CAPACITY : {capacity}\n")
        f.write(f"NODE_COORD_SECTION\n")
        
        # Write depot
        depot = nodes_dict[1]
        f.write(f"1 {depot['x']} {depot['y']}\n")
        
        # Write customers
        hgs_id = 2
        for _, row in cluster_df.iterrows():
            f.write(f"{hgs_id} {row['x']} {row['y']}\n")
            hgs_id += 1
            
        f.write(f"DEMAND_SECTION\n")
        f.write(f"1 0\n")
        
        hgs_id = 2
        for _, row in cluster_df.iterrows():
            demand = nodes_dict[int(row['original_id'])]['demand']
            f.write(f"{hgs_id} {demand}\n")
            hgs_id += 1
            
        f.write("DEPOT_SECTION\n1\n-1\nEOF\n")

def run_hgs(vrp_path, hgs_exec, time_limit):
    sol_path = vrp_path.with_suffix('.sol')
    
    # HGS syntax: hgs <vrp> <sol> -t <time>
    cmd = [hgs_exec, str(vrp_path), str(sol_path), "-t", str(time_limit)]
    
    try:
        # Run HGS
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        
        # Parse cost from .sol file if it exists
        cost = None
        if sol_path.exists():
            with open(sol_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("Cost"):
                        cost = float(line.split()[1])
                        break
        
        return {
            'vrp_file': vrp_path.name,
            'success': True,
            'cost': cost,
            'sol_path': str(sol_path),
            'log': result.stdout
        }
    except Exception as e:
        return {
            'vrp_file': vrp_path.name,
            'success': False,
            'error': str(e)
        }

def process_cluster(args):
    cluster_df, nodes_dict, capacity, out_vrp_path, hgs_exec, time_limit = args
    
    # 1. Write VRP
    write_cluster_vrp(cluster_df, nodes_dict, capacity, out_vrp_path)
    
    # 2. Run HGS
    return run_hgs(out_vrp_path, hgs_exec, time_limit)

def main():
    parser = argparse.ArgumentParser(description="HGS Dispatcher for CVRP Micro-Clusters")
    parser.add_argument("vrp_file", help="Path to original VRP instance")
    parser.add_argument("clusters_csv", help="Path to clusters mapping CSV")
    parser.add_argument("strategy_name", help="Name of the strategy (e.g., SFC, MST, Concentric)")
    parser.add_argument("--hgs", default="/home/ubuntu/cvrp/baselines/HGS-CVRP/build_linux/hgs", help="Path to HGS executable")
    parser.add_argument("--time-limit", type=int, default=60, help="HGS time limit in seconds per cluster")
    
    args = parser.parse_args()
    
    print(f"Loading original VRP: {args.vrp_file}")
    nodes_dict, capacity = parse_vrp(args.vrp_file)
    
    print(f"Loading clusters CSV: {args.clusters_csv}")
    df = pd.read_csv(args.clusters_csv)
    
    # Filter out depot
    df = df[df['macro_id'] != -1]
    
    base_dir = Path("results") / args.strategy_name
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Setting up cluster dispatching jobs...")
    jobs = []
    
    # Group by macro (wedge) and micro (cluster)
    for macro_id, macro_df in df.groupby('macro_id'):
        wedge_dir = base_dir / f"wedge_{macro_id}"
        wedge_dir.mkdir(parents=True, exist_ok=True)
        
        for micro_id, micro_df in macro_df.groupby('micro_id'):
            vrp_path = wedge_dir / f"cluster_{micro_id}.vrp"
            
            jobs.append((
                micro_df,
                nodes_dict,
                capacity,
                vrp_path,
                args.hgs,
                args.time_limit
            ))
    
    print(f"Created {len(jobs)} sub-problems. Running HGS with parallel workers...")
    
    results_list = []
    
    # Use max cores available
    num_cores = os.cpu_count()
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        # submit all jobs
        futures = {executor.submit(process_cluster, job): job[3] for job in jobs}
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            vrp_path = futures[future]
            try:
                res = future.result()
                macro_id = vrp_path.parent.name.split('_')[1]
                micro_id = vrp_path.stem.split('_')[1]
                
                results_list.append({
                    'macro_id': macro_id,
                    'micro_id': micro_id,
                    'cost': res.get('cost', None),
                    'success': res.get('success', False)
                })
                
                completed += 1
                print(f"[{completed}/{len(jobs)}] Solved {vrp_path.stem} | Cost: {res.get('cost', 'N/A')}")
                
            except Exception as e:
                print(f"Job for {vrp_path.name} failed: {e}")
                
    total_time = time.time() - start_time
    print(f"All clusters solved in {total_time:.2f} seconds.")
    
    # Aggregate Costs
    res_df = pd.DataFrame(results_list)
    res_df['cost'] = pd.to_numeric(res_df['cost'], errors='coerce')
    
    summary_path = base_dir / f"summary_{args.strategy_name}.csv"
    res_df.to_csv(summary_path, index=False)
    
    print("\n--- Wedge Aggregations ---")
    wedge_costs = res_df.groupby('macro_id')['cost'].sum()
    print(wedge_costs)
    
    total_cost = wedge_costs.sum()
    print(f"Total Aggregated Cost for Strategy {args.strategy_name}: {total_cost}")
    
    # Append the totals to the bottom of the CSV
    with open(summary_path, 'a') as f:
        f.write("\n# WEDGE TOTALS\n")
        for m_id, cost in wedge_costs.items():
            f.write(f"wedge_{m_id},,{cost}\n")
        f.write(f"# TOTAL_COST,,{total_cost}\n")
        
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
