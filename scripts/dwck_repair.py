import os
import argparse
import pandas as pd
import numpy as np
import subprocess
import time
import math
import shutil

def read_vrp(filepath):
    print(f"Reading {filepath}...")
    metadata = {}
    nodes = {}
    demands = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
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
    customer_demands = [demands[cid] for cid in customer_ids]
    
    return metadata, depot_coord, depot_demand, customer_ids, customer_coords, customer_demands

def write_vrp_chunk(output_path, metadata, depot_coord, depot_demand, chunk_coords, chunk_demands):
    total_nodes = len(chunk_coords) + 1
    with open(output_path, 'w') as f:
        f.write(f"NAME : {os.path.basename(output_path)}\n")
        f.write(f"COMMENT : DWCK Repaired chunk for HGS\n")
        f.write(f"TYPE : CVRP\n")
        f.write(f"DIMENSION : {total_nodes}\n")
        f.write(f"EDGE_WEIGHT_TYPE : {metadata.get('EDGE_WEIGHT_TYPE', 'EUC_2D')}\n")
        f.write(f"CAPACITY : {metadata.get('CAPACITY', '1000')}\n")
        f.write("NODE_COORD_SECTION\n")
        f.write(f"1 {depot_coord[0]} {depot_coord[1]}\n")
        for idx, coord in enumerate(chunk_coords):
            f.write(f"{idx + 2} {coord[0]} {coord[1]}\n")
        f.write("DEMAND_SECTION\n")
        f.write(f"1 {depot_demand}\n")
        for idx, demand in enumerate(chunk_demands):
            f.write(f"{idx + 2} {demand}\n")
        f.write("DEPOT_SECTION\n 1\n -1\nEOF\n")

def demand_weighted_kmeans(coords, demands, capacity, k_clusters):
    n_nodes = len(coords)
    total_demand = sum(demands)
    target_budget = (total_demand / k_clusters) * 1.05 # 5% headroom
    
    centroids = []
    probs = np.array(demands) / sum(demands)
    first_idx = np.random.choice(n_nodes, p=probs)
    centroids.append(coords[first_idx])
    
    for _ in range(1, k_clusters):
        dists = np.array([min(np.sum((c - ctr)**2) for ctr in centroids) for c in coords])
        probs = dists * np.array(demands)
        if sum(probs) == 0:
            probs = np.ones(n_nodes) / n_nodes
        else:
            probs = probs / sum(probs)
        idx = np.random.choice(n_nodes, p=probs)
        centroids.append(coords[idx])
        
    centroids = np.array(centroids)
    labels = np.zeros(n_nodes, dtype=int) - 1
    
    for iteration in range(30):
        labels.fill(-1)
        cluster_demand = np.zeros(k_clusters)
        sorted_indices = np.argsort(-np.array(demands))
        
        for i in sorted_indices:
            d = demands[i]
            coord = coords[i]
            dists = np.sum((centroids - coord)**2, axis=1)
            sorted_centroids = np.argsort(dists)
            
            assigned = False
            for j in sorted_centroids:
                if cluster_demand[j] + d <= target_budget:
                    labels[i] = j
                    cluster_demand[j] += d
                    assigned = True
                    break
            
            if not assigned:
                least_loaded = np.argmin(cluster_demand)
                labels[i] = least_loaded
                cluster_demand[least_loaded] += d
                
        new_centroids = np.zeros_like(centroids)
        for j in range(k_clusters):
            mask = (labels == j)
            if np.sum(mask) > 0:
                d_mask = np.array(demands)[mask]
                if np.sum(d_mask) > 0:
                    new_centroids[j] = np.average(coords[mask], axis=0, weights=d_mask)
                else:
                    new_centroids[j] = np.mean(coords[mask], axis=0)
            else:
                new_centroids[j] = centroids[j]
                
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    return labels

def run_dwck_repair_pipeline(csv_path, hgs_exe, dry_run=False):
    df = pd.read_csv(csv_path)
    
    # Identify errored chunks
    # Status/Notes containing 'Failed', 'Crash', 'ERROR', or Cost == 'ERROR'
    bad_mask = df['Cost'].astype(str) == 'ERROR'
    bad_chunks = df[bad_mask]
    
    print(f"Found {len(bad_chunks)} bad chunks to repair.")
    if dry_run:
        return
        
    for index, row in bad_chunks.iterrows():
        instance = row['Original Instance']
        chunk_file = row['Chunk File']
        
        # The vrp file might be named with _sweep if it was a sweep file, but we should go to the original
        # Let's get the base vrp file
        base_chunk_name = chunk_file.split('_sweep')[0]
        if not base_chunk_name.endswith('.vrp'):
            base_chunk_name += '.vrp'
            
        vrp_path = f"/home/ubuntu/cvrp/data/instances/hgs_partitions/{instance}_partitions/{base_chunk_name}"
        bak_path = vrp_path + ".BAK"
        
        target_file = bak_path if os.path.exists(bak_path) else vrp_path
        
        if not os.path.exists(target_file):
            print(f"Cannot find source file for {chunk_file} (tried {target_file})")
            continue
            
        metadata, depot_coord, depot_demand, cust_ids, cust_coords, cust_demands = read_vrp(target_file)
        
        num_customers = len(cust_coords)
        capacity = float(metadata.get('CAPACITY', '1000'))
        
        # Determine K to get around 1500 nodes per sub-chunk
        k_clusters = max(2, int(math.ceil(num_customers / 1500.0)))
        print(f"Repairing {chunk_file}: {num_customers} nodes -> splitting into {k_clusters} DWCK sub-chunks.")
        
        labels = demand_weighted_kmeans(cust_coords, cust_demands, capacity, k_clusters)
        
        total_cost = 0.0
        total_time = 0.0
        sub_costs = []
        
        output_dir = os.path.dirname(target_file)
        all_success = True
        
        for cluster_id in range(k_clusters):
            indices = np.where(labels == cluster_id)[0]
            chunk_coords = cust_coords[indices]
            chunk_demands = [cust_demands[i] for i in indices]
            
            sub_chunk_path = os.path.join(output_dir, f"{base_chunk_name.replace('.vrp', '')}_dwck_{cluster_id}.vrp")
            write_vrp_chunk(sub_chunk_path, metadata, depot_coord, depot_demand, chunk_coords, chunk_demands)
            
            # Run HGS
            sol_path = sub_chunk_path.replace('.vrp', '.sol')
            cmd = [hgs_exe, sub_chunk_path, sol_path, '-t', '180', '-seed', '42']
            
            start_time = time.time()
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=195, text=True)
                exec_time = time.time() - start_time
                total_time += exec_time
                
                # Parse cost from stdout
                cost = None
                for line in result.stdout.split('\n'):
                    if line.startswith("Cost"):
                        cost = float(line.split()[1])
                        break
                if cost is None:
                    print(f"  -> Sub-chunk {cluster_id} failed to find cost.")
                    all_success = False
                    break
                
                print(f"  -> Sub-chunk {cluster_id} solved: Cost {cost}, Time {exec_time:.2f}s")
                sub_costs.append(cost)
                total_cost += cost
                
            except subprocess.TimeoutExpired:
                print(f"  -> Sub-chunk {cluster_id} timed out.")
                all_success = False
                break
                
        if all_success:
            df.at[index, 'Cost'] = total_cost
            df.at[index, 'Exec Time (s)'] = total_time
            df.at[index, 'Status/Notes'] = 'OK (DWCK Repair)'
            print(f"Successfully repaired {chunk_file}! New cost: {total_cost}")
        else:
            print(f"Failed to fully repair {chunk_file}.")
            
    df.to_csv(csv_path, index=False)
    print("Repair complete. CSV updated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--hgs", type=str, default="/home/ubuntu/cvrp/baselines/HGS-CVRP/build/hgs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    run_dwck_repair_pipeline(args.csv, args.hgs, args.dry_run)
