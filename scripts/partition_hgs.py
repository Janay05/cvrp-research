import argparse
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def read_vrp(filepath):
    """Reads a standard .vrp file and separates the metadata, depot, and customers."""
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
            
    # Logic Step 1: Isolate the Depot (Always Node 1 in standard CVRP)
    depot_coord = nodes.pop(1)
    depot_demand = demands.pop(1, 0.0)
    
    # What remains in 'nodes' and 'demands' are purely customers.
    customer_ids = list(nodes.keys())
    customer_coords = np.array(list(nodes.values()))
    customer_demands = [demands[cid] for cid in customer_ids]
    
    return metadata, depot_coord, depot_demand, customer_ids, customer_coords, customer_demands

def write_vrp_chunk(output_path, metadata, depot_coord, depot_demand, chunk_coords, chunk_demands):
    """Writes a brand new, fully valid .vrp file for a specific cluster."""
    # Logic Step 4: Re-indexing. Depot is 1. Customers are 2 to N.
    total_nodes = len(chunk_coords) + 1
    
    with open(output_path, 'w') as f:
        # Write VRP Header
        f.write(f"NAME : {os.path.basename(output_path)}\n")
        f.write(f"COMMENT : Partitioned chunk for HGS\n")
        f.write(f"TYPE : CVRP\n")
        f.write(f"DIMENSION : {total_nodes}\n")
        f.write(f"EDGE_WEIGHT_TYPE : {metadata.get('EDGE_WEIGHT_TYPE', 'EUC_2D')}\n")
        f.write(f"CAPACITY : {metadata.get('CAPACITY', '1000')}\n")
        
        # Write Coordinates (Depot + Customers)
        f.write("NODE_COORD_SECTION\n")
        f.write(f"1 {depot_coord[0]} {depot_coord[1]}\n")
        for idx, coord in enumerate(chunk_coords):
            f.write(f"{idx + 2} {coord[0]} {coord[1]}\n")
            
        # Write Demands
        f.write("DEMAND_SECTION\n")
        f.write(f"1 {depot_demand}\n")
        for idx, demand in enumerate(chunk_demands):
            f.write(f"{idx + 2} {demand}\n")
            
        f.write("DEPOT_SECTION\n 1\n -1\nEOF\n")

def main():
    parser = argparse.ArgumentParser(description="Partition a CVRP dataset into HGS-ready chunks.")
    parser.add_argument("filepath", type=str, help="Path to the original .vrp file")
    parser.add_argument("--chunk_size", type=int, default=2000, help="Target number of customers per chunk")
    args = parser.parse_args()

    # 1. Parse the original file
    metadata, depot_coord, depot_demand, cust_ids, cust_coords, cust_demands = read_vrp(args.filepath)
    num_customers = len(cust_coords)
    
    # 2. Logic Step 2: Calculate K
    k_clusters = max(1, num_customers // args.chunk_size)
    print(f"Total Customers: {num_customers:,}. Dividing into {k_clusters} clusters...")

    # 3. Logic Step 3: Run MiniBatch K-Means
    print("Running geographic clustering...")
    kmeans = MiniBatchKMeans(n_clusters=k_clusters, batch_size=10000, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(cust_coords)

    # 4. Create an output folder for the chunks
    base_name = os.path.basename(args.filepath).replace('.vrp', '')
    output_dir = f"{base_name}_partitions"
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. Group customers by cluster and write files
    print(f"Writing {k_clusters} new .vrp files to ./{output_dir}/")
    for cluster_id in range(k_clusters):
        # Find which customers belong to this specific cluster
        indices = np.where(labels == cluster_id)[0]
        
        chunk_coords = cust_coords[indices]
        chunk_demands = [cust_demands[i] for i in indices]
        
        output_file = os.path.join(output_dir, f"{base_name}_chunk_{cluster_id}.vrp")
        write_vrp_chunk(output_file, metadata, depot_coord, depot_demand, chunk_coords, chunk_demands)

    print("Success! Dataset successfully partitioned for HGS testing.")

if __name__ == "__main__":
    main()