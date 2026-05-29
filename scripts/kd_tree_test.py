import argparse
import time
import numpy as np
import os
import csv
from scipy.spatial import cKDTree

def load_coordinates(filepath):
    """
    Parses a standard CVRP instance file to extract X, Y coordinates.
    """
    coords = []
    print(f"Reading dataset: {filepath}...")
    
    with open(filepath, 'r') as file:
        reading_nodes = False
        for line in file:
            line = line.strip()
            if line.startswith('NODE_COORD_SECTION'):
                reading_nodes = True
                continue
            if line.startswith('DEMAND_SECTION') or line.startswith('EOF'):
                reading_nodes = False
                continue
                
            if reading_nodes and line:
                parts = line.split()
                if len(parts) >= 3:
                    coords.append([float(parts[1]), float(parts[2])])
                    
    data = np.array(coords)
    print(f"Successfully loaded {len(data):,} nodes.")
    return data

def main():
    # 1. Setup Terminal Arguments
    parser = argparse.ArgumentParser(description="Test KD-Tree performance on CVRP instances.")
    parser.add_argument("filepath", type=str, help="Absolute or relative path to the CVRP instance file")
    parser.add_argument("--leaf_size", type=int, default=16, help="Number of points at which the algorithm switches to brute-force")
    parser.add_argument("--neighbors", type=int, default=50, help="Number of nearest neighbors to query")
    parser.add_argument("--samples", type=int, default=5, help="Number of random customers to query")
    
    args = parser.parse_args()

    # 2. Load the Data
    try:
        data = load_coordinates(args.filepath)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if len(data) == 0:
        print("Error: No coordinates found. Check your file format.")
        return

    node_count = len(data)

    # 3. Test Build Time
    print(f"\n--- Building KD-Tree (Leaf Size: {args.leaf_size}) ---")
    start_build = time.perf_counter()
    tree = cKDTree(data, leafsize=args.leaf_size)
    end_build = time.perf_counter()
    build_time = end_build - start_build
    print(f"Build Time: {build_time:.4f} seconds")

    # 4. Test Query Time
    print(f"\n--- Querying {args.neighbors} Nearest Neighbors for {args.samples} random customers ---")
    random_indices = np.random.choice(len(data), size=args.samples, replace=False)
    test_points = data[random_indices]

    start_query = time.perf_counter()
    distances, indices = tree.query(test_points, k=args.neighbors + 1)
    end_query = time.perf_counter()
    
    avg_query_time_ms = ((end_query - start_query) / args.samples) * 1000
    print(f"Total Query Time: {((end_query - start_query) * 1000):.4f} milliseconds")
    print(f"Average Time per Query: {avg_query_time_ms:.4f} milliseconds")

    # 5. Analyze the Distribution
    print("\n--- Geographic Distribution Analysis ---")
    neighbor_distances = distances[:, 1:] 
    
    median_dist = np.median(neighbor_distances)
    max_dist = np.max(neighbor_distances)
    outlier_ratio = max_dist / median_dist if median_dist > 0 else 0
    
    print(f"Global Minimum Distance: {np.min(neighbor_distances):.2f}")
    print(f"Global Maximum Distance: {max_dist:.2f}")
    print(f"Global Mean Distance:    {np.mean(neighbor_distances):.2f}")
    print(f"Global Median Distance:  {median_dist:.2f}")
    print(f"Outlier Ratio (Max/Med): {outlier_ratio:.2f}x")

    # 6. Automate the Spreadsheet Logging
    csv_filename = "CVRP_KDTree_Tests.csv"
    instance_name = os.path.basename(args.filepath)
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write headers if the file is brand new
        if not file_exists:
            writer.writerow(["Instance Name", "Node Count", "Leaf Size", "Build Time (s)", "Avg Query Time (ms)", "Median Distance", "Max Distance", "Outlier Ratio"])
        
        # Write the data row for this test
        writer.writerow([instance_name, node_count, args.leaf_size, round(build_time, 4), round(avg_query_time_ms, 4), round(median_dist, 2), round(max_dist, 2), round(outlier_ratio, 2)])
        
    print(f"\n[Success] Results appended to {csv_filename}!")

if __name__ == "__main__":
    main()