import os
import re
import csv
import time
import argparse
import subprocess
from pathlib import Path
import concurrent.futures

def run_hgs_on_chunk(exe_path, chunk_path, time_limit):
    """
    Runs HGS and extracts the cost directly from the .sol file or fallback terminal logs.
    """
    sol_path = chunk_path.with_suffix('.sol')
    cmd = [
        str(exe_path), 
        str(chunk_path), 
        str(sol_path), 
        "-t", str(time_limit),
        "-seed", "42"
    ]
    
    start_time = time.perf_counter()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=time_limit + 10)
        output = result.stdout
        exec_time = time.perf_counter() - start_time
        
        cost = None
        
        # 1. First choice: Read the final cost from the .sol file
        if sol_path.exists():
            with open(sol_path, "r", encoding="utf-8") as f:
                for line in reversed(f.readlines()):
                    if line.strip().lower().startswith("cost"):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                cost = float(parts[1])
                                break
                            except ValueError:
                                pass
                                
        # 2. Fallback: If the .sol file wasn't written, scrape the terminal for the last Feasible cost
        if cost is None:
            matches = re.findall(r'Feas\s+\d+\s+([0-9]+\.[0-9]+)', output)
            if matches:
                cost = float(matches[-1]) # Grab the very last valid cost recorded
                
        if cost is not None:
            return {"status": "success", "cost": cost, "time": exec_time}
        else:
            return {"status": "error", "error": "No feasible route found in 60s."}
            
    except Exception as e:
        return {"status": "error", "error": f"Crash: {str(e)}"}
    finally:
        # Clean up all solution files so your hard drive doesn't get cluttered
        if sol_path.exists():
            sol_path.unlink()
        pg_file = chunk_path.with_suffix('.sol.PG.csv')
        if pg_file.exists():
            pg_file.unlink()

def main():
    parser = argparse.ArgumentParser(description="Run HGS on partitioned CVRP folders.")
    parser.add_argument("--exe", type=str, required=True, help="Path to the HGS executable")
    parser.add_argument("--dir", type=str, required=True, help="Base directory containing the partition folders")
    parser.add_argument("--out_csv", type=str, default="HGS_Partition_Costs.csv", help="Where to save the results")
    parser.add_argument("--workers", type=int, default=2, help="Number of chunks to run simultaneously")
    parser.add_argument("--time_limit", type=int, default=60, help="Time limit in seconds per chunk")
    args = parser.parse_args()

    base_dir = Path(args.dir)
    exe_path = Path(args.exe)

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        return
    if not exe_path.exists():
        print(f"Error: HGS executable not found at {exe_path}.")
        return

    # Initialize CSV
    header = ["Original Instance", "Chunk File", "Cost", "Exec Time (s)", "Status/Notes"]
    results_map = {}
    if os.path.isfile(args.out_csv):
        with open(args.out_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0 and row == header:
                    continue
                if len(row) >= 2:
                    results_map[(row[0], row[1])] = row

    def write_csv():
        with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            def sort_key(k):
                inst, chk = k
                return (inst, 1 if chk == "[TOTAL AGGREGATE]" else 0, chk)
            for k in sorted(results_map.keys(), key=sort_key):
                writer.writerow(results_map[k])

    # Initial write to ensure file exists and is clean
    write_csv()

    partition_folders = [f for f in base_dir.iterdir() if f.is_dir()]
    print(f"Found {len(partition_folders)} partitioned instances to process.")

    for folder in partition_folders:
        original_instance = folder.name.replace("_partitions", "")
        print(f"\n--- Processing {original_instance} ---")
        
        all_chunk_files = list(folder.glob("*.vrp"))
        chunks_to_run = []
        
        total_folder_cost = 0.0
        total_folder_time = 0.0
        successful_chunks = 0
        
        for c in all_chunk_files:
            key = (original_instance, c.name)
            row = results_map.get(key)
            if row and len(row) >= 3 and str(row[2]) not in ("ERROR", "INCOMPLETE"):
                # Already completed successfully
                try:
                    total_folder_cost += float(row[2])
                    total_folder_time += float(row[3])
                except ValueError:
                    pass
                successful_chunks += 1
            else:
                chunks_to_run.append(c)

        if not chunks_to_run:
            print(f"All {len(all_chunk_files)} chunks already completed.")
            if successful_chunks == len(all_chunk_files) and len(all_chunk_files) > 0:
                results_map[(original_instance, "[TOTAL AGGREGATE]")] = [
                    original_instance, "[TOTAL AGGREGATE]", round(total_folder_cost, 2), round(total_folder_time, 2), "ALL CHUNKS SUCCESSFUL"
                ]
                write_csv()
            continue
            
        print(f"Found {len(chunks_to_run)} chunks to run (out of {len(all_chunk_files)}). Running {args.workers} workers in parallel...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_chunk = {executor.submit(run_hgs_on_chunk, exe_path, chunk, args.time_limit): chunk for chunk in chunks_to_run}
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_path = future_to_chunk[future]
                result = future.result()
                
                if result["status"] == "success":
                    cost = result["cost"]
                    exec_time = result["time"]
                    results_map[(original_instance, chunk_path.name)] = [
                        original_instance, chunk_path.name, round(cost, 2), round(exec_time, 2), "OK"
                    ]
                    total_folder_cost += cost
                    total_folder_time += exec_time
                    successful_chunks += 1
                    print(f"  [OK] {chunk_path.name}: Cost {cost:.2f} ({exec_time:.1f}s)")
                else:
                    results_map[(original_instance, chunk_path.name)] = [
                        original_instance, chunk_path.name, "ERROR", 0, result["error"]
                    ]
                    print(f"  [FAIL] {chunk_path.name}: {result['error']}")
                
                write_csv()
            
            if successful_chunks == len(all_chunk_files) and len(all_chunk_files) > 0:
                results_map[(original_instance, "[TOTAL AGGREGATE]")] = [
                    original_instance, "[TOTAL AGGREGATE]", round(total_folder_cost, 2), round(total_folder_time, 2), "ALL CHUNKS SUCCESSFUL"
                ]
                print(f"\n>> {original_instance} COMPLETED: Total Cost = {total_folder_cost:.2f}")
            else:
                results_map[(original_instance, "[TOTAL AGGREGATE]")] = [
                    original_instance, "[TOTAL AGGREGATE]", "INCOMPLETE", round(total_folder_cost, 2), f"Only {successful_chunks}/{len(all_chunk_files)} succeeded"
                ]
                print(f"\n>> {original_instance} INCOMPLETE: Errors occurred in some chunks.")
            
            write_csv()

if __name__ == "__main__":
    main()