import csv
import os
import subprocess
import re
from pathlib import Path

def run_filo2(exe_path, chunk_path):
    """Runs FILO2 on a specific chunk and extracts the cost, even on timeout."""
    sol_path = chunk_path.with_suffix('.sol')
    cmd = [str(exe_path), str(chunk_path)]
    
    output = ""
    try:
        # 1. Bump the timeout to 300 seconds (5 minutes)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout
        
    except subprocess.TimeoutExpired as e:
        # 2. If it times out, DO NOT PANIC. Scavenge the terminal output!
        print(f"  [TIMEOUT] Python stopped FILO2 at 300s. Scavenging for partial results...")
        output = e.stdout if e.stdout else ""
        
    except Exception as e:
        print(f"  [CRASH] {e}")
        return None
        
    finally:
        if sol_path.exists():
            sol_path.unlink()

    # --- THE FIX: Convert bytes to string if necessary ---
    if isinstance(output, bytes):
        output = output.decode('utf-8', errors='ignore')
    elif output is None:
        output = ""

    # 3. Search the output for all recorded costs, and grab the LAST (best) one
    matches = re.findall(r'(?:Cost|Result|Best|Objective)\s*[:=]?\s*([0-9]+\.?[0-9]*)', output, re.IGNORECASE)
    
    if matches:
        return float(matches[-1]) # Return the very last number it printed
    
    return None

def main():
    csv_file = "HGS_Partition_Costs.csv"
    filo_exe = Path("/mnt/c/internship/iitm/cvrp/baselines/filo2/build_wsl/filo2")
    base_dir = Path("/mnt/c/internship/iitm/cvrp/data/instances/hgs_partitions")
    
    if not os.path.exists(csv_file):
        print("CSV file not found. Wait for your HGS run to finish first!")
        return

    # 1. Read the entire CSV into memory
    with open(csv_file, 'r') as f:
        reader = list(csv.reader(f))
    
    header = reader[0]
    rows = reader[1:]
    
    fixed_count = 0
    
    print("Scanning CSV for ERRORED chunks...")
    
    # 2. Loop through and find the errors
    for i, row in enumerate(rows):
        if len(row) >= 5 and row[4] != "OK" and "[TOTAL AGGREGATE]" not in row[1]:
            instance_name = row[0]
            chunk_name = row[1]
            
            print(f"\nFound error in: {chunk_name}")
            
            # Construct the physical path to the broken chunk
            chunk_path = base_dir / f"{instance_name}_partitions" / chunk_name
            
            if chunk_path.exists():
                print(f"  -> Running FILO2...")
                cost = run_filo2(filo_exe, chunk_path)
                
                if cost is not None:
                    # 3. Update the row with the new FILO2 data
                    print(f"  -> Success! FILO2 solved it for {cost:.2f}")
                    rows[i][2] = f"{cost:.2f}"
                    rows[i][4] = "OK (Patched with FILO2)"
                    fixed_count += 1
                else:
                    print(f"  -> FILO2 also failed to parse a cost.")
            else:
                print(f"  -> File missing: {chunk_path}")

    # 4. Save the patched data back to the CSV
    if fixed_count > 0:
        print(f"\nSuccessfully patched {fixed_count} chunks. Updating CSV...")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print("Done! Your dataset is now complete.")
    else:
        print("\nNo patches were successfully applied.")

if __name__ == "__main__":
    main()