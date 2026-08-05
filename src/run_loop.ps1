$ErrorActionPreference = "Stop"

$num_runs = 20
$consistent = $true
$first_cost = -1

for ($i = 1; $i -le $num_runs; $i++) {
    Write-Host "--- Run $i ---"
    
    # Run the CVRP solver
    $solver_output = .\build\Release\cvrp_parallel.exe -p 4 | Out-String
    
    # Check if the solver succeeded
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Solver crashed on run $i!" -ForegroundColor Red
        Write-Host $solver_output
        exit 1
    }
    
    # Run the python verifier
    $verifier_output = python verifier.py | Out-String
    
    # Check if python verifier succeeded
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Verification failed on run $i!" -ForegroundColor Red
        Write-Host $verifier_output
        exit 1
    }
    
    # Extract the cost
    $cost_line = $verifier_output -split "`n" | Where-Object { $_ -match "Total computed cost: (\d+)" }
    if ($cost_line -match "Total computed cost: (\d+)") {
        $cost = [int]$matches[1]
        Write-Host "Computed Cost: $cost" -ForegroundColor Green
        
        if ($first_cost -eq -1) {
            $first_cost = $cost
        } elseif ($cost -ne $first_cost) {
            Write-Host "DATA RACE DETECTED! Run $i got cost $cost, but first run got $first_cost." -ForegroundColor Red
            $consistent = $false
            exit 1
        }
    } else {
        Write-Host "Could not parse cost from verifier on run $i!" -ForegroundColor Red
        Write-Host $verifier_output
        exit 1
    }
}

if ($consistent) {
    Write-Host "`nSUCCESS: All $num_runs runs produced the exact same cost: $first_cost" -ForegroundColor Cyan
    Write-Host "No non-deterministic data races detected in the output!" -ForegroundColor Cyan
}
