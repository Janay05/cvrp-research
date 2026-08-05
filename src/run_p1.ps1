$ErrorActionPreference = "Stop"

Write-Host "--- Running P=1 Single-Threaded CVRP ---"
$solver_output = .\build\Release\cvrp_parallel.exe -p 1 | Out-String

if ($LASTEXITCODE -ne 0) {
    Write-Host "Solver crashed on P=1 run!" -ForegroundColor Red
    Write-Host $solver_output
    exit 1
}

$verifier_output = python verifier.py | Out-String
if ($LASTEXITCODE -ne 0) {
    Write-Host "Verification failed on P=1 run!" -ForegroundColor Red
    Write-Host $verifier_output
    exit 1
}

Write-Host "P=1 Run completed successfully!" -ForegroundColor Green
Write-Host $solver_output
Write-Host "Verifier confirmation:"
Write-Host $verifier_output
