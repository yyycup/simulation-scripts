& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -B -u -m cluster_plant_v2.validation.results.heat_current_mass_transport_20260908.run_remaining_cases
$taskExitCode = $LASTEXITCODE
Set-Content -LiteralPath (Join-Path $PSScriptRoot 'parallel_exit_code.txt') -Value $taskExitCode
exit $taskExitCode
