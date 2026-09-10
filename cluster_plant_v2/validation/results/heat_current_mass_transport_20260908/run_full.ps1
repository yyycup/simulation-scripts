& 'C:\Users\24776\miniforge3\envs\btms\python.exe' -B -u -m cluster_plant_v2.validation.validate_heat_current_stage5 --conservative-transport --no-figures --output-dir cluster_plant_v2/validation/results/heat_current_mass_transport_20260908/full
$taskExitCode = $LASTEXITCODE
Set-Content -LiteralPath (Join-Path $PSScriptRoot 'exit_code.txt') -Value $taskExitCode
exit $taskExitCode
