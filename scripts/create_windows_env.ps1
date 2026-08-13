param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$conda = Get-Command conda -ErrorAction SilentlyContinue
if (-not $conda) {
    throw 'Conda was not found. Install Miniforge or provide conda on PATH.'
}

Write-Output 'This installs the isolated environment and is approval-sensitive.'
Write-Output "After user approval run: conda env create -f `"$ProjectRoot\environment.windows.yml`""
