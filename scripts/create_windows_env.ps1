param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$MiniforgeVersion = '26.3.2-3',
    [string]$MiniforgeSha256 = '14a8635465b5190537ddad6286746ffebbc55a1ed2a7bb14a506595fe3191e1e',
    [string]$TorchWheelSha256 = '633005a3700e81b5be0df2a7d3c1d48aced23ed927653797a3bd2b144a3aeeb6'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$driveRoot = [System.IO.Path]::GetPathRoot($ProjectRoot)
$pathAlias = Join-Path $driveRoot '.cwr-eg-project-local'
if (-not (Test-Path -LiteralPath $pathAlias)) {
    New-Item -ItemType Junction -Path $pathAlias -Target $ProjectRoot | Out-Null
}
$aliasItem = Get-Item -LiteralPath $pathAlias -Force
$aliasTarget = [System.IO.Path]::GetFullPath([string]$aliasItem.Target)
if (-not $aliasItem.Attributes.HasFlag([System.IO.FileAttributes]::ReparsePoint) -or $aliasTarget -ne $ProjectRoot) {
    throw "Path alias does not point to this project: $pathAlias"
}

$miniforgeRoot = Join-Path $pathAlias '.miniforge'
$environmentPrefix = Join-Path $pathAlias '.conda\cwr-eg-win-py311'
$installerDirectory = Join-Path $ProjectRoot 'artifacts\installers'
$installerPath = Join-Path $installerDirectory 'Miniforge3-Windows-x86_64.exe'
$environmentFile = Join-Path $pathAlias 'environment.windows.yml'
$condaPath = Join-Path $miniforgeRoot 'Scripts\conda.exe'
$environmentPython = Join-Path $environmentPrefix 'python.exe'
$downloadUrl = "https://github.com/conda-forge/miniforge/releases/download/$MiniforgeVersion/Miniforge3-Windows-x86_64.exe"
$torchWheelDirectory = Join-Path $pathAlias 'artifacts\wheels'
$torchWheelPath = Join-Path $torchWheelDirectory 'torch-2.9.1+cu128-cp311-cp311-win_amd64.whl'
$torchWheelUrl = 'https://download-r2.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp311-cp311-win_amd64.whl'

if (-not (Test-Path -LiteralPath $condaPath)) {
    New-Item -ItemType Directory -Path $installerDirectory -Force | Out-Null
    if (-not (Test-Path -LiteralPath $installerPath)) {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath
    }
    $actualHash = (Get-FileHash -LiteralPath $installerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $MiniforgeSha256.ToLowerInvariant()) {
        throw "Miniforge installer SHA-256 mismatch: $actualHash"
    }
    $installer = Start-Process -FilePath $installerPath -ArgumentList @(
        '/InstallationType=JustMe',
        '/RegisterPython=0',
        '/AddToPath=0',
        '/S',
        "/D=$miniforgeRoot"
    ) -WindowStyle Hidden -Wait -PassThru
    if ($installer.ExitCode -ne 0) {
        throw "Miniforge installer failed with exit code $($installer.ExitCode)"
    }
}

$env:CONDA_PKGS_DIRS = Join-Path $pathAlias '.conda\pkgs'
if (-not (Test-Path -LiteralPath $environmentPython)) {
    Push-Location $pathAlias
    try {
        & $condaPath env create --prefix $environmentPrefix --file $environmentFile --yes
        if ($LASTEXITCODE -ne 0) {
            throw "Conda environment creation failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

Push-Location $pathAlias
try {
    New-Item -ItemType Directory -Path $torchWheelDirectory -Force | Out-Null
    if (-not (Test-Path -LiteralPath $torchWheelPath)) {
        & curl.exe --fail --location --retry 10 --retry-all-errors --retry-delay 5 `
            --connect-timeout 30 --speed-limit 1024 --speed-time 60 `
            --continue-at - --output $torchWheelPath $torchWheelUrl
        if ($LASTEXITCODE -ne 0) {
            throw "PyTorch wheel download failed with exit code $LASTEXITCODE"
        }
    }
    $actualTorchWheelHash = (Get-FileHash -LiteralPath $torchWheelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualTorchWheelHash -ne $TorchWheelSha256.ToLowerInvariant()) {
        throw "PyTorch wheel SHA-256 mismatch: $actualTorchWheelHash"
    }
    & $environmentPython -m pip install `
        --index-url 'https://pypi.org/simple' `
        $torchWheelPath
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch CUDA dependency installation failed with exit code $LASTEXITCODE"
    }
    & $environmentPython -m pip install `
        --index-url 'https://pypi.org/simple' `
        --editable ".[gpu,dev]"
    if ($LASTEXITCODE -ne 0) {
        throw "Pip dependency installation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Output "ProjectPathAlias=$pathAlias"
Write-Output "MiniforgeRoot=$miniforgeRoot"
Write-Output "EnvironmentPrefix=$environmentPrefix"
& $condaPath --version
& $environmentPython --version
