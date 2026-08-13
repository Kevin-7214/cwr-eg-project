param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$externalRoot = Join-Path $ProjectRoot 'external'
New-Item -ItemType Directory -Force -Path $externalRoot | Out-Null

$repositories = @(
    @{
        Name = 'MarkLLM'
        Url = 'https://github.com/THU-BPM/MarkLLM.git'
        Commit = 'c45ddc40f7b761beabe55a1b8dc4690e531d1c6d'
    },
    @{
        Name = 'lm-watermarking'
        Url = 'https://github.com/jwkirchenbauer/lm-watermarking.git'
        Commit = '82922516930c02f8aa322765defdb5863d07a00e'
    },
    @{
        Name = 'llm-watermark-location'
        Url = 'https://github.com/XuandongZhao/llm-watermark-location.git'
        Commit = '87cab921dc5fcdef62ce3b6410a791d387780d2e'
    },
    @{
        Name = 'detect-gpt'
        Url = 'https://github.com/eric-mitchell/detect-gpt.git'
        Commit = '447d2ce8177004203f42d1da87d7f93a2e31ad52'
    },
    @{
        Name = 'Binoculars'
        Url = 'https://github.com/ahans30/Binoculars.git'
        Commit = 'c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8'
    }
)

foreach ($repository in $repositories) {
    $target = Join-Path $externalRoot $repository.Name
    if (-not (Test-Path -LiteralPath $target -PathType Container)) {
        git clone --filter=blob:none --no-checkout $repository.Url $target
    }
    $origin = git -C $target remote get-url origin
    if ($origin -ne $repository.Url) {
        throw "Unexpected origin for $($repository.Name): $origin"
    }
    git -C $target fetch --depth 1 origin $repository.Commit
    git -C $target checkout --detach $repository.Commit
    $actual = git -C $target rev-parse HEAD
    if ($actual -ne $repository.Commit) {
        throw "Revision mismatch for $($repository.Name): $actual"
    }
}

Write-Output 'Pinned external repositories are ready.'
