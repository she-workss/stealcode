[CmdletBinding()]
Param(
    [Parameter()][Alias('a')][ValidateSet('x86_64', 'aarch64')][string]$Architecture = 'x86_64',
    [Parameter()][string]$Version = '0.0.0',
    [Parameter()][ValidateSet('stable', 'nightly')][string]$Channel = 'stable'
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# `$PSScriptRoot` is `<repo>\script`; every path below is derived from it,
# so this script works regardless of where the repo is checked out or which
# machine (dev box or CI runner) it runs on.
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$Target = "$Architecture-pc-windows-msvc"
$CargoOutDir = Join-Path $RepoRoot "target\$Target\release"
$StagingDir = Join-Path $RepoRoot "target\windows-staging\$Channel-$Architecture"
$OutputDir = Join-Path $RepoRoot 'target'

function Prepare-Staging {
    if (Test-Path $StagingDir) {
        Remove-Item -Path $StagingDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $StagingDir -Force | Out-Null

    $ResourcesSource = Join-Path $RepoRoot 'crates\cli\assets\resources\windows'
    Copy-Item -Path (Join-Path $ResourcesSource '*') -Destination $StagingDir -Recurse -Force
    Copy-Item -Path (Join-Path $RepoRoot 'LICENSE') -Destination (Join-Path $StagingDir 'LICENSE') -Force
    Copy-Item -Path (Join-Path $RepoRoot 'README.md') -Destination (Join-Path $StagingDir 'README.md') -Force
    Copy-Item -Path (Join-Path $RepoRoot 'crates\cli\assets\icons\prod\icon.ico') -Destination (Join-Path $StagingDir 'icon.ico') -Force

    rustup target add $Target
}

function Build-Binaries {
    Write-Output "Building stealcode (features=all) + auto_update_helper for $Target (version $Version, channel $Channel)"
    $env:STEALCODE_VERSION = $Version
    $env:STEALCODE_RELEASE_CHANNEL = $Channel
    cargo build --release --package cli --features all --package auto_update_helper --target $Target
    Copy-Item -Path (Join-Path $CargoOutDir 'stealcode.exe') -Destination (Join-Path $StagingDir 'stealcode.exe') -Force
    Copy-Item -Path (Join-Path $CargoOutDir 'auto_update_helper.exe') -Destination (Join-Path $StagingDir 'auto_update_helper.exe') -Force
}

function Build-Installer {
    $issPath = Join-Path $StagingDir 'stealcode.iss'
    $innoSetupPath = 'C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
    # Windows Server 2022 runners have Inno Setup 6 preinstalled at this
    # path (see actions/runner-images Windows2022-Readme.md). Re-check that
    # doc if you switch runner images.
    if (-not (Test-Path $innoSetupPath)) {
        throw "Inno Setup not found at $innoSetupPath - install it or update this script's path"
    }

    $defs = @(
        "/dVersion=$Version",
        "/dMyAppArch=$Architecture",
        "/dChannel=$Channel",
        "/dResourcesDir=$StagingDir\",
        "/dOutputDir=$OutputDir\"
    )

    Write-Output "Running Inno Setup: $innoSetupPath $issPath $defs"
    $process = Start-Process -FilePath $innoSetupPath -ArgumentList (@($issPath) + $defs) -NoNewWindow -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Inno Setup failed with exit code $($process.ExitCode)"
    }

    $builtInstaller = Join-Path $OutputDir "StealCode-$Architecture.exe"
    if (-not (Test-Path $builtInstaller)) {
        throw "Expected Inno Setup to produce $builtInstaller, but it wasn't found"
    }
    Write-Output "Built $builtInstaller"
}

Prepare-Staging
Build-Binaries
Build-Installer
