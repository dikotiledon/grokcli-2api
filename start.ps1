# Go main process. Prefer WSL/Docker for registration/captcha sidecars.
# Local Windows path: build (if needed) -> migrate up -> start binary.
# Mirrors Docker entrypoint auto-migrate so empty Postgres does not fail-closed.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Import-DotEnv {
  param([string]$Path)
  if (-not (Test-Path $Path)) { return }
  Get-Content -LiteralPath $Path | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#") -or ($line -notmatch "=")) { return }
    $idx = $line.IndexOf("=")
    $key = $line.Substring(0, $idx).Trim()
    $val = $line.Substring($idx + 1).Trim()
    if (($val.StartsWith('"') -and $val.EndsWith('"')) -or ($val.StartsWith("'") -and $val.EndsWith("'"))) {
      $val = $val.Substring(1, $val.Length - 2)
    }
    if (-not [string]::IsNullOrWhiteSpace($key)) {
      Set-Item -Path "Env:$key" -Value $val
    }
  }
}

Import-DotEnv -Path (Join-Path $PSScriptRoot ".env")

$binDir = Join-Path $PSScriptRoot "bin"
$appExe = Join-Path $binDir "grok2api.exe"
$migExe = Join-Path $binDir "grok2api-migrate.exe"
$cmdDir = Join-Path $PSScriptRoot "cmd"

if (-not (Test-Path $appExe)) {
  Write-Host "Building $appExe ..."
  New-Item -ItemType Directory -Force -Path $binDir | Out-Null
  go build -o $appExe (Join-Path $cmdDir "grok2api")
  if ($LASTEXITCODE -ne 0) { throw "go build grok2api failed with exit code $LASTEXITCODE" }
}

# Auto-migrate (same default as Docker entrypoint: GROK2API_AUTO_MIGRATE=1)
$autoMigrate = if ($env:GROK2API_AUTO_MIGRATE) { $env:GROK2API_AUTO_MIGRATE.ToLower() } else { "1" }
if ($autoMigrate -in @("0", "false", "no", "off")) {
  Write-Host "GROK2API_AUTO_MIGRATE=$autoMigrate; skip schema migrate"
} else {
  if (-not (Test-Path $migExe)) {
    Write-Host "Building $migExe ..."
    New-Item -ItemType Directory -Force -Path $binDir | Out-Null
    go build -o $migExe (Join-Path $cmdDir "grok2api-migrate")
    if ($LASTEXITCODE -ne 0) { throw "go build grok2api-migrate failed with exit code $LASTEXITCODE" }
  }
  $migDir = if ($env:GROK2API_MIGRATIONS_DIR) { $env:GROK2API_MIGRATIONS_DIR } else { (Join-Path $PSScriptRoot "migrations") }
  if (-not [System.IO.Path]::IsPathRooted($migDir)) {
    $migDir = Join-Path $PSScriptRoot $migDir
  }
  if (-not (Test-Path $migDir)) {
    throw "migrations directory missing: $migDir"
  }
  if (-not $env:DATABASE_URL -and -not $env:GROK2API_DATABASE_URL) {
    throw "DATABASE_URL (or GROK2API_DATABASE_URL) is required for migrate"
  }
  Write-Host "Applying migrations (dir=$migDir) ..."
  $up = Start-Process -FilePath $migExe -ArgumentList @("-dir", $migDir, "up") -WorkingDirectory $PSScriptRoot -Wait -PassThru -NoNewWindow
  if ($up.ExitCode -ne 0) {
    throw "grok2api-migrate up failed with exit code $($up.ExitCode)"
  }
  $verify = Start-Process -FilePath $migExe -ArgumentList @("-dir", $migDir, "verify") -WorkingDirectory $PSScriptRoot -Wait -PassThru -NoNewWindow
  if ($verify.ExitCode -ne 0) {
    throw "grok2api-migrate verify failed with exit code $($verify.ExitCode)"
  }
}

$env:GROK2API_RUNTIME = "go"
$env:GROK2API_GO_PUBLIC_READ = "1"
$env:GROK2API_GO_CHAT = "1"
$env:GROK2API_GO_MESSAGES = "1"
$env:GROK2API_GO_RESPONSES = "1"
$env:GROK2API_GO_ADMIN_READ = "1"
$env:GROK2API_GO_ADMIN_WRITE = "1"
$env:GROK2API_GO_MAINTAINER = "1"
$env:GROK2API_GO_WRITES = "1"
& $appExe
