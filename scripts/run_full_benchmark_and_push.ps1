Param(
  [int]$NEstimators = 5
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$outCsv = Join-Path $repoRoot 'output\semantic_forest_benchmark_full.csv'
$outAvgCsv = Join-Path $repoRoot 'output\semantic_forest_benchmark_full_avg.csv'
$wrapperLog = Join-Path $repoRoot 'output\\benchmark_wrapper.log'
$benchOutLog = Join-Path $repoRoot 'output\\benchmark_run.log'
$benchErrLog = Join-Path $repoRoot 'output\\benchmark_run.err.log'

# Clean previous outputs (ignore if missing)
Remove-Item -LiteralPath $outCsv -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $outAvgCsv -Force -ErrorAction SilentlyContinue

# Prefer repo-local venv python if available
$venvPy = Join-Path $repoRoot '.venv\Scripts\python.exe'
$pythonExe = if (Test-Path $venvPy) { $venvPy } else { 'python' }

$scriptPath = Join-Path $repoRoot 'experiments\verify_semantic_forest_multi.py'

# Start benchmark detached; redirect output so this wrapper stays quiet

function Quote-Arg([string]$s) {
  if ($null -eq $s) { return '""' }
  # Escape embedded double-quotes for Windows command line
  $escaped = $s -replace '"', '\\"'
  if ($escaped -match '\s') {
    return '"' + $escaped + '"'
  }
  return $escaped
}

$argParts = @(
  (Quote-Arg $scriptPath),
  '--overwrite',
  '--all-tasks',
  '--sample-size', '0',
  '--n-estimators', (Quote-Arg "$NEstimators"),
  '--out', (Quote-Arg $outCsv),
  '--out-avg', (Quote-Arg $outAvgCsv)
)

$argString = $argParts -join ' '

"Starting benchmark..." | Out-File -FilePath $wrapperLog -Encoding utf8
"Command: $pythonExe $argString" | Out-File -FilePath $wrapperLog -Append -Encoding utf8

# Start benchmark detached; redirect logs.
$proc = Start-Process -FilePath $pythonExe -ArgumentList $argString -WorkingDirectory $repoRoot -PassThru -RedirectStandardOutput $benchOutLog -RedirectStandardError $benchErrLog
"Benchmark PID: $($proc.Id)" | Out-File -FilePath $wrapperLog -Append -Encoding utf8

"Benchmark launched (no commit/push)." | Out-File -FilePath $wrapperLog -Append -Encoding utf8
