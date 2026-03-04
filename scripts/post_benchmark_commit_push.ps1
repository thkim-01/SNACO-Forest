Param(
  [Parameter(Mandatory = $true)][int]$BenchmarkPid,
  [Parameter(Mandatory = $true)][string]$RepoRoot,
  [Parameter(Mandatory = $true)][string]$OutCsv,
  [Parameter(Mandatory = $true)][string]$OutAvgCsv,
  [Parameter(Mandatory = $true)][string]$BenchErrLog,
  [Parameter(Mandatory = $true)][string]$WrapperLog
)

$ErrorActionPreference = 'Stop'
Set-Location $RepoRoot

try {
  "Post-process: waiting for PID=$BenchmarkPid" | Out-File -FilePath $WrapperLog -Append -Encoding utf8

  # Wait for benchmark to exit (if already exited, this returns immediately)
  Wait-Process -Id $BenchmarkPid -ErrorAction SilentlyContinue

  "Post-process: benchmark process ended" | Out-File -FilePath $WrapperLog -Append -Encoding utf8

  # Basic validation: outputs exist and look like a full run (not a partial Ctrl-C)
  if (-not (Test-Path -LiteralPath $OutCsv)) {
    throw "Output CSV not found: $OutCsv"
  }
  if (-not (Test-Path -LiteralPath $OutAvgCsv)) {
    throw "Averages CSV not found: $OutAvgCsv"
  }

  # Heuristic: full all-tasks run should produce many rows.
  # Expected roughly: BBBP(1)+BACE(1)+ClinTox(2)+HIV(1)+Tox21(12)+SIDER(~27+) => 44+ rows.
  $lineCount = (Get-Content -LiteralPath $OutCsv).Count
  if ($lineCount -lt 40) {
    $errSnippet = ''
    if (Test-Path -LiteralPath $BenchErrLog) {
      $errSnippet = (Get-Content -LiteralPath $BenchErrLog -Tail 50) -join "`n"
    }
    throw "Output CSV seems incomplete (lines=$lineCount). Tail of stderr:`n$errSnippet"
  }

  # Extract commit_message from first data row (avoid loading entire CSV)
  $lines = Get-Content -LiteralPath $OutCsv -TotalCount 2
  if ($lines.Count -lt 2) {
    throw "Output CSV has no data rows: $OutCsv"
  }
  $row = $lines | ConvertFrom-Csv | Select-Object -First 1
  $msg = [string]$row.commit_message
  if (-not $msg) {
    throw "commit_message is empty in output CSV. Cannot auto-commit."
  }

  "Post-process: committing with message: $msg" | Out-File -FilePath $WrapperLog -Append -Encoding utf8

  # Commit & push
  git add -A | Out-Null

  $diff = git diff --cached --name-only
  if ($diff) {
    git commit -m $msg | Out-Null
  } else {
    "Post-process: nothing staged to commit" | Out-File -FilePath $WrapperLog -Append -Encoding utf8
  }

  $branch = (git rev-parse --abbrev-ref HEAD).Trim()
  try {
    git push | Out-Null
  } catch {
    git push -u origin $branch | Out-Null
  }

  "DONE: benchmark + commit + push" | Out-File -FilePath $WrapperLog -Append -Encoding utf8
} catch {
  $msg = $_.Exception.Message
  "Post-process ERROR: $msg" | Out-File -FilePath $WrapperLog -Append -Encoding utf8
  exit 1
}
