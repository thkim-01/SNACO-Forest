Param(
  [string]$ZipPath = ""
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$initFiles = Get-ChildItem -Path $repoRoot -Recurse -File -Filter '__init__.py'

foreach ($file in $initFiles) {
  if ($file.Length -eq 0) {
    @(
      '"""Package initializer."""',
      '',
      '# Intentionally non-empty to preserve this file in archives.'
    ) | Set-Content -LiteralPath $file.FullName -Encoding utf8
  }
}

if (-not [string]::IsNullOrWhiteSpace($ZipPath)) {
  $resolvedZip = if ([System.IO.Path]::IsPathRooted($ZipPath)) {
    $ZipPath
  } else {
    Join-Path $repoRoot $ZipPath
  }

  if (Test-Path -LiteralPath $resolvedZip) {
    Remove-Item -LiteralPath $resolvedZip -Force
  }

  Compress-Archive -Path (Join-Path $repoRoot '*') -DestinationPath $resolvedZip -Force
  Write-Host "Archive created: $resolvedZip"
}

Write-Host "Checked $($initFiles.Count) __init__.py files; empty files were filled if needed."
