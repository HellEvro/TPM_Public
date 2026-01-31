# На ПК где main ушёл вперёд на 14 коммитов — выровнять локальный main под origin/main.
# Запускать из корня репозитория InfoBot (или указать путь).
# После: локальные коммиты станут "висячими", git gc их со временем уберёт.

$ErrorActionPreference = "Stop"
$root = if ($args[0]) { $args[0] } else { Split-Path -Parent (Split-Path -Parent $PSCommandPath) }
Set-Location $root

Write-Host "Repo: $root"
Write-Host "Fetching origin..."
git fetch origin
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Resetting main to origin/main (hard)..."
git reset --hard origin/main
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Done. main = origin/main (f0a0c509)."
git log --oneline -1

Write-Host "Очистка reflog и висячих объектов..."
git reflog expire --expire=now --all
git gc --prune=now
Write-Host "Готово."
