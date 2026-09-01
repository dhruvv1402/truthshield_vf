<#
.SYNOPSIS
  Publish Truth Shield to a Hugging Face Space (Docker SDK).

.DESCRIPTION
  Creates the Space if it does not exist, copies in the files the image needs,
  and pushes. Re-run it any time to redeploy — it is idempotent.

.EXAMPLE
  .\deploy\deploy_hf_space.ps1 -User yourname -Token hf_xxx

.EXAMPLE
  .\deploy\deploy_hf_space.ps1 -User yourname -Token hf_xxx -Space truth-shield -Force
#>
param(
    [Parameter(Mandatory = $true)][string]$User,
    [Parameter(Mandatory = $true)][string]$Token,
    [string]$Space = "truth-shield",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$staging  = Join-Path $env:TEMP "truthshield-hf-$Space"

Write-Host "==> Space: $User/$Space" -ForegroundColor Cyan

# --- 1. create the Space (409 = already there, which is fine) ----------------
$body = @{ type = "space"; name = $Space; sdk = "docker"; private = $false } | ConvertTo-Json
try {
    Invoke-RestMethod -Method Post -Uri "https://huggingface.co/api/repos/create" `
        -Headers @{ Authorization = "Bearer $Token" } `
        -ContentType "application/json" -Body $body | Out-Null
    Write-Host "    created" -ForegroundColor Green
}
catch {
    if ($_.Exception.Response.StatusCode.value__ -eq 409) {
        Write-Host "    already exists - will update" -ForegroundColor Yellow
    }
    else {
        throw "Space creation failed: $($_.Exception.Message)"
    }
}

# --- 2. clone it -------------------------------------------------------------
if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
$remote = "https://${User}:${Token}@huggingface.co/spaces/$User/$Space"
Write-Host "==> Cloning Space into $staging"
git clone --quiet $remote $staging
if ($LASTEXITCODE -ne 0) { throw "git clone failed" }

# --- 3. stage the files the Dockerfile needs ---------------------------------
Write-Host "==> Staging files"
Get-ChildItem $staging -Exclude ".git" | Remove-Item -Recurse -Force

Copy-Item (Join-Path $repoRoot "Dockerfile")      $staging
Copy-Item (Join-Path $repoRoot ".dockerignore")   $staging
Copy-Item (Join-Path $repoRoot "LICENSE")         $staging -ErrorAction SilentlyContinue
Copy-Item (Join-Path $repoRoot "deploy\space_README.md") (Join-Path $staging "README.md")

# backend: source only - never the local venv or the fetched weights
$backendDst = Join-Path $staging "backend"
New-Item -ItemType Directory -Path $backendDst -Force | Out-Null
Get-ChildItem (Join-Path $repoRoot "backend") -Exclude @(".venv", "models", "__pycache__", ".env") |
    Copy-Item -Destination $backendDst -Recurse -Force

# frontend: source only - node_modules and dist are rebuilt in the image
$frontendDst = Join-Path $staging "frontend"
New-Item -ItemType Directory -Path $frontendDst -Force | Out-Null
Get-ChildItem (Join-Path $repoRoot "frontend") -Exclude @("node_modules", "dist") |
    Copy-Item -Destination $frontendDst -Recurse -Force

# --- 4. push -----------------------------------------------------------------
Push-Location $staging
try {
    git config user.email "$User@users.noreply.huggingface.co"
    git config user.name  $User
    git add -A
    if ((git status --porcelain).Length -eq 0 -and -not $Force) {
        Write-Host "==> Nothing changed; Space is already up to date." -ForegroundColor Yellow
    }
    else {
        git commit --quiet -m "Deploy Truth Shield ($(Get-Date -Format s))" --allow-empty
        Write-Host "==> Pushing (this uploads only source; weights are fetched during the build)"
        git push --quiet origin main
        if ($LASTEXITCODE -ne 0) { throw "git push failed" }
    }
}
finally { Pop-Location }

Write-Host ""
Write-Host "Done. The Space is now building (first build takes ~10-15 min)." -ForegroundColor Green
Write-Host "  Build logs : https://huggingface.co/spaces/$User/$Space" -ForegroundColor Cyan
Write-Host "  Live app   : https://$User-$Space.hf.space" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next: add NVIDIA_API_KEY and GNEWS_API_KEY under" -ForegroundColor Yellow
Write-Host "  Settings -> Variables and secrets  (the Space restarts automatically)." -ForegroundColor Yellow
