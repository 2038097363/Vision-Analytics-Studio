$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Runtime = Join-Path $Root "runtime\plants"
$Venv = Join-Path $Runtime ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
if (-not (Test-Path $Python)) {
    python -m venv --system-site-packages $Venv
}
& $Python -m pip install -r (Join-Path $Root "engine\requirements-plants-evaluation.txt")
$env:HF_HOME = Join-Path $Runtime "hf-cache"
& $Python -c "from huggingface_hub import snapshot_download; snapshot_download('imageomics/bioclip', revision='ce901ab3c6a913f9e9ef94ce6d27761069f4f01c', local_dir=r'$Root\models\plants\weights')"
Write-Host "BioCLIP evaluation runtime is ready."
