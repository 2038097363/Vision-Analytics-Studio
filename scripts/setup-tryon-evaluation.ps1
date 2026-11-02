$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $projectRoot "runtime\tryon"
$venvPython = Join-Path $runtimeRoot ".venv\Scripts\python.exe"
$sourceRoot = Join-Path $runtimeRoot "source"
$sourceRevision = "7c0f10af3f91ad4048fe9729c470a13ef905d25a"
$weightsRoot = Join-Path $projectRoot "models\tryon\weights"
$hfHome = Join-Path $runtimeRoot "hf-cache"

New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null

python -c "import torch; assert torch.cuda.is_available(), 'CUDA PyTorch is required'; print(torch.__version__, torch.cuda.get_device_name(0))"
if (-not (Test-Path -LiteralPath $venvPython)) {
    python -m venv --system-site-packages (Join-Path $runtimeRoot ".venv")
}

if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot ".git"))) {
    git clone https://github.com/fashn-AI/fashn-vton-1.5.git $sourceRoot
}
git -C $sourceRoot fetch --depth 1 origin $sourceRevision
git -C $sourceRoot checkout --detach $sourceRevision

& $venvPython -m pip install --retries 8 --timeout 60 -r (Join-Path $projectRoot "engine\requirements-tryon-evaluation.txt")
& $venvPython -m pip install --no-deps -e $sourceRoot

$env:HF_HOME = $hfHome
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
& $venvPython (Join-Path $projectRoot "scripts\download-tryon-weights.py") --weights-dir $weightsRoot --hf-home $hfHome

$statusRequest = @{
    protocol_version = 1
    request_id = "setup-status"
    op = "tryon.status"
    payload = @{
        weights_dir = $weightsRoot
        human_parser_dir = Join-Path $hfHome "hub\models--fashn-ai--fashn-human-parser\snapshots\1f80c34dbab321c5730dda5c3fea279fd3e97498"
        probe_cuda = $true
    }
} | ConvertTo-Json -Depth 4 -Compress
$statusRequest | & $venvPython (Join-Path $projectRoot "engine\tryon_worker.py")
