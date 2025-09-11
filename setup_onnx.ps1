# Setup ONNX Runtime environment for Windows PowerShell

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$OnnxDir = Join-Path $ScriptDir "onnxruntime-win-x64-1.22.1"
$LibDir = Join-Path $OnnxDir "lib"

# Add ONNX Runtime lib directory to PATH
$env:PATH = "$LibDir;$env:PATH"

# Set ORT environment variables
$env:ORT_LIB_LOCATION = $LibDir
$env:ONNXRUNTIME_ROOT = $OnnxDir

Write-Host "ONNX Runtime environment configured:" -ForegroundColor Green
Write-Host "  ONNX_DIR: $OnnxDir" -ForegroundColor Yellow
Write-Host "  ORT_LIB_LOCATION: $LibDir" -ForegroundColor Yellow
Write-Host "  ONNXRUNTIME_ROOT: $OnnxDir" -ForegroundColor Yellow
Write-Host ""
Write-Host "You can now run: cargo build --release" -ForegroundColor Cyan