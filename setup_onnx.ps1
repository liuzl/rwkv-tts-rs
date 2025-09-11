# Setup ONNX Runtime environment for Windows PowerShell

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$OnnxDir = Join-Path $ScriptDir "onnxruntime-win-x64-1.22.1"
$LibDir = Join-Path $OnnxDir "lib"

# Add ONNX Runtime lib directory to PATH
$env:PATH = "$LibDir;$env:PATH"

# Set ORT environment variables
$env:ORT_LIB_LOCATION = $LibDir
$env:ONNXRUNTIME_ROOT = $OnnxDir
$OrtDllPath = Join-Path $LibDir "onnxruntime.dll"
$env:ORT_DYLIB_PATH = $OrtDllPath
setx ORT_DYLIB_PATH $OrtDllPath | Out-Null

Write-Host "ONNX Runtime environment configured:" -ForegroundColor Green
Write-Host "  ONNX_DIR: $OnnxDir" -ForegroundColor Yellow
Write-Host "  ORT_LIB_LOCATION: $LibDir" -ForegroundColor Yellow
Write-Host "  ONNXRUNTIME_ROOT: $OnnxDir" -ForegroundColor Yellow
Write-Host "  ORT_DYLIB_PATH: $OrtDllPath" -ForegroundColor Yellow
Write-Host "User ORT_DYLIB_PATH=" ([Environment]::GetEnvironmentVariable("ORT_DYLIB_PATH","User")) -ForegroundColor Cyan
Write-Host "Current Session ORT_DYLIB_PATH=$env:ORT_DYLIB_PATH" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now run: cargo build --release" -ForegroundColor Cyan