@echo off
REM Setup ONNX Runtime environment for Windows

set SCRIPT_DIR=%~dp0
set ONNX_DIR=%SCRIPT_DIR%onnxruntime-win-x64-1.22.1

REM Add ONNX Runtime lib directory to PATH
set PATH=%ONNX_DIR%\lib;%PATH%

REM Set ORT environment variables
set ORT_LIB_LOCATION=%ONNX_DIR%\lib
set ONNXRUNTIME_ROOT=%ONNX_DIR%

echo ONNX Runtime environment configured:
echo   ONNX_DIR: %ONNX_DIR%
echo   ORT_LIB_LOCATION: %ORT_LIB_LOCATION%
echo   ONNXRUNTIME_ROOT: %ONNXRUNTIME_ROOT%
echo.
echo You can now run: cargo build --release