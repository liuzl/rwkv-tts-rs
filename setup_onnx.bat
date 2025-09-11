@echo off
REM Setup ONNX Runtime environment for Windows

set SCRIPT_DIR=%~dp0
REM 优先选择“第三方库源码/onnxruntime-win-x64-1.22.1”，否则回退到默认目录
set ONNX_DIR=
if exist "%SCRIPT_DIR%onnxruntime-win-x64-1.22.1" (
  set ONNX_DIR=%SCRIPT_DIR%onnxruntime-win-x64-1.22.1
) else (
  set ONNX_DIR=%SCRIPT_DIR%onnxruntime-win-x64-1.22.1
)


REM Add ONNX Runtime lib directory to PATH
set PATH=%ONNX_DIR%\lib;%PATH%

REM Set ORT environment variables
set ORT_LIB_LOCATION=%ONNX_DIR%\lib
set ONNXRUNTIME_ROOT=%ONNX_DIR%
set ORT_DYLIB_PATH=%ONNX_DIR%\lib\onnxruntime.dll
setx ORT_DYLIB_PATH "%ONNX_DIR%\lib\onnxruntime.dll" >nul

echo ONNX Runtime environment configured:
echo   ONNX_DIR: %ONNX_DIR%
echo   ORT_LIB_LOCATION: %ORT_LIB_LOCATION%
echo   ONNXRUNTIME_ROOT: %ONNXRUNTIME_ROOT%
echo   ORT_DYLIB_PATH: %ORT_DYLIB_PATH%
echo User ORT_DYLIB_PATH: %ORT_DYLIB_PATH%
echo Current Session ORT_DYLIB_PATH: %ORT_DYLIB_PATH%
echo.
echo You can now run: cargo build --release