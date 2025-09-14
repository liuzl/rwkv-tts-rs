# RWKV TTS Rust

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.78+-orange.svg)](https://www.rust-lang.org)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co/cgisky/rwkv-tts/tree/main)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/your-repo/rwkv-tts-rs/releases)

RWKV-based Text-to-Speech implementation in Rust with embedded Web UI.

**Based on**: This project is a Rust implementation inspired by the original Python project [yueyulin/respark](https://huggingface.co/yueyulin/respark), which is a TTS system with RWKV-7 LM modeling audio tokens.

## Features

- 🚀 **Single-file deployment** - All-in-one executable with embedded Web UI
- 🌐 **Web Interface** - User-friendly browser-based interface for TTS generation
- ⚡ **High-performance** TTS generation using RWKV models
- 🎯 **Zero-shot voice cloning** with reference audio
- 🎛️ **Customizable voice properties** (pitch, speed, emotion, age, gender)
- 🌍 **Multi-language support** with automatic model downloading
- 🔄 **Mirror support** - Automatic fallback to China mirrors for faster downloads
- 📱 **Cross-platform** - Windows, Linux, and macOS support


## Quick Start

### 1. Build the Project

```bash
# Linux/macOS
sh build.sh

# Windows
.\build.ps1
```

The build script will:
- Automatically download required models from Hugging Face
- Support mirror fallback for users in China
- Compile the single-file executable with embedded Web UI

### 2. Run the Web Server

```bash
# Start the web server (default port: 3000)
cargo run --release --bin rwkvtts_server

# Or specify a custom port
cargo run --release --bin rwkvtts_server -- --port 8080
```

### 3. Access the Web Interface

Open your browser and navigate to:
- Default: http://localhost:3000
- Custom port: http://localhost:8080

The Web UI provides an intuitive interface for:
- Text input and TTS generation
- Voice parameter adjustment (age, gender, emotion, pitch, speed)
- Zero-shot voice cloning with reference audio upload
- Real-time audio playback and download

> **Model Source**: https://huggingface.co/cgisky/rwkv-tts/
> **Mirror Support**: Automatic fallback to https://hf-mirror.com for users in China

## Usage

### Web Interface (Recommended)

The easiest way to use RWKV TTS is through the embedded Web interface:

1. Start the server: `cargo run --release --bin rwkvtts_server`
2. Open http://localhost:3000 in your browser
3. Enter your text and adjust voice parameters
4. Click "Generate" to create speech
5. Play or download the generated audio

### Server Configuration

The RWKV TTS server supports various configuration options through command line parameters:

#### Command Line Parameters

**Server Configuration:**
- `--port <PORT>`: 服务器监听端口 (Server listening port, default: `3000`)

**Model Configuration:**
- `--model-path <PATH>`: 模型文件路径 (Model file path, default: `assets/model/rwkvtts-Int8_22.prefab`)
- `--vocab-path <PATH>`: 词汇表文件路径 (Vocabulary file path, default: `assets/model/tokenizer.json`)
- `--quant-layers <NUMBER>`: 指定量化层数 (Quantization layers, default: `24`)
- `--quant-type <TYPE>`: 指定量化类型 (Quantization type: none, int8, nf4, sf4, default: `int8`)
  - 推荐使用 `int8` 以获得最佳稳定性

**Performance Configuration:**
- `--batch-size <NUMBER>`: 批处理最大大小 (Maximum batch size, default: `10`)
- `--batch-timeout <MS>`: 批处理超时时间（毫秒） (Batch timeout in milliseconds, default: `20`)
- `--inference-timeout <MS>`: 推理超时时间（毫秒） (Inference timeout in milliseconds, default: `120000`)

#### Usage Examples

**Start with Default Settings:**
```bash
cargo run --release --bin rwkvtts_server
```

**Custom Port:**
```bash
cargo run --release --bin rwkvtts_server -- --port 8080
```

**Custom Model Path:**
```bash
cargo run --release --bin rwkvtts_server -- --model-path ./custom/model.prefab --vocab-path ./custom/tokenizer.json
```

**Performance Tuning:**
```bash
cargo run --release --bin rwkvtts_server -- --batch-size 20 --batch-timeout 50 --quant-type int8
```

**Production Deployment:**
```bash
cargo run --release --bin rwkvtts_server -- --port 80 --batch-size 50 --inference-timeout 60000
```

### API Usage

Once the server is running, you can use the TTS service through:

1. **Web Interface**: Navigate to `http://localhost:3000` (or your custom port)
2. **HTTP API**: Send POST requests to `http://localhost:3000/api/tts`
3. **Health Check**: GET `http://localhost:3000/api/health`
4. **Status**: GET `http://localhost:3000/api/status`

The Web UI provides intuitive controls for:
- Text input and voice parameter adjustment
- Zero-shot voice cloning with reference audio upload
- Real-time audio generation and playback
- Download generated audio files

## HTTP API 文档

### 1. TTS 语音合成 API
**路径**: `POST /api/tts`

**支持的请求格式**:
- JSON格式（application/json）
- Multipart表单格式（multipart/form-data，支持文件上传）

**JSON请求参数**:
```json
{
  "text": "要转换的文本",
  "temperature": 1.0,
  "top_p": 0.3,
  "seed": 42,
  "age": "youth-adult",
  "gender": "male",
  "emotion": "NEUTRAL",
  "pitch": "medium_pitch",
  "prompt_text": "可选的提示词"
}
```

**Multipart表单参数**:
- `text`: 要转换的文本（必需）
- `temperature`: 温度参数（可选，默认1.0）
- `top_p`: Top-p参数（可选，默认0.3）
- `seed`: 随机种子（可选）
- `age`: 年龄特征（可选，默认"youth-adult"）
- `gender`: 性别特征（可选，默认"male"）
- `emotion`: 情感特征（可选，默认"NEUTRAL"）
- `pitch`: 音调（可选，默认"medium_pitch"）
- `ref_audio`: 参考音频文件（可选，用于零样本语音克隆）

**参数说明**:
- `age`: "child", "youth-adult", "elderly"
- `gender`: "male", "female"
- `emotion`: "NEUTRAL", "HAPPY", "SAD", "ANGRY", "SURPRISED"
- `pitch`: "low_pitch", "medium_pitch", "high_pitch", "very_high_pitch"

**响应格式**:
```json
{
  "success": true,
  "message": "TTS生成成功",
  "audio_base64": "base64编码的WAV音频数据",
  "duration_ms": 1500,
  "rtf": 0.25
}
```

### 2. 健康检查 API
**路径**: `GET /api/health`

**响应格式**:
```json
{
  "status": "healthy"
}
```

### 3. 服务器状态 API
**路径**: `GET /api/status`

**响应格式**:
```json
{
  "status": "running",
  "version": "0.2.0",
  "uptime_seconds": 3600,
  "total_requests": 150
}
```

### 4. Web界面
**路径**: `GET /`

返回嵌入式Web UI界面

### 5. 静态资源
**路径**: `GET /static/<path>`

提供Web UI所需的静态资源文件

**使用示例**:

```bash
# JSON格式请求
curl -X POST http://localhost:3000/api/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是一个测试",
    "temperature": 1.0,
    "top_p": 0.3,
    "age": "youth-adult",
    "gender": "female",
    "emotion": "HAPPY"
  }'

# Multipart表单请求（带参考音频）
curl -X POST http://localhost:3000/api/tts \
  -F "text=你好，这是一个测试" \
  -F "temperature=1.0" \
  -F "gender=female" \
  -F "ref_audio=@reference.wav"

# 健康检查
curl http://localhost:3000/api/health

# 服务器状态
curl http://localhost:3000/api/status
```

## Requirements

- **Rust 1.78 or later** - Required for compilation
- **ONNX Runtime library (version 1.22)** - For neural network inference
  - Windows: Automatically configured by build script
  - Linux: Install via package manager or download prebuilt binaries
  - macOS: Install via Homebrew `brew install onnxruntime` or download prebuilt binaries
- **Internet connection** - For initial model download (models are cached locally)
- **Modern web browser** - For accessing the Web UI (Chrome, Firefox, Safari, Edge)

## Installation Details

### Automatic Setup

The build scripts (`build.sh` / `build.ps1`) handle everything automatically:
- Download and configure ONNX Runtime
- Download TTS models with mirror fallback
- Compile the single-file executable
- Embed the Web UI into the binary

### Manual ONNX Runtime Setup (if needed)

**Windows:**
```powershell
# The build script handles this automatically
.\build.ps1
```

**Linux/macOS:**
```bash
# Install ONNX Runtime
# Ubuntu/Debian: apt install libonnxruntime-dev
# macOS: brew install onnxruntime
# Or download from: https://github.com/microsoft/onnxruntime/releases

# Then build
sh build.sh
```

### Model Download

Models are automatically downloaded on first run:
- **Primary source**: https://huggingface.co/cgisky/rwkv-tts/
- **Mirror fallback**: https://hf-mirror.com (for users in China)
- **Local cache**: `./assets/model/` (reused on subsequent runs)

## Troubleshooting

### Build Issues

**Problem**: `LINK : fatal error LNK1181: 无法打开输入文件'onnxruntime.lib'`

**Solution**: Run the build script which automatically handles ONNX Runtime setup:
```bash
# Windows
.\build.ps1

# Linux/macOS  
sh build.sh
```

### Model Download Issues

**Problem**: Slow or failed model downloads

**Solution**: The system automatically tries mirror fallback:
1. Primary: https://huggingface.co/cgisky/rwkv-tts/
2. Fallback: https://hf-mirror.com (China mirror)

**Problem**: "Model not found" errors

**Solution**: Ensure internet connection and run the build script to download models automatically.

### Web Interface Issues

**Problem**: Cannot access web interface

**Solutions**:
1. Check if the server is running: `cargo run --release --bin rwkvtts_server`
2. Verify the port (default: 3000): http://localhost:3000
3. Try a different port: `cargo run --release --bin rwkvtts_server -- --port 8080`
4. Check firewall settings

### Performance Issues

**Problem**: Slow TTS generation

**Solutions**:
1. Ensure you're using `--release` flag for optimal performance
2. Close other resource-intensive applications
3. Use shorter text inputs for faster generation

### General Tips

- Always use `cargo run --release` for better performance
- Models are cached locally after first download
- Check system requirements (Rust 1.78+, modern browser)
- For detailed logs, check the console output when running the server

## License

MIT License