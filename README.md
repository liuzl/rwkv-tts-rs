# RWKV TTS Rust

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.78+-orange.svg)](https://www.rust-lang.org)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co/cgisky/rwkv-tts/tree/main)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/cgisky1980/rwkv-tts-rs/releases)

**High-performance Text-to-Speech with RWKV Language Models** | **基于RWKV语言模型的高性能语音合成**

[English](#rwkv-tts-rust) | [中文](#rwkv-tts-rust-1)

</div>

## 🌟 Project Showcase | 项目展示

RWKV TTS Rust is a cutting-edge Text-to-Speech implementation built with Rust, featuring:
- 🚀 **Single-file deployment** with embedded Web UI
- ⚡ **High-performance** TTS generation using RWKV models
- 🎯 **Zero-shot voice cloning** with reference audio
- 🎛️ **Customizable voice properties** (pitch, speed, emotion, age, gender)
- 🌍 **Multi-language support** with automatic model downloading
- 🔄 **Mirror support** for faster downloads in China

RWKV TTS Rust 是一个使用 Rust 构建的前沿语音合成实现，具有：
- 🚀 **单文件部署**，内嵌 Web UI 界面
- ⚡ **高性能** TTS 生成，基于 RWKV 模型
- 🎯 **零样本语音克隆**，支持参考音频
- 🎛️ **可定制语音属性**（音调、语速、情感、年龄、性别）
- 🌍 **多语言支持**，自动下载模型
- 🔄 **镜像支持**，为中国用户提供更快下载

## 🚀 Quick Start | 快速开始

### 1. Build the Project | 构建项目

```bash
# Linux/macOS
sh build.sh

# Windows
.\build.ps1
```

The build script will automatically:
- Download required models from Hugging Face
- Support mirror fallback for users in China
- Compile the single-file executable with embedded Web UI

构建脚本将自动：
- 从 Hugging Face 下载所需模型
- 为中国用户提供镜像回退支持
- 编译包含内嵌 Web UI 的单文件可执行程序

### 2. Run the Web Server | 运行 Web 服务器

```bash
# Start the web server (default port: 3000)
cargo run --release --bin rwkvtts_server

# Or specify a custom port
cargo run --release --bin rwkvtts_server -- --port 8080
```

### 3. Access the Web Interface | 访问 Web 界面

Open your browser and navigate to:
- Default: http://localhost:3000
- Custom port: http://localhost:8080

在浏览器中打开以下地址：
- 默认: http://localhost:3000
- 自定义端口: http://localhost:8080

The Web UI provides an intuitive interface for:
- Text input and TTS generation
- Voice parameter adjustment (age, gender, emotion, pitch, speed)
- Zero-shot voice cloning with reference audio upload
- Real-time audio playback and download

Web UI 提供直观的界面功能：
- 文本输入和 TTS 生成
- 语音参数调整（年龄、性别、情感、音调、语速）
- 零样本语音克隆，支持参考音频上传
- 实时音频播放和下载

## 🎯 Core Features | 核心功能

### 🔊 High-Performance TTS | 高性能语音合成
- Utilizes RWKV-7 language models for superior audio quality
- Optimized Rust implementation for maximum performance
- Dynamic batching for efficient concurrent processing

- 使用 RWKV-7 语言模型实现卓越音质
- 优化的 Rust 实现，性能最大化
- 动态批处理，高效并发处理

### 🎭 Voice Customization | 语音定制
Customize voice characteristics with multiple parameters:
- **Age**: child, youth-adult, elderly
- **Gender**: male, female
- **Emotion**: NEUTRAL, HAPPY, SAD, ANGRY, SURPRISED
- **Pitch**: low, medium, high, very high
- **Speed**: adjustable from very slow to very fast

多种参数定制语音特征：
- **年龄**: 儿童、青年、老年
- **性别**: 男性、女性
- **情感**: 中性、快乐、悲伤、愤怒、惊讶
- **音调**: 低、中、高、很高
- **语速**: 从很慢到很快可调

### 🎯 Zero-Shot Voice Cloning | 零样本语音克隆
Clone voices using reference audio without training:
- Upload reference audio files (WAV/MP3)
- Extract voice characteristics automatically
- Generate speech in the cloned voice instantly

使用参考音频克隆语音，无需训练：
- 上传参考音频文件（WAV/MP3）
- 自动提取语音特征
- 即时生成克隆语音

### 🌐 Multi-Language Support | 多语言支持
- Automatic model downloading with mirror support
- Cross-platform compatibility (Windows, Linux, macOS)
- Web-based interface for easy access

- 自动模型下载，支持镜像
- 跨平台兼容性（Windows、Linux、macOS）
- 基于 Web 的界面，易于访问

## 📖 Usage | 使用说明

### Web Interface (Recommended) | Web 界面（推荐）

The easiest way to use RWKV TTS is through the embedded Web interface:

使用 RWKV TTS 最简单的方法是通过内嵌的 Web 界面：

1. Start the server: `cargo run --release --bin rwkvtts_server`
2. Open http://localhost:3000 in your browser
3. Enter your text and adjust voice parameters
4. Click "Generate" to create speech
5. Play or download the generated audio

1. 启动服务器: `cargo run --release --bin rwkvtts_server`
2. 在浏览器中打开 http://localhost:3000
3. 输入文本并调整语音参数
4. 点击"生成"创建语音
5. 播放或下载生成的音频

### Server Configuration | 服务器配置

The RWKV TTS server supports various configuration options:

RWKV TTS 服务器支持多种配置选项：

#### Command Line Parameters | 命令行参数

**Server Configuration:**
- `--port <PORT>`: Server listening port (default: `3000`)

**Model Configuration:**
- `--model-path <PATH>`: Model file path (default: `assets/model/rwkvtts-Int8_22.prefab`)
- `--vocab-path <PATH>`: Vocabulary file path (default: `assets/model/tokenizer.json`)
- `--quant-layers <NUMBER>`: Quantization layers (default: `24`)
- `--quant-type <TYPE>`: Quantization type: none, int8, nf4, sf4 (default: `int8`)

**Performance Configuration:**
- `--batch-size <NUMBER>`: Maximum batch size (default: `10`)
- `--batch-timeout <MS>`: Batch timeout in milliseconds (default: `20`)
- `--inference-timeout <MS>`: Inference timeout in milliseconds (default: `120000`)

#### Usage Examples | 使用示例

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

### API Usage | API 使用

Once the server is running, you can use the TTS service through:

服务器运行后，可通过以下方式使用 TTS 服务：

1. **Web Interface**: Navigate to `http://localhost:3000`
2. **HTTP API**: Send POST requests to `http://localhost:3000/api/tts`
3. **Health Check**: GET `http://localhost:3000/api/health`
4. **Status**: GET `http://localhost:3000/api/status`

## 🌐 HTTP API Documentation | HTTP API 文档

### 1. TTS Speech Synthesis API | TTS 语音合成 API
**Path**: `POST /api/tts`

**Supported Request Formats**:
- JSON format (application/json)
- Multipart form format (multipart/form-data, supports file upload)

**JSON Request Parameters**:
```json
{
  "text": "Text to convert",
  "temperature": 1.0,
  "top_p": 0.3,
  "seed": 42,
  "age": "youth-adult",
  "gender": "male",
  "emotion": "NEUTRAL",
  "pitch": "medium_pitch",
  "prompt_text": "Optional prompt text"
}
```

**Multipart Form Parameters**:
- `text`: Text to convert (required)
- `temperature`: Temperature parameter (optional, default 1.0)
- `top_p`: Top-p parameter (optional, default 0.3)
- `seed`: Random seed (optional)
- `age`: Age characteristic (optional, default "youth-adult")
- `gender`: Gender characteristic (optional, default "male")
- `emotion`: Emotion characteristic (optional, default "NEUTRAL")
- `pitch`: Pitch (optional, default "medium_pitch")
- `ref_audio`: Reference audio file (optional, for zero-shot voice cloning)

**Response Format**:
```json
{
  "success": true,
  "message": "TTS generation successful",
  "audio_base64": "base64 encoded WAV audio data",
  "duration_ms": 1500,
  "rtf": 0.25
}
```

### 2. Health Check API | 健康检查 API
**Path**: `GET /api/health`

**Response Format**:
```json
{
  "status": "healthy"
}
```

### 3. Server Status API | 服务器状态 API
**Path**: `GET /api/status`

**Response Format**:
```json
{
  "status": "running",
  "version": "0.2.0",
  "uptime_seconds": 3600,
  "total_requests": 150
}
```

## 🛠️ Requirements | 环境要求

- **Rust 1.78 or later** - Required for compilation
- **ONNX Runtime library (version 1.22)** - For neural network inference
- **Internet connection** - For initial model download (models are cached locally)
- **Modern web browser** - For accessing the Web UI (Chrome, Firefox, Safari, Edge)

- **Rust 1.78 或更高版本** - 编译所需
- **ONNX Runtime 库 (版本 1.22)** - 神经网络推理
- **网络连接** - 初始模型下载（模型本地缓存）
- **现代浏览器** - 访问 Web UI（Chrome、Firefox、Safari、Edge）

## 📦 Installation Details | 安装详情

### Automatic Setup | 自动设置

The build scripts (`build.sh` / `build.ps1`) handle everything automatically:
- Download and configure ONNX Runtime
- Download TTS models with mirror fallback
- Compile the single-file executable
- Embed the Web UI into the binary

构建脚本 (`build.sh` / `build.ps1`) 自动处理所有步骤：
- 下载并配置 ONNX Runtime
- 下载 TTS 模型，支持镜像回退
- 编译单文件可执行程序
- 将 Web UI 内嵌到二进制文件中

### Manual ONNX Runtime Setup (if needed) | 手动设置 ONNX Runtime（如需要）

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

### Model Download | 模型下载

Models are automatically downloaded on first run:
- **Primary source**: https://huggingface.co/cgisky/rwkv-tts/
- **Mirror fallback**: https://hf-mirror.com (for users in China)
- **Local cache**: `./assets/model/` (reused on subsequent runs)

首次运行时自动下载模型：
- **主要来源**: https://huggingface.co/cgisky/rwkv-tts/
- **镜像回退**: https://hf-mirror.com（为中国用户）
- **本地缓存**: `./assets/model/`（后续运行重复使用）

## 🤝 Developer Resources | 开发者资源

### Building from Source | 从源码构建

```bash
# Clone the repository
git clone https://github.com/cgisky1980/rwkv-tts-rs.git
cd rwkv-tts-rs

# Build the project
# Linux/macOS
sh build.sh

# Windows
.\build.ps1
```

### Project Structure | 项目结构

```
rwkv-tts-rs/
├── assets/           # Model files and resources
├── bin/              # Server binary
├── src/              # Rust source code
├── static/           # Embedded Web UI
├── python/           # Python utilities and CLI
├── Cargo.toml        # Rust package configuration
├── build.sh          # Build script for Linux/macOS
├── build.ps1         # Build script for Windows
└── README.md         # This file
```

### API Documentation | API 文档

For detailed API documentation, please refer to the source code and inline comments.

详细 API 文档请参考源代码和内联注释。

## 🆘 Troubleshooting | 故障排除

### Build Issues | 构建问题

**Problem**: `LINK : fatal error LNK1181: cannot open input file 'onnxruntime.lib'`

**Solution**: Run the build script which automatically handles ONNX Runtime setup:
```bash
# Windows
.\build.ps1

# Linux/macOS  
sh build.sh
```

### Model Download Issues | 模型下载问题

**Problem**: Slow or failed model downloads

**Solution**: The system automatically tries mirror fallback:
1. Primary: https://huggingface.co/cgisky/rwkv-tts/
2. Fallback: https://hf-mirror.com (China mirror)

### Web Interface Issues | Web 界面问题

**Problem**: Cannot access web interface

**Solutions**:
1. Check if the server is running: `cargo run --release --bin rwkvtts_server`
2. Verify the port (default: 3000): http://localhost:3000
3. Try a different port: `cargo run --release --bin rwkvtts_server -- --port 8080`
4. Check firewall settings

### Performance Issues | 性能问题

**Problem**: Slow TTS generation

**Solutions**:
1. Ensure you're using `--release` flag for optimal performance
2. Close other resource-intensive applications
3. Use shorter text inputs for faster generation

## 📄 License | 许可证

MIT License

---

<div align="center">
  <p> Built with ❤️ using Rust and RWKV </p>
  <p> 使用 Rust 和 RWKV 构建 </p>
</div>