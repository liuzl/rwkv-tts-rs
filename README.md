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
# Start the web server (default port: 8080)
cargo run --release --bin rwkvtts_server

# Or specify a custom port
cargo run --release --bin rwkvtts_server -- --port 3000
```

### 3. Access the Web Interface

Open your browser and navigate to:
- Default: http://localhost:8080
- Custom port: http://localhost:3000

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
2. Open http://localhost:8080 in your browser
3. Enter your text and adjust voice parameters
4. Click "Generate" to create speech
5. Play or download the generated audio

### Command Line Interface

For batch processing or automation, you can use the CLI:

#### Basic Usage
```bash
cargo run --release --bin rwkvtts_server -- --text "Hello, world!" --output output.wav
```

#### Command Line Parameters

**Required Parameters:**
- `-t, --text <TEXT>`: 要转换的文本 (Text to convert)

**Optional Parameters:**

**Model Configuration:**
- `-m, --model <PATH>`: 模型文件路径 (Model file path, default: `./assets/model`)
- `-v, --vocab <PATH>`: 词表文件路径 (Vocabulary file path, default: `./assets/model/tokenizer.json`)
- `-o, --output <PATH>`: 输出音频文件路径 (Output audio file path, default: `./output`)

**Generation Parameters:**
- `--temperature <FLOAT>`: 采样温度 (Sampling temperature, default: `1.0`)
- `--top-p <FLOAT>`: Top-p采样参数 (Top-p sampling parameter, default: `0.95`)
- `--top-k <INT>`: Top-k采样参数 (Top-k sampling parameter, default: `0`)
- `--max-tokens <INT>`: 最大生成token数 (Maximum tokens to generate, default: `8000`)

**Voice Characteristics:**
- `--age <AGE>`: 说话人年龄 (Speaker age)
  - 可选值: `child`, `teenager`, `youth-adult`, `middle-aged`, `elderly`
  - 数值区间: 
    - `child`: 0-12岁
    - `teenager`: 13-19岁
    - `youth-adult`: 20-39岁 (默认)
    - `middle-aged`: 40-64岁
    - `elderly`: 65岁以上
- `--gender <GENDER>`: 说话人性别 (Speaker gender)
  - 可选值: `female` (默认), `male`
- `--emotion <EMOTION>`: 情感 (Emotion)
  - 可选值: `NEUTRAL` (默认), `ANGRY`, `DISGUSTED`, `FEARFUL`, `HAPPY`, `SAD`, `SURPRISED`, `ANNOYED`, `TIRED`, `LAUGHING`, `TERRIFIED`, `SHOUTING`, `WHISPERING`, `UNFRIENDLY`, `ENUNCIATED`, `SINGING`, `QUESTIONING`, `CONFUSED`, `SERIOUS`, `SMILING`, `EXCITED`, `FRIENDLY`, `HUMOROUS`, `CONTEMPT`, `UNKNOWN`
- `--pitch <FLOAT>`: 音调 (Pitch)
  - 数值范围: 建议80-400Hz
  - 系统会根据性别和年龄自动分类为:
    - `low_pitch` (低音调)
    - `medium_pitch` (中音调) 
    - `high_pitch` (高音调)
    - `very_high_pitch` (极高音调)
  - 分类区间示例 (女性青年):
    - low_pitch: <191Hz
    - medium_pitch: 191-211Hz
    - high_pitch: 211-232Hz
    - very_high_pitch: >232Hz
  - 分类区间示例 (男性青年):
    - low_pitch: <115Hz
    - medium_pitch: 115-131Hz
    - high_pitch: 131-153Hz
    - very_high_pitch: >153Hz
  - 默认值: `200.0`
- `--speed <FLOAT>`: 语速 (Speech speed)
  - 数值范围: 1.0-10.0
  - 分类区间:
    - `very_slow`: ≤3.5
    - `slow`: 3.5-4.0
    - `medium`: 4.0-4.5
    - `fast`: 4.5-5.0
    - `very_fast`: >5.0
  - 默认值: `4.2`

**Zero-shot Voice Cloning:**
- `--zero-shot`: 启用Zero-shot模式 (Enable zero-shot mode)
- `--ref-audio <PATH>`: 参考音频路径 (Reference audio path for zero-shot mode)
- `--prompt-text <TEXT>`: 提示文本 (Prompt text for zero-shot mode, default: `希望你以后能够做的，比我还好呦！`)

**Validation:**
- `--validate`: 使用ASR验证生成的音频是否正确 (Use ASR to validate generated audio)

#### Examples

**Basic TTS:**
```bash
cargo run --release --bin rwkvtts_server -- --text "你好，世界！" --output ./output
```

**Custom Voice Settings:**
```bash
cargo run --release --bin rwkvtts_server -- --text "Hello, world!" --gender male --age youth-adult --emotion happy --speed 3.5
```

**Zero-shot Voice Cloning:**
```bash
cargo run --release --bin rwkvtts_server -- --text "Clone this voice" --zero-shot --ref-audio ./reference.wav --prompt-text "Sample text"
```

**Start Web Server:**
```bash
# Default port (8080)
cargo run --release --bin rwkvtts_server

# Custom port
cargo run --release --bin rwkvtts_server -- --port 3000
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
2. Verify the port (default: 8080): http://localhost:8080
3. Try a different port: `cargo run --release --bin rwkvtts_server -- --port 3000`
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