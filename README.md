# RWKV TTS Rust

RWKV-based Text-to-Speech implementation in Rust.

**Based on**: This project is a Rust implementation inspired by the original Python project [yueyulin/respark](https://huggingface.co/yueyulin/respark), which is a TTS system with RWKV-7 LM modeling audio tokens.

## Features

- High-performance TTS generation using RWKV models
- Command-line interface for batch processing
- Interactive CLI for real-time TTS
- Support for multiple languages and voice characteristics
- Zero-shot voice cloning with reference audio
- Customizable voice properties (pitch, speed, energy)

## Model Download

**Important**: Before using this TTS system, you need to download the required RWKV TTS models.

Download the models from: **https://huggingface.co/cgisky/rwkv-tts/**

Place the downloaded model files in the `assets/model/` directory. See `assets/model/download_instructions.txt` for detailed instructions.

## Usage

### CLI Tool

#### Basic Usage
```bash
cargo run --bin tts_cli -- --text "Hello, world!" --output output.wav
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
- `--age <AGE>`: 说话人年龄 (Speaker age, default: `youth-adult`)
- `--gender <GENDER>`: 说话人性别 (Speaker gender, default: `female`)
- `--emotion <EMOTION>`: 情感 (Emotion, default: `NEUTRAL`)
- `--pitch <FLOAT>`: 音调 (Pitch, default: `200.0`)
- `--speed <FLOAT>`: 语速 (Speech speed, default: `4.2`)

**Zero-shot Voice Cloning:**
- `--zero-shot`: 启用Zero-shot模式 (Enable zero-shot mode)
- `--ref-audio <PATH>`: 参考音频路径 (Reference audio path for zero-shot mode)
- `--prompt-text <TEXT>`: 提示文本 (Prompt text for zero-shot mode, default: `希望你以后能够做的，比我还好呦！`)

**Validation:**
- `--validate`: 使用ASR验证生成的音频是否正确 (Use ASR to validate generated audio)

#### Examples

**Basic TTS:**
```bash
cargo run --bin tts_cli -- --text "你好，世界！" --output ./output
```

**Custom Voice Settings:**
```bash
cargo run --bin tts_cli -- --text "Hello, world!" --gender male --age adult --emotion happy --speed 3.5
```

**Zero-shot Voice Cloning:**
```bash
cargo run --bin tts_cli -- --text "Clone this voice" --zero-shot --ref-audio ./reference.wav --prompt-text "Sample text"
```

**With Validation:**
```bash
cargo run --bin tts_cli -- --text "Validate this output" --validate
```

### Interactive CLI
```bash
cargo run --bin interactive_tts_cli
```

## Installation

1. Clone this repository
2. Download models from https://huggingface.co/cgisky/rwkv-tts/
3. Place model files in `assets/model/` directory
4. Build the project:

```bash
cargo build --release
```

## Requirements

- Rust 1.70 or later
- ONNX Runtime library (version 1.22) for neural network inference
  - Windows: Download from [Microsoft ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases)
  - Linux: Install via package manager or download prebuilt binaries
  - macOS: Install via Homebrew `brew install onnxruntime` or download prebuilt binaries
- Audio processing libraries

## ONNX Runtime Setup

### Windows

This project includes ONNX Runtime 1.22.1 for Windows. To configure the environment:

**Option 1: Using PowerShell (Recommended)**
```powershell
.\setup_onnx.ps1
cargo build --release
```

**Option 2: Using Command Prompt**
```cmd
setup_onnx.bat
cargo build --release
```

**Option 3: Manual Setup**
The build script (`build.rs`) automatically configures the ONNX Runtime paths during compilation.

### Linux/macOS

Install ONNX Runtime through your package manager or download prebuilt binaries from the official releases.

## License

MIT License