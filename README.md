# RWKV TTS Rust

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.78+-orange.svg)](https://www.rust-lang.org)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co/cgisky/rwkv-tts/tree/main)

RWKV-based Text-to-Speech implementation in Rust.

**Based on**: This project is a Rust implementation inspired by the original Python project [yueyulin/respark](https://huggingface.co/yueyulin/respark), which is a TTS system with RWKV-7 LM modeling audio tokens.

## Features

- High-performance TTS generation using RWKV models
- Command-line interface for batch processing
- Interactive CLI for real-time TTS
- Support for multiple languages and voice characteristics
- Zero-shot voice cloning with reference audio
- Customizable voice properties (pitch, speed, energy)


## Installation

```bash
# linux/macOS
sh build.sh

# windows
.\build.ps1
```

> Model Source: https://huggingface.co/cgisky/rwkv-tts/

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


### Interactive CLI
```bash
cargo run --bin interactive_tts_cli
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

## Troubleshooting

### LINK : fatal error LNK1181: 无法打开输入文件'onnxruntime.lib'

这个错误通常发生在 Windows 平台上编译时，表示链接器无法找到 ONNX Runtime 库文件。

#### 问题原因分析

1. **库文件缺失**: ONNX Runtime 库文件未正确下载或放置在预期位置
2. **路径配置错误**: `build.rs` 中配置的库路径与实际文件位置不匹配
3. **环境变量未设置**: 缺少必要的环境变量指向 ONNX Runtime 库

#### 解决方案

**方案 1: 设置环境变量 (推荐)**

如果你已经下载了 ONNX Runtime 库，可以通过设置环境变量来指定库路径：

```powershell
# PowerShell
$env:ORT_LIB_LOCATION = "C:\path\to\your\onnxruntime\lib"
cargo build --release
```

```cmd
# Command Prompt
set ORT_LIB_LOCATION=C:\path\to\your\onnxruntime\lib
cargo build --release
```

**方案 2: 手动下载并放置库文件**

1. 从 [Microsoft ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases) 下载适合你平台的版本
2. 解压到项目根目录下的以下路径之一：
   - `./第三方库源码/onnxruntime-win-x64-1.22.1/` (Windows x64)
   - `./第三方库源码/onnxruntime-win-arm64-1.22.1/` (Windows ARM64)
   - `./onnxruntime-win-x64-1.22.1/` (Windows x64)
   - `./onnxruntime-win-arm64-1.22.1/` (Windows ARM64)

**方案 3: 使用项目提供的设置脚本**

```powershell
# PowerShell
.\setup_onnx.ps1
cargo build --release
```

```cmd
# Command Prompt
setup_onnx.bat
cargo build --release
```

#### 不同平台的具体操作步骤

**Windows x64:**
1. 下载 `onnxruntime-win-x64-1.22.1.zip`
2. 解压到 `./onnxruntime-win-x64-1.22.1/`
3. 确保 `lib/onnxruntime.lib` 文件存在
4. 运行 `cargo build --release`

**Windows ARM64:**
1. 下载 `onnxruntime-win-arm64-1.22.1.zip`
2. 解压到 `./onnxruntime-win-arm64-1.22.1/`
3. 确保 `lib/onnxruntime.lib` 文件存在
4. 运行 `cargo build --release`

**验证安装:**
```powershell
# 检查库文件是否存在
Test-Path "./onnxruntime-win-x64-1.22.1/lib/onnxruntime.lib"
# 应该返回 True
```

如果问题仍然存在，请检查：
1. 下载的 ONNX Runtime 版本是否与项目要求匹配 (1.22.1)
2. 文件路径是否正确
3. 是否有足够的磁盘空间
4. 防病毒软件是否阻止了文件访问

## License

MIT License