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
```bash
cargo run --bin tts_cli -- --text "Hello, world!" --output output.wav
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
- ONNX Runtime library for neural network inference
  - Windows: Download from [Microsoft ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases)
  - Linux: Install via package manager or download prebuilt binaries
  - macOS: Install via Homebrew `brew install onnxruntime` or download prebuilt binaries
- Audio processing libraries

## License

MIT License