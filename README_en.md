# RWKV TTS Rust

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.78+-orange.svg)](https://www.rust-lang.org)

**High-performance Text-to-Speech with RWKV Language Models**

</div>

## Introduction

RWKV TTS Rust is a high-performance text-to-speech implementation built with Rust, featuring RWKV language models. Key features include:

- 🚀 Single-file deployment with embedded Web UI
- ⚡ High-performance TTS generation
- 🎯 Voice cloning with reference audio support
- 🎛️ Customizable voice properties (pitch, speed, emotion, age, gender)
- 🌍 Multi-language support with automatic model downloading

![Web UI Screenshot](1.png)

## Compilation

### Automatic Compilation

Use the provided build scripts for automatic compilation:

```bash
# Linux/macOS
sh build.sh

# Windows (using PowerShell)
.\build.ps1
```

The build script will automatically:
- Download required models
- Configure ONNX Runtime
- Compile a single executable file with embedded Web UI

### Manual Compilation

```bash
# Build release version
cargo build --release

# Run the server
cargo run --release --bin rwkvtts_server
```

## Running

After compilation, you can run the server with:

```bash
# Using default port 3000
cargo run --release --bin rwkvtts_server

# Or specify a custom port
cargo run --release --bin rwkvtts_server -- --port 8080
```

Then open http://localhost:3000 in your browser to access the Web interface.

## Download from Releases

For users who prefer not to compile the project, you can download pre-compiled executables from the [Releases page](https://github.com/cgisky1980/rwkv-tts-rs/releases):

1. Visit the [Releases page](https://github.com/cgisky1980/rwkv-tts-rs/releases)
2. Download the pre-compiled version for your operating system
3. Extract the downloaded file
4. Run the executable:
   - Windows: Double-click `rwkvtts_server.exe` or run in command line
   - Linux/macOS: Run `./rwkvtts_server` in terminal

The required model files will be automatically downloaded on first run.

## License

MIT License

---

<div align="center">
  <p><a href="README.md">中文版本</a> | English Version</p>
</div>