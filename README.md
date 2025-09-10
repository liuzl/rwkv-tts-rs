# RWKV TTS Rust

RWKV-based Text-to-Speech implementation in Rust.

## Features

- High-performance TTS generation using RWKV models
- Command-line interface for batch processing
- Interactive CLI for real-time TTS
- Support for multiple languages and voice characteristics

## Usage

### CLI Tool
```bash
cargo run --bin tts_cli -- --text "Hello, world!" --output output.wav
```

### Interactive CLI
```bash
cargo run --bin interactive_tts_cli
```

## Building

```bash
cargo build --release
```

## License

MIT License