//! RWKV TTS CLI 工具
//! 基于RWKV模型的文本转语音命令行工具

use std::path::Path;

use clap::{Arg, Command};

use anyhow::Result;

// 引入本地模块
use rwkv_tts_rs::{TtsPipeline, TtsPipelineArgs};


/// 生成唯一的文件名
fn get_unique_filename(output_dir: &str, text: &str, extension: &str) -> String {
    let output_dir = Path::new(output_dir);
    std::fs::create_dir_all(output_dir).unwrap_or_default();

    let prefix = if text.len() >= 3 {
        text.chars().take(3).collect::<String>()
    } else {
        text.to_string()
    };

    let prefix = regex::Regex::new(r"[^\w]")
        .unwrap()
        .replace_all(&prefix, "");
    let base_name = prefix.to_string();

    let mut index = 0;
    loop {
        let filename = if index == 0 {
            format!("{}{}", base_name, extension)
        } else {
            format!("{}_{}{}", base_name, index, extension)
        };

        let filepath = output_dir.join(&filename);
        if !filepath.exists() {
            return filepath.to_string_lossy().to_string();
        }
        index += 1;
    }
}

/// 解析命令行参数
pub fn parse_args() -> TtsPipelineArgs {
    let matches = Command::new("RWKV TTS CLI")
        .version("1.0")
        .author("AI00 Team")
        .about("RWKV文本到语音转换命令行工具 - 集成ONNX模型")
        .arg(
            Arg::new("text")
                .short('t')
                .long("text")
                .value_name("TEXT")
                .help("要转换的文本")
                .required(true),
        )
        .arg(
            Arg::new("model")
                .short('m')
                .long("model")
                .value_name("PATH")
                .help("模型文件路径")
                .default_value("./assets/model"),
        )
        .arg(
            Arg::new("vocab")
                .short('v')
                .long("vocab")
                .value_name("PATH")
                .help("词表文件路径")
                .default_value("./assets/model/tokenizer.json"),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("PATH")
                .help("输出音频文件路径")
                .default_value("./output"),
        )
        .arg(
            Arg::new("temperature")
                .long("temperature")
                .value_name("FLOAT")
                .help("采样温度")
                .default_value("1.0"),
        )
        .arg(
            Arg::new("top_p")
                .long("top-p")
                .value_name("FLOAT")
                .help("Top-p采样参数")
                .default_value("0.90"),
        )
        .arg(
            Arg::new("top_k")
                .long("top-k")
                .value_name("INT")
                .help("Top-k采样参数")
                .default_value("0"),
        )
        .arg(
            Arg::new("max_tokens")
                .long("max-tokens")
                .value_name("INT")
                .help("最大生成token数")
                .default_value("8000"),
        )
        .arg(
            Arg::new("age")
                .long("age")
                .value_name("AGE")
                .help("说话人年龄")
                .default_value("youth-adult"),
        )
        .arg(
            Arg::new("gender")
                .long("gender")
                .value_name("GENDER")
                .help("说话人性别")
                .default_value("female"),
        )
        .arg(
            Arg::new("emotion")
                .long("emotion")
                .value_name("EMOTION")
                .help("情感")
                .default_value("NEUTRAL"),
        )
        .arg(
            Arg::new("pitch")
                .long("pitch")
                .value_name("FLOAT")
                .help("音调")
                .default_value("200.0"),
        )
        .arg(
            Arg::new("speed")
                .long("speed")
                .value_name("FLOAT")
                .help("语速")
                .default_value("4.2"),
        )
        .arg(
            Arg::new("seed")
                .long("seed")
                .value_name("INT")
                .help("随机种子（启用确定性采样）"),
        )
        .arg(
            Arg::new("validate")
                .long("validate")
                .value_name("VALIDATE")
                .help("使用ASR验证生成的音频是否正确")
                .action(clap::ArgAction::SetTrue),
        )
        // Zero-shot模式参数
        .arg(
            Arg::new("zero_shot")
                .long("zero-shot")
                .value_name("ZERO_SHOT")
                .help("启用Zero-shot模式")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("ref_audio")
                .long("ref-audio")
                .value_name("REF_AUDIO")
                .help("参考音频路径（Zero-shot模式）")
                .default_value(""),
        )
        .arg(
            Arg::new("prompt_text")
                .long("prompt-text")
                .value_name("PROMPT_TEXT")
                .help("提示文本（Zero-shot模式）")
                .default_value("希望你以后能够做的，比我还好呦！"),
        )
        .get_matches();

    TtsPipelineArgs {
        text: matches.get_one::<String>("text").unwrap().clone(),
        model_path: matches.get_one::<String>("model").unwrap().clone(),
        vocab_path: matches.get_one::<String>("vocab").unwrap().clone(),
        output_path: matches.get_one::<String>("output").unwrap().clone(),
        temperature: matches
            .get_one::<String>("temperature")
            .unwrap()
            .parse()
            .unwrap_or(1.0),
        top_p: matches
            .get_one::<String>("top_p")
            .unwrap()
            .parse()
            .unwrap_or(0.90),
        top_k: matches
            .get_one::<String>("top_k")
            .unwrap()
            .parse()
            .unwrap_or(0),
        max_tokens: matches
            .get_one::<String>("max_tokens")
            .unwrap()
            .parse()
            .unwrap_or(8000),
        age: matches.get_one::<String>("age").unwrap().clone(),
        gender: matches.get_one::<String>("gender").unwrap().clone(),
        emotion: matches.get_one::<String>("emotion").unwrap().clone(),
        pitch: matches
            .get_one::<String>("pitch")
            .unwrap()
            .parse()
            .unwrap_or(200.0),
        speed: matches
            .get_one::<String>("speed")
            .unwrap()
            .parse()
            .unwrap_or(4.2),
        zero_shot: matches.get_flag("zero_shot"),
        ref_audio_path: matches
            .get_one::<String>("ref_audio")
            .cloned()
            .unwrap_or_default(),
        prompt_text: matches
            .get_one::<String>("prompt_text")
            .cloned()
            .unwrap_or_default(),
        validate: matches.get_flag("validate"),
        seed: matches
            .get_one::<String>("seed")
            .and_then(|s| s.parse::<u64>().ok()),
    }
}

#[tokio::main]
pub async fn main() -> Result<()> {
    println!("🚀 RWKV TTS 命令行工具启动 - 集成ONNX模型");

    // 解析命令行参数
    let args = parse_args();

    // 检查文件是否存在
    if !Path::new(&args.vocab_path).exists() {
        eprintln!("❌ 错误: 词表文件不存在: {}", args.vocab_path);
        std::process::exit(1);
    }

    if !Path::new(&args.model_path).exists() {
        eprintln!("❌ 错误: 模型目录不存在: {}", args.model_path);
        std::process::exit(1);
    }

    // 检查ONNX模型文件
    let onnx_files = [
        "BiCodecTokenize.onnx",
        "BiCodecDetokenize.onnx",
        "wav2vec2-large-xlsr-53.onnx",
    ];
    for file in &onnx_files {
        let path = Path::new(&args.model_path).join(file);
        if !path.exists() {
            eprintln!("❌ 错误: ONNX模型文件不存在: {:?}", path);
            std::process::exit(1);
        }
    }

    // 如果是Zero-shot模式且提供了参考音频路径，检查参考音频文件是否存在
    if args.zero_shot
        && !args.ref_audio_path.is_empty()
        && !Path::new(&args.ref_audio_path).exists()
    {
        eprintln!("❌ 错误: 参考音频文件不存在: {}", args.ref_audio_path);
        std::process::exit(1);
    }

    println!("📋 参数配置:");
    println!("  文本: {}", args.text);
    println!("  模型路径: {}", args.model_path);
    println!("  词表路径: {}", args.vocab_path);
    println!("  输出路径: {}", args.output_path);
    println!("  温度: {}", args.temperature);
    println!("  Top-p: {}", args.top_p);
    println!("  Top-k: {}", args.top_k);
    println!("  最大token数: {}", args.max_tokens);
    println!("  年龄: {}", args.age);
    println!("  性别: {}", args.gender);
    println!("  情感: {}", args.emotion);
    println!("  音调: {}", args.pitch);
    println!("  语速: {}", args.speed);
    match args.seed {
        Some(s) => println!("  采样Seed: {} (启用确定性采样)", s),
        None => println!("  采样Seed: 随机"),
    }
    println!("  Zero-shot模式: {}", args.zero_shot);
    if args.zero_shot && !args.ref_audio_path.is_empty() {
        println!("  参考音频路径: {}", args.ref_audio_path);
        println!("  提示文本: {}", args.prompt_text);
    }

    // 特殊命令：运行验证测试
    if args.text == "RUN_VALIDATION_TEST" {
        return run_tts_validation_test().await;
    }

    // 创建TTS流水线
    let mut pipeline = TtsPipeline::new(&args).await?;

    // 生成语音
    let audio_samples = pipeline.generate_speech(&args).await?;

    // 生成输出文件名
    let output_filename = get_unique_filename(&args.output_path, &args.text, ".wav");

    // 保存音频文件 - 使用正确的采样率16000Hz（与Python版本一致）
    pipeline.save_audio(&audio_samples, &output_filename, 16000)?;

    println!("✅ TTS生成完成！音频已保存到: {}", output_filename);

    // 如果启用了验证功能，则使用ASR验证生成的音频
    if args.validate {
        println!("🔍 开始验证生成的音频...");
        println!("🔄 ASR验证功能暂未实现");
    }

    Ok(())
}

/// TTS验证测试函数
async fn run_tts_validation_test() -> Result<()> {
    println!("🧪 开始TTS验证测试");

    // 测试用例
    let test_cases = vec![
        ("A", "单字符英文测试"),
        ("好", "单字符中文测试"),
        ("Hello", "简单英文测试"),
        ("你好", "简单中文测试"),
        ("Hello World", "多词英文测试"),
        ("你好世界", "多词中文测试"),
        (
            "The quick brown fox jumps over the lazy dog",
            "长英文句子测试",
        ),
        ("今天天气真不错，适合出去散步", "长中文句子测试"),
    ];

    let mut passed_tests = 0;
    let total_tests = test_cases.len();

    for (text, description) in test_cases {
        println!("\n🔍 测试用例: {} - '{}'", description, text);

        // 创建默认参数
        let args = TtsPipelineArgs {
            text: text.to_string(),
            model_path: "./assets/model".to_string(),
            vocab_path: "./assets/model/tokenizer.json".to_string(),
            output_path: "./output".to_string(),
            temperature: 1.0,
            top_p: 0.90,
            top_k: 50,
            max_tokens: 3000,
            age: "youth-adult".to_string(),
            gender: "female".to_string(),
            emotion: "NEUTRAL".to_string(),
            pitch: 200.0,
            speed: 4.2,
            zero_shot: false,
            ref_audio_path: String::new(),
            prompt_text: String::new(),
            validate: false,
            seed: None,
        };

        // 尝试创建TTS流水线
        match TtsPipeline::new(&args).await {
            Ok(mut pipeline) => {
                // 尝试生成语音
                match pipeline.generate_speech(&args).await {
                    Ok(audio_samples) => {
                        println!(
                            "  ✅ {} 测试通过，生成了 {} 个音频样本",
                            description,
                            audio_samples.len()
                        );

                        // 验证音频数据
                        if audio_samples.is_empty() {
                            println!("  ❌ 音频样本为空");
                            continue;
                        }

                        // 检查是否有NaN或无穷大值
                        let nan_count = audio_samples.iter().filter(|&&x| x.is_nan()).count();
                        let inf_count = audio_samples.iter().filter(|&&x| x.is_infinite()).count();
                        if nan_count > 0 || inf_count > 0 {
                            println!(
                                "  ❌ 音频数据中包含 {} 个 NaN 值和 {} 个无穷大值",
                                nan_count, inf_count
                            );
                            continue;
                        }

                        // 生成输出文件名
                        let output_filename = get_unique_filename(&args.output_path, text, ".wav");

                        // 保存音频文件
                        match pipeline.save_audio(&audio_samples, &output_filename, 16000) {
                            Ok(_) => {
                                println!("  💾 音频已保存到: {}", output_filename);
                                passed_tests += 1;
                            }
                            Err(e) => {
                                println!("  ❌ 保存音频失败: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        println!("  ❌ 生成语音失败: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("  ❌ 创建流水线失败: {}", e);
            }
        }
    }

    println!(
        "\n✅ 测试完成: 通过 {}/{} 个测试用例",
        passed_tests, total_tests
    );

    Ok(())
}
