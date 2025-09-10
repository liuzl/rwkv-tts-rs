//! 交互式TTS CLI工具
//! 提供类似Python版本的交互式界面

use std::io::{self, Write};
use std::path::Path;
use std::process;

// 引入TTS生成器和相关组件
use rwkv_tts_rs::tts_generator::Args;
use rwkv_tts_rs::tts_generator::TTSGenerator;

/// 命令行参数结构
#[derive(Debug)]
struct CliArgs {
    model_path: String,
}

/// TTS参数
#[derive(Debug, Clone)]
struct TtsParams {
    text: String,
    age: String,
    gender: String,
    emotion: String,
    pitch: String,
    speed: String,
    output_dir: String,
    zero_shot: bool,
    ref_audio_path: String,
    prompt_text: String,
}

/// 解析命令行参数
fn parse_args() -> CliArgs {
    let matches = clap::Command::new("交互式RWKV TTS CLI")
        .version("1.0")
        .author("AI00 Team")
        .about("RWKV文本到语音转换交互式命令行工具")
        .arg(
            clap::Arg::new("model")
                .short('m')
                .long("model")
                .value_name("PATH")
                .help("模型文件路径")
                .default_value("./assets/model"),
        )
        .get_matches();

    CliArgs {
        model_path: matches.get_one::<String>("model").unwrap().clone(),
    }
}

/// 显示欢迎信息
fn show_welcome() {
    println!("🚀 欢迎使用 RWKV TTS 交互式音频生成工具!");
    println!("💡 使用方向键选择，回车确认，Ctrl+C退出");
}

/// 获取用户输入
fn get_user_input(
    prompt: &str,
    default: &str,
) -> std::result::Result<String, Box<dyn std::error::Error>> {
    print!("{} [默认: {}]: ", prompt, default);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim();

    if input.is_empty() {
        Ok(default.to_string())
    } else {
        Ok(input.to_string())
    }
}

/// 从选项中选择
fn select_from_options(
    prompt: &str,
    options: &[&str],
    default_index: usize,
) -> std::result::Result<String, Box<dyn std::error::Error>> {
    println!("{}", prompt);
    for (i, option) in options.iter().enumerate() {
        if i == default_index {
            println!("  {}. {} (默认)", i + 1, option);
        } else {
            println!("  {}. {}", i + 1, option);
        }
    }

    loop {
        print!("请选择 (1-{}): ", options.len());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            return Ok(options[default_index].to_string());
        }

        if let Ok(index) = input.parse::<usize>() {
            if index > 0 && index <= options.len() {
                return Ok(options[index - 1].to_string());
            }
        }

        println!("无效选择，请重新输入");
    }
}

/// 确认操作
fn confirm_action(prompt: &str) -> std::result::Result<bool, Box<dyn std::error::Error>> {
    loop {
        print!("{} [y/N]: ", prompt);
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim().to_lowercase();

        match input.as_str() {
            "y" | "yes" => return Ok(true),
            "n" | "no" | "" => return Ok(false),
            _ => println!("请输入 y 或 n"),
        }
    }
}

/// 交互式参数选择
async fn interactive_parameter_selection(
    model_path: &str,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    loop {
        println!("\n{}", "=".repeat(60));
        println!("🎵 RWKV TTS 参数配置");
        println!("{}", "=".repeat(60));

        // 选择生成模式
        let generation_mode = select_from_options(
            "🎯 请选择生成模式:",
            &["传统模式 (使用属性参数)", "Zero Shot 模式 (使用参考音频)"],
            0,
        )?;

        let is_zero_shot = generation_mode == "Zero Shot 模式 (使用参考音频)";

        // 文本输入
        let text = get_user_input("📝 请输入要转换的文本", "你好，世界！")?;

        // 输出目录
        let output_dir = get_user_input("📁 请输入输出目录", "./generated_audio")?;

        let params = if is_zero_shot {
            // Zero Shot 模式参数
            let ref_audio_path = get_user_input("🎵 请输入参考音频路径", "zero_shot_prompt.wav")?;

            let prompt_text = get_user_input(
                "💬 请输入提示文本 (可选)",
                "希望你以后能够做的，比我还好呦！",
            )?;

            // 确认生成
            let confirm = confirm_action(&format!(
                "🚀 确认生成音频 (Zero Shot 模式)?\n文本: {}\n参考音频: {}\n提示文本: {}\n输出目录: {}",
                text, ref_audio_path, prompt_text, output_dir
            ))?;

            if !confirm {
                continue;
            }

            TtsParams {
                text,
                age: String::new(),
                gender: String::new(),
                emotion: String::new(),
                pitch: String::new(),
                speed: String::new(),
                output_dir,
                zero_shot: true,
                ref_audio_path,
                prompt_text,
            }
        } else {
            // 传统模式参数
            let age = select_from_options(
                "👶 请选择年龄:",
                &["child", "teenager", "youth-adult", "middle-aged", "elderly"],
                2, // youth-adult
            )?;

            let gender = select_from_options(
                "👤 请选择性别:",
                &["female", "male"],
                0, // female
            )?;

            let emotion = select_from_options(
                "😊 请选择情感:",
                &[
                    "NEUTRAL",
                    "ANGRY",
                    "HAPPY",
                    "SAD",
                    "FEARFUL",
                    "DISGUSTED",
                    "SURPRISED",
                ],
                0, // NEUTRAL
            )?;

            let pitch = select_from_options(
                "🎵 请选择音高:",
                &["low_pitch", "medium_pitch", "high_pitch", "very_high_pitch"],
                1, // medium_pitch
            )?;

            let speed = select_from_options(
                "⚡ 请选择速度:",
                &["very_slow", "slow", "medium", "fast", "very_fast"],
                2, // medium
            )?;

            // 确认生成
            let confirm = confirm_action(&format!(
                "🚀 确认生成音频?\n文本: {}\n参数: 年龄={}, 性别={}, 情感={}, 音高={}, 速度={}\n输出目录: {}",
                text, age, gender, emotion, pitch, speed, output_dir
            ))?;

            if !confirm {
                continue;
            }

            TtsParams {
                text,
                age,
                gender,
                emotion,
                pitch,
                speed,
                output_dir,
                zero_shot: false,
                ref_audio_path: String::new(),
                prompt_text: String::new(),
            }
        };

        // 生成音频
        match generate_audio(model_path, &params).await {
            Ok(output_path) => {
                println!("✅ 音频生成成功，保存至: {}", output_path);
            }
            Err(e) => {
                println!("❌ 生成失败: {:?}", e);
            }
        }

        // 询问是否继续
        let continue_generation = confirm_action("🔄 是否继续生成音频?")?;
        if !continue_generation {
            break;
        }
    }

    Ok(())
}

/// 生成音频
async fn generate_audio(
    model_path: &str,
    params: &TtsParams,
) -> std::result::Result<String, Box<dyn std::error::Error>> {
    println!("🔧 使用参数生成音频:");
    println!("  模型路径: {}", model_path);
    println!("  文本: {}", params.text);
    if params.zero_shot {
        println!("  模式: Zero Shot");
        println!("  参考音频: {}", params.ref_audio_path);
        println!("  提示文本: {}", params.prompt_text);
    } else {
        println!("  模式: 传统");
        println!("  年龄: {}", params.age);
        println!("  性别: {}", params.gender);
        println!("  情感: {}", params.emotion);
        println!("  音高: {}", params.pitch);
        println!("  速度: {}", params.speed);
    }

    // 检查文本是否为空
    if params.text.trim().is_empty() {
        return Err("输入文本不能为空".into());
    }

    // 检查模型路径是否存在
    if !Path::new(model_path).exists() {
        return Err(format!("模型路径不存在: {}", model_path).into());
    }

    // 创建TTS参数
    let args = Args {
        text: params.text.clone(),
        model_path: model_path.to_string(),
        vocab_path: format!(
            "{}/rwkv_vocab_v20230424_sparktts_spct_tokens.txt",
            model_path
        ),
        output_path: params.output_dir.clone(),
        temperature: 1.0,
        top_p: 0.95,
        top_k: 50,
        max_tokens: 3000,
        age: params.age.clone(),
        gender: params.gender.clone(),
        emotion: params.emotion.clone(),
        pitch: params.pitch.clone(),
        speed: params.speed.clone(),
        validate: false,
        zero_shot: params.zero_shot,
        ref_audio_path: params.ref_audio_path.clone(),
        prompt_text: params.prompt_text.clone(),
    };

    // 检查参考音频文件是否存在（Zero-shot模式）
    if params.zero_shot
        && !params.ref_audio_path.is_empty()
        && !Path::new(&params.ref_audio_path).exists()
    {
        return Err(format!("参考音频文件不存在: {}", params.ref_audio_path).into());
    }

    // 创建TTS生成器并生成音频
    let generator =
        TTSGenerator::new_async(args.model_path.clone(), args.vocab_path.clone()).await?;
    let audio_samples = generator.generate(&args.text, &args).await?;

    // 生成唯一的输出文件名
    let output_path = get_unique_filename(&params.output_dir, &params.text, ".wav");
    println!("💾 音频将保存到: {}", output_path);

    // 保存音频文件
    generator.save_audio(&audio_samples, &output_path, 16000)?;

    Ok(output_path)
}

/// 生成唯一的文件名
fn get_unique_filename(output_dir: &str, text: &str, extension: &str) -> String {
    let output_dir = Path::new(output_dir);
    std::fs::create_dir_all(output_dir).unwrap_or_default();

    let prefix = if text.len() >= 3 {
        text.chars().take(3).collect::<String>()
    } else {
        text.to_string()
    };

    let prefix: String = prefix.chars().filter(|c| c.is_alphanumeric()).collect();
    let base_name = if prefix.is_empty() {
        "audio".to_string()
    } else {
        prefix
    };

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

pub fn main() {
    show_welcome();

    // 解析命令行参数
    let args = parse_args();

    // 检查模型路径是否存在
    if !Path::new(&args.model_path).exists() {
        eprintln!("❌ 错误: 模型路径不存在: {}", args.model_path);
        process::exit(1);
    }

    println!("📋 模型路径: {}", args.model_path);

    // 启动交互式界面
    // 使用同步方式调用异步函数
    let result = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async { interactive_parameter_selection(&args.model_path).await });

    if let Err(e) = result {
        eprintln!("❌ 错误: {:?}", e);
        process::exit(1);
    }

    println!("👋 感谢使用 RWKV TTS!");
}
