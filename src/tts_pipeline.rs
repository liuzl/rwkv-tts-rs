//! TTS流水线
//! 实现完整的TTS流程，包括文本处理、RWKV模型推理和音频生成

use crate::{
    properties_util,
    ref_audio_utilities::RefAudioUtilities,
    rwkv_sampler::{RwkvSampler, SamplerArgs},
};
use anyhow::Result;
use std::path::Path;

/// TTS流水线参数
#[derive(Debug, Clone)]
pub struct TtsPipelineArgs {
    pub text: String,
    pub model_path: String,
    pub vocab_path: String,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub max_tokens: usize,
    pub age: String,
    pub gender: String,
    pub emotion: String,
    pub pitch: f32,
    pub speed: f32,
    pub zero_shot: bool,
    pub ref_audio_path: String,
    pub prompt_text: String,
    // 新增字段：输出目录（用于保存生成的音频文件）
    pub output_path: String,
    // 新增字段：是否启用验证
    pub validate: bool,
}

impl Default for TtsPipelineArgs {
    fn default() -> Self {
        Self {
            text: String::new(),
            model_path: String::new(),
            vocab_path: String::new(),
            temperature: 1.0,
            top_p: 0.95,
            top_k: 0,
            max_tokens: 8000,
            age: "youth-adult".to_string(),
            gender: "female".to_string(),
            emotion: "NEUTRAL".to_string(),
            pitch: 200.0,
            speed: 4.2,
            zero_shot: false,
            ref_audio_path: String::new(),
            prompt_text: String::new(),
            output_path: String::from("./output"),
            validate: false,
        }
    }
}

/// TTS流水线
pub struct TtsPipeline {
    rwkv_sampler: RwkvSampler,
    ref_audio_utilities: Option<RefAudioUtilities>,
}

impl TtsPipeline {
    /// 创建新的TTS流水线
    ///
    /// # Arguments
    /// * `args` - TTS流水线参数
    ///
    /// # Returns
    /// * `Result<TtsPipeline>` - TTS流水线实例或错误
    pub async fn new(args: &TtsPipelineArgs) -> Result<Self> {
        // 创建RWKV采样器
        let rwkv_sampler = RwkvSampler::new(&args.model_path, &args.vocab_path).await?;

        // 如果是Zero-shot模式，创建参考音频处理工具；
        // 否则若存在BiCodecDetokenize.onnx，则也加载以支持解码。
        let ref_audio_utilities = if args.zero_shot && !args.ref_audio_path.is_empty() {
            // 检查参考音频文件是否存在
            if Path::new(&args.ref_audio_path).exists() {
                // 创建参考音频处理工具（包含可选的解码器）
                Some(RefAudioUtilities::new(
                    &format!("{}/BiCodecTokenize.onnx", args.model_path),
                    &format!("{}/wav2vec2-large-xlsr-53.onnx", args.model_path),
                    6.0, // ref_segment_duration
                    320, // latent_hop_length
                    Some(&format!("{}/BiCodecDetokenize.onnx", args.model_path)),
                )?)
            } else {
                None
            }
        } else {
            // 非zero-shot也尝试加载解码器（若存在）
            let detok_path = format!("{}/BiCodecDetokenize.onnx", args.model_path);
            if Path::new(&detok_path).exists() {
                Some(RefAudioUtilities::new(
                    &format!("{}/BiCodecTokenize.onnx", args.model_path),
                    &format!("{}/wav2vec2-large-xlsr-53.onnx", args.model_path),
                    6.0,
                    320,
                    Some(&detok_path),
                )?)
            } else {
                None
            }
        };

        Ok(Self {
            rwkv_sampler,
            ref_audio_utilities,
        })
    }

    /// 处理文本
    ///
    /// # Arguments
    /// * `text` - 输入文本
    ///
    /// # Returns
    /// * `String` - 处理后的文本
    fn process_text(&self, text: &str) -> String {
        // 这里可以添加文本预处理逻辑
        text.to_string()
    }

    /// 生成TTS属性tokens
    ///
    /// # Arguments
    /// * `args` - TTS流水线参数
    ///
    /// # Returns
    /// * `Vec<i32>` - 属性token ID数组
    fn generate_property_tokens(&self, args: &TtsPipelineArgs) -> Vec<i32> {
        if args.zero_shot {
            // Zero-shot模式下，tokenize已在generate_speech中处理，这里仅返回空数组
            vec![]
        } else {
            // 解析age字符串为数字
            let age_num = args.age.parse::<u8>().unwrap_or(25);
            properties_util::convert_properties_to_tokens(
                args.speed,
                args.pitch,
                age_num,
                &args.gender,
                &args.emotion,
            )
        }
    }

    /// 生成语音
    ///
    /// # Arguments
    /// * `args` - TTS流水线参数
    ///
    /// # Returns
    /// * `Result<Vec<f32>>` - 音频数据或错误
    pub async fn generate_speech(&mut self, args: &TtsPipelineArgs) -> Result<Vec<f32>> {
        println!("🚀 开始TTS生成流程");
        println!("  文本: {}", args.text);
        println!("  模型路径: {}", args.model_path);
        println!("  词表路径: {}", args.vocab_path);
        println!("  Zero-shot模式: {}", args.zero_shot);

        // 处理文本
        let processed_text = self.process_text(&args.text);
        println!("  处理后文本: {}", processed_text);

        // 生成属性tokens
        let (property_tokens_str, property_tokens) = {
            // 因为tokenize需要&mut self，所以这里临时可变借用ref_audio_utilities
            if args.zero_shot {
                if let Some(ref mut utils) = self.ref_audio_utilities {
                    match utils.tokenize(&args.ref_audio_path) {
                        Ok((global_tokens, semantic_tokens)) => {
                            let tokens_str = format!(
                                "GLOBAL:{} SEMANTIC:{}",
                                global_tokens
                                    .iter()
                                    .map(|t| t.to_string())
                                    .collect::<Vec<_>>()
                                    .join(","),
                                semantic_tokens
                                    .iter()
                                    .map(|t| t.to_string())
                                    .collect::<Vec<_>>()
                                    .join(","),
                            );
                            (tokens_str, vec![])
                        }
                        Err(_) => {
                            let age_num = args.age.parse::<u8>().unwrap_or(25);
                            let tokens = properties_util::convert_properties_to_tokens(
                                args.speed,
                                args.pitch,
                                age_num,
                                &args.gender,
                                &args.emotion,
                            );
                            let tokens_str = format!("TOKENS:{}", tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
                            (tokens_str, tokens)
                        }
                    }
                } else {
                    let age_num = args.age.parse::<u8>().unwrap_or(25);
                    let tokens = properties_util::convert_properties_to_tokens(
                        args.speed,
                        args.pitch,
                        age_num,
                        &args.gender,
                        &args.emotion,
                    );
                    let tokens_str = format!("TOKENS:{}", tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
                    (tokens_str, tokens)
                }
            } else {
                let tokens = self.generate_property_tokens(args);
                let tokens_str = format!("TOKENS:{}", tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
                (tokens_str, tokens)
            }
        };
        println!("  属性tokens: {}", property_tokens_str);
        // 处理属性tokens（如果包含GLOBAL/SEMANTIC则解析，否则使用已生成的property_tokens）
        let mut final_property_tokens = property_tokens;
        let mut ref_global_tokens: Option<Vec<i32>> = None;
        let mut ref_semantic_tokens: Option<Vec<i32>> = None;
        if property_tokens_str.starts_with("GLOBAL:") {
            // 解析形如 "GLOBAL:a,b,c SEMANTIC:x,y,z" 的格式
            let parts: Vec<&str> = property_tokens_str.split_whitespace().collect();
            for part in parts {
                if let Some(rest) = part.strip_prefix("GLOBAL:") {
                    let vals: Vec<i32> = rest
                        .split(',')
                        .filter_map(|s| s.parse::<i32>().ok())
                        .collect();
                    ref_global_tokens = Some(vals);
                } else if let Some(rest) = part.strip_prefix("SEMANTIC:") {
                    let vals: Vec<i32> = rest
                        .split(',')
                        .filter_map(|s| s.parse::<i32>().ok())
                        .collect();
                    ref_semantic_tokens = Some(vals);
                }
            }
            // 对于GLOBAL/SEMANTIC格式，清空property_tokens
            final_property_tokens = vec![];
        } else if property_tokens_str.starts_with("TOKENS:") {
            // 已经有了property_tokens，无需额外处理
        } else {
            // 将属性tokens字符串通过tokenizer编码为整数ID序列
            let ids_u32 = self
                .rwkv_sampler
                .tokenizer()
                .encode(property_tokens_str.as_bytes())
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            final_property_tokens = ids_u32.into_iter().map(|x| x as i32).collect();
        }

        // 创建采样参数
        let sampler_args = SamplerArgs {
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            max_tokens: args.max_tokens,
        };

        // 重置RWKV运行时状态
        self.rwkv_sampler.reset();

        // 使用TTS专用采样生成tokens
        println!("  开始RWKV TTS token生成...");
        let (global_tokens, semantic_tokens) = self
            .rwkv_sampler
            .generate_tts_tokens(
                &processed_text,
                &final_property_tokens,
                ref_global_tokens.as_deref(),
                ref_semantic_tokens.as_deref(),
                &sampler_args,
            )
            .await?;
        println!(
            "  生成global tokens: {} 个, semantic tokens: {} 个",
            global_tokens.len(),
            semantic_tokens.len()
        );

        // 若未生成任何token，则返回静音占位，避免调用detokenizer失败
        if global_tokens.is_empty() && semantic_tokens.is_empty() {
            println!("  未生成任何TTS tokens，返回静音占位");
            return Ok(vec![0.0; 16000]);
        }

        // 使用BiCodecDetokenize解码为音频
        if let Some(ref mut utils) = self.ref_audio_utilities {
            println!("  开始BiCodecDetokenize解码...");
            // detokenizer 期望codec原始token域 [0..8191]，不需要做 -4096 平移
            // 为安全起见仅进行裁剪，避免越界
            let semantic_clipped: Vec<i32> =
                semantic_tokens.iter().map(|&v| v.clamp(0, 8191)).collect();
            let global_clipped: Vec<i32> =
                global_tokens.iter().map(|&v| v.clamp(0, 8191)).collect();
            let audio = utils.detokenize_audio(&global_clipped, &semantic_clipped)?;
            println!("  解码完成，音频长度: {} 样本", audio.len());
            Ok(audio)
        } else {
            println!("  未启用参考音频解码，返回静音占位");
            Ok(vec![0.0; 16000])
        }
    }

    /// 保存音频到WAV文件
    ///
    /// # Arguments
    /// * `audio_samples` - 音频数据
    /// * `output_path` - 输出文件路径
    /// * `sample_rate` - 采样率
    ///
    /// # Returns
    /// * `Result<()>` - 保存结果或错误
    pub fn save_audio(
        &self,
        audio_samples: &[f32],
        output_path: &str,
        sample_rate: u32,
    ) -> Result<()> {
        // 保存音频到WAV文件
        println!("  保存音频到: {}", output_path);

        // 使用hound库保存WAV文件
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = hound::WavWriter::create(output_path, spec)?;
        for &sample in audio_samples {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;

        println!("  音频保存完成");
        Ok(())
    }
}
