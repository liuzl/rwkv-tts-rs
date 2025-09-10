//! RWKV TTS Core Library
//!
//! This library provides the core functionality for text-to-speech generation using RWKV models.

// Core modules
pub mod properties_util;
pub mod ref_audio_utilities;
pub mod rwkv_sampler;
pub mod tts_pipeline;

// Re-export key components
pub use properties_util::*;
pub use ref_audio_utilities::RefAudioUtilities;
pub use rwkv_sampler::{RwkvSampler, SamplerArgs};
pub use tts_pipeline::{TtsPipeline, TtsPipelineArgs};

/// TTS Generator module
pub mod tts_generator {
    // TTS Generator implementation

    use crate::{RefAudioUtilities, RwkvSampler, TtsPipeline, TtsPipelineArgs};
    use anyhow::Result;

    /// Args结构体定义
    #[derive(Debug)]
    pub struct Args {
        pub text: String,
        pub model_path: String,
        pub vocab_path: String,
        pub output_path: String,
        pub temperature: f32,
        pub top_p: f32,
        pub top_k: usize,
        pub max_tokens: usize,
        pub age: String,
        pub gender: String,
        pub emotion: String,
        pub pitch: String,
        pub speed: String,
        pub validate: bool,
        pub zero_shot: bool,
        pub ref_audio_path: String,
        pub prompt_text: String,
    }

    /// TTS生成器结构体
    pub struct TTSGenerator {
        /// RWKV采样器
        pub rwkv_sampler: Option<RwkvSampler>,
        /// 参考音频处理工具
        pub ref_audio_utilities: Option<RefAudioUtilities>,
    }

    impl TTSGenerator {
        /// 创建新的TTS生成器
        pub fn new() -> Self {
            Self {
                rwkv_sampler: None,
                ref_audio_utilities: None,
            }
        }

        /// 异步创建新的TTS生成器
        pub async fn new_async(model_path: String, vocab_path: String) -> Result<Self> {
            // 创建RWKV采样器
            let rwkv_sampler = RwkvSampler::new(&model_path, &vocab_path).await?;

            Ok(Self {
                rwkv_sampler: Some(rwkv_sampler),
                ref_audio_utilities: None,
            })
        }

        /// 设置RWKV采样器
        pub fn with_rwkv_sampler(mut self, sampler: RwkvSampler) -> Self {
            self.rwkv_sampler = Some(sampler);
            self
        }

        /// 设置参考音频处理工具
        pub fn with_ref_audio_utilities(mut self, utilities: RefAudioUtilities) -> Self {
            self.ref_audio_utilities = Some(utilities);
            self
        }

        /// 生成语音
        pub async fn generate(&self, text: &str, args: &Args) -> Result<Vec<f32>> {
            // 创建TTS流水线参数
            let pipeline_args = TtsPipelineArgs {
                text: text.to_string(),
                model_path: args.model_path.clone(),
                vocab_path: args.vocab_path.clone(),
                temperature: args.temperature,
                top_p: args.top_p,
                top_k: args.top_k,
                max_tokens: args.max_tokens,
                age: args.age.clone(),
                gender: args.gender.clone(),
                emotion: args.emotion.clone(),
                pitch: 200.0, // 这里需要从参数中获取实际的音高值，暂时使用默认值
                speed: 4.2,   // 这里需要从参数中获取实际的语速值，暂时使用默认值
                zero_shot: args.zero_shot,
                ref_audio_path: args.ref_audio_path.clone(),
                prompt_text: args.prompt_text.clone(),
                output_path: args.output_path.clone(),
                validate: args.validate,
            };

            // 创建TTS流水线
            let mut pipeline = TtsPipeline::new(&pipeline_args).await?;

            // 生成语音
            let audio_samples = pipeline.generate_speech(&pipeline_args).await?;

            Ok(audio_samples)
        }

        /// 保存音频到WAV文件
        pub fn save_audio(
            &self,
            audio_samples: &[f32],
            output_path: &str,
            sample_rate: u32,
        ) -> Result<()> {
            // 保存音频到WAV文件
            println!("保存音频到: {}", output_path);

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

            Ok(())
        }
    }

    impl Default for TTSGenerator {
        fn default() -> Self {
            Self::new()
        }
    }
}