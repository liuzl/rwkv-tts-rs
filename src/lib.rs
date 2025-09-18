//! RWKV TTS Core Library
//!
//! This library provides the core functionality for text-to-speech generation using RWKV models.

// Core modules
// pub mod batch_manager; // 已移动到备份目录
pub mod properties_util;
pub mod ref_audio_utilities;
pub mod rwkv_sampler;
pub mod tts_state_manager;
// pub mod tts_pipeline; // 已移动到备份目录
pub mod tts_pipeline_fixes;

// New concurrent architecture modules
pub mod global_sampler_manager;
pub mod onnx_session_pool;
// pub mod batch_request_scheduler; // 已移动到备份目录
pub mod dynamic_batch_manager;
pub mod lightweight_tts_pipeline;
pub mod voice_feature;
pub mod voice_feature_manager;

// Refactored batch manager modules
pub mod batch_types;
pub mod feature_extractor;
pub mod sampler_manager;
pub mod shared_runtime;

// Inference modules
pub mod normal_mode_inference;
pub mod zero_shot_inference;

// 新的状态管理架构
pub use tts_state_manager::{
    TtsInferContext, TtsInferOptions, TtsStateId, TtsStateManager, TtsStateStats,
};

// Re-export key components
// pub use batch_manager::{BatchManager, BatchConfig, BatchStats}; // 已移动到备份目录
pub use properties_util::*;
pub use ref_audio_utilities::RefAudioUtilities;
pub use rwkv_sampler::{RwkvSampler, SamplerArgs, TtsBatchRequest};
// pub use tts_pipeline::{TtsPipeline, TtsPipelineArgs}; // 已移动到备份目录

/// TTS Generator module
pub mod tts_generator {
    // TTS Generator implementation

    use crate::{RefAudioUtilities, RwkvSampler};

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
            // 创建RWKV采样器，不使用量化配置
            let quant_config = None;
            let rwkv_sampler =
                RwkvSampler::new(&model_path, &vocab_path, quant_config, 256).await?;

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

        /// 生成语音 (暂时禁用，因为TtsPipeline已移动到备份目录)
        pub async fn generate(&self, _text: &str, _args: &Args) -> Result<Vec<f32>> {
            Err(anyhow::anyhow!(
                "TtsPipeline已移动到备份目录，请使用lightweight_tts_pipeline"
            ))
        }

        /// 保存音频到WAV文件
        pub fn save_audio(
            &self,
            audio_samples: &[f32],
            output_path: &str,
            sample_rate: u32,
        ) -> Result<()> {
            // 保存音频到WAV文件
            // 保存音频到指定路径

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
