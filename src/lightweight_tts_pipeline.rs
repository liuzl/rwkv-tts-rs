//! 轻量级TTS流水线
//! 复用全局资源，不再每次创建新的模型实例

use crate::{
    dynamic_batch_manager::get_global_dynamic_batch_manager,
    onnx_session_pool::get_global_onnx_manager,
    properties_util,
    rwkv_sampler::{SamplerArgs, TtsBatchRequest},
};
use anyhow::Result;
use ndarray::{Array1, Array2, Array3};
use ort::{session::SessionInputValue, value::Value};
use std::path::Path;

/// 轻量级TTS流水线参数
#[derive(Debug, Clone)]
pub struct LightweightTtsPipelineArgs {
    pub text: String,
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
    pub output_path: String,
    pub validate: bool,
    pub seed: Option<u64>,
}

impl Default for LightweightTtsPipelineArgs {
    fn default() -> Self {
        Self {
            text: String::new(),
            temperature: 1.0,
            top_p: 0.90,
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
            seed: None,
        }
    }
}

/// 轻量级TTS流水线，复用全局资源
#[derive(Debug)]
pub struct LightweightTtsPipeline {}

impl LightweightTtsPipeline {
    /// 创建新的轻量级TTS流水线
    pub fn new() -> Self {
        Self {}
    }

    /// 处理文本
    fn process_text(&self, text: &str) -> String {
        text.to_string()
    }

    /// 生成TTS属性tokens
    fn generate_property_tokens(&self, args: &LightweightTtsPipelineArgs) -> Vec<i32> {
        if args.zero_shot {
            vec![] // Zero-shot模式下由参考音频处理
        } else {
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

    /// 处理参考音频（Zero-shot模式）
    async fn process_reference_audio(&self, ref_audio_path: &str) -> Result<(Vec<i32>, Vec<i32>)> {
        if ref_audio_path.is_empty() || !Path::new(ref_audio_path).exists() {
            return Err(anyhow::anyhow!("参考音频文件不存在: {}", ref_audio_path));
        }

        let onnx_manager = get_global_onnx_manager()?;

        // 加载音频文件
        let audio_data = self.load_audio_file(ref_audio_path).await?;

        // 使用BiCodec Tokenize会话
        let bicodec_session = onnx_manager.acquire_bicodec_tokenize_session().await?;
        let (global_tokens, semantic_tokens) = self
            .tokenize_audio_with_session(&audio_data, bicodec_session)
            .await?;

        Ok((global_tokens, semantic_tokens))
    }

    /// 加载音频文件
    async fn load_audio_file(&self, audio_path: &str) -> Result<Vec<f32>> {
        use hound::WavReader;
        use std::path::Path;

        if !Path::new(audio_path).exists() {
            return Err(anyhow::anyhow!("音频文件不存在: {}", audio_path));
        }

        let audio_path = audio_path.to_string();
        let result = tokio::task::spawn_blocking(move || -> Result<Vec<f32>> {
            let mut reader = WavReader::open(&audio_path)?;
            let spec = reader.spec();

            // 读取音频样本并转换为f32
            let samples: Result<Vec<f32>, _> = reader
                .samples::<i16>()
                .map(|s| s.map(|sample| sample as f32 / i16::MAX as f32))
                .collect();
            let mut audio = samples?;

            // 转换为单声道
            if spec.channels > 1 {
                let len = audio.len() / spec.channels as usize;
                let mut mono_audio = Vec::with_capacity(len);
                for i in 0..len {
                    mono_audio.push(audio[i * spec.channels as usize]);
                }
                audio = mono_audio;
            }

            // 重采样到16kHz（简化实现）
            if spec.sample_rate != 16000 {
                let original_len = audio.len();
                let target_len = (original_len as f32 * 16000.0 / spec.sample_rate as f32) as usize;
                let mut resampled = Vec::with_capacity(target_len);
                for i in 0..target_len {
                    let idx = i * original_len / target_len;
                    resampled.push(audio[idx]);
                }
                audio = resampled;
            }

            Ok(audio)
        })
        .await??;

        Ok(result)
    }

    /// 使用ONNX会话进行音频tokenize
    async fn tokenize_audio_with_session(
        &self,
        audio_data: &[f32],
        mut session_guard: crate::onnx_session_pool::SessionGuard,
    ) -> Result<(Vec<i32>, Vec<i32>)> {
        let audio_data = audio_data.to_vec();
        let result = tokio::task::spawn_blocking(move || -> Result<(Vec<i32>, Vec<i32>)> {
            // 转换音频数据为ndarray
            let wav = Array1::from(audio_data);

            // 提取wav2vec2特征（这里需要wav2vec2会话，暂时使用简化处理）
            let feature_dim = 1024;
            let t = wav.len() / 320; // 假设hop_length=320
            let feat = Array2::<f32>::zeros((t, feature_dim));

            // 提取参考音频的mel频谱图（简化处理）
            let ref_segment_length = 16000 * 6; // 6秒参考音频
            let _ref_wav = if wav.len() >= ref_segment_length {
                wav.slice(ndarray::s![..ref_segment_length]).to_owned()
            } else {
                // 重复音频到足够长度
                let repeat_times = ref_segment_length / wav.len() + 1;
                let mut repeated = Vec::with_capacity(wav.len() * repeat_times);
                for _ in 0..repeat_times {
                    repeated.extend(wav.iter());
                }
                Array1::from(repeated)
                    .slice(ndarray::s![..ref_segment_length])
                    .to_owned()
            };

            let ref_mel = Array2::<f32>::zeros((128, 301)); // 简化的mel频谱图

            // 准备ONNX输入
            let ref_mel_input = ref_mel.insert_axis(ndarray::Axis(0));
            let feat_input = feat.insert_axis(ndarray::Axis(0));

            let ref_mel_dyn = ref_mel_input.into_dyn();
            let feat_dyn = feat_input.into_dyn();

            let ref_mel_shape: Vec<i64> = ref_mel_dyn.shape().iter().map(|&d| d as i64).collect();
            let ref_mel_vec: Vec<f32> = ref_mel_dyn.into_raw_vec();
            let ref_mel_tensor = Value::from_array((ref_mel_shape, ref_mel_vec))?;

            let feat_shape: Vec<i64> = feat_dyn.shape().iter().map(|&d| d as i64).collect();
            let feat_vec: Vec<f32> = feat_dyn.into_raw_vec();
            let feat_tensor = Value::from_array((feat_shape, feat_vec))?;

            // 运行ONNX推理
            let outputs = session_guard.session_mut().run(ort::inputs![
                "ref_wav_mel" => SessionInputValue::from(ref_mel_tensor),
                "feat" => SessionInputValue::from(feat_tensor)
            ])?;

            let (_s_sem, semantic_tokens_slice) = outputs[0].try_extract_tensor::<i64>()?;
            let (_s_glb, global_tokens_slice) = outputs[1].try_extract_tensor::<i64>()?;

            let semantic_tokens: Vec<i32> =
                semantic_tokens_slice.iter().map(|&x| x as i32).collect();
            let global_tokens: Vec<i32> = global_tokens_slice.iter().map(|&x| x as i32).collect();

            Ok((global_tokens, semantic_tokens))
        })
        .await??;

        Ok(result)
    }

    /// 解码音频
    async fn decode_audio(
        &self,
        global_tokens: &[i32],
        semantic_tokens: &[i32],
    ) -> Result<Vec<f32>> {
        let onnx_manager = get_global_onnx_manager()?;

        // 获取BiCodec Detokenize会话
        let detokenize_session = onnx_manager.acquire_bicodec_detokenize_session().await?;

        // 执行解码
        let audio = self
            .detokenize_audio_with_session(global_tokens, semantic_tokens, detokenize_session)
            .await?;

        Ok(audio)
    }

    /// 使用ONNX会话进行音频解码
    async fn detokenize_audio_with_session(
        &self,
        global_tokens: &[i32],
        semantic_tokens: &[i32],
        mut session_guard: crate::onnx_session_pool::SessionGuard,
    ) -> Result<Vec<f32>> {
        let global_tokens = global_tokens.to_vec();
        let semantic_tokens = semantic_tokens.to_vec();

        let result = tokio::task::spawn_blocking(move || -> Result<Vec<f32>> {
            // 转换tokens为i64
            let global_i64: Vec<i64> = global_tokens.iter().map(|&v| v as i64).collect();
            let semantic_i64: Vec<i64> = semantic_tokens.iter().map(|&v| v as i64).collect();

            // 按照BiCodec模型的输入格式准备数据
            // global_tokens: (1, 1, Lg)
            let global_tokens_array = Array3::from_shape_vec((1, 1, global_i64.len()), global_i64)?;
            // semantic_tokens: (1, Ls)
            let semantic_tokens_array =
                Array2::from_shape_vec((1, semantic_i64.len()), semantic_i64)?;

            // 转换为动态数组
            let global_dyn = global_tokens_array.into_dyn();
            let semantic_dyn = semantic_tokens_array.into_dyn();

            let global_shape: Vec<i64> = global_dyn.shape().iter().map(|&d| d as i64).collect();
            let global_vec: Vec<i64> = global_dyn.into_raw_vec();
            let global_tensor = Value::from_array((global_shape, global_vec))?;

            let semantic_shape: Vec<i64> = semantic_dyn.shape().iter().map(|&d| d as i64).collect();
            let semantic_vec: Vec<i64> = semantic_dyn.into_raw_vec();
            let semantic_tensor = Value::from_array((semantic_shape, semantic_vec))?;

            // 运行ONNX推理
            let outputs = session_guard.session_mut().run(ort::inputs![
                "semantic_tokens" => SessionInputValue::from(semantic_tensor),
                "global_tokens" => SessionInputValue::from(global_tensor)
            ])?;

            let (_shape, audio_slice) = outputs[0].try_extract_tensor::<f32>()?;
            let audio_vec: Vec<f32> = audio_slice.to_vec();

            Ok(audio_vec)
        })
        .await??;

        Ok(result)
    }

    /// 生成语音（使用批处理调度器）
    pub async fn generate_speech(&self, args: &LightweightTtsPipelineArgs) -> Result<Vec<f32>> {
        let total_start = std::time::Instant::now();

        println!("🚀 开始轻量级TTS生成流程");
        println!("  文本: {}", args.text);
        println!("  Zero-shot模式: {}", args.zero_shot);

        // 1. 处理文本
        let text_start = std::time::Instant::now();
        let processed_text = self.process_text(&args.text);
        let text_time = text_start.elapsed();
        println!(
            "  ⏱️  文本处理耗时: {:.2}ms",
            text_time.as_secs_f64() * 1000.0
        );

        // 2. 处理属性tokens或参考音频
        let ref_start = std::time::Instant::now();
        let (property_tokens, ref_global_tokens, ref_semantic_tokens) = if args.zero_shot {
            let (global, semantic) = self.process_reference_audio(&args.ref_audio_path).await?;
            (vec![], Some(global), Some(semantic))
        } else {
            let tokens = self.generate_property_tokens(args);
            (tokens, None, None)
        };
        let ref_time = ref_start.elapsed();
        if args.zero_shot {
            println!(
                "  ⏱️  参考音频处理耗时: {:.2}ms",
                ref_time.as_secs_f64() * 1000.0
            );
        } else {
            println!(
                "  ⏱️  属性tokens生成耗时: {:.2}ms",
                ref_time.as_secs_f64() * 1000.0
            );
        }

        // 3. 创建采样参数
        let sampler_args = SamplerArgs {
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            max_tokens: args.max_tokens,
            seed: args.seed,
        };

        // 4. 创建批处理请求
        let request = TtsBatchRequest {
            text: processed_text,
            property_tokens,
            ref_global_tokens,
            ref_semantic_tokens,
            args: sampler_args,
        };

        // 5. 提交到动态批处理管理器并等待RWKV推理
        let inference_start = std::time::Instant::now();
        let manager = get_global_dynamic_batch_manager()?;
        let (global_tokens, semantic_tokens) = manager
            .generate_tts(
                request.text,
                request.property_tokens,
                request.ref_global_tokens,
                request.ref_semantic_tokens,
                request.args,
            )
            .await?;
        let inference_time = inference_start.elapsed();
        println!(
            "  ⏱️  RWKV模型推理耗时: {:.2}ms",
            inference_time.as_secs_f64() * 1000.0
        );

        println!(
            "  生成global tokens: {} 个, semantic tokens: {} 个",
            global_tokens.len(),
            semantic_tokens.len()
        );

        // 6. 解码音频
        if global_tokens.is_empty() && semantic_tokens.is_empty() {
            println!("  未生成任何TTS tokens，返回静音占位");
            return Ok(vec![0.0; 16000]);
        }

        let decode_start = std::time::Instant::now();
        let audio = self.decode_audio(&global_tokens, &semantic_tokens).await?;
        let decode_time = decode_start.elapsed();
        println!(
            "  ⏱️  音频解码耗时: {:.2}ms",
            decode_time.as_secs_f64() * 1000.0
        );

        let total_time = total_start.elapsed();
        let audio_duration = audio.len() as f64 / 16000.0; // 假设16kHz采样率
        let rtf = total_time.as_secs_f64() / audio_duration;

        println!(
            "  ⏱️  总耗时: {:.2}ms, 音频时长: {:.2}s, RTF: {:.3}",
            total_time.as_secs_f64() * 1000.0,
            audio_duration,
            rtf
        );

        // 性能优化建议
        if rtf > 0.3 {
            println!("  ⚠️  性能提示: RTF > 0.3，建议优化:");
            if inference_time.as_secs_f64() > total_time.as_secs_f64() * 0.6 {
                println!(
                    "     - RWKV推理占用{:.1}%时间，考虑使用更激进的量化或更小的模型",
                    inference_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
                );
            }
            if decode_time.as_secs_f64() > total_time.as_secs_f64() * 0.3 {
                println!(
                    "     - 音频解码占用{:.1}%时间，考虑优化BiCodec模型或使用GPU加速",
                    decode_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
                );
            }
            if args.zero_shot && ref_time.as_secs_f64() > total_time.as_secs_f64() * 0.2 {
                println!(
                    "     - 参考音频处理占用{:.1}%时间，考虑缓存或预处理参考音频",
                    ref_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
                );
            }
        }

        println!("  轻量级TTS生成完成，音频长度: {} 样本", audio.len());
        Ok(audio)
    }

    /// 保存音频到WAV文件
    pub fn save_audio(
        &self,
        audio_samples: &[f32],
        output_path: &str,
        sample_rate: u32,
    ) -> Result<()> {
        println!("  保存音频到: {}", output_path);

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

impl Default for LightweightTtsPipeline {
    fn default() -> Self {
        Self::new()
    }
}
