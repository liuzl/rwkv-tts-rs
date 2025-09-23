//! 轻量级TTS流水线
//! 复用全局资源，不再每次创建新的模型实例

use crate::{
    dynamic_batch_manager::get_global_dynamic_batch_manager,
    onnx_session_pool::get_global_onnx_manager,
    properties_util,
    rwkv_sampler::{SamplerArgs, TtsBatchRequest},
    voice_feature_manager::VoiceFeatureManager,
};
use anyhow::Result;
use ndarray::{Array1, Array2};
use ort::{session::SessionInputValue, value::Value};
use std::path::Path;
use tracing;

/// 轻量级TTS流水线参数
#[derive(Debug, Clone)]
pub struct LightweightTtsPipelineArgs {
    pub text: String,
    pub prompt_text: String,
    pub ref_audio_path: String,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub max_tokens: usize,
    pub age: String,
    pub gender: String,
    pub emotion: String,
    pub pitch: String,
    pub speed: String,
    pub zero_shot: bool,
    pub validate: bool,
    pub seed: Option<u64>,
    // 新增：voice_id用于从缓存获取tokens
    pub voice_id: Option<String>,
    // 新增：直接传入的音色特征tokens
    pub voice_global_tokens: Option<Vec<i32>>,
    pub voice_semantic_tokens: Option<Vec<i32>>,
}

impl Default for LightweightTtsPipelineArgs {
    fn default() -> Self {
        Self {
            text: String::new(),
            prompt_text: String::new(),
            ref_audio_path: String::new(),
            temperature: 1.0,
            top_p: 0.90,
            top_k: 0,
            max_tokens: 8000,
            age: "youth-adult".to_string(),
            gender: "female".to_string(),
            emotion: "NEUTRAL".to_string(),
            pitch: "medium".to_string(),
            speed: "medium".to_string(),
            zero_shot: false,
            validate: false,
            seed: None,
            voice_id: None,
            voice_global_tokens: None,
            voice_semantic_tokens: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_args_default() {
        let args = LightweightTtsPipelineArgs::default();

        // 测试新添加的字段
        assert_eq!(args.prompt_text, "");
        assert_eq!(args.ref_audio_path, "");

        // 测试其他基本字段
        assert_eq!(args.text, "");
        assert_eq!(args.temperature, 1.0);
        assert_eq!(args.top_p, 0.90);
        assert_eq!(args.top_k, 0);
        assert_eq!(args.max_tokens, 8000);
        assert_eq!(args.age, "youth-adult");
        assert_eq!(args.gender, "female");
        assert_eq!(args.emotion, "NEUTRAL");
        assert_eq!(args.pitch, "medium");
        assert_eq!(args.speed, "medium");
        assert!(!args.zero_shot);
        assert!(!args.validate);
        assert_eq!(args.seed, None);
        assert_eq!(args.voice_id, None);
        assert_eq!(args.voice_global_tokens, None);
        assert_eq!(args.voice_semantic_tokens, None);
    }

    #[test]
    fn test_pipeline_args_custom() {
        let args = LightweightTtsPipelineArgs {
            prompt_text: "这是提示文本".to_string(),
            ref_audio_path: "/path/to/audio.wav".to_string(),
            text: "这是要合成的文本".to_string(),
            zero_shot: true,
            ..Default::default()
        };

        // 验证字段设置正确
        assert_eq!(args.prompt_text, "这是提示文本");
        assert_eq!(args.ref_audio_path, "/path/to/audio.wav");
        assert_eq!(args.text, "这是要合成的文本");
        assert!(args.zero_shot);
    }

    #[test]
    fn test_pipeline_args_clone() {
        let args = LightweightTtsPipelineArgs {
            prompt_text: "测试克隆".to_string(),
            ref_audio_path: "/test/path.wav".to_string(),
            ..Default::default()
        };

        let cloned_args = args.clone();

        assert_eq!(args.prompt_text, cloned_args.prompt_text);
        assert_eq!(args.ref_audio_path, cloned_args.ref_audio_path);
    }

    #[test]
    fn test_process_text_zero_shot() {
        let pipeline = LightweightTtsPipeline::new();
        let result = pipeline.process_text_zero_shot("用户文本", "提示文本");
        assert_eq!(result, "提示文本用户文本");
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

    /// 处理文本（Zero-shot模式）
    /// 注意：Zero-shot模式下结合参考音频的提示文本和用户输入文本
    /// 返回格式为"prompt_text + user_text"的组合，以改善语音合成效果
    pub fn process_text_zero_shot(&self, text: &str, prompt_text: &str) -> String {
        let combined_text = format!("{}{}", prompt_text, text);
        #[cfg(debug_assertions)]
        {
            // Zero-shot模式：使用组合文本处理
        }
        combined_text
    }

    /// 生成TTS属性tokens
    fn generate_property_tokens(&self, args: &LightweightTtsPipelineArgs) -> Vec<i32> {
        // 如果提供了预提取的音色特征tokens或处于zero_shot模式，则不使用传统属性参数
        if (args.voice_global_tokens.is_some() && args.voice_semantic_tokens.is_some())
            || args.zero_shot
        {
            tracing::info!("🎭 使用预提取音色特征或zero_shot模式，跳过传统属性参数");
            vec![] // 使用预提取音色特征或zero_shot模式时，传统属性参数不起作用
        } else {
            // 添加调试日志，打印传入的参数
            tracing::info!(
                "🎵 生成属性tokens - age: {}, gender: {}, emotion: {}, pitch: {}, speed: {}",
                args.age,
                args.gender,
                args.emotion,
                args.pitch,
                args.speed
            );

            // 直接使用传入的pitch和speed字符串，无需分类转换
            // 注意：函数定义的参数顺序是(age, gender, emotion, pitch, speed) - 与Python/C++版本一致
            let tokens = properties_util::convert_standard_properties_to_tokens(
                &args.age,     // age - 直接传递字符串年龄
                &args.gender,  // gender
                &args.emotion, // emotion
                &args.pitch,   // pitch - 直接传递字符串
                &args.speed,   // speed - 直接传递字符串
            );

            tracing::info!("🎯 生成的属性tokens: {:?}", tokens);
            tokens
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

    /// 加载音频文件（支持WAV和MP3格式）
    async fn load_audio_file(&self, audio_path: &str) -> Result<Vec<f32>> {
        use std::path::Path;

        if !Path::new(audio_path).exists() {
            return Err(anyhow::anyhow!("音频文件不存在: {}", audio_path));
        }

        let audio_path = audio_path.to_string();
        let result = tokio::task::spawn_blocking(move || -> Result<Vec<f32>> {
            let path = Path::new(&audio_path);
            let extension = path
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("")
                .to_lowercase();

            let (mut audio, sample_rate, channels) = match extension.as_str() {
                "wav" => {
                    // 使用hound处理WAV文件
                    use hound::WavReader;
                    let mut reader = WavReader::open(&audio_path)?;
                    let spec = reader.spec();

                    let samples: Result<Vec<f32>, _> = reader
                        .samples::<i16>()
                        .map(|s| s.map(|sample| sample as f32 / i16::MAX as f32))
                        .collect();
                    let audio = samples?;

                    (audio, spec.sample_rate, spec.channels as usize)
                }
                "mp3" => {
                    // 使用symphonia处理MP3文件
                    use std::fs::File;
                    use symphonia::core::audio::{AudioBufferRef, Signal};
                    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_MP3};
                    use symphonia::core::formats::FormatOptions;
                    use symphonia::core::io::MediaSourceStream;
                    use symphonia::core::meta::MetadataOptions;
                    use symphonia::core::probe::Hint;

                    let file = File::open(&audio_path)?;
                    let mss = MediaSourceStream::new(Box::new(file), Default::default());

                    let mut hint = Hint::new();
                    hint.with_extension("mp3");

                    let meta_opts: MetadataOptions = Default::default();
                    let fmt_opts: FormatOptions = Default::default();

                    let probed = symphonia::default::get_probe()
                        .format(&hint, mss, &fmt_opts, &meta_opts)?;

                    let mut format = probed.format;
                    let track = format
                        .tracks()
                        .iter()
                        .find(|t| t.codec_params.codec == CODEC_TYPE_MP3)
                        .ok_or_else(|| anyhow::anyhow!("未找到MP3音轨"))?;

                    let track_id = track.id;
                    let mut decoder = symphonia::default::get_codecs()
                        .make(&track.codec_params, &DecoderOptions { verify: false })?;

                    let mut audio_data = Vec::new();
                    let mut sample_rate = 44100u32;
                    let mut channels = 2usize;

                    while let Ok(packet) = format.next_packet() {
                        if packet.track_id() != track_id {
                            continue;
                        }

                        match decoder.decode(&packet)? {
                            AudioBufferRef::F32(buf) => {
                                sample_rate = buf.spec().rate;
                                channels = buf.spec().channels.count();
                                for &sample in buf.chan(0) {
                                    audio_data.push(sample);
                                }
                                if channels > 1 {
                                    for &sample in buf.chan(1) {
                                        audio_data.push(sample);
                                    }
                                }
                            }
                            AudioBufferRef::S16(buf) => {
                                sample_rate = buf.spec().rate;
                                channels = buf.spec().channels.count();
                                for &sample in buf.chan(0) {
                                    audio_data.push(sample as f32 / i16::MAX as f32);
                                }
                                if channels > 1 {
                                    for &sample in buf.chan(1) {
                                        audio_data.push(sample as f32 / i16::MAX as f32);
                                    }
                                }
                            }
                            _ => return Err(anyhow::anyhow!("不支持的音频格式")),
                        }
                    }

                    (audio_data, sample_rate, channels)
                }
                _ => {
                    return Err(anyhow::anyhow!("不支持的音频格式: {}", extension));
                }
            };

            // 转换为单声道
            if channels > 1 {
                let len = audio.len() / channels;
                let mut mono_audio = Vec::with_capacity(len);
                for i in 0..len {
                    mono_audio.push(audio[i * channels]);
                }
                audio = mono_audio;
            }

            // 重采样到16kHz
            if sample_rate != 16000 {
                let original_len = audio.len();
                let target_len = (original_len as f32 * 16000.0 / sample_rate as f32) as usize;
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
    pub async fn tokenize_audio_with_session(
        &self,
        audio_data: &[f32],
        mut session_guard: crate::onnx_session_pool::SessionGuard,
    ) -> Result<(Vec<i32>, Vec<i32>)> {
        // 预先获取wav2vec2会话（异步），随后在阻塞线程中使用
        let onnx_manager = get_global_onnx_manager()?;
        let mut wav2vec2_guard = onnx_manager.acquire_wav2vec2_session().await?;

        let audio_data = audio_data.to_vec();
        let result = tokio::task::spawn_blocking(move || -> Result<(Vec<i32>, Vec<i32>)> {
            // 转换音频数据为ndarray
            let wav = Array1::from(audio_data);

            // 提取参考片段（长度与Ref实现一致）
            let ref_wav = Self::get_ref_clip(&wav);

            // 修复：使用原始wav数据进行wav2vec2特征提取，与C++和Python实现保持一致
            // C++实现中直接使用原始audio进行extract_wav2vec2_features(audio)
            // Python实现中使用原始wav进行extract_wav2vec2_features(wav)
            // 注意：zero_mean_unit_variance_normalize已经在extract_wav2vec2_features内部进行了

            // 应用零均值单位方差归一化预处理 - 与C++和Python实现保持一致
            let normalized_wav =
                crate::ref_audio_utilities::RefAudioUtilities::zero_mean_unit_variance_normalize(
                    wav.to_vec(),
                );
            let feat_input = Array1::from(normalized_wav).insert_axis(ndarray::Axis(0));
            let feat_dyn = feat_input.into_dyn();
            let feat_shape: Vec<i64> = feat_dyn.shape().iter().map(|&d| d as i64).collect();
            let feat_vec = feat_dyn.into_raw_vec();
            let feat_tensor = Value::from_array((feat_shape, feat_vec))?;

            let wav2vec2_outputs = wav2vec2_guard
                .session_mut()
                .run(ort::inputs![SessionInputValue::from(feat_tensor)])?;
            let (feat_out_shape, feat_data) = wav2vec2_outputs[0].try_extract_tensor::<f32>()?;

            if !(feat_out_shape.len() == 3 && feat_out_shape[0] == 1) {
                return Err(anyhow::anyhow!(
                    "Unexpected wav2vec2 output shape: {:?}",
                    feat_out_shape
                ));
            }
            let time_steps = feat_out_shape[1] as usize;
            let feature_dim = feat_out_shape[2] as usize;
            if feature_dim != 1024 {
                return Err(anyhow::anyhow!(
                    "Expected feature dimension 1024, got {}",
                    feature_dim
                ));
            }
            let feat = Array2::from_shape_vec((time_steps, feature_dim), feat_data.to_vec())?;

            // 提取mel频谱图（精确对齐Ref实现）
            // 修复：使用与C++完全一致的参数
            let ref_mel =
                crate::tts_pipeline_fixes::TtsPipelineFixes::extract_mel_spectrogram_consistent(
                    &ref_wav,
                )?;

            // 准备ONNX输入
            // 确保数据是行优先布局（C-order），与C++实现保持一致
            let ref_mel_c_order = if ref_mel.is_standard_layout() {
                ref_mel.clone()
            } else {
                ref_mel.as_standard_layout().to_owned()
            };

            let feat_c_order = if feat.is_standard_layout() {
                feat.clone()
            } else {
                feat.as_standard_layout().to_owned()
            };

            let ref_mel_input = ref_mel_c_order.insert_axis(ndarray::Axis(0));
            let feat_input2 = feat_c_order.insert_axis(ndarray::Axis(0));

            let ref_mel_dyn = ref_mel_input.into_dyn();
            let feat_dyn2 = feat_input2.into_dyn();

            let ref_mel_shape: Vec<i64> = ref_mel_dyn.shape().iter().map(|&d| d as i64).collect();
            let ref_mel_vec: Vec<f32> = ref_mel_dyn.into_raw_vec();
            let ref_mel_tensor = Value::from_array((ref_mel_shape, ref_mel_vec))?;

            let feat_shape2: Vec<i64> = feat_dyn2.shape().iter().map(|&d| d as i64).collect();
            let feat_vec2: Vec<f32> = feat_dyn2.into_raw_vec();
            let feat_tensor2 = Value::from_array((feat_shape2, feat_vec2))?;

            // 运行BiCodec Tokenize ONNX推理
            let outputs = session_guard.session_mut().run(ort::inputs![
                "ref_wav_mel" => SessionInputValue::from(ref_mel_tensor),
                "feat" => SessionInputValue::from(feat_tensor2)
            ])?;

            // 修复输出解析顺序：与Python和C++实现保持一致
            // Python: semantic_tokens = outputs[0], global_tokens = outputs[1]
            // C++: semantic_tokens = output_tensors["semantic_tokens"], global_tokens = output_tensors["global_tokens"]

            // 先尝试按名称解析输出
            let mut semantic_tokens: Vec<i32> = vec![];
            let mut global_tokens: Vec<i32> = vec![];

            // 检查输出名称来确定正确的解析顺序
            for (_name, output) in outputs.iter() {
                // 获取输出的形状信息
                let shape = output.shape();

                // semantic_tokens 通常是形状为 [1, length] 的张量
                // global_tokens 通常是形状为 [1, 1, 32] 的张量
                if shape.len() == 2 && shape[0] == 1 {
                    // 这很可能是 semantic_tokens
                    semantic_tokens = match output.try_extract_tensor::<i64>() {
                        Ok((_s_sem, semantic_tokens_slice)) => {
                            semantic_tokens_slice.iter().map(|&x| x as i32).collect()
                        }
                        Err(_) => {
                            let (_s_sem, semantic_tokens_slice) =
                                output.try_extract_tensor::<i32>()?;
                            semantic_tokens_slice.to_vec()
                        }
                    };
                } else if shape.len() == 3 && shape[0] == 1 && shape[1] == 1 {
                    // 这很可能是 global_tokens
                    global_tokens = match output.try_extract_tensor::<i64>() {
                        Ok((_s_glb, global_tokens_slice)) => {
                            global_tokens_slice.iter().map(|&x| x as i32).collect()
                        }
                        Err(_) => {
                            let (_s_glb, global_tokens_slice) =
                                output.try_extract_tensor::<i32>()?;
                            global_tokens_slice.to_vec()
                        }
                    };
                }
            }

            // 如果按形状没有找到，使用索引方式作为备选
            if semantic_tokens.is_empty() && global_tokens.is_empty() && outputs.len() >= 2 {
                semantic_tokens = match outputs[0].try_extract_tensor::<i64>() {
                    Ok((_s_sem, semantic_tokens_slice)) => {
                        semantic_tokens_slice.iter().map(|&x| x as i32).collect()
                    }
                    Err(_) => {
                        let (_s_sem, semantic_tokens_slice) =
                            outputs[0].try_extract_tensor::<i32>()?;
                        semantic_tokens_slice.to_vec()
                    }
                };

                global_tokens = match outputs[1].try_extract_tensor::<i64>() {
                    Ok((_s_glb, global_tokens_slice)) => {
                        global_tokens_slice.iter().map(|&x| x as i32).collect()
                    }
                    Err(_) => {
                        let (_s_glb, global_tokens_slice) =
                            outputs[1].try_extract_tensor::<i32>()?;
                        global_tokens_slice.to_vec()
                    }
                };
            }

            Ok((global_tokens, semantic_tokens))
        })
        .await??;

        Ok(result)
    }

    /// 创建梅尔滤波器组
    #[allow(dead_code)]
    fn create_mel_filterbank_slaney_with_fmax(
        n_mels: usize,
        n_fft: usize,
        sample_rate: f32,
        fmin: f32,
        fmax: f32,
    ) -> Array2<f32> {
        let n_freqs = n_fft / 2 + 1;
        let mut filterbank = Array2::zeros((n_mels, n_freqs));

        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

        let mel_min = hz_to_mel(fmin);
        let mel_max = hz_to_mel(fmax);

        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + i as f32 * (mel_max - mel_min) / (n_mels + 1) as f32)
            .collect();

        let hz_points: Vec<f32> = mel_points.iter().map(|&mel| mel_to_hz(mel)).collect();
        let bin_points: Vec<f32> = hz_points
            .iter()
            .map(|&hz| hz * n_fft as f32 / sample_rate)
            .collect();

        for m in 1..=n_mels {
            let left = bin_points[m - 1];
            let center = bin_points[m];
            let right = bin_points[m + 1];

            for k in 0..n_freqs {
                let k_f = k as f32;
                if k_f >= left && k_f <= right {
                    if k_f <= center {
                        if center > left {
                            filterbank[[m - 1, k]] = (k_f - left) / (center - left);
                        }
                    } else if right > center {
                        filterbank[[m - 1, k]] = (right - k_f) / (right - center);
                    }
                }
            }

            // Slaney归一化：面积归一化为 2/(fhi-flo)
            let fhi = hz_points[m + 1];
            let flo = hz_points[m - 1];
            let norm_factor = 2.0 / (fhi - flo);
            for k in 0..n_freqs {
                filterbank[[m - 1, k]] *= norm_factor;
            }
        }

        filterbank
    }

    /// 计算功率谱
    #[allow(dead_code)]
    fn compute_power_spectrum(frame: &[f32]) -> Vec<f32> {
        let n_fft = frame.len();
        let n_freqs = n_fft / 2 + 1;
        let mut power_spectrum = vec![0.0f32; n_freqs];

        for (k, power) in power_spectrum.iter_mut().enumerate().take(n_freqs) {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;

            for (n, &sample) in frame.iter().enumerate().take(n_fft) {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / n_fft as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }

            *power = real * real + imag * imag;
        }

        power_spectrum
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

    /// 批量解码音频（CPU优化：减少会话获取开销）
    async fn decode_audio_batch(
        &self,
        batch_requests: &[(Vec<i32>, Vec<i32>)],
    ) -> Result<Vec<Vec<f32>>> {
        let onnx_manager = get_global_onnx_manager()?;
        let batch_size = batch_requests.len();
        let mut results = Vec::with_capacity(batch_size);

        // 批量获取会话，减少锁竞争
        let session_guards = onnx_manager
            .acquire_bicodec_detokenize_sessions_batch(batch_size)
            .await?;

        // 并行执行解码（使用CPU多核心）
        let mut tasks = Vec::with_capacity(batch_size);
        for ((global_tokens, semantic_tokens), session_guard) in
            batch_requests.iter().zip(session_guards.into_iter())
        {
            let global_tokens_clone = global_tokens.clone();
            let semantic_tokens_clone = semantic_tokens.clone();
            let mut session_guard_clone = session_guard;

            let task = tokio::task::spawn_blocking(move || {
                // 在阻塞线程中执行CPU密集型操作
                let global_shape: Vec<i64> =
                    [1i64, 1i64, global_tokens_clone.len() as i64].to_vec();
                let global_vec_i64: Vec<i64> =
                    global_tokens_clone.iter().map(|&x| x as i64).collect();
                let global_tensor = match Value::from_array((global_shape, global_vec_i64)) {
                    Ok(tensor) => tensor,
                    Err(e) => {
                        eprintln!("创建全局tensor失败: {}", e);
                        return vec![];
                    }
                };

                let semantic_shape: Vec<i64> = [1i64, semantic_tokens_clone.len() as i64].to_vec();
                let semantic_vec_i64: Vec<i64> =
                    semantic_tokens_clone.iter().map(|&x| x as i64).collect();
                let semantic_tensor = match Value::from_array((semantic_shape, semantic_vec_i64)) {
                    Ok(tensor) => tensor,
                    Err(e) => {
                        eprintln!("创建语义tensor失败: {}", e);
                        return vec![];
                    }
                };

                let outputs = match session_guard_clone.session_mut().run(ort::inputs![
                    "semantic_tokens" => SessionInputValue::from(semantic_tensor),
                    "global_tokens" => SessionInputValue::from(global_tensor)
                ]) {
                    Ok(outputs) => outputs,
                    Err(e) => {
                        eprintln!("ONNX推理失败: {}", e);
                        return vec![];
                    }
                };

                match outputs[0].try_extract_tensor::<f32>() {
                    Ok((_shape, audio_slice)) => audio_slice.to_vec(),
                    Err(e) => {
                        eprintln!("提取音频tensor失败: {}", e);
                        vec![]
                    }
                }
            });
            tasks.push(task);
        }

        // 等待所有任务完成
        for task in tasks {
            let audio_result = task
                .await
                .map_err(|e| anyhow::anyhow!("批处理解码任务失败: {}", e))?;
            results.push(audio_result);
        }

        Ok(results)
    }

    /// 使用ONNX会话进行音频解码
    async fn detokenize_audio_with_session(
        &self,
        global_tokens: &[i32],
        semantic_tokens: &[i32],
        mut session_guard: crate::onnx_session_pool::SessionGuard,
    ) -> Result<Vec<f32>> {
        // 优化：移除spawn_blocking，直接在异步上下文中执行
        // 直接转换为i64，减少中间步骤和内存分配
        let global_shape: Vec<i64> = [1i64, 1i64, global_tokens.len() as i64].to_vec();
        let global_vec_i64: Vec<i64> = global_tokens.iter().map(|&x| x as i64).collect();
        let global_tensor = Value::from_array((global_shape, global_vec_i64))?;

        let semantic_shape: Vec<i64> = [1i64, semantic_tokens.len() as i64].to_vec();
        let semantic_vec_i64: Vec<i64> = semantic_tokens.iter().map(|&x| x as i64).collect();
        let semantic_tensor = Value::from_array((semantic_shape, semantic_vec_i64))?;

        // 直接运行ONNX推理，避免spawn_blocking的开销
        let outputs = session_guard.session_mut().run(ort::inputs![
            "semantic_tokens" => SessionInputValue::from(semantic_tensor),
            "global_tokens" => SessionInputValue::from(global_tensor)
        ])?;

        let (_shape, audio_slice) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(audio_slice.to_vec())
    }

    /// 生成语音（使用批处理调度器）
    pub async fn generate_speech(&self, args: &LightweightTtsPipelineArgs) -> Result<Vec<f32>> {
        let total_start = std::time::Instant::now();

        // 1. 处理文本
        let text_start = std::time::Instant::now();
        let processed_text = if args.zero_shot {
            self.process_text_zero_shot(&args.text, &args.prompt_text)
        } else {
            self.process_text(&args.text)
        };
        let text_processing_time = text_start.elapsed();

        // 2. 处理属性tokens或参考音频
        let ref_start = std::time::Instant::now();
        let (property_tokens, ref_global_tokens, ref_semantic_tokens) =
            // 优先使用voice_id从缓存获取tokens
            if let Some(voice_id) = &args.voice_id {
                // 创建VoiceFeatureManager实例（假设使用默认RAF目录）
                let voice_manager = VoiceFeatureManager::new("./raf")?;
                match voice_manager.get_voice_tokens(voice_id).await {
                    Ok((global_tokens, semantic_tokens)) => {
                        (vec![], Some(global_tokens), Some(semantic_tokens))
                    }
                    Err(_) => {
                        // 回退到直接传入的tokens或其他方式
                        if let (Some(global_tokens), Some(semantic_tokens)) = (&args.voice_global_tokens, &args.voice_semantic_tokens) {
                            (vec![], Some(global_tokens.clone()), Some(semantic_tokens.clone()))
                        } else if args.zero_shot {
                            let (global, semantic) = self.process_reference_audio(&args.ref_audio_path).await?;
                            (vec![], Some(global), Some(semantic))
                        } else {
                            let tokens = self.generate_property_tokens(args);
                            (tokens, None, None)
                        }
                    }
                }
            }
            // 优先使用直接传入的音色特征tokens
            else if let (Some(global_tokens), Some(semantic_tokens)) = (&args.voice_global_tokens, &args.voice_semantic_tokens) {
                (vec![], Some(global_tokens.clone()), Some(semantic_tokens.clone()))
            } else if args.zero_shot {
                // 在zero-shot模式下，优化为一次性获取所有需要的信息
                // 直接使用传入的音色特征tokens（如果提供了的话）
                if let (Some(global_tokens), Some(semantic_tokens)) = (&args.voice_global_tokens, &args.voice_semantic_tokens) {
                    (vec![], Some(global_tokens.clone()), Some(semantic_tokens.clone()))
                } else {
                    // 处理参考音频文件
                    let (global, semantic) = self.process_reference_audio(&args.ref_audio_path).await?;
                    (vec![], Some(global), Some(semantic))
                }
            } else {
                let tokens = self.generate_property_tokens(args);
                println!("generate_property_tokens: {:?}", tokens);
                (tokens, None, None)
            };
        let reference_processing_time = ref_start.elapsed();

        // 3. 创建采样参数
        let sampler_args = SamplerArgs {
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            max_tokens: args.max_tokens,
            seed: args.seed,
            voice_fidelity: 0.8, // 默认音色保真度
            layered_randomness: crate::rwkv_sampler::LayeredRandomnessConfig::default(),
            token_chunk_size: 512, // 使用默认值
        };

        // 4. 创建批处理请求
        let request = TtsBatchRequest {
            text: processed_text,
            property_tokens,
            ref_global_tokens,
            ref_semantic_tokens,
            args: sampler_args,
            voice_id: args.voice_id.clone(),
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
                request.voice_id,
                request.args,
            )
            .await?;
        let inference_time = inference_start.elapsed();

        // 6. 解码音频
        if global_tokens.is_empty() && semantic_tokens.is_empty() {
            return Ok(vec![0.0; 16000]);
        }

        let decode_start = std::time::Instant::now();
        let audio = self.decode_audio(&global_tokens, &semantic_tokens).await?;
        let audio_decoding_time = decode_start.elapsed();

        let total_time = total_start.elapsed();
        let audio_duration = audio.len() as f64 / 16000.0; // 假设16kHz采样率
        let _rtf = total_time.as_secs_f64() / audio_duration;

        // 输出详细的耗时统计
        println!("⏱️  TTS生成详细耗时统计:");
        println!("  文本处理耗时: {:.2}ms", text_processing_time.as_millis());
        println!(
            "  参考音频处理耗时: {:.2}ms",
            reference_processing_time.as_millis()
        );
        println!("  RWKV推理耗时: {:.2}ms", inference_time.as_millis());
        println!("  音频解码耗时: {:.2}ms", audio_decoding_time.as_millis());
        println!("  总耗时: {:.2}ms", total_time.as_millis());

        Ok(audio)
    }

    /// 批量生成语音（CPU优化：支持批处理推理和音频解码）
    pub async fn generate_speech_batch(
        &self,
        batch_args: Vec<LightweightTtsPipelineArgs>,
    ) -> Result<Vec<Vec<f32>>> {
        let total_start = std::time::Instant::now();
        let batch_size = batch_args.len();

        // 1. 处理所有请求的文本和参考音频
        let mut batch_requests = Vec::with_capacity(batch_size);
        let mut processed_texts = Vec::with_capacity(batch_size);
        let mut ref_processing_results = Vec::with_capacity(batch_size);

        for args in &batch_args {
            // 处理文本
            let processed_text = if args.zero_shot {
                self.process_text_zero_shot(&args.text, &args.prompt_text)
            } else {
                self.process_text(&args.text)
            };
            processed_texts.push(processed_text);

            // 处理参考音频或属性tokens
            let ref_result = if let Some(voice_id) = &args.voice_id {
                let voice_manager = VoiceFeatureManager::new("./raf")?;
                match voice_manager.get_voice_tokens(voice_id).await {
                    Ok((global_tokens, semantic_tokens)) => {
                        (vec![], Some(global_tokens), Some(semantic_tokens))
                    }
                    Err(_) => {
                        if let (Some(global_tokens), Some(semantic_tokens)) =
                            (&args.voice_global_tokens, &args.voice_semantic_tokens)
                        {
                            (
                                vec![],
                                Some(global_tokens.clone()),
                                Some(semantic_tokens.clone()),
                            )
                        } else if args.zero_shot {
                            let (global, semantic) =
                                self.process_reference_audio(&args.ref_audio_path).await?;
                            (vec![], Some(global), Some(semantic))
                        } else {
                            let tokens = self.generate_property_tokens(args);
                            (tokens, None, None)
                        }
                    }
                }
            } else if let (Some(global_tokens), Some(semantic_tokens)) =
                (&args.voice_global_tokens, &args.voice_semantic_tokens)
            {
                (
                    vec![],
                    Some(global_tokens.clone()),
                    Some(semantic_tokens.clone()),
                )
            } else if args.zero_shot {
                if let (Some(global_tokens), Some(semantic_tokens)) =
                    (&args.voice_global_tokens, &args.voice_semantic_tokens)
                {
                    (
                        vec![],
                        Some(global_tokens.clone()),
                        Some(semantic_tokens.clone()),
                    )
                } else {
                    let (global, semantic) =
                        self.process_reference_audio(&args.ref_audio_path).await?;
                    (vec![], Some(global), Some(semantic))
                }
            } else {
                let tokens = self.generate_property_tokens(args);
                (tokens, None, None)
            };
            ref_processing_results.push(ref_result);
        }

        // 2. 创建批处理请求
        for (i, args) in batch_args.iter().enumerate() {
            let (property_tokens, ref_global_tokens, ref_semantic_tokens) =
                &ref_processing_results[i];

            let sampler_args = SamplerArgs {
                temperature: args.temperature,
                top_p: args.top_p,
                top_k: args.top_k,
                max_tokens: args.max_tokens,
                seed: args.seed,
                voice_fidelity: 0.8,
                layered_randomness: crate::rwkv_sampler::LayeredRandomnessConfig::default(),
                token_chunk_size: 512,
            };

            let request = TtsBatchRequest {
                text: processed_texts[i].clone(),
                property_tokens: property_tokens.clone(),
                ref_global_tokens: ref_global_tokens.clone(),
                ref_semantic_tokens: ref_semantic_tokens.clone(),
                args: sampler_args,
                voice_id: args.voice_id.clone(),
            };
            batch_requests.push(request);
        }

        // 3. 批量执行RWKV推理
        let manager = get_global_dynamic_batch_manager()?;
        let inference_results = manager.generate_tts_batch(batch_requests).await?;

        // 4. 批量解码音频
        let audio_results = self.decode_audio_batch(&inference_results).await?;

        let total_time = total_start.elapsed();
        println!(
            "⏱️  批量TTS生成完成: {}个请求, 总耗时: {:.2}ms",
            batch_size,
            total_time.as_millis()
        );

        Ok(audio_results)
    }

    /// 保存音频到文件（支持WAV和MP3格式）
    pub fn save_audio(
        &self,
        audio_samples: &[f32],
        output_path: &str,
        sample_rate: u32,
    ) -> Result<()> {
        use std::path::Path;

        #[cfg(debug_assertions)]
        {
            // 保存音频到指定路径
        }

        let path = Path::new(output_path);
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("wav")
            .to_lowercase();

        match extension.as_str() {
            "mp3" => self.save_audio_mp3(audio_samples, output_path, sample_rate),
            "wav" => self.save_audio_wav(audio_samples, output_path, sample_rate),
            _ => self.save_audio_wav(audio_samples, output_path, sample_rate),
        }
    }

    /// 保存音频到WAV文件
    fn save_audio_wav(
        &self,
        audio_samples: &[f32],
        output_path: &str,
        sample_rate: u32,
    ) -> Result<()> {
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

        #[cfg(debug_assertions)]
        {
            // WAV音频保存完成
        }
        Ok(())
    }

    /// 保存音频到MP3文件
    fn save_audio_mp3(
        &self,
        audio_samples: &[f32],
        output_path: &str,
        sample_rate: u32,
    ) -> Result<()> {
        use mp3lame_encoder::{Builder, FlushNoGap};
        use std::fs::File;
        use std::io::Write;

        // 转换f32样本到i16
        let i16_samples: Vec<i16> = audio_samples
            .iter()
            .map(|&sample| {
                let clamped = sample.clamp(-1.0, 1.0);
                (clamped * i16::MAX as f32) as i16
            })
            .collect();

        // 配置MP3编码器
        let mut builder = Builder::new().ok_or_else(|| anyhow::anyhow!("创建MP3编码器失败"))?;

        builder
            .set_num_channels(1)
            .map_err(|e| anyhow::anyhow!("设置声道数失败: {}", e))?;

        builder
            .set_sample_rate(sample_rate)
            .map_err(|e| anyhow::anyhow!("设置采样率失败: {}", e))?;

        builder
            .set_brate(mp3lame_encoder::Bitrate::Kbps128)
            .map_err(|e| anyhow::anyhow!("设置比特率失败: {}", e))?;

        builder
            .set_quality(mp3lame_encoder::Quality::Best)
            .map_err(|e| anyhow::anyhow!("设置质量失败: {}", e))?;

        let mut encoder = builder
            .build()
            .map_err(|e| anyhow::anyhow!("构建MP3编码器失败: {}", e))?;

        // 创建输出文件
        let mut output_file =
            File::create(output_path).map_err(|e| anyhow::anyhow!("创建输出文件失败: {}", e))?;

        // 编码音频数据
        use mp3lame_encoder::InterleavedPcm;
        use std::mem::MaybeUninit;

        let mut mp3_buffer: Vec<MaybeUninit<u8>> =
            vec![MaybeUninit::uninit(); i16_samples.len() * 2];
        let pcm_input = InterleavedPcm(&i16_samples);
        let encoded_size = encoder
            .encode(pcm_input, &mut mp3_buffer)
            .map_err(|e| anyhow::anyhow!("MP3编码失败: {}", e))?;

        // 安全地转换MaybeUninit<u8>到u8
        let encoded_data: Vec<u8> = mp3_buffer[..encoded_size]
            .iter()
            .map(|x| unsafe { x.assume_init() })
            .collect();

        output_file
            .write_all(&encoded_data)
            .map_err(|e| anyhow::anyhow!("写入MP3数据失败: {}", e))?;

        // 刷新编码器并写入剩余数据
        let mut flush_buffer: Vec<MaybeUninit<u8>> = vec![MaybeUninit::uninit(); 7200];
        let flush_size = encoder
            .flush::<FlushNoGap>(&mut flush_buffer)
            .map_err(|e| anyhow::anyhow!("刷新MP3编码器失败: {}", e))?;

        if flush_size > 0 {
            let flush_data: Vec<u8> = flush_buffer[..flush_size]
                .iter()
                .map(|x| unsafe { x.assume_init() })
                .collect();

            output_file
                .write_all(&flush_data)
                .map_err(|e| anyhow::anyhow!("写入MP3刷新数据失败: {}", e))?;
        }

        #[cfg(debug_assertions)]
        {
            // MP3音频保存完成
        }
        Ok(())
    }
}

impl Default for LightweightTtsPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl LightweightTtsPipeline {
    fn get_ref_clip(wav: &Array1<f32>) -> Array1<f32> {
        // 与Ref实现保持一致：长度 = (ref_segment_duration * sample_rate) // latent_hop_length * latent_hop_length
        let sample_rate: u32 = 16000;
        let ref_segment_duration: f32 = 6.0;
        let latent_hop_length: u32 = 320;

        let ref_segment_length = ((ref_segment_duration * sample_rate as f32) as u32
            / latent_hop_length
            * latent_hop_length) as usize;

        let wav_length = wav.len();
        if ref_segment_length > wav_length {
            // 如果音频不足指定长度，重复音频直到达到要求
            let repeat_times = ref_segment_length / wav_length + 1;
            let mut repeated = Vec::with_capacity(wav_length * repeat_times);
            for _ in 0..repeat_times {
                repeated.extend(wav.iter());
            }
            Array1::from(repeated)
                .slice(ndarray::s![..ref_segment_length])
                .to_owned()
        } else {
            // 截取指定长度
            wav.slice(ndarray::s![..ref_segment_length]).to_owned()
        }
    }

    #[allow(dead_code)]
    fn extract_mel_spectrogram_simple(wav: &Array1<f32>) -> Result<Array2<f32>> {
        // 参数与Ref实现一致
        let n_mels: usize = 128;
        let n_fft: usize = 1024;
        let hop_length: usize = 320;
        let win_length: usize = 1024;
        let sample_rate: f32 = 16000.0;

        // center=true 的填充
        let pad_width = n_fft / 2;
        let mut padded_wav = vec![0.0f32; wav.len() + 2 * pad_width];
        for (i, &sample) in wav.iter().enumerate() {
            padded_wav[pad_width + i] = sample;
        }

        let wav_len = padded_wav.len();
        let n_frames = if wav_len <= n_fft {
            1
        } else {
            (wav_len - n_fft) / hop_length + 1
        };

        // Hann窗
        let window: Vec<f32> = if win_length == n_fft {
            (0..n_fft)
                .map(|i| {
                    let angle = 2.0 * std::f32::consts::PI * i as f32 / (n_fft - 1) as f32;
                    0.5 * (1.0 - angle.cos())
                })
                .collect()
        } else {
            let mut window = vec![0.0f32; n_fft];
            let start_pad = (n_fft - win_length) / 2;
            for i in 0..win_length {
                let angle = 2.0 * std::f32::consts::PI * i as f32 / (win_length - 1) as f32;
                window[start_pad + i] = 0.5 * (1.0 - angle.cos());
            }
            window
        };

        // 修复：使用与C++和Python实现一致的参数
        // C++: melSpectrogram(ref_wav_samples, 16000, 1024, 320, 128, 10, 8000, 1.0f, true, false)
        // Python: extract_mel_spectrogram(wav, n_mels=128, n_fft=1024, hop_length=320, win_length=1024)
        // 注意：C++中fmin=10, fmax=8000, power=1.0, center=true, norm=false(slaney)
        let mel_filters =
            Self::create_mel_filterbank_slaney_with_fmax(n_mels, n_fft, sample_rate, 10.0, 8000.0);

        let mut mel_spectrogram = Array2::zeros((n_mels, n_frames));

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;
            let end = (start + n_fft).min(wav_len);

            // 提取帧并应用窗函数
            let mut frame = vec![0.0f32; n_fft];
            for i in 0..(end - start) {
                frame[i] = padded_wav[start + i] * window[i];
            }
            // 零填充剩余部分
            for item in frame.iter_mut().take(n_fft).skip(end - start) {
                *item = 0.0;
            }

            // 计算功率谱（简化版）
            let power_spectrum = Self::compute_power_spectrum(&frame);

            // 应用梅尔滤波器
            for mel_idx in 0..n_mels {
                let mut mel_energy = 0.0f32;
                for freq_idx in 0..power_spectrum.len() {
                    mel_energy += power_spectrum[freq_idx] * mel_filters[[mel_idx, freq_idx]];
                }
                mel_spectrogram[[mel_idx, frame_idx]] = mel_energy;
            }
        }

        Ok(mel_spectrogram)
    }
}

impl LightweightTtsPipeline {
    // 将可能带有统一偏移（如+8196）的codebook token安全归一到模型声码器所需的原始索引空间
    #[allow(dead_code)]
    fn normalize_codebook_offset(tokens: &[i32], offset: i32) -> Vec<i32> {
        if tokens.is_empty() {
            return vec![];
        }
        let min_v = tokens.iter().copied().min().unwrap_or(0);
        if min_v >= offset {
            tokens.iter().map(|&t| t - offset).collect()
        } else {
            tokens.to_vec()
        }
    }
}
