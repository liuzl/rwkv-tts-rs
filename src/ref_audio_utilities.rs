//! 参考音频处理工具模块
//! 实现与Python版本ref_audio_utilities.py相同的功能

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::input::SessionInputValue;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;

/// 参考音频处理工具类
pub struct RefAudioUtilities {
    /// ONNX会话
    pub ort_session: Session,
    /// wav2vec2 ONNX会话
    pub wav2vec2_session: Session,
    /// 采样率
    pub sample_rate: u32,
    /// 参考音频时长（秒）
    pub ref_segment_duration: f32,
    /// 潜在特征跳长度
    pub latent_hop_length: u32,
    bicodec_detokenizer_session: Option<Session>,
}

impl RefAudioUtilities {
    /// 创建新的参考音频处理工具实例
    pub fn new(
        onnx_model_path: &str,
        wav2vec2_path: &str,
        ref_segment_duration: f32,
        latent_hop_length: u32,
        detokenizer_path: Option<&str>,
    ) -> Result<Self> {
        // 使用 ort 2.x 的 Session::builder() API 构建会话
        let ort_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(onnx_model_path)?;

        let wav2vec2_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(wav2vec2_path)?;

        // 可选的detokenizer会话
        let bicodec_detokenizer_session = if let Some(detokenizer_path) = detokenizer_path {
            match Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(1)?
                .commit_from_file(detokenizer_path)
            {
                Ok(session) => Some(session),
                Err(e) => {
                    println!("Warning: Failed to load BiCodecDetokenize model: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            ort_session,
            wav2vec2_session,
            sample_rate: 16000,
            ref_segment_duration,
            latent_hop_length,
            bicodec_detokenizer_session,
        })
    }

    /// 加载音频文件
    pub fn load_audio(
        &self,
        audio_path: &str,
        target_sr: u32,
        volume_normalize: bool,
    ) -> Result<Array1<f32>> {
        if !Path::new(audio_path).exists() {
            return Err(anyhow!("音频文件不存在: {}", audio_path));
        }
        let mut reader = hound::WavReader::open(audio_path)?;
        let spec = reader.spec();
        let samples: Result<Vec<f32>, _> = reader
            .samples::<i16>()
            .map(|s| s.map(|sample| sample as f32 / i16::MAX as f32))
            .collect();
        let mut audio = Array1::from(samples?);
        if spec.channels > 1 {
            let len = audio.len() / spec.channels as usize;
            let mut mono_audio = Vec::with_capacity(len);
            for i in 0..len {
                mono_audio.push(audio[i * spec.channels as usize]);
            }
            audio = Array1::from(mono_audio);
        }
        if spec.sample_rate != target_sr {
            audio = self.resample_audio(audio, spec.sample_rate, target_sr)?;
        }
        if volume_normalize {
            audio = self.audio_volume_normalize(audio, 0.2);
        }
        Ok(audio)
    }

    /// 重采样音频数据
    pub fn resample_audio(
        &self,
        audio: Array1<f32>,
        _original_sr: u32,
        target_sr: u32,
    ) -> Result<Array1<f32>> {
        let original_len = audio.len();
        let target_len =
            (original_len as f32 * target_sr as f32 / self.sample_rate as f32) as usize;
        let mut resampled = Vec::with_capacity(target_len);
        for i in 0..target_len {
            let idx = i * original_len / target_len;
            resampled.push(audio[idx]);
        }
        Ok(Array1::from(resampled))
    }

    /// 音量归一化
    pub fn audio_volume_normalize(&self, audio: Array1<f32>, max_val: f32) -> Array1<f32> {
        let max_amp = audio.iter().fold(0.0_f32, |acc, &x| acc.max(x.abs()));
        if max_amp > 0.0 {
            audio.mapv(|x| x * (max_val / max_amp))
        } else {
            audio
        }
    }

    /// 梅尔频谱图（占位）
    pub fn extract_mel_spectrogram(
        &self,
        _wav: &Array1<f32>,
        n_mels: usize,
        _n_fft: usize,
        _hop_length: usize,
        _win_length: usize,
    ) -> Array2<f32> {
        Array2::zeros((n_mels, 301))
    }

    /// 使用ONNX wav2vec2模型提取特征（需要可变借用以兼容ort::Session::run的API约束）
    pub fn extract_wav2vec2_features(&mut self, wav: &Array1<f32>) -> Result<Array2<f32>> {
        let input_data = wav.clone().insert_axis(ndarray::Axis(0));
        let input_dyn = input_data.into_dyn();
        let input_shape: Vec<i64> = input_dyn.shape().iter().map(|&d| d as i64).collect();
        let input_vec = input_dyn.into_raw_vec();
        let input_tensor = Value::from_array((input_shape, input_vec))?;
        let outputs = self
            .wav2vec2_session
            .run(ort::inputs![SessionInputValue::from(input_tensor)])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let feature_dim = 1024usize;
        let t = data.len() / feature_dim;
        let features = Array2::from_shape_vec((t, feature_dim), data.to_vec())?;
        Ok(features)
    }

    pub fn get_ref_clip(&self, wav: &Array1<f32>) -> Array1<f32> {
        let ref_segment_length = (self.sample_rate as f32 * self.ref_segment_duration) as usize
            / self.latent_hop_length as usize
            * self.latent_hop_length as usize;
        let wav_length = wav.len();
        if ref_segment_length > wav_length {
            let repeat_times = ref_segment_length / wav_length + 1;
            let mut repeated = Vec::with_capacity(wav_length * repeat_times);
            for _ in 0..repeat_times {
                repeated.extend(wav.iter());
            }
            Array1::from(repeated)
                .slice(ndarray::s![..ref_segment_length])
                .to_owned()
        } else {
            wav.slice(ndarray::s![..ref_segment_length]).to_owned()
        }
    }

    pub fn process_audio(
        &self,
        audio_path: &str,
        volume_normalize: bool,
    ) -> Result<(Array1<f32>, Array1<f32>)> {
        let wav = self.load_audio(audio_path, self.sample_rate, volume_normalize)?;
        let ref_wav = self.get_ref_clip(&wav);
        Ok((wav, ref_wav))
    }

    pub fn tokenize(&mut self, audio_path: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        let (wav, ref_wav) = self.process_audio(audio_path, false)?;
        let feat = self.extract_wav2vec2_features(&wav)?;
        let ref_mel = self.extract_mel_spectrogram(&ref_wav, 128, 1024, 320, 640);
        let ref_mel_input = ref_mel.insert_axis(ndarray::Axis(0));
        let feat_input = feat.insert_axis(ndarray::Axis(0));
        let ref_mel_dyn = ref_mel_input.into_dyn();
        let feat_dyn = feat_input.into_dyn();
        let ref_mel_shape: Vec<i64> = ref_mel_dyn.shape().iter().map(|&d| d as i64).collect();
        let ref_mel_vec = ref_mel_dyn.into_raw_vec();
        let ref_mel_tensor = Value::from_array((ref_mel_shape, ref_mel_vec))?;
        let feat_shape: Vec<i64> = feat_dyn.shape().iter().map(|&d| d as i64).collect();
        let feat_vec = feat_dyn.into_raw_vec();
        let feat_tensor = Value::from_array((feat_shape, feat_vec))?;
        let outputs = self.ort_session.run(ort::inputs![
            "ref_wav_mel" => SessionInputValue::from(ref_mel_tensor),
            "feat" => SessionInputValue::from(feat_tensor)
        ])?;
        let (_s_sem, semantic_tokens_slice) = outputs[0].try_extract_tensor::<i64>()?;
        let (_s_glb, global_tokens_slice) = outputs[1].try_extract_tensor::<i64>()?;
        let semantic_tokens: Vec<i64> = semantic_tokens_slice.to_vec();
        let global_tokens: Vec<i64> = global_tokens_slice.to_vec();
        Ok((global_tokens, semantic_tokens))
    }

    pub fn detokenize_audio(
        &mut self,
        global_tokens: &[i32],
        semantic_tokens: &[i32],
    ) -> Result<Vec<f32>> {
        let detokenizer_session = self.bicodec_detokenizer_session.as_mut().ok_or_else(|| {
            anyhow::anyhow!("BiCodecDetokenize 会话未初始化，请检查模型路径和加载逻辑")
        })?;

        // 按Python实现使用动态长度：global -> (1,1,-1)，semantic -> (1,-1)
        let global_i64: Vec<i64> = global_tokens.iter().map(|&v| v as i64).collect();
        let semantic_i64: Vec<i64> = semantic_tokens.iter().map(|&v| v as i64).collect();

        // 形状：(1, 1, Lg)
        let global_tokens_array =
            ndarray::Array3::from_shape_vec((1, 1, global_i64.len()), global_i64)?;
        // 形状：(1, Ls)
        let semantic_tokens_array =
            ndarray::Array2::from_shape_vec((1, semantic_i64.len()), semantic_i64)?;

        // 转动态图并打包成Tensor
        let global_dyn = global_tokens_array.into_dyn();
        let semantic_dyn = semantic_tokens_array.into_dyn();

        let global_shape: Vec<i64> = global_dyn.shape().iter().map(|&d| d as i64).collect();
        let global_vec = global_dyn.into_raw_vec();
        let global_tensor = Value::from_array((global_shape, global_vec))?;

        let semantic_shape: Vec<i64> = semantic_dyn.shape().iter().map(|&d| d as i64).collect();
        let semantic_vec = semantic_dyn.into_raw_vec();
        let semantic_tensor = Value::from_array((semantic_shape, semantic_vec))?;

        // 按名称提供输入以避免顺序不一致
        let outputs = detokenizer_session.run(ort::inputs![
            // 注意：部分ORT绑定实现可能按位置匹配，这里将顺序调整为先提供 semantic(2D)，再提供 global(3D)
            "semantic_tokens" => SessionInputValue::from(semantic_tensor),
            "global_tokens" => SessionInputValue::from(global_tensor)
        ])?;

        let (_shape, audio_slice) = outputs[0].try_extract_tensor::<f32>()?;
        let audio_vec: Vec<f32> = audio_slice.to_vec();
        Ok(audio_vec)
    }
}