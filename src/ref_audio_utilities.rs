//! 参考音频处理工具模块
//! 实现与Python版本ref_audio_utilities.py相同的功能

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::input::SessionInputValue;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};

use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// 参考音频处理工具类
pub struct RefAudioUtilities {
    ort_session: Option<Session>,
    wav2vec2_session: Option<Session>,
    sample_rate: u32,
    #[allow(dead_code)]
    ref_segment_duration: f32,
    latent_hop_length: u32,
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
        // 测试模式：如果路径包含"dummy"，则跳过实际模型加载
        #[cfg(test)]
        if onnx_model_path.contains("dummy") || wav2vec2_path.contains("dummy") {
            return Ok(Self {
                ort_session: None,
                wav2vec2_session: None,
                sample_rate: 16000,
                ref_segment_duration,
                latent_hop_length,
                bicodec_detokenizer_session: None,
            });
        }

        // 根据CPU核心数动态设置线程数，提升性能
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4); // 默认4核

        // 设置合理的线程数：inter_threads控制并行操作数，intra_threads控制单个操作内的并行度
        let inter_threads = std::cmp::min(cpu_cores / 2, 4); // 最多4个inter线程
        let intra_threads = std::cmp::max(cpu_cores / 4, 2); // 至少2个intra线程

        // 创建会话构建器的辅助函数
        let create_session_builder = || -> Result<ort::session::builder::SessionBuilder> {
            let mut builder =
                Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;

            // 在非macOS系统上设置线程数以提升性能
            #[cfg(not(target_os = "macos"))]
            {
                builder = builder
                    .with_inter_threads(inter_threads)?
                    .with_intra_threads(intra_threads)?;
            }

            // macOS系统保持默认设置以避免mutex问题
            #[cfg(target_os = "macos")]
            {
                builder = builder.with_inter_threads(1)?;
            }

            Ok(builder)
        };

        // 使用 ort 2.x 的 Session::builder() API 构建会话
        let ort_session = create_session_builder()?.commit_from_file(onnx_model_path)?;

        let wav2vec2_session = create_session_builder()?.commit_from_file(wav2vec2_path)?;

        // 可选的detokenizer会话
        let bicodec_detokenizer_session = if let Some(detokenizer_path) = detokenizer_path {
            {
                let result = create_session_builder()?.commit_from_file(detokenizer_path);
                if let Err(_e) = &result {
                    #[cfg(debug_assertions)]
                    println!("Warning: Failed to load BiCodecDetokenize model: {}", _e);
                }
                result.ok()
            }
        } else {
            None
        };

        Ok(Self {
            ort_session: Some(ort_session),
            wav2vec2_session: Some(wav2vec2_session),
            sample_rate: 16000,
            ref_segment_duration,
            latent_hop_length,
            bicodec_detokenizer_session,
        })
    }

    /// 加载音频文件并进行预处理 - 支持WAV和MP3格式
    pub fn load_audio(
        &self,
        audio_path: &str,
        target_sr: u32,
        volume_normalize: bool,
    ) -> Result<Array1<f32>> {
        #[cfg(debug_assertions)]
        println!("[DEBUG] Loading audio file: {}", audio_path);

        if !Path::new(audio_path).exists() {
            return Err(anyhow!("音频文件不存在: {}", audio_path));
        }

        // 检查文件大小
        let metadata =
            std::fs::metadata(audio_path).map_err(|e| anyhow!("无法读取文件元数据: {}", e))?;
        if metadata.len() == 0 {
            return Err(anyhow!("音频文件为空: {}", audio_path));
        }
        if metadata.len() > 100 * 1024 * 1024 {
            // 100MB限制
            return Err(anyhow!("音频文件过大 (>100MB): {}", audio_path));
        }

        // 检查文件扩展名以确定格式
        let path = Path::new(audio_path);
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        let (audio_samples, sample_rate, channels) = match extension.as_str() {
            "mp3" => {
                // 使用symphonia解码MP3文件
                self.load_audio_with_symphonia(audio_path)?
            }
            "wav" => self.load_audio_with_hound(audio_path)?,
            _ => {
                // 使用hound解码WAV文件（默认处理）
                self.load_audio_with_hound(audio_path)?
            }
        };

        // 验证音频数据的合理性
        if audio_samples.is_empty() {
            return Err(anyhow!("音频文件不包含有效的音频数据"));
        }
        if audio_samples.len() < channels as usize {
            return Err(anyhow!("音频数据不完整：样本数少于声道数"));
        }

        let mut audio = Array1::from(audio_samples);

        // 多声道转单声道 - 与C++实现一致（取第一个通道）
        if channels > 1 {
            #[cfg(debug_assertions)]
            println!("[DEBUG] Converting {} channels to mono", channels);
            let len = audio.len() / channels as usize;
            let mut mono_audio = Vec::with_capacity(len);
            for i in 0..len {
                mono_audio.push(audio[i * channels as usize]);
            }
            audio = Array1::from(mono_audio);
        }

        // 验证转换后的音频数据
        if audio.is_empty() {
            return Err(anyhow!("音频处理后数据为空"));
        }

        // 检查音频数据的数值范围
        let max_val = audio.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
        if max_val > 10.0 {
            #[cfg(debug_assertions)]
            println!(
                "[WARNING] 音频数据可能未正确归一化，最大绝对值: {:.6}",
                max_val
            );
        }

        // 重采样到目标采样率 - 与C++的wav->resample(16000)保持一致
        if sample_rate != target_sr {
            #[cfg(debug_assertions)]
            println!(
                "[DEBUG] Resampling from {} Hz to {} Hz",
                sample_rate, target_sr
            );
            audio = self.resample_audio_high_quality(audio, sample_rate, target_sr)?;

            // 验证重采样结果
            if audio.is_empty() {
                return Err(anyhow!("重采样后音频数据为空"));
            }
        }

        // 音量归一化
        if volume_normalize {
            #[cfg(debug_assertions)]
            println!("[DEBUG] Applying volume normalization");
            audio = self.audio_volume_normalize(audio, 0.2);
        }

        #[cfg(debug_assertions)]
        println!("[DEBUG] Final audio length: {} samples", audio.len());
        Ok(audio)
    }

    /// 使用hound库加载WAV文件
    fn load_audio_with_hound(&self, audio_path: &str) -> Result<(Vec<f32>, u32, u16)> {
        let mut reader = hound::WavReader::open(audio_path).map_err(|e| {
            anyhow!(
                "无法打开WAV文件 '{}': {}\n提示：请确保文件未损坏且格式正确",
                audio_path,
                e
            )
        })?;
        let spec = reader.spec();

        #[cfg(debug_assertions)]
        println!(
            "[DEBUG] WAV spec: channels={}, sample_rate={}, bits_per_sample={}",
            spec.channels, spec.sample_rate, spec.bits_per_sample
        );

        // 验证音频格式
        if spec.bits_per_sample != 16 && spec.bits_per_sample != 24 && spec.bits_per_sample != 32 {
            return Err(anyhow!(
                "不支持的位深度: {} (支持16/24/32位)",
                spec.bits_per_sample
            ));
        }

        // 验证音频规格的合理性
        if spec.channels == 0 || spec.channels > 8 {
            return Err(anyhow!("不支持的声道数: {} (支持1-8声道)", spec.channels));
        }
        if spec.sample_rate == 0 || spec.sample_rate > 192000 {
            return Err(anyhow!(
                "不支持的采样率: {} Hz (支持1-192000 Hz)",
                spec.sample_rate
            ));
        }

        // 读取音频样本并转换为f32
        let samples: Result<Vec<f32>, _> = if spec.bits_per_sample == 16 {
            let samples: Result<Vec<i16>, _> = reader.samples().collect();
            Ok(samples?.into_iter().map(|s| s as f32 / 32768.0).collect())
        } else if spec.bits_per_sample == 24 {
            let samples: Result<Vec<i32>, _> = reader.samples().collect();
            Ok(samples?.into_iter().map(|s| s as f32 / 8388608.0).collect())
        } else if spec.bits_per_sample == 32 {
            match spec.sample_format {
                hound::SampleFormat::Float => {
                    let samples: Result<Vec<f32>, _> = reader.samples().collect();
                    samples
                }
                hound::SampleFormat::Int => {
                    let samples: Result<Vec<i32>, _> = reader.samples().collect();
                    Ok(samples?
                        .into_iter()
                        .map(|s| s as f32 / 2147483648.0)
                        .collect())
                }
            }
        } else {
            return Err(anyhow!("不支持的位深度: {}", spec.bits_per_sample));
        };

        let audio_samples = samples
            .map_err(|e| anyhow!("读取音频样本失败: {}\n提示：文件可能已损坏或格式不正确", e))?;

        Ok((audio_samples, spec.sample_rate, spec.channels))
    }

    /// 使用symphonia库加载MP3文件
    fn load_audio_with_symphonia(&self, audio_path: &str) -> Result<(Vec<f32>, u32, u16)> {
        // 打开文件
        let file = std::fs::File::open(audio_path)
            .map_err(|e| anyhow!("无法打开MP3文件 '{}': {}", audio_path, e))?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        // 创建格式提示
        let mut hint = Hint::new();
        hint.with_extension("mp3");

        // 探测格式
        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &fmt_opts, &meta_opts)
            .map_err(|e| anyhow!("无法探测MP3文件格式: {}", e))?;

        let mut format = probed.format;

        // 查找默认音频轨道
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| anyhow!("MP3文件中未找到音频轨道"))?;

        let track_id = track.id;
        let codec_params = &track.codec_params;

        // 获取音频参数
        let sample_rate = codec_params.sample_rate.unwrap_or(44100);
        let channels = codec_params.channels.map(|ch| ch.count()).unwrap_or(2) as u16;

        #[cfg(debug_assertions)]
        println!(
            "[DEBUG] MP3 spec: channels={}, sample_rate={}",
            channels, sample_rate
        );

        // 创建解码器
        let dec_opts: DecoderOptions = Default::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(codec_params, &dec_opts)
            .map_err(|e| anyhow!("无法创建MP3解码器: {}", e))?;

        // 解码音频数据
        let mut audio_samples = Vec::new();

        loop {
            // 获取下一个数据包
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(symphonia::core::errors::Error::ResetRequired) => {
                    // 解码器需要重置，但我们可以继续
                    continue;
                }
                Err(symphonia::core::errors::Error::IoError(ref e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    // 文件结束
                    break;
                }
                Err(e) => return Err(anyhow!("读取MP3数据包失败: {}", e)),
            };

            // 跳过非目标轨道的数据包
            if packet.track_id() != track_id {
                continue;
            }

            // 解码数据包
            match decoder.decode(&packet) {
                Ok(decoded) => {
                    // 转换音频缓冲区为f32样本
                    match decoded {
                        AudioBufferRef::F32(buf) => {
                            for &sample in buf.chan(0) {
                                audio_samples.push(sample);
                            }
                            // 如果是立体声，交错存储
                            if channels > 1 {
                                for ch in 1..channels as usize {
                                    if ch < buf.spec().channels.count() {
                                        for &sample in buf.chan(ch) {
                                            audio_samples.push(sample);
                                        }
                                    }
                                }
                            }
                        }
                        AudioBufferRef::U8(buf) => {
                            for &sample in buf.chan(0) {
                                audio_samples.push((sample as f32 - 128.0) / 128.0);
                            }
                            if channels > 1 {
                                for ch in 1..channels as usize {
                                    if ch < buf.spec().channels.count() {
                                        for &sample in buf.chan(ch) {
                                            audio_samples.push((sample as f32 - 128.0) / 128.0);
                                        }
                                    }
                                }
                            }
                        }
                        AudioBufferRef::U16(buf) => {
                            for &sample in buf.chan(0) {
                                audio_samples.push((sample as f32 - 32768.0) / 32768.0);
                            }
                            if channels > 1 {
                                for ch in 1..channels as usize {
                                    if ch < buf.spec().channels.count() {
                                        for &sample in buf.chan(ch) {
                                            audio_samples.push((sample as f32 - 32768.0) / 32768.0);
                                        }
                                    }
                                }
                            }
                        }
                        AudioBufferRef::U24(buf) => {
                            for &sample in buf.chan(0) {
                                let val = sample.inner() as f32;
                                audio_samples.push((val - 8388608.0) / 8388608.0);
                            }
                            if channels > 1 {
                                for ch in 1..channels as usize {
                                    if ch < buf.spec().channels.count() {
                                        for &sample in buf.chan(ch) {
                                            let val = sample.inner() as f32;
                                            audio_samples.push((val - 8388608.0) / 8388608.0);
                                        }
                                    }
                                }
                            }
                        }
                        AudioBufferRef::U32(buf) => {
                            for &sample in buf.chan(0) {
                                audio_samples.push((sample as f32 - 2147483648.0) / 2147483648.0);
                            }
                            if channels > 1 {
                                for ch in 1..channels as usize {
                                    if ch < buf.spec().channels.count() {
                                        for &sample in buf.chan(ch) {
                                            audio_samples.push(
                                                (sample as f32 - 2147483648.0) / 2147483648.0,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        AudioBufferRef::S8(buf) => {
                            for &sample in buf.chan(0) {
                                audio_samples.push(sample as f32 / 128.0);
                            }
                            if channels > 1 {
                                for ch in 1..channels as usize {
                                    if ch < buf.spec().channels.count() {
                                        for &sample in buf.chan(ch) {
                                            audio_samples.push(sample as f32 / 128.0);
                                        }
                                    }
                                }
                            }
                        }
                        AudioBufferRef::S16(buf) => {
                            for &sample in buf.chan(0) {
                                audio_samples.push(sample as f32 / 32768.0);
                            }
                            if channels > 1 {
                                for ch in 1..channels as usize {
                                    if ch < buf.spec().channels.count() {
                                        for &sample in buf.chan(ch) {
                                            audio_samples.push(sample as f32 / 32768.0);
                                        }
                                    }
                                }
                            }
                        }
                        AudioBufferRef::S24(buf) => {
                            for &sample in buf.chan(0) {
                                let val = sample.inner() as f32;
                                audio_samples.push(val / 8388608.0);
                            }
                            if channels > 1 {
                                for ch in 1..channels as usize {
                                    if ch < buf.spec().channels.count() {
                                        for &sample in buf.chan(ch) {
                                            let val = sample.inner() as f32;
                                            audio_samples.push(val / 8388608.0);
                                        }
                                    }
                                }
                            }
                        }
                        AudioBufferRef::S32(buf) => {
                            for &sample in buf.chan(0) {
                                audio_samples.push(sample as f32 / 2147483648.0);
                            }
                            if channels > 1 {
                                for ch in 1..channels as usize {
                                    if ch < buf.spec().channels.count() {
                                        for &sample in buf.chan(ch) {
                                            audio_samples.push(sample as f32 / 2147483648.0);
                                        }
                                    }
                                }
                            }
                        }
                        AudioBufferRef::F64(buf) => {
                            for &sample in buf.chan(0) {
                                audio_samples.push(sample as f32);
                            }
                            if channels > 1 {
                                for ch in 1..channels as usize {
                                    if ch < buf.spec().channels.count() {
                                        for &sample in buf.chan(ch) {
                                            audio_samples.push(sample as f32);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(symphonia::core::errors::Error::IoError(_)) => {
                    // 忽略IO错误，继续处理
                    continue;
                }
                Err(symphonia::core::errors::Error::DecodeError(_)) => {
                    // 忽略解码错误，继续处理
                    continue;
                }
                Err(_e) => {
                    #[cfg(debug_assertions)]
                    println!("[WARNING] MP3解码错误: {}", _e);
                    continue;
                }
            }
        }

        if audio_samples.is_empty() {
            return Err(anyhow!("MP3文件解码后无音频数据"));
        }

        Ok((audio_samples, sample_rate, channels))
    }

    /// 高质量重采样音频数据 - 与C++实现保持一致
    pub fn resample_audio_high_quality(
        &self,
        audio: Array1<f32>,
        original_sr: u32,
        target_sr: u32,
    ) -> Result<Array1<f32>> {
        if original_sr == target_sr {
            return Ok(audio);
        }

        let original_len = audio.len();
        let ratio = original_sr as f64 / target_sr as f64;
        let target_len = (original_len as f64 / ratio).round() as usize;

        // 使用Sinc插值进行高质量重采样（模拟专业音频库的行为）
        let mut resampled = Vec::with_capacity(target_len);
        let sinc_window_size = 8; // 窗口大小

        for i in 0..target_len {
            let src_idx = i as f64 * ratio;
            let center = src_idx.round() as isize;
            let mut sum = 0.0f32;
            let mut weight_sum = 0.0f32;

            // 应用Sinc插值窗口
            for j in -sinc_window_size..=sinc_window_size {
                let sample_idx = center + j;
                if sample_idx >= 0 && (sample_idx as usize) < original_len {
                    let x = src_idx - sample_idx as f64;
                    let weight = if x.abs() < f64::EPSILON {
                        1.0f32
                    } else {
                        let pi_x = std::f64::consts::PI * x;
                        let sinc = (pi_x.sin() / pi_x) as f32;
                        // 应用Hann窗口减少振铃效应
                        let hann = 0.5
                            * (1.0
                                + (2.0 * std::f64::consts::PI * x
                                    / (2.0 * sinc_window_size as f64))
                                    .cos()) as f32;
                        sinc * hann
                    };

                    sum += audio[sample_idx as usize] * weight;
                    weight_sum += weight;
                }
            }

            // 归一化权重
            let sample = if weight_sum > f32::EPSILON {
                sum / weight_sum
            } else {
                0.0
            };

            resampled.push(sample);
        }

        Ok(Array1::from(resampled))
    }

    /// 重采样音频数据（使用线性插值，保留以兼容）
    pub fn resample_audio(
        &self,
        audio: Array1<f32>,
        original_sr: u32,
        target_sr: u32,
    ) -> Result<Array1<f32>> {
        if original_sr == target_sr {
            return Ok(audio);
        }

        let original_len = audio.len();
        let ratio = original_sr as f64 / target_sr as f64;
        let target_len = (original_len as f64 / ratio).round() as usize;
        let mut resampled = Vec::with_capacity(target_len);

        for i in 0..target_len {
            let src_idx = i as f64 * ratio;
            let idx_floor = src_idx.floor() as usize;
            let idx_ceil = (idx_floor + 1).min(original_len - 1);
            let frac = src_idx - idx_floor as f64;

            if idx_floor >= original_len {
                resampled.push(0.0);
            } else if idx_floor == idx_ceil || frac < f64::EPSILON {
                resampled.push(audio[idx_floor]);
            } else {
                // 线性插值
                let val = audio[idx_floor] * (1.0 - frac as f32) + audio[idx_ceil] * frac as f32;
                resampled.push(val);
            }
        }

        Ok(Array1::from(resampled))
    }

    /// 音量归一化 - 与Python实现保持一致
    pub fn audio_volume_normalize(&self, audio: Array1<f32>, coeff: f32) -> Array1<f32> {
        let mut audio = audio;

        // 获取音频绝对值并排序
        let mut temp: Vec<f32> = audio.iter().map(|&x| x.abs()).collect();
        temp.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // 如果最大值小于0.1，缩放到0.1
        if temp[temp.len() - 1] < 0.1 {
            let scaling_factor = temp[temp.len() - 1].max(1e-3); // 防止除零
            audio = audio.mapv(|x| x / scaling_factor * 0.1);
        }

        // 过滤掉小于0.01的值
        temp.retain(|&x| x > 0.01);
        let l = temp.len();

        // 如果有效值少于等于10个，直接返回
        if l <= 10 {
            return audio;
        }

        // 计算90%到99%范围内的平均值
        let start_idx = (0.9 * l as f32) as usize;
        let end_idx = (0.99 * l as f32) as usize;
        let volume: f32 =
            temp[start_idx..end_idx].iter().sum::<f32>() / (end_idx - start_idx) as f32;

        // 归一化到目标系数水平，限制缩放因子在0.1到10之间
        let scale_factor = (coeff / volume).clamp(0.1, 10.0);
        audio = audio.mapv(|x| x * scale_factor);

        // 确保最大绝对值不超过1
        let max_value = audio.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        if max_value > 1.0 {
            audio = audio.mapv(|x| x / max_value);
        }

        #[cfg(debug_assertions)]
        println!("[DEBUG] Volume normalization: original_max={:.4}, volume={:.4}, scale_factor={:.4}, final_max={:.4}", 
                temp[temp.len() - 1], volume, scale_factor,
                audio.iter().fold(0.0f32, |acc, &x| acc.max(x.abs())));

        audio
    }

    /// 简单音量归一化（保留以兼容）
    pub fn audio_volume_normalize_simple(&self, audio: Array1<f32>, max_val: f32) -> Array1<f32> {
        let max_amp = audio.iter().fold(0.0_f32, |acc, &x| acc.max(x.abs()));
        if max_amp > 0.0 {
            audio.mapv(|x| x * (max_val / max_amp))
        } else {
            audio
        }
    }

    /// 零均值单位方差归一化 - 与C++实现完全一致
    /// C++实现：
    /// float mean = std::accumulate(input_values.begin(), input_values.end(), 0.0f) / input_values.size();
    /// float std = std::sqrt(std::accumulate(input_values.begin(), input_values.end(), 0.0f, [mean](float a, float b) {
    ///     return a + (b - mean) * (b - mean);
    /// }) / input_values.size() + 1e-7f);
    /// for (int i = 0; i < input_values.size(); i++) {
    ///     input_values[i] = (input_values[i] - mean) / std;
    /// }
    pub fn zero_mean_unit_variance_normalize(mut input_values: Vec<f32>) -> Vec<f32> {
        // 计算均值 - 与C++完全一致
        let mean = input_values.iter().sum::<f32>() / input_values.len() as f32;

        // 计算标准差 - 与C++实现完全一致
        let variance_sum = input_values
            .iter()
            .fold(0.0f32, |acc, &b| acc + (b - mean) * (b - mean));
        let std = (variance_sum / input_values.len() as f32 + 1e-7f32).sqrt();

        // 归一化 - 与C++完全一致
        for value in input_values.iter_mut() {
            *value = (*value - mean) / std;
        }

        #[cfg(debug_assertions)]
        println!(
            "[DEBUG] Zero-mean unit-variance normalize: mean={:.6}, std={:.6}, size={}",
            mean,
            std,
            input_values.len()
        );

        input_values
    }

    /// 梅尔频谱图提取 - 与Python librosa实现保持一致
    pub fn extract_mel_spectrogram(
        &self,
        wav: &Array1<f32>,
        n_mels: usize,
        n_fft: usize,
        hop_length: usize,
        win_length: usize,
    ) -> Array2<f32> {
        // C++代码中center=true，需要进行中心填充
        // 填充长度为n_fft//2，两端各填充n_fft//2个零
        let pad_width = n_fft / 2;
        let mut padded_wav = vec![0.0f32; wav.len() + 2 * pad_width];

        // 复制原始音频到中间位置
        for (i, &sample) in wav.iter().enumerate() {
            padded_wav[pad_width + i] = sample;
        }

        let wav_len = padded_wav.len();
        #[cfg(debug_assertions)]
        println!(
            "[DEBUG] After center padding: original_len={}, padded_len={}, pad_width={}",
            wav.len(),
            wav_len,
            pad_width
        );

        // 使用与librosa相同的帧数计算方式（基于填充后的长度）
        let n_frames = if wav_len <= n_fft {
            1
        } else {
            (wav_len - n_fft) / hop_length + 1
        };

        #[cfg(debug_assertions)]
        println!(
            "[DEBUG] Mel spectrogram extraction: wav_len={}, n_fft={}, hop_length={}, n_frames={}",
            wav_len, n_fft, hop_length, n_frames
        );

        // 创建汉宁窗 - 与librosa默认窗口一致
        let window: Vec<f32> = if win_length == n_fft {
            (0..n_fft)
                .map(|i| {
                    let angle = 2.0 * std::f32::consts::PI * i as f32 / (n_fft - 1) as f32;
                    0.5 * (1.0 - angle.cos())
                })
                .collect()
        } else {
            // 如果win_length != n_fft，需要进行窗口填充
            let mut window = vec![0.0f32; n_fft];
            let start_pad = (n_fft - win_length) / 2;
            for i in 0..win_length {
                let angle = 2.0 * std::f32::consts::PI * i as f32 / (win_length - 1) as f32;
                window[start_pad + i] = 0.5 * (1.0 - angle.cos());
            }
            window
        };

        // 创建梅尔滤波器组 - 使用slaney归一化，fmin=10，fmax=8000（与C++保持一致）
        let mel_filters = self.create_mel_filterbank_slaney_with_fmax(
            n_mels,
            n_fft,
            self.sample_rate as f32,
            10.0,
            8000.0,
        );

        let mut mel_spectrogram = Array2::zeros((n_mels, n_frames));

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;
            let end = (start + n_fft).min(wav_len);

            // 提取帧并应用窗函数（使用填充后的音频）
            let mut frame = vec![0.0f32; n_fft];
            for i in 0..(end - start) {
                frame[i] = padded_wav[start + i] * window[i];
            }
            // 零填充剩余部分
            for item in frame.iter_mut().take(n_fft).skip(end - start) {
                *item = 0.0;
            }

            // 计算功率谱 - 使用更精确的FFT实现
            let power_spectrum = self.compute_power_spectrum_accurate(&frame);

            // 应用梅尔滤波器
            for mel_idx in 0..n_mels {
                let mut mel_energy = 0.0f32;
                for freq_idx in 0..power_spectrum.len() {
                    mel_energy += power_spectrum[freq_idx] * mel_filters[[mel_idx, freq_idx]];
                }
                // 不进行对数变换，使用线性尺度（与C++的melSpectrogram函数一致）
                mel_spectrogram[[mel_idx, frame_idx]] = mel_energy;
            }
        }

        #[cfg(debug_assertions)]
        println!(
            "[DEBUG] Mel spectrogram shape: [{}, {}]",
            mel_spectrogram.nrows(),
            mel_spectrogram.ncols()
        );
        mel_spectrogram
    }

    /// 创建梅尔滤波器组 - 使用slaney归一化，支持指定fmax
    fn create_mel_filterbank_slaney_with_fmax(
        &self,
        n_mels: usize,
        n_fft: usize,
        sample_rate: f32,
        fmin: f32,
        fmax: f32,
    ) -> Array2<f32> {
        let n_freqs = n_fft / 2 + 1;
        let mut filterbank = Array2::zeros((n_mels, n_freqs));

        // 梅尔刻度转换函数
        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

        let mel_min = hz_to_mel(fmin);
        let mel_max = hz_to_mel(fmax);

        // 创建梅尔刻度上的等间距点
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + i as f32 * (mel_max - mel_min) / (n_mels + 1) as f32)
            .collect();

        // 转换回Hz并映射到FFT bin
        let hz_points: Vec<f32> = mel_points.iter().map(|&mel| mel_to_hz(mel)).collect();
        let bin_points: Vec<f32> = hz_points
            .iter()
            .map(|&hz| hz * n_fft as f32 / sample_rate)
            .collect();

        // 构建三角滤波器
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

            // Slaney归一化：每个滤波器的面积归一化为2/(fhi-flo)
            let fhi = hz_points[m + 1];
            let flo = hz_points[m - 1];
            let norm_factor = 2.0 / (fhi - flo);
            for k in 0..n_freqs {
                filterbank[[m - 1, k]] *= norm_factor;
            }
        }

        filterbank
    }

    /// 创建梅尔滤波器组 - 使用slaney归一化（保留原方法以兼容）
    #[allow(dead_code)]
    fn create_mel_filterbank_slaney(
        &self,
        n_mels: usize,
        n_fft: usize,
        sample_rate: f32,
        fmin: f32,
    ) -> Array2<f32> {
        let fmax = sample_rate / 2.0;
        self.create_mel_filterbank_slaney_with_fmax(n_mels, n_fft, sample_rate, fmin, fmax)
    }

    /// 创建梅尔滤波器组（保留原方法以兼容）
    #[allow(dead_code)]
    fn create_mel_filterbank(&self, n_mels: usize, n_fft: usize, sample_rate: f32) -> Array2<f32> {
        self.create_mel_filterbank_slaney(n_mels, n_fft, sample_rate, 0.0)
    }

    /// 计算功率谱 - 更精确的实现
    fn compute_power_spectrum_accurate(&self, frame: &[f32]) -> Vec<f32> {
        let n_fft = frame.len();
        let n_freqs = n_fft / 2 + 1;
        let mut power_spectrum = vec![0.0f32; n_freqs];

        // 使用更精确的DFT计算，包含适当的归一化
        for (k, power) in power_spectrum.iter_mut().enumerate().take(n_freqs) {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;

            for (n, &sample) in frame.iter().enumerate().take(n_fft) {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / n_fft as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }

            // 计算幅度谱的平方（power=1对应幅度谱）
            let magnitude = (real * real + imag * imag).sqrt();
            *power = magnitude;
        }

        power_spectrum
    }

    /// 计算功率谱（简化实现，保留以兼容）
    #[allow(dead_code)]
    fn compute_power_spectrum(&self, frame: &[f32]) -> Vec<f32> {
        let n_fft = frame.len();
        let n_freqs = n_fft / 2 + 1;
        let mut power_spectrum = vec![0.0f32; n_freqs];

        // 简化的DFT计算（仅计算所需频率）
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

    /// 使用ONNX wav2vec2模型提取特征（需要可变借用以兼容ort::Session::run的API约束）
    pub fn extract_wav2vec2_features(&mut self, audio_data: &[f32]) -> Result<Array2<f32>> {
        let wav2vec2_session = self
            .wav2vec2_session
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("wav2vec2 session not initialized"))?;

        // 应用零均值单位方差归一化预处理 - 与C++实现保持一致
        let normalized_audio = Self::zero_mean_unit_variance_normalize(audio_data.to_vec());

        let input_data = Array1::from(normalized_audio).insert_axis(ndarray::Axis(0));
        let input_dyn = input_data.into_dyn();
        let input_shape: Vec<i64> = input_dyn.shape().iter().map(|&d| d as i64).collect();
        let input_vec = input_dyn.into_raw_vec();

        println!("[DEBUG] wav2vec2 input shape: {:?}", input_shape);

        let input_tensor = Value::from_array((input_shape, input_vec))?;

        let outputs = wav2vec2_session.run(ort::inputs![SessionInputValue::from(input_tensor)])?;
        let (output_shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        println!(
            "[DEBUG] wav2vec2 output shape: {:?}, data length: {}",
            output_shape,
            data.len()
        );

        // 与Python版本保持一致：输出形状应该是 [1, time_steps, 1024]
        // Python: features = outputs[0][0]  # 移除batch维度，得到 [time_steps, 1024]
        if output_shape.len() == 3 && output_shape[0] == 1 {
            let time_steps = output_shape[1] as usize;
            let feature_dim = output_shape[2] as usize;

            if feature_dim != 1024 {
                return Err(anyhow::anyhow!(
                    "Expected feature dimension 1024, got {}",
                    feature_dim
                ));
            }

            // 移除batch维度，与Python版本一致
            let features = Array2::from_shape_vec((time_steps, feature_dim), data.to_vec())?;
            println!(
                "[DEBUG] wav2vec2 features after removing batch dim: {:?}",
                features.shape()
            );
            Ok(features)
        } else {
            Err(anyhow::anyhow!(
                "Unexpected wav2vec2 output shape: {:?}",
                output_shape
            ))
        }
    }

    pub fn get_ref_clip(&self, wav: &Array1<f32>) -> Array1<f32> {
        // 使用与C++和Python版本完全一致的计算方式
        // C++: ref_segment_duration * sample_rate / latent_hop_length * latent_hop_length
        // Python: ref_segment_duration * sample_rate // latent_hop_length * latent_hop_length
        let ref_segment_length = ((self.ref_segment_duration * self.sample_rate as f32) as u32
            / self.latent_hop_length
            * self.latent_hop_length) as usize;

        println!(
            "[DEBUG] get_ref_clip - ref_segment_duration: {}, sample_rate: {}, latent_hop_length: {}, calculated length: {}",
            self.ref_segment_duration, self.sample_rate, self.latent_hop_length, ref_segment_length
        );

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

    /// 确保音频长度的一致性处理
    /// 确保音频长度的一致性处理
    #[allow(dead_code)]
    fn ensure_consistent_length(&self, audio: Array1<f32>) -> Array1<f32> {
        let len = audio.len();
        // 确保长度是hop_length的倍数，以保证特征提取的一致性
        let hop_length = self.latent_hop_length as usize;
        let aligned_len = (len / hop_length) * hop_length;

        if aligned_len == 0 {
            // 如果音频太短，填充到最小长度
            let min_len = hop_length;
            let mut padded = vec![0.0f32; min_len];
            let copy_len = len.min(min_len);
            padded[..copy_len].copy_from_slice(&audio.as_slice().unwrap()[..copy_len]);
            Array1::from(padded)
        } else if aligned_len < len {
            // 截断到对齐长度
            audio.slice(ndarray::s![..aligned_len]).to_owned()
        } else {
            audio
        }
    }

    pub fn process_audio(
        &mut self,
        audio_path: &str,
        volume_normalize: bool,
    ) -> Result<(Array1<f32>, Array1<f32>)> {
        let wav = self.load_audio(audio_path, self.sample_rate, volume_normalize)?;
        let ref_wav = self.get_ref_clip(&wav);
        Ok((wav, ref_wav))
    }

    pub fn tokenize(&mut self, audio_path: &str) -> Result<(Vec<i32>, Vec<i32>)> {
        println!("[DEBUG] Tokenizing audio: {}", audio_path);

        // 确定性音频预处理：启用音量归一化以确保一致性（修复潜在噪音问题）
        let (wav, ref_wav) = self.process_audio(audio_path, true)?;
        println!(
            "[DEBUG] Audio loaded - wav length: {}, ref_wav length: {}",
            wav.len(),
            ref_wav.len()
        );
        println!(
            "[DEBUG] Wav stats - min: {:.6}, max: {:.6}, mean: {:.6}",
            wav.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            wav.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
            wav.iter().sum::<f32>() / wav.len() as f32
        );
        println!(
            "[DEBUG] Ref wav stats - min: {:.6}, max: {:.6}, mean: {:.6}",
            ref_wav.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            ref_wav.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
            ref_wav.iter().sum::<f32>() / ref_wav.len() as f32
        );

        // 注意：不对wav进行长度对齐处理，保持与Python版本一致
        // Python版本没有ensure_consistent_length调用，直接使用原始音频长度
        // let wav = self.ensure_consistent_length(wav);
        println!("[DEBUG] Using original wav length: {}", wav.len());

        let feat = self.extract_wav2vec2_features(wav.as_slice().unwrap())?;
        println!(
            "[DEBUG] Wav2vec2 features extracted - shape: {:?}",
            feat.shape()
        );
        println!(
            "[DEBUG] Feat stats - min: {:.6}, max: {:.6}, mean: {:.6}",
            feat.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            feat.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
            feat.iter().sum::<f32>() / feat.len() as f32
        );

        // 使用与C++完全相同的梅尔频谱图参数
        // C++: melSpectrogram(ref_wav_samples, 16000, 1024, 320, 128, 10, 8000, 1.0f, true, false)
        // 对应参数: (audio, sample_rate, n_fft, hop_length, n_mels, fmin, fmax, power, center, norm)
        // 注意：C++中win_length默认等于n_fft，fmax=8000，power=1.0，center=true，norm=false(slaney)
        let ref_mel = self.extract_mel_spectrogram(&ref_wav, 128, 1024, 320, 1024); // n_mels=128, n_fft=1024, hop_length=320, win_length=1024
        println!(
            "[DEBUG] Mel spectrogram extracted - shape: {:?}",
            ref_mel.shape()
        );
        println!(
            "[DEBUG] Mel stats - min: {:.6}, max: {:.6}, mean: {:.6}",
            ref_mel.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            ref_mel.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
            ref_mel.iter().sum::<f32>() / ref_mel.len() as f32
        );

        // 准备ref_mel张量：形状应该是[1, 128, 301]
        // 关键修复：确保数据布局与C++一致
        // C++中使用memcpy按行复制：memcpy(nchw_tensor_mel->host<float>() + i * 301, ref_wav_mel[i].data(), ref_wav_mel[i].size() * sizeof(float))
        // 这意味着C++期望的是行优先布局，即每行301个元素连续存储

        // 检查当前mel_spectrogram的形状和布局
        println!(
            "[DEBUG] Original mel_spectrogram shape: {:?}, layout: {:?}",
            ref_mel.shape(),
            ref_mel.strides()
        );

        // 确保数据是行优先布局（C-order）
        let ref_mel_c_order = if ref_mel.is_standard_layout() {
            ref_mel.clone()
        } else {
            // 如果不是标准布局，转换为C-order
            println!("[DEBUG] Converting mel_spectrogram to C-order layout");
            ref_mel.as_standard_layout().to_owned()
        };

        let ref_mel_input = ref_mel_c_order.insert_axis(ndarray::Axis(0));
        let ref_mel_dyn = ref_mel_input.into_dyn();
        let ref_mel_shape: Vec<i64> = ref_mel_dyn.shape().iter().map(|&d| d as i64).collect();
        let ref_mel_vec = ref_mel_dyn.into_raw_vec();

        // 验证数据布局：打印前几行的前几个元素
        println!("[DEBUG] ref_mel tensor data layout verification:");
        for row in 0..3.min(ref_mel_shape[1] as usize) {
            let start_idx = row * ref_mel_shape[2] as usize;
            let end_idx = (start_idx + 5).min(ref_mel_vec.len());
            println!(
                "[DEBUG] Row {}: {:?}",
                row,
                &ref_mel_vec[start_idx..end_idx]
            );
        }

        let ref_mel_tensor = Value::from_array((ref_mel_shape.clone(), ref_mel_vec))?;

        // 准备feat张量：形状应该是[1, t, 1024]，与C++完全一致
        // C++: input_shape_feat = {1, static_cast<int>(wav2vec2_features.size() / 1024), 1024}
        // C++: memcpy(nchw_tensor_feat->host<float>(), wav2vec2_features.data(), wav2vec2_features.size() * sizeof(float))

        // 检查feat张量的形状和布局
        println!(
            "[DEBUG] Original feat shape: {:?}, layout: {:?}",
            feat.shape(),
            feat.strides()
        );

        // 确保feat数据是行优先布局（C-order）
        let feat_c_order = if feat.is_standard_layout() {
            feat.clone()
        } else {
            println!("[DEBUG] Converting feat to C-order layout");
            feat.as_standard_layout().to_owned()
        };

        let feat_input = feat_c_order.insert_axis(ndarray::Axis(0));
        let feat_dyn = feat_input.into_dyn();
        let feat_shape: Vec<i64> = feat_dyn.shape().iter().map(|&d| d as i64).collect();
        let feat_vec = feat_dyn.into_raw_vec();

        #[cfg(debug_assertions)]
        {
            // 验证feat数据布局：打印前几行的前几个元素
            println!("[DEBUG] feat tensor data layout verification:");
            for row in 0..3.min(feat_shape[1] as usize) {
                let start_idx = row * feat_shape[2] as usize;
                let end_idx = (start_idx + 5).min(feat_vec.len());
                println!("[DEBUG] Row {}: {:?}", row, &feat_vec[start_idx..end_idx]);
            }

            // 验证张量形状与C++一致
            println!("[DEBUG] Tensor shapes verification:");
            println!(
                "[DEBUG] - ref_mel: {:?} (expected: [1, 128, 301])",
                ref_mel_shape
            );
            println!("[DEBUG] - feat: {:?} (expected: [1, t, 1024])", feat_shape);

            // 验证feat张量的数据统计（在创建张量之前）
            println!(
                "[DEBUG] Feat tensor data stats - min: {:.6}, max: {:.6}, mean: {:.6}",
                feat_vec.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                feat_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                feat_vec.iter().sum::<f32>() / feat_vec.len() as f32
            );
        }

        let feat_tensor = Value::from_array((feat_shape.clone(), feat_vec))?;

        #[cfg(debug_assertions)]
        println!(
            "[DEBUG] Input tensors prepared - ref_mel shape: {:?}, feat shape: {:?}",
            ref_mel_shape, feat_shape
        );

        let ort_session = self
            .ort_session
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("ort session not initialized"))?;

        let outputs = ort_session.run(ort::inputs![
            "ref_wav_mel" => SessionInputValue::from(ref_mel_tensor),
            "feat" => SessionInputValue::from(feat_tensor)
        ])?;

        #[cfg(debug_assertions)]
        {
            println!("[DEBUG] Model inference completed, processing outputs...");
            println!("[DEBUG] Number of outputs: {}", outputs.len());

            // 检查输出张量的形状和类型
            for (i, (_name, output)) in outputs.iter().enumerate() {
                println!(
                    "[DEBUG] Output {}: shape = {:?}, data_type = {:?}",
                    i,
                    output.shape(),
                    output.data_type()
                );
            }
        }

        // 修复输出解析顺序：与Python和C++实现保持一致
        // Python: semantic_tokens = outputs[0], global_tokens = outputs[1]
        // C++: semantic_tokens = output_tensors["semantic_tokens"], global_tokens = output_tensors["global_tokens"]

        // 先尝试按Python/C++顺序解析（semantic_tokens在前，global_tokens在后）
        // 修复：根据实际输出名称来解析，确保顺序正确
        let mut semantic_tokens: Vec<i32> = vec![];
        let mut global_tokens: Vec<i32> = vec![];

        // 检查输出名称来确定正确的解析顺序
        for (i, (name, output)) in outputs.iter().enumerate() {
            #[cfg(debug_assertions)]
            println!(
                "[DEBUG] Processing output {}: name = {:?}, shape = {:?}",
                i,
                name,
                output.shape()
            );

            if name == "semantic_tokens" || i == 0 {
                semantic_tokens = match output.try_extract_tensor::<i64>() {
                    Ok((_s_sem, semantic_tokens_slice)) => {
                        semantic_tokens_slice.iter().map(|&x| x as i32).collect()
                    }
                    Err(_) => {
                        let (_s_sem, semantic_tokens_slice) = output.try_extract_tensor::<i32>()?;
                        semantic_tokens_slice.to_vec()
                    }
                };
                #[cfg(debug_assertions)]
                println!(
                    "[DEBUG] Extracted semantic_tokens: length = {}",
                    semantic_tokens.len()
                );
            } else if name == "global_tokens" || i == 1 {
                global_tokens = match output.try_extract_tensor::<i64>() {
                    Ok((_s_glb, global_tokens_slice)) => {
                        global_tokens_slice.iter().map(|&x| x as i32).collect()
                    }
                    Err(_) => {
                        let (_s_glb, global_tokens_slice) = output.try_extract_tensor::<i32>()?;
                        global_tokens_slice.to_vec()
                    }
                };
                #[cfg(debug_assertions)]
                println!(
                    "[DEBUG] Extracted global_tokens: length = {}",
                    global_tokens.len()
                );
            }
        }

        // 如果按名称没有找到，使用索引方式作为备选
        if semantic_tokens.is_empty() && global_tokens.is_empty() && outputs.len() >= 2 {
            #[cfg(debug_assertions)]
            println!("[DEBUG] Falling back to index-based parsing");
            semantic_tokens = match outputs[0].try_extract_tensor::<i64>() {
                Ok((_s_sem, semantic_tokens_slice)) => {
                    semantic_tokens_slice.iter().map(|&x| x as i32).collect()
                }
                Err(_) => {
                    let (_s_sem, semantic_tokens_slice) = outputs[0].try_extract_tensor::<i32>()?;
                    semantic_tokens_slice.to_vec()
                }
            };

            global_tokens = match outputs[1].try_extract_tensor::<i64>() {
                Ok((_s_glb, global_tokens_slice)) => {
                    global_tokens_slice.iter().map(|&x| x as i32).collect()
                }
                Err(_) => {
                    let (_s_glb, global_tokens_slice) = outputs[1].try_extract_tensor::<i32>()?;
                    global_tokens_slice.to_vec()
                }
            };
        }

        #[cfg(debug_assertions)]
        {
            // 统计global_tokens的唯一值
            let unique_values: std::collections::HashSet<i32> =
                global_tokens.iter().cloned().collect();
            println!(
                "[DEBUG] Global tokens unique values count: {}, values: {:?}",
                unique_values.len(),
                unique_values.iter().take(10).collect::<Vec<_>>()
            );
            println!(
                "[DEBUG] Global tokens raw data (first 10): {:?}",
                &global_tokens[..global_tokens.len().min(10)]
            );
        }

        #[cfg(debug_assertions)]
        {
            println!(
                "[DEBUG] Tokenization completed - global tokens: {:?}, semantic tokens length: {}",
                global_tokens,
                semantic_tokens.len()
            );
            if !semantic_tokens.is_empty() {
                println!(
                    "[DEBUG] Semantic tokens sample (first 10): {:?}",
                    &semantic_tokens[..semantic_tokens.len().min(10)]
                );
            }

            // 添加tokens范围检查
            let mut out_of_range_global = 0;
            for &token in &global_tokens {
                if !(0..4096).contains(&token) {
                    out_of_range_global += 1;
                }
            }
            let mut out_of_range_semantic = 0;
            for &token in &semantic_tokens {
                if !(0..2048).contains(&token) {
                    out_of_range_semantic += 1;
                }
            }
            println!(
                "[DEBUG] Token range check: out_of_range_global={}, out_of_range_semantic={}",
                out_of_range_global, out_of_range_semantic
            );
            if out_of_range_global > 0 || out_of_range_semantic > 0 {
                println!("[WARNING] Invalid tokens detected in reference audio processing");
            }
        }
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

        // 优化：直接转换为i64，避免中间的i32向量拷贝
        let global_len = global_tokens.len();
        let semantic_len = semantic_tokens.len();

        // 预分配容量以避免重复分配
        let mut global_vec_i64 = Vec::with_capacity(global_len);
        let mut semantic_vec_i64 = Vec::with_capacity(semantic_len);

        // 直接转换，避免中间拷贝
        global_vec_i64.extend(global_tokens.iter().map(|&x| x as i64));
        semantic_vec_i64.extend(semantic_tokens.iter().map(|&x| x as i64));

        // 直接构建tensor，避免ndarray的中间步骤
        let global_shape = vec![1i64, 1i64, global_len as i64];
        let semantic_shape = vec![1i64, semantic_len as i64];

        let global_tensor = Value::from_array((global_shape, global_vec_i64))?;
        let semantic_tensor = Value::from_array((semantic_shape, semantic_vec_i64))?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// 创建测试用的音频数据
    fn create_test_audio_data(sample_rate: u32, duration_secs: f32) -> Vec<f32> {
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let mut audio = Vec::with_capacity(num_samples);

        // 生成确定性的正弦波信号
        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let freq = 440.0; // A4音符
            let sample = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5;
            audio.push(sample);
        }

        audio
    }

    /// 将音频数据写入临时WAV文件
    #[allow(dead_code)]
    fn write_test_wav_file(audio_data: &[f32], sample_rate: u32) -> Result<NamedTempFile> {
        let mut temp_file = NamedTempFile::new()?;

        // 简化的WAV文件头（44字节）
        let num_samples = audio_data.len() as u32;
        let byte_rate = sample_rate * 2; // 16位单声道
        let data_size = num_samples * 2;
        let file_size = 36 + data_size;

        // RIFF头
        temp_file.write_all(b"RIFF")?;
        temp_file.write_all(&file_size.to_le_bytes())?;
        temp_file.write_all(b"WAVE")?;

        // fmt块
        temp_file.write_all(b"fmt ")?;
        temp_file.write_all(&16u32.to_le_bytes())?; // fmt块大小
        temp_file.write_all(&1u16.to_le_bytes())?; // PCM格式
        temp_file.write_all(&1u16.to_le_bytes())?; // 单声道
        temp_file.write_all(&sample_rate.to_le_bytes())?;
        temp_file.write_all(&byte_rate.to_le_bytes())?;
        temp_file.write_all(&2u16.to_le_bytes())?; // 块对齐
        temp_file.write_all(&16u16.to_le_bytes())?; // 位深度

        // data块
        temp_file.write_all(b"data")?;
        temp_file.write_all(&data_size.to_le_bytes())?;

        // 音频数据（转换为16位PCM）
        for &sample in audio_data {
            let pcm_sample = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            temp_file.write_all(&pcm_sample.to_le_bytes())?;
        }

        temp_file.flush()?;
        Ok(temp_file)
    }

    #[test]
    fn test_audio_volume_normalize_consistency() {
        // 创建测试音频数据
        let audio_data = create_test_audio_data(16000, 1.0);
        let audio_array = Array1::from(audio_data);

        let utilities =
            RefAudioUtilities::new("dummy_onnx_path", "dummy_wav2vec2_path", 6.0, 320, None)
                .expect("Failed to create RefAudioUtilities");

        // 多次运行音量归一化，验证结果一致性
        let result1 = utilities.audio_volume_normalize(audio_array.clone(), 0.8);
        let result2 = utilities.audio_volume_normalize(audio_array.clone(), 0.8);
        let result3 = utilities.audio_volume_normalize(audio_array.clone(), 0.8);

        // 验证结果完全一致
        assert_eq!(result1.len(), result2.len());
        assert_eq!(result2.len(), result3.len());

        for i in 0..result1.len() {
            assert!(
                (result1[i] - result2[i]).abs() < f32::EPSILON,
                "Volume normalization inconsistent at index {}: {} vs {}",
                i,
                result1[i],
                result2[i]
            );
            assert!(
                (result2[i] - result3[i]).abs() < f32::EPSILON,
                "Volume normalization inconsistent at index {}: {} vs {}",
                i,
                result2[i],
                result3[i]
            );
        }
    }

    #[test]
    fn test_resample_audio_consistency() {
        let utilities =
            RefAudioUtilities::new("dummy_onnx_path", "dummy_wav2vec2_path", 3.0, 320, None)
                .expect("Failed to create RefAudioUtilities");

        // 创建测试音频数据
        let audio_data = create_test_audio_data(22050, 1.0);
        let audio_array = Array1::from(audio_data);

        // 多次重采样，验证结果一致性
        let result1 = utilities
            .resample_audio(audio_array.clone(), 22050, 16000)
            .unwrap();
        let result2 = utilities
            .resample_audio(audio_array.clone(), 22050, 16000)
            .unwrap();
        let result3 = utilities
            .resample_audio(audio_array.clone(), 22050, 16000)
            .unwrap();

        // 验证结果完全一致
        assert_eq!(result1.len(), result2.len());
        assert_eq!(result2.len(), result3.len());

        for i in 0..result1.len() {
            assert!(
                (result1[i] - result2[i]).abs() < f32::EPSILON,
                "Resample inconsistent at index {}: {} vs {}",
                i,
                result1[i],
                result2[i]
            );
            assert!(
                (result2[i] - result3[i]).abs() < f32::EPSILON,
                "Resample inconsistent at index {}: {} vs {}",
                i,
                result2[i],
                result3[i]
            );
        }
    }

    #[test]
    fn test_ensure_consistent_length() {
        let utilities =
            RefAudioUtilities::new("dummy_onnx_path", "dummy_wav2vec2_path", 3.0, 320, None)
                .expect("Failed to create RefAudioUtilities");

        // 测试不同长度的音频
        let short_audio = Array1::from(vec![0.1, 0.2, 0.3]); // 长度 < hop_length
        let medium_audio = Array1::from(vec![0.0; 500]); // 长度不是hop_length的倍数
        let aligned_audio = Array1::from(vec![0.0; 640]); // 长度是hop_length的倍数

        let result1 = utilities.ensure_consistent_length(short_audio.clone());
        let result2 = utilities.ensure_consistent_length(medium_audio.clone());
        let result3 = utilities.ensure_consistent_length(aligned_audio.clone());

        // 验证长度对齐
        assert_eq!(result1.len() % 320, 0, "Short audio not properly aligned");
        assert_eq!(result2.len() % 320, 0, "Medium audio not properly aligned");
        assert_eq!(result3.len() % 320, 0, "Aligned audio not properly aligned");

        // 验证最小长度
        assert!(
            result1.len() >= 320,
            "Short audio not padded to minimum length"
        );

        // 验证截断正确性
        assert_eq!(result2.len(), 320, "Medium audio not truncated correctly");

        // 验证已对齐音频保持不变
        assert_eq!(result3.len(), 640, "Aligned audio length changed");
    }

    #[test]
    fn test_mel_spectrogram_consistency() {
        let utilities =
            RefAudioUtilities::new("dummy_onnx_path", "dummy_wav2vec2_path", 6.0, 320, None)
                .expect("Failed to create RefAudioUtilities");

        // 创建测试音频数据
        let audio_data = create_test_audio_data(16000, 1.0);
        let audio_array = Array1::from(audio_data);

        // 获取参考音频片段
        let ref_wav = utilities.get_ref_clip(&audio_array);

        // 验证参考音频长度正确（使用与C++一致的计算方式）
        let expected_length = 6 * 16000; // ref_segment_duration * sample_rate = 96000
        assert_eq!(
            ref_wav.len(),
            expected_length,
            "ref_wav length should be {} (6 seconds * 16000 Hz)",
            expected_length
        );

        // 提取梅尔频谱图（使用与C++一致的参数）
        let mel_spec = utilities.extract_mel_spectrogram(&ref_wav, 128, 1024, 320, 1024);

        // 验证梅尔频谱图维度（基于96000样本的音频）
        // center=true 时，等效于在两端各填充 n_fft/2，总长度增加 n_fft
        // librosa 的帧数计算为 floor((len + n_fft - n_fft)/hop) + 1 = floor(len / hop) + 1
        let expected_frames = ref_wav.len() / 320 + 1;
        assert_eq!(
            mel_spec.shape(),
            &[128, expected_frames],
            "Mel spectrogram should have shape [128, {}]",
            expected_frames
        );

        // 多次提取梅尔频谱图，验证结果一致性（使用与C++一致的参数）
        let result1 = utilities.extract_mel_spectrogram(&audio_array, 128, 1024, 320, 1024);
        let result2 = utilities.extract_mel_spectrogram(&audio_array, 128, 1024, 320, 1024);
        let result3 = utilities.extract_mel_spectrogram(&audio_array, 128, 1024, 320, 1024);

        // 验证形状一致
        assert_eq!(result1.shape(), result2.shape());
        assert_eq!(result2.shape(), result3.shape());

        // 验证数值一致
        for ((v1, v2), v3) in result1.iter().zip(result2.iter()).zip(result3.iter()) {
            assert!(
                (v1 - v2).abs() < f32::EPSILON,
                "Mel spectrogram inconsistent: {} vs {}",
                v1,
                v2
            );
            assert!(
                (v2 - v3).abs() < f32::EPSILON,
                "Mel spectrogram inconsistent: {} vs {}",
                v2,
                v3
            );
        }
    }
}
