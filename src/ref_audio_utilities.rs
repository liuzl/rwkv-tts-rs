//! 参考音频处理工具模块
//! 实现与Python版本ref_audio_utilities.py相同的功能

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::input::SessionInputValue;
use ort::session::Session;
use ort::value::Value;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
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
                    // Warning: Failed to load BiCodecDetokenize model
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
        // Loading audio file

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

        // 检查音频最小长度要求（至少0.1秒）
        let min_samples = (sample_rate as f32 * 0.1) as usize;
        if audio_samples.len() < min_samples {
            return Err(anyhow!(
                "音频太短：{:.3}秒（最少需要0.1秒），样本数：{}",
                audio_samples.len() as f32 / sample_rate as f32,
                audio_samples.len()
            ));
        }

        let mut audio = Array1::from(audio_samples);

        // 多声道转单声道 - 与C++实现一致（取第一个通道）
        if channels > 1 {
            // Converting channels to mono
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
            // 音频数据可能未正确归一化
        }

        // 重采样到目标采样率 - 与C++的wav->resample(16000)保持一致
        if sample_rate != target_sr {
            // Resampling audio
            audio = self.resample_audio_high_quality(audio, sample_rate, target_sr)?;

            // 验证重采样结果
            if audio.is_empty() {
                return Err(anyhow!("重采样后音频数据为空"));
            }
        }

        // 音量归一化
        if volume_normalize {
            // Applying volume normalization
            audio = self.audio_volume_normalize(audio, 0.2);
        }

        // 静音处理：仅裁剪开头和结尾静音，避免强制填充导致对齐偏移
        audio = self.trim_silence_only(audio, 0.01);

        // Final audio length processed
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

        // WAV spec loaded

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

        // MP3 spec loaded

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
                    // MP3解码错误
                    continue;
                }
            }
        }

        if audio_samples.is_empty() {
            return Err(anyhow!("MP3文件解码后无音频数据"));
        }

        Ok((audio_samples, sample_rate, channels))
    }

    /// 高质量重采样音频数据 - 使用rubato库实现专业级重采样
    pub fn resample_audio_high_quality(
        &self,
        audio: Array1<f32>,
        original_sr: u32,
        target_sr: u32,
    ) -> Result<Array1<f32>> {
        if original_sr == target_sr {
            return Ok(audio);
        }

        // 使用rubato库进行高质量重采样
        // 配置参数以获得最佳音质，与Python的soxr库相当
        let params = SincInterpolationParameters {
            sinc_len: 256,  // 更长的sinc长度提供更好的频率响应
            f_cutoff: 0.95, // 稍微保守的截止频率避免混叠
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256, // 高过采样因子提供更好的精度
            window: WindowFunction::BlackmanHarris2, // 优秀的频域特性
        };

        // 创建重采样器
        let mut resampler = SincFixedIn::<f32>::new(
            target_sr as f64 / original_sr as f64,
            2.0, // 最大比率变化
            params,
            audio.len(),
            1, // 单声道
        )
        .map_err(|e| anyhow!("创建重采样器失败: {}", e))?;

        // 准备输入数据（rubato需要Vec<Vec<f32>>格式）
        let input_data = vec![audio.to_vec()];

        // 执行重采样
        let output_data = resampler
            .process(&input_data, None)
            .map_err(|e| anyhow!("重采样处理失败: {}", e))?;

        // 提取重采样后的数据
        if output_data.is_empty() || output_data[0].is_empty() {
            return Err(anyhow!("重采样输出为空"));
        }

        Ok(Array1::from(output_data[0].clone()))
    }

    /// 重采样音频数据（现在使用高质量rubato库实现）
    pub fn resample_audio(
        &self,
        audio: Array1<f32>,
        original_sr: u32,
        target_sr: u32,
    ) -> Result<Array1<f32>> {
        // 直接调用高质量重采样方法，确保所有重采样都使用相同的高质量算法
        self.resample_audio_high_quality(audio, original_sr, target_sr)
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

        // Volume normalization applied

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

    /// 零均值单位方差归一化 - 与C++实现完全一致，增强数值稳定性
    /// C++实现：
    /// float mean = std::accumulate(input_values.begin(), input_values.end(), 0.0f) / input_values.size();
    /// float std = std::sqrt(std::accumulate(input_values.begin(), input_values.end(), 0.0f, [mean](float a, float b) {
    ///     return a + (b - mean) * (b - mean);
    /// }) / input_values.size() + 1e-7f);
    /// for (int i = 0; i < input_values.size(); i++) {
    ///     input_values[i] = (input_values[i] - mean) / std;
    /// }
    pub fn zero_mean_unit_variance_normalize(mut input_values: Vec<f32>) -> Vec<f32> {
        // 数值稳定性检查：处理空向量或极短向量
        if input_values.is_empty() {
            return input_values;
        }

        if input_values.len() == 1 {
            // 单个值的情况，直接返回零
            input_values[0] = 0.0;
            return input_values;
        }

        // 计算均值 - 与C++完全一致
        let mean = input_values.iter().sum::<f32>() / input_values.len() as f32;

        // 检查是否所有值都相同（方差为零的情况）
        let all_same = input_values.iter().all(|&x| (x - mean).abs() < 1e-10);
        if all_same {
            // 所有值都相同，直接设为零
            input_values.fill(0.0);
            return input_values;
        }

        // 计算标准差 - 与C++实现完全一致，但增加更大的epsilon以提高数值稳定性
        let variance_sum = input_values
            .iter()
            .fold(0.0f32, |acc, &b| acc + (b - mean) * (b - mean));
        let variance = variance_sum / input_values.len() as f32;

        // 使用固定的epsilon值，与Python版本保持一致
        let epsilon = 1e-7f32;
        let std = (variance + epsilon).sqrt();

        // 归一化 - 与C++完全一致
        for value in input_values.iter_mut() {
            *value = (*value - mean) / std;
        }

        // Zero-mean unit-variance normalize applied with numerical stability

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
        // After center padding

        // 使用与librosa相同的帧数计算方式（基于填充后的长度）
        let n_frames = if wav_len <= n_fft {
            1
        } else {
            (wav_len - n_fft) / hop_length + 1
        };

        // Mel spectrogram extraction parameters

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

        // 创建梅尔滤波器组 - 使用slaney归一化，fmin=10，fmax=sample_rate/2.0（与Python版本保持一致）
        let mel_filters = self.create_mel_filterbank_slaney_with_fmax(
            n_mels,
            n_fft,
            self.sample_rate as f32,
            10.0,
            self.sample_rate as f32 / 2.0,
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

        // Mel spectrogram shape processed
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

    /// 计算功率谱 - 更精确的实现，真正计算功率谱而非幅度谱
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

            // 计算真正的功率谱：幅度的平方
            *power = real * real + imag * imag;

            // 对于非零频率和奈奎斯特频率，需要适当的归一化
            if k > 0 && k < n_freqs - 1 {
                // 对于中间频率，由于我们只计算正频率部分，需要乘以2来补偿负频率部分
                *power *= 2.0;
            }

            // 归一化：除以N^2以匹配标准功率谱定义
            *power /= (n_fft * n_fft) as f32;
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

        // wav2vec2 input shape调试信息

        let input_tensor = Value::from_array((input_shape, input_vec))?;

        let outputs = wav2vec2_session.run(ort::inputs![SessionInputValue::from(input_tensor)])?;
        let (output_shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        // wav2vec2 output shape调试信息

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
            // wav2vec2 features after removing batch dim
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

        let ref_segment_length = ((self.ref_segment_duration * self.sample_rate as f32) as u32
            / self.latent_hop_length
            * self.latent_hop_length) as usize;

        // get_ref_clip parameters calculated

        let wav_length = wav.len();

        // 验证音频长度的合理性
        if wav_length == 0 {
            // 如果音频为空，返回零填充的参考片段
            return Array1::zeros(ref_segment_length);
        }

        if ref_segment_length == 0 {
            // 如果参考长度为0，返回空数组
            return Array1::zeros(0);
        }

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
        self.tokenize_with_options(audio_path, true)
    }

    /// 带选项的tokenize方法，允许配置音量归一化
    pub fn tokenize_with_options(
        &mut self,
        audio_path: &str,
        volume_normalize: bool,
    ) -> Result<(Vec<i32>, Vec<i32>)> {
        // Tokenizing audio

        // 音频预处理：可配置音量归一化选项
        let (wav, ref_wav) = self.process_audio(audio_path, volume_normalize)?;

        let feat = self.extract_wav2vec2_features(wav.as_slice().unwrap())?;

        // 使用与C++实现完全一致的梅尔频谱提取，避免参数差异导致不稳定
        let ref_mel =
            crate::tts_pipeline_fixes::TtsPipelineFixes::extract_mel_spectrogram_consistent(
                &ref_wav,
            )?;

        // 确保数据是行优先布局（C-order）
        let ref_mel_c_order = if ref_mel.is_standard_layout() {
            ref_mel.clone()
        } else {
            // 如果不是标准布局，转换为C-order
            // Converting mel_spectrogram to C-order layout
            ref_mel.as_standard_layout().to_owned()
        };

        let ref_mel_input = ref_mel_c_order.insert_axis(ndarray::Axis(0));
        let ref_mel_dyn = ref_mel_input.into_dyn();
        let ref_mel_shape: Vec<i64> = ref_mel_dyn.shape().iter().map(|&d| d as i64).collect();
        let ref_mel_vec = ref_mel_dyn.into_raw_vec();

        let ref_mel_tensor = Value::from_array((ref_mel_shape.clone(), ref_mel_vec))?;

        // 准备feat张量：形状应该是[1, t, 1024]，与C++完全一致

        let feat_c_order = if feat.is_standard_layout() {
            feat.clone()
        } else {
            // Converting feat to C-order layout
            feat.as_standard_layout().to_owned()
        };

        let feat_input = feat_c_order.insert_axis(ndarray::Axis(0));
        let feat_dyn = feat_input.into_dyn();
        let feat_shape: Vec<i64> = feat_dyn.shape().iter().map(|&d| d as i64).collect();
        let feat_vec = feat_dyn.into_raw_vec();

        let feat_tensor = Value::from_array((feat_shape.clone(), feat_vec))?;

        // Input tensors prepared

        let ort_session = self
            .ort_session
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("ort session not initialized"))?;

        let outputs = ort_session.run(ort::inputs![
            "ref_wav_mel" => SessionInputValue::from(ref_mel_tensor),
            "feat" => SessionInputValue::from(feat_tensor)
        ])?;

        let mut semantic_tokens: Vec<i32> = vec![];
        let mut global_tokens: Vec<i32> = vec![];

        // 1) 首先严格按名称解析，避免位置顺序不一致导致错位
        for (name, output) in outputs.iter() {
            if name == "semantic_tokens" {
                semantic_tokens = match output.try_extract_tensor::<i64>() {
                    Ok((_s_sem, semantic_tokens_slice)) => {
                        semantic_tokens_slice.iter().map(|&x| x as i32).collect()
                    }
                    Err(_) => {
                        let (_s_sem, semantic_tokens_slice) = output.try_extract_tensor::<i32>()?;
                        semantic_tokens_slice.to_vec()
                    }
                };
            } else if name == "global_tokens" {
                global_tokens = match output.try_extract_tensor::<i64>() {
                    Ok((_s_glb, global_tokens_slice)) => {
                        global_tokens_slice.iter().map(|&x| x as i32).collect()
                    }
                    Err(_) => {
                        let (_s_glb, global_tokens_slice) = output.try_extract_tensor::<i32>()?;
                        global_tokens_slice.to_vec()
                    }
                };
            }
        }

        // 2) 如果名称未匹配到，按形状辅助判定（semantic为[1, L]；global为[1, 1, 32]）
        if semantic_tokens.is_empty() || global_tokens.is_empty() {
            for (_name, output) in outputs.iter() {
                let shape = output.shape();
                if semantic_tokens.is_empty() && shape.len() == 2 && shape[0] == 1 {
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
                    continue;
                }
                if global_tokens.is_empty() && shape.len() == 3 && shape[0] == 1 && shape[1] == 1 {
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
        }

        // 3) 兜底：若仍无法按名称/形状区分，则按索引[0]=semantic，[1]=global
        if (semantic_tokens.is_empty() || global_tokens.is_empty()) && outputs.len() >= 2 {
            if semantic_tokens.is_empty() {
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
            }
            if global_tokens.is_empty() {
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
        }

        // 4) 范围校验与修正日志（保持与生成阶段一致的约束）
        // global: [0..4096)
        if !global_tokens.is_empty() {
            let mut out_of_range: Vec<i32> = Vec::new();
            for &t in &global_tokens {
                if !(0..4096).contains(&t) {
                    out_of_range.push(t);
                }
            }
            if !out_of_range.is_empty() {
                log::warn!(
                    "🚨 参考global tokens越界：{:?}，将进行clamp到[0..4095]",
                    out_of_range
                );
                for v in global_tokens.iter_mut() {
                    *v = (*v).clamp(0, 4095);
                }
            } else {
                log::info!("✅ 参考global tokens在词表范围内（vocab_size=4096）");
            }
        }

        // semantic: [0..=8192]（包含EOS=8192），仅记录越界并clamp，不移除EOS
        if !semantic_tokens.is_empty() {
            let mut out_of_range: Vec<i32> = Vec::new();
            for &t in &semantic_tokens {
                if !(0..=crate::rwkv_sampler::TTS_EOS_TOKEN).contains(&t) {
                    out_of_range.push(t);
                }
            }
            if !out_of_range.is_empty() {
                log::warn!(
                    "🚨 参考semantic tokens越界：{:?}，将clamp到[0..={}](含EOS)",
                    out_of_range,
                    crate::rwkv_sampler::TTS_EOS_TOKEN
                );
                for v in semantic_tokens.iter_mut() {
                    *v = (*v).clamp(0, crate::rwkv_sampler::TTS_EOS_TOKEN);
                }
            } else {
                log::info!(
                    "✅ 参考semantic tokens在范围内（含EOS={}）",
                    crate::rwkv_sampler::TTS_EOS_TOKEN
                );
            }
        }

        // Global tokens unique values counted
        // Global tokens raw data checked

        // Tokenization completed
        // Semantic tokens sample checked
        // Token range check performed
        // Check for invalid tokens in reference audio processing
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

    /// 检测音频开头和结尾的静音长度
    /// 返回 (开头静音样本数, 结尾静音样本数)
    fn detect_silence(&self, audio: &Array1<f32>, threshold: f32) -> (usize, usize) {
        let samples = audio.as_slice().unwrap();
        let len = samples.len();

        if len == 0 {
            return (0, 0);
        }

        // 检测开头静音
        let mut start_silence = 0;
        for &sample in samples.iter() {
            if sample.abs() > threshold {
                break;
            }
            start_silence += 1;
        }

        // 检测结尾静音
        let mut end_silence = 0;
        for &sample in samples.iter().rev() {
            if sample.abs() > threshold {
                break;
            }
            end_silence += 1;
        }

        // 确保不会超过音频总长度
        if start_silence + end_silence >= len {
            // 如果整个音频都是静音，平均分配
            let half = len / 2;
            return (half, len - half);
        }

        (start_silence, end_silence)
    }

    /// 智能处理音频开头和结尾的静音，确保各保持0.5秒
    /// target_silence_duration: 目标静音时长（秒）
    /// sample_rate: 采样率
    /// 仅裁剪开头与结尾静音，不进行补零，保持原始有效音频时长
    fn trim_silence_only(&self, audio: Array1<f32>, silence_threshold: f32) -> Array1<f32> {
        let (start_silence, end_silence) = self.detect_silence(&audio, silence_threshold);
        let samples = audio.as_slice().unwrap();
        let total_len = samples.len();

        // 计算有效音频片段范围
        let audio_start = start_silence.min(total_len);
        let audio_end = total_len.saturating_sub(end_silence);

        if audio_start >= audio_end {
            // 整段静音，直接返回原长度的零（保持行为简洁、可预期）
            return Array1::zeros(total_len);
        }

        Array1::from(samples[audio_start..audio_end].to_vec())
    }
}
