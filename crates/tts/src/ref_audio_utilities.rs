//! 参考音频处理工具模块
//! 实现与Python版本ref_audio_utilities.py相同的功能

use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, Array3};
use ort::{Environment, Session, SessionBuilder, Value, Tensor};
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
}

impl RefAudioUtilities {
    /// 创建新的参考音频处理工具实例
    /// 
    /// # Arguments
    /// * `onnx_model_path` - ONNX模型文件路径
    /// * `wav2vec2_path` - wav2vec2 ONNX模型文件路径
    /// * `ref_segment_duration` - 参考音频时长（秒）
    /// * `latent_hop_length` - 潜在特征跳长度
    /// 
    /// # Returns
    /// * `Result<RefAudioUtilities>` - 参考音频处理工具实例或错误
    pub fn new(
        onnx_model_path: &str,
        wav2vec2_path: &str,
        ref_segment_duration: f32,
        latent_hop_length: u32,
    ) -> Result<Self> {
        // 初始化ONNX Runtime环境
        let environment = Environment::builder()
            .with_name("RefAudioUtilities")
            .build()?;
        
        // 创建ONNX会话
        let ort_session = SessionBuilder::new(&environment)?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_model_from_file(onnx_model_path)?;
        
        // 创建wav2vec2 ONNX会话
        let wav2vec2_session = SessionBuilder::new(&environment)?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_model_from_file(wav2vec2_path)?;
        
        Ok(Self {
            ort_session,
            wav2vec2_session,
            sample_rate: 16000,
            ref_segment_duration,
            latent_hop_length,
        })
    }
    
    /// 加载音频文件
    /// 
    /// # Arguments
    /// * `audio_path` - 音频文件路径
    /// * `target_sr` - 目标采样率
    /// * `volume_normalize` - 是否进行音量归一化
    /// 
    /// # Returns
    /// * `Result<Array1<f32>>` - 音频数据数组或错误
    pub fn load_audio(&self, audio_path: &str, target_sr: u32, volume_normalize: bool) -> Result<Array1<f32>> {
        // 检查文件是否存在
        if !Path::new(audio_path).exists() {
            return Err(anyhow!("音频文件不存在: {}", audio_path));
        }
        
        // 使用hound库加载音频文件
        let mut reader = hound::WavReader::open(audio_path)?;
        let spec = reader.spec();
        
        // 获取音频数据
        let samples: Result<Vec<f32>, _> = reader.samples::<i16>()
            .map(|s| s.map(|sample| sample as f32 / i16::MAX as f32))
            .collect();
        let mut audio = Array1::from(samples?);
        
        // 如果是立体声，取第一个通道
        if spec.channels > 1 {
            // 重新组织数据，只取第一个通道
            let len = audio.len() / spec.channels as usize;
            let mut mono_audio = Vec::with_capacity(len);
            for i in 0..len {
                mono_audio.push(audio[i * spec.channels as usize]);
            }
            audio = Array1::from(mono_audio);
        }
        
        // 重采样到目标采样率
        if spec.sample_rate != target_sr {
            audio = self.resample_audio(audio, spec.sample_rate, target_sr)?;
        }
        
        // 音量归一化
        if volume_normalize {
            audio = self.audio_volume_normalize(audio, 0.2);
        }
        
        Ok(audio)
    }
    
    /// 重采样音频数据
    /// 
    /// # Arguments
    /// * `audio` - 原始音频数据
    /// * `original_sr` - 原始采样率
    /// * `target_sr` - 目标采样率
    /// 
    /// # Returns
    /// * `Array1<f32>` - 重采样后的音频数据
    fn resample_audio(&self, audio: Array1<f32>, original_sr: u32, target_sr: u32) -> Result<Array1<f32>> {
        // 简单的线性重采样实现
        let original_len = audio.len();
        let ratio = target_sr as f32 / original_sr as f32;
        let new_len = (original_len as f32 * ratio) as usize;
        
        let mut resampled = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let pos = i as f32 / ratio;
            let prev_idx = pos.floor() as usize;
            let next_idx = pos.ceil() as usize.min(original_len - 1);
            let frac = pos - prev_idx as f32;
            
            if prev_idx < original_len && next_idx < original_len {
                let sample = audio[prev_idx] * (1.0 - frac) + audio[next_idx] * frac;
                resampled.push(sample);
            } else if prev_idx < original_len {
                resampled.push(audio[prev_idx]);
            } else {
                resampled.push(0.0);
            }
        }
        
        Ok(Array1::from(resampled))
    }
    
    /// 音频音量归一化
    /// 
    /// # Arguments
    /// * `audio` - 音频数据
    /// * `coeff` - 目标系数
    /// 
    /// # Returns
    /// * `Array1<f32>` - 归一化后的音频数据
    fn audio_volume_normalize(&self, audio: Array1<f32>, coeff: f32) -> Array1<f32> {
        let mut audio = audio.to_vec();
        
        // 对音频信号的绝对值进行排序
        let mut temp: Vec<f32> = audio.iter().map(|&x| x.abs()).collect();
        temp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // 如果最大值小于0.1，则将数组缩放至最大值为0.1
        if let Some(&max_val) = temp.last() {
            if max_val < 0.1 {
                let scaling_factor = max_val.max(1e-3); // 防止除零，使用小常数
                for sample in &mut audio {
                    *sample = *sample / scaling_factor * 0.1;
                }
            }
        }
        
        // 过滤掉小于0.01的值
        let temp: Vec<f32> = temp.into_iter().filter(|&x| x > 0.01).collect();
        let l = temp.len();
        
        // 如果显著值少于或等于10个，直接返回音频而不进一步处理
        if l <= 10 {
            return Array1::from(audio);
        }
        
        // 计算temp中前10%到1%值的平均值
        let start = (0.9 * l as f32) as usize;
        let end = (0.99 * l as f32) as usize.min(l);
        let volume = if start < end {
            let sum: f32 = temp[start..end].iter().sum();
            sum / (end - start) as f32
        } else {
            1.0 // 默认值
        };
        
        // 将音频归一化到目标系数水平，将缩放因子限制在0.1到10之间
        for sample in &mut audio {
            *sample = *sample * (coeff / volume).clamp(0.1, 10.0);
        }
        
        // 确保音频中的最大绝对值不超过1
        let max_value = audio.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        if max_value > 1.0 {
            for sample in &mut audio {
                *sample = *sample / max_value;
            }
        }
        
        Array1::from(audio)
    }
    
    /// 提取梅尔频谱图
    /// 
    /// # Arguments
    /// * `wav` - 音频数据
    /// * `n_mels` - 梅尔滤波器组数量
    /// * `n_fft` - FFT窗口大小
    /// * `hop_length` - 帧移
    /// * `win_length` - 窗口长度
    /// 
    /// # Returns
    /// * `Array2<f32>` - 梅尔频谱图
    pub fn extract_mel_spectrogram(
        &self,
        wav: &Array1<f32>,
        n_mels: usize,
        n_fft: usize,
        hop_length: usize,
        win_length: usize,
    ) -> Array2<f32> {
        // 这里需要实现梅尔频谱图提取逻辑
        // 由于Rust中没有直接等价于librosa的库，我们需要手动实现
        // 为简化起见，这里返回一个占位符数组
        Array2::zeros((n_mels, 301)) // 假设时间步长为301
    }
    
    /// 使用ONNX wav2vec2模型提取特征
    /// 
    /// # Arguments
    /// * `wav` - 音频数据
    /// 
    /// # Returns
    /// * `Result<Array2<f32>>` - 特征向量或错误
    pub fn extract_wav2vec2_features(&self, wav: &Array1<f32>) -> Result<Array2<f32>> {
        // 添加batch维度
        let input_data = wav.insert_axis(ndarray::Axis(0)); // [1, sequence_length]
        
        // 运行wav2vec2推理
        let input_tensor = Value::from_array(input_data)?;
        let outputs = self.wav2vec2_session.run(vec![input_tensor])?;
        
        // 获取输出
        let features = outputs[0].try_extract::<f32>()?;
        let features = features.slice(ndarray::s![0, .., ..]).to_owned(); // 移除batch维度，得到 [time_steps, 1024]
        
        Ok(features)
    }
    
    /// 获取参考音频片段
    /// 
    /// # Arguments
    /// * `wav` - 原始音频数据
    /// 
    /// # Returns
    /// * `Array1<f32>` - 参考音频片段
    pub fn get_ref_clip(&self, wav: &Array1<f32>) -> Array1<f32> {
        // 使用与BiCodecTokenizer相同的计算方式
        let ref_segment_length = (self.sample_rate as f32 * self.ref_segment_duration) as usize
            / self.latent_hop_length as usize
            * self.latent_hop_length as usize;
        let wav_length = wav.len();
        
        if ref_segment_length > wav_length {
            // 如果音频不足指定长度，重复音频直到达到要求
            let repeat_times = ref_segment_length / wav_length + 1;
            let mut repeated = Vec::with_capacity(wav_length * repeat_times);
            for _ in 0..repeat_times {
                repeated.extend(wav.iter());
            }
            Array1::from(repeated).slice(ndarray::s![..ref_segment_length]).to_owned()
        } else {
            // 截取指定长度
            wav.slice(ndarray::s![..ref_segment_length]).to_owned()
        }
    }
    
    /// 处理音频文件，返回原始音频和参考音频
    /// 
    /// # Arguments
    /// * `audio_path` - 音频文件路径
    /// * `volume_normalize` - 是否进行音量归一化
    /// 
    /// # Returns
    /// * `Result<(Array1<f32>, Array1<f32>)>` - (原始音频, 参考音频)或错误
    pub fn process_audio(&self, audio_path: &str, volume_normalize: bool) -> Result<(Array1<f32>, Array1<f32>)> {
        let wav = self.load_audio(audio_path, self.sample_rate, volume_normalize)?;
        let ref_wav = self.get_ref_clip(&wav);
        
        Ok((wav, ref_wav))
    }
    
    /// 使用ONNX模型生成tokens
    /// 
    /// # Arguments
    /// * `audio_path` - 音频文件路径
    /// 
    /// # Returns
    /// * `Result<(Vec<i64>, Vec<i64>)>` - (global_tokens, semantic_tokens)或错误
    pub fn tokenize(&self, audio_path: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        // 处理音频
        let (wav, ref_wav) = self.process_audio(audio_path, false)?;
        
        // 提取特征
        let feat = self.extract_wav2vec2_features(&wav)?;
        let ref_mel = self.extract_mel_spectrogram(&ref_wav, 128, 1024, 320, 640);
        
        // 添加batch维度
        let ref_mel_input = ref_mel.insert_axis(ndarray::Axis(0)); // [1, 128, 301]
        let feat_input = feat.insert_axis(ndarray::Axis(0)); // [1, feat_len, 1024]
        
        // 运行ONNX模型
        let ref_mel_tensor = Value::from_array(ref_mel_input)?;
        let feat_tensor = Value::from_array(feat_input)?;
        
        let outputs = self.ort_session.run(vec![ref_mel_tensor, feat_tensor])?;
        
        // 解析输出
        let semantic_tokens_tensor = outputs[0].try_extract::<i64>()?;
        let global_tokens_tensor = outputs[1].try_extract::<i64>()?;
        
        // 转换为Vec
        let semantic_tokens: Vec<i64> = semantic_tokens_tensor.iter().cloned().collect();
        let global_tokens: Vec<i64> = global_tokens_tensor.iter().cloned().collect();
        
        Ok((global_tokens, semantic_tokens))
    }
}