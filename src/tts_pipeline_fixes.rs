//! TTS流水线修复模块
//! 修复zero-shot模式下的问题，确保与C++实现完全一致

use anyhow::Result;
use ndarray::{Array1, Array2};

/// 修复zero-shot模式下的音频处理问题
pub struct TtsPipelineFixes;

impl TtsPipelineFixes {
    /// 提取梅尔频谱图（与C++实现完全一致）
    pub fn extract_mel_spectrogram_consistent(wav: &Array1<f32>) -> Result<Array2<f32>> {
        // 参数与C++实现完全一致
        let n_mels: usize = 128;
        let n_fft: usize = 1024;
        let hop_length: usize = 320;
        let _win_length: usize = 1024; // 修复：与C++保持一致（win_length等于n_fft）
        let sample_rate: f32 = 16000.0;
        let fmin: f32 = 10.0;
        let fmax: f32 = 8000.0; // 修复：与C++保持一致

        // center=true 的填充（与C++实现一致）
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

        // Hann窗（与C++实现一致）
        let window: Vec<f32> = (0..n_fft)
            .map(|i| {
                let angle = 2.0 * std::f32::consts::PI * i as f32 / (n_fft - 1) as f32;
                0.5 * (1.0 - angle.cos())
            })
            .collect();

        // 创建梅尔滤波器组（与C++实现一致）
        let mel_filters =
            Self::create_mel_filterbank_slaney_with_fmax(n_mels, n_fft, sample_rate, fmin, fmax);

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

            // 计算功率谱（与C++实现一致，使用幅度谱而非功率谱）
            let power_spectrum = Self::compute_magnitude_spectrum(&frame);

            // 应用梅尔滤波器
            for mel_idx in 0..n_mels {
                let mut mel_energy = 0.0f32;
                for freq_idx in 0..power_spectrum.len() {
                    mel_energy += power_spectrum[freq_idx] * mel_filters[[mel_idx, freq_idx]];
                }
                // 修复：不进行对数变换，与C++的melSpectrogram函数一致
                mel_spectrogram[[mel_idx, frame_idx]] = mel_energy;
            }
        }

        Ok(mel_spectrogram)
    }

    /// 计算幅度谱（与C++实现一致）
    fn compute_magnitude_spectrum(frame: &[f32]) -> Vec<f32> {
        let n_fft = frame.len();
        let n_freqs = n_fft / 2 + 1;
        let mut magnitude_spectrum = vec![0.0f32; n_freqs];

        for (k, magnitude) in magnitude_spectrum.iter_mut().enumerate().take(n_freqs) {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;

            for (n, &sample) in frame.iter().enumerate().take(n_fft) {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / n_fft as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }

            // 计算幅度谱（与C++实现一致）
            *magnitude = (real * real + imag * imag).sqrt();
        }

        magnitude_spectrum
    }

    /// 创建梅尔滤波器组（与C++实现一致）
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
}
