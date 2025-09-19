//! 高性能采样器实现
//! 使用SIMD、内存池和快速路径优化采样性能

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::arch::x86_64::*;

/// 采样配置
#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub use_fast_path: bool,
    pub fast_path_threshold: f32,
    pub use_simd: bool,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.9,
            top_k: 50,
            use_fast_path: true,
            fast_path_threshold: 0.7,
            use_simd: true,
        }
    }
}

/// 全局内存池
static VEC_POOL: Lazy<Mutex<VecPool<f32>>> = Lazy::new(|| Mutex::new(VecPool::new(1000)));

/// 快速采样器
#[derive(Default)]
pub struct FastSampler;

impl FastSampler {
    /// 创建新的快速采样器
    pub fn new() -> Self {
        Self
    }

    /// 优化的采样函数
    pub fn optimized_sample(
        &self,
        logits: &[f32],
        config: &SamplingConfig,
        forbid_token: Option<usize>,
        rng: &mut Option<StdRng>,
    ) -> usize {
        // 快速路径检查：确定性采样
        if config.use_fast_path && self.should_use_fast_path(logits, config) {
            return self.fast_path_sample(logits, forbid_token);
        }

        // 使用内存池避免分配
        let mut pool_guard = VEC_POOL.lock();
        let mut probs = pool_guard.get_with_capacity(logits.len());

        // 计算概率分布
        if config.use_simd {
            self.compute_probs_simd(logits, config.temperature, &mut probs);
        } else {
            self.compute_probs_scalar(logits, config.temperature, &mut probs);
        }

        // 应用top-p和top-k
        self.apply_top_constraints(&mut probs, config.top_p, config.top_k);

        // 采样
        let result = self.sample_from_probs(&probs, rng);

        // 返回向量到池中
        pool_guard.return_vec(probs);

        result
    }

    /// 检查是否应该使用快速路径
    fn should_use_fast_path(&self, logits: &[f32], config: &SamplingConfig) -> bool {
        // 确定性采样条件
        if config.top_k == 1 || config.top_p < 1e-4 {
            return true;
        }

        // 检查是否有明显的主导token
        if let Some(max_idx) = self.find_max_index(logits) {
            let max_val = logits[max_idx];
            let second_max = logits
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != max_idx)
                .map(|(_, &v)| v)
                .fold(f32::NEG_INFINITY, f32::max);

            // 如果最大值明显大于次大值，使用快速路径
            (max_val - second_max) > config.fast_path_threshold
        } else {
            false
        }
    }

    /// 快速路径采样（直接选择最大值）
    fn fast_path_sample(&self, logits: &[f32], forbid_token: Option<usize>) -> usize {
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;

        for (i, &val) in logits.iter().enumerate() {
            if forbid_token.map_or(true, |ft| i != ft) && val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        max_idx
    }

    /// 使用SIMD计算概率分布
    #[cfg(target_arch = "x86_64")]
    fn compute_probs_simd(&self, logits: &[f32], temperature: f32, probs: &mut Vec<f32>) {
        unsafe {
            let inv_t = _mm_set1_ps(1.0 / temperature);
            let mut max_vec = _mm_set1_ps(f32::NEG_INFINITY);

            // 找到最大值
            let mut i = 0;
            while i + 4 <= logits.len() {
                let logits_vec = _mm_loadu_ps(&logits[i]);
                let scaled_vec = _mm_mul_ps(logits_vec, inv_t);
                max_vec = _mm_max_ps(max_vec, scaled_vec);
                i += 4;
            }

            // 处理剩余元素
            let mut max_scalar = f32::NEG_INFINITY;
            for &val in &logits[i..] {
                let scaled = val / temperature;
                if scaled > max_scalar {
                    max_scalar = scaled;
                }
            }

            // 合并SIMD和标量最大值
            let max_vals = [0.0; 4];
            _mm_storeu_ps(max_vals.as_ptr() as *mut f32, max_vec);
            let simd_max = max_vals.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let max_val = simd_max.max(max_scalar);

            // 计算exp和sum
            let mut sum_vec = _mm_set1_ps(0.0);
            i = 0;
            probs.resize(logits.len(), 0.0);

            while i + 4 <= logits.len() {
                let logits_vec = _mm_loadu_ps(&logits[i]);
                let scaled_vec = _mm_mul_ps(logits_vec, inv_t);
                let centered_vec = _mm_sub_ps(scaled_vec, _mm_set1_ps(max_val));
                let exp_vec = self.simd_exp(centered_vec);

                _mm_storeu_ps(&mut probs[i], exp_vec);
                sum_vec = _mm_add_ps(sum_vec, exp_vec);
                i += 4;
            }

            // 处理剩余元素并计算总和
            let mut sum = 0.0;
            let sum_vals = [0.0; 4];
            _mm_storeu_ps(sum_vals.as_ptr() as *mut f32, sum_vec);
            sum += sum_vals.iter().sum::<f32>();

            for j in i..logits.len() {
                let scaled = logits[j] / temperature;
                let exp_val = (scaled - max_val).exp();
                probs[j] = exp_val;
                sum += exp_val;
            }

            // 归一化
            if sum > 0.0 {
                let inv_sum = _mm_set1_ps(1.0 / sum);
                i = 0;
                while i + 4 <= probs.len() {
                    let prob_vec = _mm_loadu_ps(&probs[i]);
                    let normalized_vec = _mm_mul_ps(prob_vec, inv_sum);
                    _mm_storeu_ps(&mut probs[i], normalized_vec);
                    i += 4;
                }

                for j in i..probs.len() {
                    probs[j] /= sum;
                }
            }
        }
    }

    /// 标量版本的概率计算
    fn compute_probs_scalar(&self, logits: &[f32], temperature: f32, probs: &mut Vec<f32>) {
        let inv_t = 1.0 / temperature;
        let mut max_val = f32::NEG_INFINITY;

        // 找到最大值
        for &val in logits {
            let scaled = val * inv_t;
            if scaled > max_val {
                max_val = scaled;
            }
        }

        // 计算exp和sum
        let mut sum = 0.0;
        probs.resize(logits.len(), 0.0);

        for (i, &val) in logits.iter().enumerate() {
            let scaled = val * inv_t;
            let exp_val = (scaled - max_val).exp();
            probs[i] = exp_val;
            sum += exp_val;
        }

        // 归一化
        if sum > 0.0 {
            for prob in probs.iter_mut() {
                *prob /= sum;
            }
        }
    }

    /// 应用top-p和top-k约束
    fn apply_top_constraints(&self, probs: &mut [f32], top_p: f32, top_k: usize) {
        if top_p < 1.0 {
            self.apply_top_p(probs, top_p);
        }

        if top_k > 0 && top_k < probs.len() {
            self.apply_top_k(probs, top_k);
        }
    }

    /// 应用top-p（核采样）
    fn apply_top_p(&self, probs: &mut [f32], top_p: f32) {
        // 创建索引并排序
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

        let mut cumulative = 0.0;
        let mut cutoff = probs.len();

        for (i, &idx) in indices.iter().enumerate() {
            cumulative += probs[idx];
            if cumulative >= top_p {
                cutoff = i + 1;
                break;
            }
        }

        // 将cutoff之后的概率设为0
        for &idx in &indices[cutoff..] {
            probs[idx] = 0.0;
        }

        // 重新归一化
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for prob in probs.iter_mut() {
                *prob /= sum;
            }
        }
    }

    /// 应用top-k
    fn apply_top_k(&self, probs: &mut [f32], top_k: usize) {
        // 创建索引并排序
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

        // 将top_k之后的概率设为0
        for &idx in &indices[top_k..] {
            probs[idx] = 0.0;
        }

        // 重新归一化
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for prob in probs.iter_mut() {
                *prob /= sum;
            }
        }
    }

    /// 从概率分布中采样
    fn sample_from_probs(&self, probs: &[f32], rng: &mut Option<StdRng>) -> usize {
        let random_val = if let Some(rng_ref) = rng {
            rng_ref.gen::<f32>()
        } else {
            StdRng::from_entropy().gen::<f32>()
        };

        let mut cumulative = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if random_val <= cumulative {
                return i;
            }
        }

        probs.len() - 1
    }

    /// 找到最大值的索引
    fn find_max_index(&self, logits: &[f32]) -> Option<usize> {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
    }

    /// SIMD exp近似计算
    #[cfg(target_arch = "x86_64")]
    unsafe fn simd_exp(&self, x: __m128) -> __m128 {
        // 使用多项式近似计算exp
        const A: f32 = 1.0;
        const B: f32 = 1.0;
        const C: f32 = 0.5;
        const D: f32 = 0.1666667;
        const E: f32 = 0.04166667;

        let x2 = _mm_mul_ps(x, x);
        let x3 = _mm_mul_ps(x2, x);
        let x4 = _mm_mul_ps(x3, x);

        let term1 = _mm_set1_ps(A);
        let term2 = _mm_mul_ps(_mm_set1_ps(B), x);
        let term3 = _mm_mul_ps(_mm_set1_ps(C), x2);
        let term4 = _mm_mul_ps(_mm_set1_ps(D), x3);
        let term5 = _mm_mul_ps(_mm_set1_ps(E), x4);

        let result = _mm_add_ps(term1, term2);
        let result = _mm_add_ps(result, term3);
        let result = _mm_add_ps(result, term4);
        let result = _mm_add_ps(result, term5);

        result
    }
}

/// 内存池用于避免频繁分配
struct VecPool<T> {
    pool: Vec<Vec<T>>,
    max_size: usize,
}

impl<T> VecPool<T> {
    fn new(max_size: usize) -> Self {
        Self {
            pool: Vec::new(),
            max_size,
        }
    }

    fn get_with_capacity(&mut self, capacity: usize) -> Vec<T> {
        if let Some(mut vec) = self.pool.pop() {
            vec.clear();
            vec.reserve(capacity);
            vec
        } else {
            Vec::with_capacity(capacity)
        }
    }

    fn return_vec(&mut self, vec: Vec<T>) {
        if self.pool.len() < self.max_size {
            self.pool.push(vec);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_fast_path_detection() {
        let sampler = FastSampler::new();
        let config = SamplingConfig::default();

        // 测试明显主导的情况
        let mut logits = vec![-1.0; 100];
        logits[50] = 10.0; // 明显的主导token

        assert!(sampler.should_use_fast_path(&logits, &config));

        // 测试平局的情况
        let logits = vec![1.0; 100];
        assert!(!sampler.should_use_fast_path(&logits, &config));
    }

    #[test]
    fn test_fast_path_sampling() {
        let sampler = FastSampler::new();

        let mut logits = vec![-1.0; 100];
        logits[42] = 5.0;

        let result = sampler.fast_path_sample(&logits, None);
        assert_eq!(result, 42);

        // 测试禁止token
        let result = sampler.fast_path_sample(&logits, Some(42));
        assert_ne!(result, 42);
    }

    #[test]
    fn test_sampling_consistency() {
        let sampler = FastSampler::new();
        let config = SamplingConfig::default();
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut rng = Some(StdRng::seed_from_u64(12345));
        let result1 = sampler.optimized_sample(&logits, &config, None, &mut rng);

        let mut rng = Some(StdRng::seed_from_u64(12345));
        let result2 = sampler.optimized_sample(&logits, &config, None, &mut rng);

        assert_eq!(result1, result2);
    }
}
