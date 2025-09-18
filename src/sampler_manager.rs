//! 采样管理模块
//!
//! 本模块负责处理TTS推理中的参数处理和采样逻辑，
//! 包括采样策略、参数验证和结果后处理等。

use anyhow::Result;
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
// 删除未使用的导入

use crate::batch_types::TtsInferOptions;

/// 采样策略枚举
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// 贪心采样（选择概率最高的token）
    Greedy,
    /// Top-k采样
    TopK { k: usize },
    /// Top-p采样（核采样）
    TopP { p: f32 },
    /// 温度采样
    Temperature { temperature: f32 },
    /// 混合采样策略
    Mixed {
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
    },
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        Self::Mixed {
            temperature: 1.0,
            top_k: Some(50),
            top_p: Some(0.9),
        }
    }
}

/// 采样参数
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// 采样策略
    pub strategy: SamplingStrategy,
    /// 随机种子
    pub seed: Option<u64>,
    /// 最大生成长度
    pub max_length: usize,
    /// 停止token列表
    pub stop_tokens: Vec<u16>,
    /// 重复惩罚
    pub repetition_penalty: f32,
    /// 频率惩罚
    pub frequency_penalty: f32,
    /// 存在惩罚
    pub presence_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            strategy: SamplingStrategy::default(),
            seed: None,
            max_length: 512,
            stop_tokens: vec![],
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

/// 采样管理器
pub struct SamplerManager {
    /// 默认采样参数
    default_params: SamplingParams,
    /// 参数缓存（使用RwLock优化读多写少场景）
    params_cache: Arc<tokio::sync::RwLock<HashMap<String, SamplingParams>>>,
    /// 随机数生成器
    rng: Arc<Mutex<rand::rngs::StdRng>>,
}

impl SamplerManager {
    /// 创建新的采样管理器
    pub fn new(default_params: Option<SamplingParams>) -> Self {
        use rand::SeedableRng;

        Self {
            default_params: default_params.unwrap_or_default(),
            params_cache: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            rng: Arc::new(Mutex::new(rand::rngs::StdRng::from_entropy())),
        }
    }

    /// 从推理选项解析采样参数
    pub async fn parse_sampling_params(&self, options: &TtsInferOptions) -> Result<SamplingParams> {
        let mut params = self.default_params.clone();

        // 直接从选项中获取参数
        params.strategy = match params.strategy {
            SamplingStrategy::Mixed { top_k, top_p, .. } => SamplingStrategy::Mixed {
                temperature: options.temperature,
                top_k,
                top_p,
            },
            _ => SamplingStrategy::Temperature {
                temperature: options.temperature,
            },
        };

        // 更新策略参数
        params.strategy = match params.strategy {
            SamplingStrategy::Mixed { .. } => SamplingStrategy::Mixed {
                temperature: options.temperature,
                top_k: Some(options.top_k),
                top_p: Some(options.top_p),
            },
            _ => SamplingStrategy::Mixed {
                temperature: options.temperature,
                top_k: Some(options.top_k),
                top_p: Some(options.top_p),
            },
        };

        // 验证参数
        self.validate_params(&params)?;

        // 解析采样参数
        Ok(params)
    }

    /// 验证采样参数
    fn validate_params(&self, params: &SamplingParams) -> Result<()> {
        match &params.strategy {
            SamplingStrategy::TopK { k } => {
                if *k == 0 {
                    return Err(anyhow::anyhow!("top_k 必须大于 0"));
                }
            }
            SamplingStrategy::TopP { p } => {
                if *p <= 0.0 || *p > 1.0 {
                    return Err(anyhow::anyhow!("top_p 必须在 (0, 1] 范围内"));
                }
            }
            SamplingStrategy::Temperature { temperature } => {
                if *temperature <= 0.0 {
                    return Err(anyhow::anyhow!("temperature 必须大于 0"));
                }
            }
            SamplingStrategy::Mixed {
                temperature,
                top_k,
                top_p,
            } => {
                if *temperature <= 0.0 {
                    return Err(anyhow::anyhow!("temperature 必须大于 0"));
                }
                if let Some(k) = top_k {
                    if *k == 0 {
                        return Err(anyhow::anyhow!("top_k 必须大于 0"));
                    }
                }
                if let Some(p) = top_p {
                    if *p <= 0.0 || *p > 1.0 {
                        return Err(anyhow::anyhow!("top_p 必须在 (0, 1] 范围内"));
                    }
                }
            }
            SamplingStrategy::Greedy => {}
        }

        if params.max_length == 0 {
            return Err(anyhow::anyhow!("max_length 必须大于 0"));
        }

        if params.repetition_penalty < 0.0 {
            return Err(anyhow::anyhow!("repetition_penalty 必须非负"));
        }

        Ok(())
    }

    /// 执行采样
    pub async fn sample(
        &self,
        logits: &[f32],
        params: &SamplingParams,
        generated_tokens: &[u16],
    ) -> Result<u16> {
        let mut processed_logits = logits.to_vec();

        // 应用惩罚
        self.apply_penalties(&mut processed_logits, params, generated_tokens)?;

        // 根据策略采样
        let token = match &params.strategy {
            SamplingStrategy::Greedy => self.greedy_sample(&processed_logits)?,
            SamplingStrategy::TopK { k } => {
                self.top_k_sample(&processed_logits, *k, params.seed)
                    .await?
            }
            SamplingStrategy::TopP { p } => {
                self.top_p_sample(&processed_logits, *p, params.seed)
                    .await?
            }
            SamplingStrategy::Temperature { temperature } => {
                self.temperature_sample(&processed_logits, *temperature, params.seed)
                    .await?
            }
            SamplingStrategy::Mixed {
                temperature,
                top_k,
                top_p,
            } => {
                self.mixed_sample(&processed_logits, *temperature, *top_k, *top_p, params.seed)
                    .await?
            }
        };

        // 采样结果
        Ok(token)
    }

    /// 应用各种惩罚
    fn apply_penalties(
        &self,
        logits: &mut [f32],
        params: &SamplingParams,
        generated_tokens: &[u16],
    ) -> Result<()> {
        if params.repetition_penalty != 1.0 {
            self.apply_repetition_penalty(logits, params.repetition_penalty, generated_tokens);
        }

        if params.frequency_penalty != 0.0 {
            self.apply_frequency_penalty(logits, params.frequency_penalty, generated_tokens);
        }

        if params.presence_penalty != 0.0 {
            self.apply_presence_penalty(logits, params.presence_penalty, generated_tokens);
        }

        Ok(())
    }

    /// 应用重复惩罚
    fn apply_repetition_penalty(&self, logits: &mut [f32], penalty: f32, generated_tokens: &[u16]) {
        for &token in generated_tokens {
            let idx = token as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= penalty;
                } else {
                    logits[idx] *= penalty;
                }
            }
        }
    }

    /// 应用频率惩罚
    fn apply_frequency_penalty(&self, logits: &mut [f32], penalty: f32, generated_tokens: &[u16]) {
        let mut token_counts = HashMap::new();
        for &token in generated_tokens {
            *token_counts.entry(token).or_insert(0) += 1;
        }

        for (&token, &count) in &token_counts {
            let idx = token as usize;
            if idx < logits.len() {
                logits[idx] -= penalty * count as f32;
            }
        }
    }

    /// 应用存在惩罚
    fn apply_presence_penalty(&self, logits: &mut [f32], penalty: f32, generated_tokens: &[u16]) {
        let mut seen_tokens = std::collections::HashSet::new();
        for &token in generated_tokens {
            seen_tokens.insert(token);
        }

        for &token in &seen_tokens {
            let idx = token as usize;
            if idx < logits.len() {
                logits[idx] -= penalty;
            }
        }
    }

    /// 贪心采样
    fn greedy_sample(&self, logits: &[f32]) -> Result<u16> {
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| anyhow::anyhow!("空的logits向量"))?;

        Ok(max_idx as u16)
    }

    /// Top-k采样
    async fn top_k_sample(&self, logits: &[f32], k: usize, seed: Option<u64>) -> Result<u16> {
        let mut indexed_logits: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &logit)| (i, logit))
            .collect();

        // 按logit值降序排序
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 取前k个
        indexed_logits.truncate(k);

        // 计算概率并采样
        let probs = self.softmax(
            &indexed_logits
                .iter()
                .map(|(_, logit)| *logit)
                .collect::<Vec<_>>(),
        );
        let selected_idx = self.sample_from_probs(&probs, seed).await?;

        Ok(indexed_logits[selected_idx].0 as u16)
    }

    /// Top-p采样
    async fn top_p_sample(&self, logits: &[f32], p: f32, seed: Option<u64>) -> Result<u16> {
        let mut indexed_logits: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &logit)| (i, logit))
            .collect();

        // 按logit值降序排序
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 计算概率
        let logit_values: Vec<f32> = indexed_logits.iter().map(|(_, logit)| *logit).collect();
        let probs = self.softmax(&logit_values);

        // 找到累积概率超过p的位置
        let mut cumsum = 0.0;
        let mut cutoff = indexed_logits.len();
        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if cumsum >= p {
                cutoff = i + 1;
                break;
            }
        }

        // 截断并重新归一化
        indexed_logits.truncate(cutoff);
        let truncated_probs = &probs[..cutoff];
        let sum: f32 = truncated_probs.iter().sum();
        let normalized_probs: Vec<f32> = truncated_probs.iter().map(|&p| p / sum).collect();

        let selected_idx = self.sample_from_probs(&normalized_probs, seed).await?;
        Ok(indexed_logits[selected_idx].0 as u16)
    }

    /// 温度采样
    async fn temperature_sample(
        &self,
        logits: &[f32],
        temperature: f32,
        seed: Option<u64>,
    ) -> Result<u16> {
        let scaled_logits: Vec<f32> = logits.iter().map(|&logit| logit / temperature).collect();
        let probs = self.softmax(&scaled_logits);
        let selected_idx = self.sample_from_probs(&probs, seed).await?;
        Ok(selected_idx as u16)
    }

    /// 混合采样策略
    async fn mixed_sample(
        &self,
        logits: &[f32],
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
        seed: Option<u64>,
    ) -> Result<u16> {
        let mut working_logits = logits.to_vec();

        // 应用温度
        for logit in &mut working_logits {
            *logit /= temperature;
        }

        // 应用top_k过滤
        if let Some(k) = top_k {
            let mut indexed_logits: Vec<(usize, f32)> = working_logits
                .iter()
                .enumerate()
                .map(|(i, &logit)| (i, logit))
                .collect();

            indexed_logits
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // 将非top-k的logits设为负无穷
            for i in k..working_logits.len() {
                if let Some((original_idx, _)) = indexed_logits.get(i) {
                    working_logits[*original_idx] = f32::NEG_INFINITY;
                }
            }
        }

        // 应用top_p过滤
        if let Some(p) = top_p {
            let mut indexed_logits: Vec<(usize, f32)> = working_logits
                .iter()
                .enumerate()
                .filter(|(_, &logit)| logit != f32::NEG_INFINITY)
                .map(|(i, &logit)| (i, logit))
                .collect();

            indexed_logits
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let logit_values: Vec<f32> = indexed_logits.iter().map(|(_, logit)| *logit).collect();
            let probs = self.softmax(&logit_values);

            let mut cumsum = 0.0;
            for (i, &prob) in probs.iter().enumerate() {
                cumsum += prob;
                if cumsum >= p {
                    // 将超出top_p的logits设为负无穷
                    for (original_idx, _) in indexed_logits.iter().skip(i + 1) {
                        working_logits[*original_idx] = f32::NEG_INFINITY;
                    }
                    break;
                }
            }
        }

        // 最终采样
        let probs = self.softmax(&working_logits);
        let selected_idx = self.sample_from_probs(&probs, seed).await?;
        Ok(selected_idx as u16)
    }

    /// Softmax函数
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum: f32 = exp_logits.iter().sum();
        exp_logits.iter().map(|&x| x / sum).collect()
    }

    /// 从概率分布中采样
    async fn sample_from_probs(&self, probs: &[f32], seed: Option<u64>) -> Result<usize> {
        // 优化：减少锁持有时间，只在需要生成随机数时获取锁
        let random_val = if let Some(s) = seed {
            // 如果提供了种子，创建新的RNG，无需获取锁
            use rand::SeedableRng;
            let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(s);
            seeded_rng.gen::<f32>()
        } else {
            // 只在生成随机数时短暂持有锁
            let mut rng = self.rng.lock().await;
            let val = rng.gen::<f32>();
            drop(rng); // 显式释放锁
            val
        };

        let mut cumsum = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if random_val <= cumsum {
                return Ok(i);
            }
        }

        // 如果由于浮点精度问题没有选中任何token，返回最后一个
        Ok(probs.len() - 1)
    }

    /// 检查是否应该停止生成
    pub fn should_stop(&self, generated_tokens: &[u16], params: &SamplingParams) -> bool {
        // 检查长度限制
        if generated_tokens.len() >= params.max_length {
            return true;
        }

        // 检查停止token
        if let Some(&last_token) = generated_tokens.last() {
            if params.stop_tokens.contains(&last_token) {
                return true;
            }
        }

        false
    }

    /// 获取采样统计信息
    pub async fn stats(&self) -> SamplerStats {
        let cache = self.params_cache.read().await;
        SamplerStats {
            cached_params: cache.len(),
        }
    }
}

/// 采样器统计信息
#[derive(Debug, Clone)]
pub struct SamplerStats {
    pub cached_params: usize,
}
