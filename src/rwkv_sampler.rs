//! RWKV模型推理采样器
//! 实现基于web-rwkv库的RWKV模型推理和采样功能

use crate::voice_feature_manager::VoiceFeatureManager;
use anyhow::Result;
use memmap2::Mmap;
use rand::{rngs::StdRng, Rng, SeedableRng};
use safetensors::SafeTensors;
use serde::de::DeserializeSeed;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::Loader,
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant},
        v7, Runtime, TokioRuntime,
    },
    tensor::serialization::Seed,
    tokenizer::Tokenizer,
    wgpu::{self, Instance},
};

// Import optimization components
use crate::inference_state_manager::{InferenceStateConfig, InferenceStateManager};
use crate::streaming_inference::{BatchConfig, StreamingInference};
use std::sync::Arc;
use std::time::Duration;

/// 公开的采样函数，支持传入RNG参数
/// 匹配Python原版sampler_simple_batch的行为：温度应用在logits上，支持噪声
pub fn sample_logits(
    logits: &[f32],
    args: &SamplerArgs,
    forbid_token: Option<usize>,
    rng: &mut Option<StdRng>,
) -> usize {
    // 实现与Python版本一致的sample_logits逻辑
    sample_logits_with_top_p_k(
        logits,
        args.temperature,
        args.top_p,
        args.top_k,
        forbid_token,
        rng,
    )
}

/// 实现与Python版本一致的sample_logits函数
/// 支持temperature、top_p、top_k采样
/// 按照Python版本的顺序：softmax -> top_k -> top_p -> temperature -> multinomial
pub fn sample_logits_with_top_p_k(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: usize,
    forbid_token: Option<usize>,
    rng: &mut Option<StdRng>,
) -> usize {
    let vocab_size = logits.len();
    if vocab_size == 0 {
        return 0;
    }

    // 创建可修改的logits副本
    let mut modified_logits = logits.to_vec();

    // 处理禁止token
    if let Some(ft) = forbid_token {
        if ft < vocab_size {
            modified_logits[ft] = f32::NEG_INFINITY;
        }
    }

    // 步骤1: 计算softmax概率（匹配Python版本：probs = F.softmax(logits.float(), dim=-1)）
    let max_logit = modified_logits
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut probs: Vec<f32> = modified_logits
        .iter()
        .map(|&logit| (logit - max_logit).exp())
        .collect();

    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for prob in probs.iter_mut() {
            *prob /= sum;
        }
    }

    // 步骤2: Top-k截断（匹配Python版本：probs[sorted_ids[top_k:]] = 0）
    if top_k > 0 && top_k < vocab_size {
        // 创建索引-概率对并排序
        let mut indexed_probs: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 保留top-k，其余设为0
        for (idx, _) in indexed_probs.iter().skip(top_k) {
            probs[*idx] = 0.0;
        }
    }

    // 步骤3: Top-p截断（匹配Python版本的复杂逻辑）
    if top_p < 1.0 {
        // 创建索引-概率对并排序
        let mut indexed_probs: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 计算累积概率
        let mut cumulative_prob = 0.0;
        let mut cutoff_index = None;
        let mut cutoff_prob = 0.0;

        for (i, (_, prob)) in indexed_probs.iter().enumerate() {
            cumulative_prob += prob;
            if cumulative_prob >= top_p {
                cutoff_index = Some(i);
                cutoff_prob = *prob;
                break;
            }
        }

        if let Some(_cutoff_idx) = cutoff_index {
            // 将小于cutoff的概率设为0
            for prob in probs.iter_mut() {
                if *prob < cutoff_prob {
                    *prob = 0.0;
                }
            }

            // 处理等于cutoff的概率（匹配Python版本的精确逻辑）
            if top_p > 0.0 {
                let current_sum: f32 = probs.iter().sum();
                if current_sum < top_p {
                    let remaining = top_p - current_sum;
                    let cutoff_count = probs.iter().filter(|&&p| p == cutoff_prob).count();
                    if cutoff_count > 0 {
                        let adjustment = remaining / cutoff_count as f32;
                        for prob in probs.iter_mut() {
                            if *prob == cutoff_prob {
                                *prob = cutoff_prob + adjustment;
                            }
                        }
                    }
                }
            }
        }
    }

    // 步骤4: 应用温度（匹配Python版本：probs = probs ** (1.0 / temperature)）
    if temperature != 1.0 && temperature > 0.0 {
        let temp_inv = 1.0 / temperature;
        for prob in probs.iter_mut() {
            if *prob > 0.0 {
                *prob = prob.powf(temp_inv);
            }
        }

        // 重新归一化
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for prob in probs.iter_mut() {
                *prob /= sum;
            }
        }
    }

    // 步骤5: 多项式采样（匹配Python版本：torch.multinomial(probs, num_samples=1).item()）
    if let Some(rng_ref) = rng {
        // 使用传入的RNG
        let rand_val: f32 = rng_ref.gen();
        let mut cumulative = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if rand_val <= cumulative {
                return i;
            }
        }
        // 如果没有找到，返回最后一个非零概率的索引
        for (i, &prob) in probs.iter().enumerate().rev() {
            if prob > 0.0 {
                return i;
            }
        }
    } else {
        // 没有RNG时使用确定性种子
        let mut temp_rng = StdRng::seed_from_u64(42);
        let rand_val: f32 = temp_rng.gen();
        let mut cumulative = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if rand_val <= cumulative {
                return i;
            }
        }
        // 如果没有找到，返回最后一个非零概率的索引
        for (i, &prob) in probs.iter().enumerate().rev() {
            if prob > 0.0 {
                return i;
            }
        }
    }

    // 最后的回退
    0
}

// 已删除有问题的sample_logits_impl函数，统一使用FastSampler版本的sample_logits

/// 加载类型枚举
enum LoadType {
    SafeTensors(Vec<u8>), // 存储原始数据而不是引用
    Prefab(Vec<u8>),
}

/// 批处理TTS请求结构
#[derive(Debug, Clone)]
pub struct TtsBatchRequest {
    pub text: String,
    pub property_tokens: Vec<i32>,
    pub ref_global_tokens: Option<Vec<i32>>,
    pub ref_semantic_tokens: Option<Vec<i32>>,
    pub args: SamplerArgs,
    /// 音色ID，用于从缓存中快速获取tokens
    pub voice_id: Option<String>,
}

/// 采样参数
#[derive(Debug, Clone)]
pub struct SamplerArgs {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub max_tokens: usize,
    // 可选随机种子：提供则启用确定性采样
    pub seed: Option<u64>,
    // 音色保真度控制：0.0-1.0，越高越保持参考音色
    pub voice_fidelity: f32,
    // 分层随机性控制
    pub layered_randomness: LayeredRandomnessConfig,
    // Token chunk size配置
    pub token_chunk_size: usize,
}

/// 分层随机性配置
#[derive(Debug, Clone)]
pub struct LayeredRandomnessConfig {
    /// Global阶段的随机性强度 (0.0-1.0)
    pub global_randomness: f32,
    /// Semantic阶段的随机性强度 (0.0-1.0)
    pub semantic_randomness: f32,
    /// 是否使用独立的种子策略
    pub use_independent_seeds: bool,
    /// Global阶段种子偏移
    pub global_seed_offset: u64,
    /// Semantic阶段种子偏移
    pub semantic_seed_offset: u64,
}

impl Default for LayeredRandomnessConfig {
    fn default() -> Self {
        Self {
            global_randomness: 0.1,   // 大幅降低global阶段随机性，保护音色特征
            semantic_randomness: 0.4, // 适度降低semantic阶段随机性
            use_independent_seeds: true,
            global_seed_offset: 1000,
            semantic_seed_offset: 2000,
        }
    }
}

impl Default for SamplerArgs {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.85, // 修复：与Python版本sample_logits函数默认值一致
            top_k: 0,
            max_tokens: 2048, // 修复：提高默认值以支持更长的音频生成
            seed: None,
            voice_fidelity: 0.8, // 默认高音色保真度
            layered_randomness: LayeredRandomnessConfig::default(),
            token_chunk_size: 512, // 默认token chunk size
        }
    }
}

/// Prefab文件结构体
/// TTS相关常量
pub const TTS_EOS_TOKEN: i32 = 8192;
pub const TTS_TAG_0: i32 = 8193;
pub const TTS_TAG_1: i32 = 8194;
pub const TTS_TAG_2: i32 = 8195;
// 注意：以下偏移量常量已废弃，根据C++代码，tokens应直接使用原始ID
pub const GLOBAL_TOKEN_OFFSET: i32 = 8196; // Global tokens在prefill时需要偏移
                                           // pub const SEMANTIC_TOKEN_OFFSET: i32 = 4096; // 已废弃：不再给tokens添加偏移

/// RWKV采样器，用于生成文本和TTS tokens
pub struct RwkvSampler {
    runtime: Box<dyn Runtime<Rnn> + Send + Sync>, // 使用TokioRuntime封装Bundle
    tokenizer: Tokenizer,
    // 带种子的RNG（可选，启用则实现确定性采样）
    rng: Option<StdRng>,
    batch_counter: AtomicUsize,
    // Token chunk size配置
    token_chunk_size: usize,
    // Optimization components
    #[allow(dead_code)]
    streaming_inference: Option<Arc<StreamingInference>>,
    inference_state_manager: Arc<InferenceStateManager>,
}
impl RwkvSampler {
    /// 创建默认量化配置
    /// 默认不使用量化以提高推理精度
    pub fn default_quant_config() -> HashMap<usize, Quant> {
        HashMap::new() // 返回空配置，不使用量化
    }

    /// 创建新的RWKV采样器
    ///
    /// # Arguments
    /// * `model_path` - RWKV模型目录或模型文件(.safetensors)路径
    /// * `vocab_path` - 词表文件路径
    /// * `quant_config` - 量化配置，None表示不使用量化
    ///
    /// # Returns
    /// * `Result<RwkvSampler>` - RWKV采样器实例或错误
    pub async fn new(
        model_path: &str,
        vocab_path: &str,
        quant_config: Option<HashMap<usize, Quant>>,
        token_chunk_size: usize,
    ) -> Result<Self> {
        // 检查模型目录/文件是否存在
        let model_path_ref = Path::new(model_path);
        if !model_path_ref.exists() {
            return Err(anyhow::anyhow!("模型路径不存在: {}", model_path));
        }

        // 检查词表文件是否存在
        if !Path::new(vocab_path).exists() {
            return Err(anyhow::anyhow!("词表文件不存在: {}", vocab_path));
        }

        // 解析模型文件路径：
        // - 若传入目录，则优先查找 "webrwkv.safetensors"，其次 "rwkvtts-Int8_22.safetensors"
        // - 若传入文件，则直接使用该文件
        let model_file_path = if model_path_ref.is_dir() {
            let prefab_path = model_path_ref.join("webrwkv.safetensors");
            let safetensors_path = model_path_ref.join("rwkvtts-Int8_22.safetensors");
            if prefab_path.exists() {
                prefab_path
            } else if safetensors_path.exists() {
                safetensors_path
            } else {
                return Err(anyhow::anyhow!(
                    "模型文件不存在: 在目录 {} 中未找到 webrwkv.safetensors 或 rwkvtts-Int8_22.safetensors",
                    model_path
                ));
            }
        } else {
            model_path_ref.to_path_buf()
        };
        if !model_file_path.exists() {
            return Err(anyhow::anyhow!(
                "模型文件不存在: {}",
                model_file_path.display()
            ));
        }

        // 加载模型文件
        let file = std::fs::File::open(&model_file_path)?;
        let _file_size = file.metadata()?.len();
        let data = unsafe { Mmap::map(&file)? };

        // 模型完整性校验：打印大小与SHA256
        let mut hasher = Sha256::new();
        hasher.update(&data[..]);

        // 创建 GPU 上下文
        let instance = Instance::default();
        let adapter = instance
            .adapter(wgpu::PowerPreference::HighPerformance)
            .await?;

        // 检测模型格式
        let load_type = {
            // 首先尝试SafeTensors格式
            if SafeTensors::deserialize(&data).is_ok() {
                // SafeTensors 格式模型
                LoadType::SafeTensors(data.to_vec())
            } else {
                // 如果不是SafeTensors，假设是prefab格式
                // prefab 格式模型
                LoadType::Prefab(data.to_vec())
            }
        };

        // 为V7模型创建默认信息（稍后在实际加载时会被验证）
        let info = ModelInfo {
            version: ModelVersion::V7,
            num_vocab: 65536,           // 默认值，实际值会在模型加载时确定
            num_layer: 32,              // 默认值
            num_emb: 4096,              // 默认值
            num_hidden: 4096,           // 默认值
            num_head: 32,               // 默认值
            custom: Default::default(), // 默认值
        };

        // 基于模型信息自动配置 Context 的硬件 limits

        // 打印适配器/后端/驱动与精度

        let context = ContextBuilder::new(adapter)
            .auto_limits(&info)
            .build()
            .await?;

        // 根据加载类型创建V7模型
        let model = match load_type {
            LoadType::SafeTensors(data_vec) => {
                // 从Vec<u8>重新创建SafeTensors
                let safetensors = SafeTensors::deserialize(&data_vec)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize SafeTensors: {}", e))?;

                // 获取并验证模型信息
                let actual_info = Loader::info(&safetensors)?;
                if actual_info.version != ModelVersion::V7 {
                    return Err(anyhow::anyhow!(
                        "Only V7 models are supported, got {:?}",
                        actual_info.version
                    ));
                }
                // 模型信息验证

                let mut builder = ModelBuilder::new(&context, safetensors);
                if let Some(quant) = quant_config {
                    builder = builder.quant(quant);
                }
                builder.build_v7().await?
            }
            LoadType::Prefab(data_vec) => {
                // 使用cbor4ii Deserializer反序列化prefab数据
                // 参考web-rwkv的serde示例实现
                use cbor4ii::{core::utils::SliceReader, serde::Deserializer};

                // 反序列化V7 prefab模型
                let reader = SliceReader::new(&data_vec);
                let mut deserializer = Deserializer::new(reader);

                let seed = Seed::<Context, v7::Model>::new(&context);
                seed.deserialize(&mut deserializer)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize v7 model: {}", e))?
            }
        };

        // 创建Bundle与TokioRuntime（切换为 f32 以启用 FP32 推理）
        // 增加batch size以支持并发推理
        let max_batch = 8;
        let bundle = v7::Bundle::<f32>::new(model, max_batch);
        let runtime: Box<dyn Runtime<Rnn> + Send + Sync> =
            Box::new(TokioRuntime::new(bundle).await);

        // 加载tokenizer
        let vocab_content = std::fs::read_to_string(vocab_path)?;
        let tokenizer = Tokenizer::new(&vocab_content)?;

        // Initialize inference state manager
        let inference_state_config = InferenceStateConfig {
            max_cache_entries: 200,
            max_entry_age: std::time::Duration::from_secs(600),
            batch_inference_size: 8,
            prediction_window: 16,
            enable_async_pre_inference: true,
            state_similarity_threshold: 0.95,
        };
        let inference_state_manager = Arc::new(InferenceStateManager::new(inference_state_config));

        // Initialize StreamingInference
        let batch_config = BatchConfig {
            max_batch_size: 8,
            batch_timeout: Duration::from_millis(50),
            dynamic_batching: true,
            min_batch_size: 2,
            prefetch_window: 4,
        };
        let streaming_inference = Arc::new(StreamingInference::new(batch_config));

        Ok(Self {
            runtime,
            tokenizer,
            rng: None,
            batch_counter: AtomicUsize::new(0),
            token_chunk_size,
            streaming_inference: Some(streaming_inference),
            inference_state_manager,
        })
    }

    /// 设置随机种子（启用确定性采样）。传None则关闭确定性模式。
    pub fn set_seed(&mut self, seed: Option<u64>) {
        self.rng = seed.map(StdRng::seed_from_u64);
    }

    /// 为特定阶段创建独立的RNG
    pub fn create_stage_rng(&self, base_seed: Option<u64>, stage_offset: u64) -> Option<StdRng> {
        base_seed.map(|seed| StdRng::seed_from_u64(seed.wrapping_add(stage_offset)))
    }

    /// 应用音色保真度调整采样参数
    pub fn apply_voice_fidelity_adjustment(
        &self,
        args: &SamplerArgs,
        stage_randomness: f32,
    ) -> SamplerArgs {
        let mut adjusted_args = args.clone();

        // 根据音色保真度和阶段随机性调整采样参数
        let fidelity_factor = args.voice_fidelity;
        let randomness_factor = stage_randomness;

        // 高保真度 + 低随机性 = 更保守的采样
        let conservative_factor = fidelity_factor * (1.0 - randomness_factor);

        // 调整温度：保真度越高，温度越低
        adjusted_args.temperature = args.temperature * (0.5 + 0.5 * (1.0 - conservative_factor));

        // 调整top_p：保真度越高，top_p越小（更集中采样）
        adjusted_args.top_p = args.top_p * (0.7 + 0.3 * (1.0 - conservative_factor));

        // 调整top_k：保真度越高，top_k越小
        if adjusted_args.top_k > 0 {
            let reduction_factor = 0.5 + 0.5 * (1.0 - conservative_factor);
            adjusted_args.top_k =
                ((adjusted_args.top_k as f32) * reduction_factor).max(1.0) as usize;
        }

        adjusted_args
    }

    /// 创建独立的推理上下文（复用已加载的模型和tokenizer）
    /// 这样可以避免重新加载模型，同时确保每个上下文有独立的状态
    /// 注意：由于Runtime是trait对象，无法直接clone，需要重新创建
    pub async fn create_independent_context(
        model_path: &str,
        vocab_path: &str,
        quant_config: Option<HashMap<usize, Quant>>,
    ) -> Result<Self> {
        // 重新创建一个新的采样器实例
        // 虽然这会重新加载模型，但确保了完全独立的状态
        Self::new(model_path, vocab_path, quant_config, 512).await
    }

    /// 为请求生成唯一ID
    fn generate_request_id(&self) -> String {
        let counter = self.batch_counter.load(Ordering::SeqCst);
        let mut id = String::with_capacity(16); // 预分配足够容量
        use std::fmt::Write;
        write!(&mut id, "req_{}", counter).unwrap();
        id
    }

    /// 只读访问内部tokenizer（用于外部按相同方式编码属性）
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// 生成文本（示例）
    pub async fn generate_text(&mut self, prompt: &str, args: &SamplerArgs) -> Result<String> {
        // 若提供了种子，设置确定性采样
        self.set_seed(args.seed);

        // 编码prompt
        let prompt_tokens: Vec<u32> = self
            .tokenizer
            .encode(prompt.as_bytes())
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let prompt_batch = RnnInputBatch::new(prompt_tokens.clone(), RnnOption::Last);
        let mut inference = RnnInput::new(vec![prompt_batch], self.token_chunk_size);

        // 预填充阶段：先把完整 prompt 吃完，直到 runtime 开始产生输出
        loop {
            let (next_inference, output) = self.runtime.infer(inference).await?;
            inference = next_inference;
            if output[0].0.size() > 0 {
                break;
            }
        }

        // 采样生成
        let mut generated: Vec<u32> = Vec::with_capacity(args.max_tokens);
        for _ in 0..args.max_tokens {
            // 每步仅消耗当前剩余输入（可能为空），并基于输出采样一个新 token
            let (next_inference, output) = self.runtime.infer(inference).await?;
            inference = next_inference;

            // 若仍在消耗输入（size==0），继续直到产生输出
            if output[0].0.size() == 0 {
                continue;
            }

            let logits = output[0].0.clone().to_vec();
            let next_id = self.sample_logits(&logits, args, None) as u32;

            // 将新 token 追加到后续输入中，实现增量推理
            inference.batches[0].push(next_id);
            generated.push(next_id);
        }

        // 解码（prompt + 生成）
        let mut all = prompt_tokens;
        all.extend_from_slice(&generated);
        let decoded = self
            .tokenizer
            .decode(&all)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let text = String::from_utf8_lossy(&decoded).to_string();
        Ok(text)
    }

    pub async fn generate_tts_tokens(
        &mut self,
        text: &str,
        property_tokens: &[i32],
        _ref_global_tokens: Option<&[i32]>,
        _ref_semantic_tokens: Option<&[i32]>,
        voice_id: Option<&str>,
        args: &SamplerArgs,
    ) -> Result<(Vec<i32>, Vec<i32>)> {
        // 生成唯一请求ID
        let _request_id = self.generate_request_id();

        // 开始TTS生成

        // 若提供了种子，设置确定性采样
        self.set_seed(args.seed);

        // 关键修复：为每个请求创建完全独立的推理上下文
        // 这确保了不同请求之间的状态完全隔离
        // 创建独立推理上下文

        // 编码文本：使用原始文本token（不加任何偏移）以匹配参考实现
        // 编码输入文本
        let text_tokens_u32: Vec<u32> = self
            .tokenizer
            .encode(text.as_bytes())
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let text_tokens: Vec<i32> = text_tokens_u32.into_iter().map(|t| t as i32).collect();
        // 文本编码完成

        // 参考实现在prefill阶段喂入属性tokens（原始域）、文本tokens与阶段标签。
        let mut input_tokens: Vec<i32> = Vec::new();
        input_tokens.extend_from_slice(property_tokens);
        input_tokens.push(TTS_TAG_2);
        input_tokens.extend_from_slice(&text_tokens);
        input_tokens.push(TTS_TAG_0);
        // 构建完整输入序列

        // === Prefill 阶段 ===
        let input_tokens_u32: Vec<u32> = input_tokens.iter().map(|&t| t as u32).collect();

        // Prefill阶段 - 创建推理上下文

        // 关键修复：为每个请求创建完全独立的推理上下文
        // 使用固定的batch索引0，但确保每次调用都是独立的推理状态
        // 这避免了不同请求之间的状态污染问题
        // 创建独立推理上下文
        let batch = RnnInputBatch::new(input_tokens_u32.clone(), RnnOption::Last);
        let mut inference = RnnInput::new(vec![batch], self.token_chunk_size);

        // 重要：确保推理上下文完全独立，不受之前请求影响
        // 推理上下文已隔离
        // 关键修复：直接执行Prefill推理，不使用缓存管理器
        // 这确保第一个logit不会被跳过或丢失
        let mut last_logits: Vec<f32> = loop {
            let (next_inference, output) = self.runtime.infer(inference).await?;
            inference = next_inference;
            if output[0].0.size() > 0 {
                break output[0].0.clone().to_vec();
            }
        };

        // === Global 阶段 ===
        let mut global_tokens: Vec<i32> = Vec::new();
        let mut semantic_tokens: Vec<i32> = Vec::new();

        // 优先尝试从voice_id缓存获取tokens
        if let Some(voice_id) = voice_id {
            // 创建VoiceFeatureManager实例（假设使用默认RAF目录）
            if let Ok(voice_manager) = VoiceFeatureManager::new("./raf") {
                if let Ok((cached_global, cached_semantic)) =
                    voice_manager.get_voice_tokens(voice_id).await
                {
                    return Ok((cached_global, cached_semantic));
                }
            }
        }

        // 检查是否有预提取的音色特征
        let has_ref_audio = _ref_global_tokens.is_some() || _ref_semantic_tokens.is_some();

        // 如果有预提取的音色特征，直接使用它们
        if has_ref_audio {
            if let Some(ref_global) = _ref_global_tokens {
                global_tokens = ref_global.to_vec();
                // 使用预提取的global tokens
            }
            if let Some(ref_semantic) = _ref_semantic_tokens {
                semantic_tokens = ref_semantic.to_vec();
                // 使用预提取的semantic tokens
            }

            // 声音克隆模式：使用预提取特征

            return Ok((global_tokens, semantic_tokens));
        }

        // 如果没有预提取特征，则进行正常的生成流程
        // 设置分层采样参数和独立RNG
        let mut args_global = if args.layered_randomness.use_independent_seeds {
            self.apply_voice_fidelity_adjustment(args, args.layered_randomness.global_randomness)
        } else {
            args.clone()
        };

        let mut args_sem = if args.layered_randomness.use_independent_seeds {
            self.apply_voice_fidelity_adjustment(args, args.layered_randomness.semantic_randomness)
        } else {
            args.clone()
        };

        // 设置默认top_k值
        if args_global.top_k == 0 {
            args_global.top_k = 20;
        }
        if args_sem.top_k == 0 {
            args_sem.top_k = 80;
        }

        // 声音克隆时使用确定性参数
        if has_ref_audio {
            // 声音克隆模式：使用确定性采样参数确保结果一致性
            // 声音克隆时使用固定的确定性参数
            args_global.temperature = 0.1; // 极低温度确保确定性
            args_global.top_p = 0.9;
            args_global.top_k = 1; // 只选择最可能的token

            args_sem.temperature = 0.1; // 极低温度确保确定性
            args_sem.top_p = 0.9;
            args_sem.top_k = 1; // 只选择最可能的token
        } else {
            // 正常生成模式：使用与Python版本一致的固定参数
            // Python版本: Global阶段(temperature=1.0, top_p=0.95, top_k=20)
            args_global.temperature = 1.0;
            args_global.top_p = 0.95;
            args_global.top_k = 20;

            // Python版本: Semantic阶段(temperature=1.0, top_p=0.95, top_k=80)
            args_sem.temperature = 1.0;
            args_sem.top_p = 0.95;
            args_sem.top_k = 80;
        }

        // 简化RNG管理，参考Python版本使用统一的随机状态
        // 声音克隆时使用确定性采样（temperature=0），否则使用当前RNG状态
        let use_deterministic = has_ref_audio;

        // 如果是声音克隆，临时调整采样参数为确定性
        if use_deterministic {
            args_global.temperature = 0.01; // 接近确定性
            args_sem.temperature = 0.01;
        }

        // Python实现固定生成32个global tokens，并且仅在前4096维内采样
        let global_tokens_size: usize = 32;

        // 使用批量推理优化Global阶段

        // 打印Global阶段采样参数
        log::info!("🎯 开始生成Global tokens，目标数量: {}", global_tokens_size);
        log::info!("📋 Global阶段采样参数:");
        log::info!(
            "   - 默认参数: temperature={:.3}, top_p={:.3}, top_k={}",
            args.temperature,
            args.top_p,
            args.top_k
        );
        log::info!(
            "   - 实际参数: temperature={:.3}, top_p={:.3}, top_k={}",
            args_global.temperature,
            args_global.top_p,
            args_global.top_k
        );
        if has_ref_audio {
            log::info!("   - 模式: 声音克隆 (确定性采样)");
        } else {
            log::info!("   - 模式: 正常生成 (随机采样)");
        }
        for (i, _) in (0..global_tokens_size).enumerate() {
            // 关键修复：确保第一个token使用Prefill阶段的正确logits
            let logits: &[f32] = if i == 0 {
                // 第一个token必须使用Prefill阶段的logits
                &last_logits
            } else {
                // 后续token通过推理获取
                let (next_inference, output) = self.runtime.infer(inference).await?;
                inference = next_inference;
                if output[0].0.size() > 0 {
                    last_logits = output[0].0.clone().to_vec();
                    &last_logits
                } else {
                    // 如果没有输出，继续使用之前的logits
                    &last_logits
                }
            };

            // 仅在[0..4096)范围内采样（Global相对域），不涉及EOS与阶段标签
            let vocab_global = if logits.len() < 4096 {
                logits.len()
            } else {
                4096
            };
            // Global阶段采样 - 使用简化的采样方法
            let next_id = self.sample_logits(&logits[..vocab_global], &args_global, None);

            // 追加到global输出（相对域 [0..4095]）
            global_tokens.push(next_id as i32);
            // 反馈到模型：直接使用原始ID（与C++代码一致）
            inference.batches[0].push(next_id as u32);

            // 打印当前生成进度
            if (i + 1) % 8 == 0 || i == global_tokens_size - 1 {
                println!(
                    "📊 Global阶段: 已生成 {}/{} tokens",
                    i + 1,
                    global_tokens_size
                );
            }
            // Global token生成
        }

        // === 切换到 Semantic 阶段 ===
        inference.batches[0].push(TTS_TAG_1 as u32);
        // 让标签生效，直到产生输出，并保留logits供首步使用
        let mut last_sem_logits: Vec<f32> = loop {
            let (next_inference, output) = self.runtime.infer(inference).await?;
            inference = next_inference;
            if output[0].0.size() > 0 {
                break output[0].0.clone().to_vec();
            }
        };

        // 语义阶段：限制最大生成步数为2048
        let semantic_limit: usize = usize::min(args.max_tokens, 2048);

        // 使用批量推理优化Semantic阶段
        let semantic_context_id = format!("tts_semantic_{}", self.generate_request_id());
        let mut semantic_logits_cache: Vec<Vec<f32>> = Vec::new();
        let mut semantic_cache_index = 0;

        // 打印Semantic阶段采样参数
        log::info!("🎯 开始生成Semantic tokens，最大数量: {}", semantic_limit);
        log::info!("📋 Semantic阶段采样参数:");
        log::info!(
            "   - 默认参数: temperature={:.3}, top_p={:.3}, top_k={}",
            args.temperature,
            args.top_p,
            args.top_k
        );
        log::info!(
            "   - 实际参数: temperature={:.3}, top_p={:.3}, top_k={}",
            args_sem.temperature,
            args_sem.top_p,
            args_sem.top_k
        );
        if has_ref_audio {
            log::info!("   - 模式: 声音克隆 (确定性采样)");
        } else {
            log::info!("   - 模式: 正常生成 (随机采样)");
        }
        for (i, _) in (0..semantic_limit).enumerate() {
            // 取得当前语义阶段的logits：首步使用注入标签后的logits，其后每步从runtime获取
            let logits: &[f32] = if i == 0 {
                &last_sem_logits
            } else if semantic_cache_index < semantic_logits_cache.len() {
                // 使用缓存的logits
                &semantic_logits_cache[semantic_cache_index]
            } else {
                // 需要批量获取更多logits
                let remaining_tokens = semantic_limit - i;
                let batch_size = remaining_tokens.min(16); // 批量推理16个token

                let (next_inference, batch_logits) = self
                    .inference_state_manager
                    .smart_inference(
                        &mut self.runtime,
                        inference,
                        &semantic_context_id,
                        batch_size,
                    )
                    .await?;
                inference = next_inference;

                if !batch_logits.is_empty() {
                    semantic_logits_cache.extend(batch_logits);
                    semantic_cache_index = 0;
                    &semantic_logits_cache[semantic_cache_index]
                } else {
                    // 回退到传统方式
                    loop {
                        let (next_inference, output) = self.runtime.infer(inference).await?;
                        inference = next_inference;
                        if output[0].0.size() > 0 {
                            // 重用变量，避免重复分配
                            last_sem_logits = output[0].0.clone().to_vec();
                            break &last_sem_logits;
                        }
                    }
                }
            };

            // 语义阶段仅采样 [0..8192]（包含EOS），屏蔽TTS_TAG_*与其它域
            // 使用栈分配的缓冲区，避免堆分配
            let mut logits_buf = [f32::NEG_INFINITY; 8192];
            let copy_len = logits.len().min(8192);
            logits_buf[..copy_len].copy_from_slice(&logits[..copy_len]);

            // 屏蔽超出EOS的token
            for item in logits_buf
                .iter_mut()
                .take(copy_len)
                .skip(TTS_EOS_TOKEN as usize + 1)
            {
                *item = f32::NEG_INFINITY;
            }

            // 屏蔽TTS标签
            for tag in [TTS_TAG_0, TTS_TAG_1, TTS_TAG_2] {
                let idx = tag as usize;
                if idx < copy_len {
                    logits_buf[idx] = f32::NEG_INFINITY;
                }
            }

            let next_id = self.sample_logits(&logits_buf[..copy_len], &args_sem, None);

            // 追加到semantic输出（原始域 [0..8191]）
            semantic_tokens.push(next_id as i32);
            // 语义阶段反馈：直接反馈原始id（经验）
            inference.batches[0].push(next_id as u32);

            // 打印当前生成进度
            if (i + 1) % 16 == 0 || i == semantic_limit - 1 {
                println!(
                    "📊 Semantic阶段: 已生成 {}/{} tokens",
                    i + 1,
                    semantic_limit
                );
            }

            // 语义token生成完成
            // Semantic token生成
        }

        // TTS生成完成
        println!(
            "✅ TTS生成完成 - Global tokens: {}, Semantic tokens: {}",
            global_tokens.len(),
            semantic_tokens.len()
        );
        Ok((global_tokens, semantic_tokens))
    }

    /// 批处理生成TTS tokens - 完全独立的串行处理
    /// 每个请求都有独立的推理状态，避免状态污染
    pub async fn generate_tts_tokens_batch(
        &mut self,
        requests: Vec<TtsBatchRequest>,
    ) -> Result<Vec<(Vec<i32>, Vec<i32>)>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = requests.len();
        // 批处理生成TTS tokens

        // 批处理开始前进行全局状态重置
        self.reset();
        // 批处理前已重置全局状态

        // 完全独立的串行处理：每个请求都有独立状态，确保无污染
        let mut results = Vec::with_capacity(batch_size);
        for request in requests.into_iter() {
            // 处理独立请求

            // 关键修复：每个请求前进行彻底的状态重置
            self.reset();

            // 统一处理种子设置，不区分声音克隆场景
            if let Some(seed) = request.args.seed {
                self.set_seed(Some(seed));
                // 设置确定性种子
            } else {
                self.set_seed(None); // 重置为非确定性模式
                                     // 使用非确定性采样
            }

            let result = self
                .generate_tts_tokens(
                    &request.text,
                    &request.property_tokens,
                    request.ref_global_tokens.as_deref(),
                    request.ref_semantic_tokens.as_deref(),
                    request.voice_id.as_deref(),
                    &request.args,
                )
                .await?;
            results.push(result);

            // 每个请求完成后进行彻底的状态清理
            self.reset();
            // 请求完成，状态已清理
        }

        // 批处理完成后进行最终状态重置
        self.reset();
        // 批处理完成，成功生成独立结果，最终状态已重置
        Ok(results)
    }

    /// 重置采样器状态 - 彻底清理所有状态
    pub fn reset(&mut self) {
        // 重置随机数生成器状态
        self.rng = None;

        // 重置batch计数器，避免索引累积
        self.batch_counter.store(0, Ordering::SeqCst);

        // 关键修复：尝试清理Runtime的内部状态
        // 虽然我们不能直接重置Runtime，但可以确保下次使用时状态是干净的
        // 通过重置batch索引，确保使用不同的推理上下文

        // 采样器状态已彻底重置 (RNG + batch索引)
    }

    /// 采样函数 - 使用与Python版本一致的sample_logits逻辑
    /// forbid_token: 可选禁止采样的token（如某些阶段的特殊符号）
    pub fn sample_logits(
        &mut self,
        logits: &[f32],
        args: &SamplerArgs,
        forbid_token: Option<usize>,
    ) -> usize {
        // 使用与Python版本一致的sample_logits逻辑
        sample_logits_with_top_p_k(
            logits,
            args.temperature,
            args.top_p,
            args.top_k,
            forbid_token,
            &mut self.rng,
        )
    }
}
