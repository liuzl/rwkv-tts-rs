//! RWKV模型推理采样器
//! 实现基于web-rwkv库的RWKV模型推理和采样功能

use anyhow::Result;
use memmap2::Mmap;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use safetensors::SafeTensors;
use sha2::{Digest, Sha256};
use std::path::Path;
use web_rwkv::{
    context::{ContextBuilder, InstanceExt},
    runtime::{
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::Loader,
        model::{ContextAutoLimits, ModelBuilder},
        v7, Runtime, TokioRuntime,
    },
    tokenizer::Tokenizer,
};
use wgpu::Instance;

/// 采样参数
#[derive(Debug, Clone)]
pub struct SamplerArgs {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub max_tokens: usize,
    // 可选随机种子：提供则启用确定性采样
    pub seed: Option<u64>,
}

impl Default for SamplerArgs {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.85,
            top_k: 0,
            max_tokens: 100,
            seed: None,
        }
    }
}

/// TTS相关常量
pub const TTS_EOS_TOKEN: i32 = 8192;
pub const TTS_TAG_0: i32 = 8193;
pub const TTS_TAG_1: i32 = 8194;
pub const TTS_TAG_2: i32 = 8195;
pub const GLOBAL_TOKEN_OFFSET: i32 = 8196;

/// RWKV采样器，用于生成文本和TTS tokens
pub struct RwkvSampler {
    runtime: Box<dyn Runtime<Rnn>>, // 使用TokioRuntime封装Bundle
    tokenizer: Tokenizer,
    // 带种子的RNG（可选，启用则实现确定性采样）
    rng: Option<StdRng>,
}
impl RwkvSampler {
    /// 创建新的RWKV采样器
    ///
    /// # Arguments
    /// * `model_path` - RWKV模型目录或模型文件(.safetensors)路径
    /// * `vocab_path` - 词表文件路径
    ///
    /// # Returns
    /// * `Result<RwkvSampler>` - RWKV采样器实例或错误
    pub async fn new(model_path: &str, vocab_path: &str) -> Result<Self> {
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
        // - 若传入目录，则默认加载其中的 "webrwkv.safetensors"
        // - 若传入文件，则直接使用该文件
        let model_file_path = if model_path_ref.is_dir() {
            model_path_ref.join("webrwkv.safetensors")
        } else {
            model_path_ref.to_path_buf()
        };
        if !model_file_path.exists() {
            return Err(anyhow::anyhow!(
                "模型文件不存在: {}",
                model_file_path.display()
            ));
        }

        // 加载并反序列化SafeTensors模型
        let file = std::fs::File::open(&model_file_path)?;
        let file_size = file.metadata()?.len();
        let data = unsafe { Mmap::map(&file)? };

        // 模型完整性校验：打印大小与SHA256
        let mut hasher = Sha256::new();
        hasher.update(&data[..]);
        let hash_bytes = hasher.finalize();
        let sha256 = hash_bytes.iter().map(|b| format!("{:02x}", b)).collect::<String>();
        println!("🔒 模型检验: {}", model_file_path.display());
        println!("   - 大小: {} bytes", file_size);
        println!("   - SHA256: {}", sha256);

        let model = SafeTensors::deserialize(&data)?;

        // 基于模型信息自动配置 Context 的硬件 limits
        let info = Loader::info(&model)?;
        let instance = Instance::default();
        let adapter = instance
            .adapter(wgpu::PowerPreference::HighPerformance)
            .await?;

        // 打印适配器/后端/驱动与精度
        let adapter_info = adapter.get_info();
        println!("🖥️ 选用GPU适配器: {}", adapter_info.name);
        println!(
            "   - 后端: {:?} | 供应商: {:#06x} 设备: {:#06x} | 类型: {:?}",
            adapter_info.backend, adapter_info.vendor, adapter_info.device, adapter_info.device_type
        );
        println!(
            "   - 驱动: {} | 详情: {}",
            adapter_info.driver, adapter_info.driver_info
        );
        println!("   - 使用 FP32 推理: true (v7::Bundle::<f32>)");

        let context = ContextBuilder::new(adapter)
            .auto_limits(&info)
            .build()
            .await?;

        // 创建模型构建器并构建v7模型
        let builder = ModelBuilder::new(&context, model);
        let model = builder.build_v7().await?;

        // 创建Bundle与TokioRuntime（切换为 f32 以启用 FP32 推理）
        let bundle = v7::Bundle::<f32>::new(model, 1);
        let runtime: Box<dyn Runtime<Rnn>> = Box::new(TokioRuntime::new(bundle).await);

        // 加载tokenizer
        let vocab_content = std::fs::read_to_string(vocab_path)?;
        let tokenizer = Tokenizer::new(&vocab_content)?;

        Ok(Self { runtime, tokenizer, rng: None })
    }

    /// 设置随机种子（启用确定性采样）。传None则关闭确定性模式。
    pub fn set_seed(&mut self, seed: Option<u64>) {
        self.rng = seed.map(|s| StdRng::seed_from_u64(s));
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
        let token_chunk_size = 64usize;
        let prompt_batch = RnnInputBatch::new(prompt_tokens.clone(), RnnOption::Last);
        let mut inference = RnnInput::new(vec![prompt_batch], token_chunk_size);

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
        args: &SamplerArgs,
    ) -> Result<(Vec<i32>, Vec<i32>)> {
        // 若提供了种子，设置确定性采样
        self.set_seed(args.seed);

        // 编码文本：使用原始文本token（不加任何偏移）以匹配参考实现
        let text_tokens_u32: Vec<u32> = self
            .tokenizer
            .encode(text.as_bytes())
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let text_tokens: Vec<i32> = text_tokens_u32.into_iter().map(|t| t as i32).collect();

        // 参考实现在prefill阶段喂入属性tokens（原始域）、文本tokens与阶段标签。
        let mut input_tokens: Vec<i32> = Vec::new();
        input_tokens.extend_from_slice(property_tokens);
        input_tokens.push(TTS_TAG_2);
        input_tokens.extend_from_slice(&text_tokens);
        input_tokens.push(TTS_TAG_0);

        // === Prefill 阶段 ===
        let input_tokens_u32: Vec<u32> = input_tokens.iter().map(|&t| t as u32).collect();
        let token_chunk_size = 64usize;
        let prompt_batch = RnnInputBatch::new(input_tokens_u32.clone(), RnnOption::Last);
        let mut inference = RnnInput::new(vec![prompt_batch], token_chunk_size);
        // 消化输入直到产生输出，并保留最后一次logits
        let last_logits: Vec<f32> = loop {
            let (next_inference, output) = self.runtime.infer(inference).await?;
            inference = next_inference;
            if output[0].0.size() > 0 {
                break output[0].0.clone().to_vec();
            }
        };

        // === Global 阶段 ===
        let mut global_tokens: Vec<i32> = Vec::new();
        let mut semantic_tokens: Vec<i32> = Vec::new();
        let mut args_global = args.clone();
        let mut args_sem = args.clone();
        // 若未指定top_k（为0），按Python经验值设置：global=20, semantic=80
        if args_global.top_k == 0 {
            args_global.top_k = 20;
        }
        if args_sem.top_k == 0 {
            args_sem.top_k = 80;
        }

        // Python实现固定生成32个global tokens，并且仅在前4096维内采样
        let global_tokens_size: usize = 32;
        for i in 0..global_tokens_size {
            // 取得当前可用的logits：首步使用prefill得到的logits，其后每步从runtime获取
            let logits: Vec<f32> = if i == 0 {
                last_logits.clone()
            } else {
                // 确保拿到非空logits
                loop {
                    let (next_inference, output) = self.runtime.infer(inference).await?;
                    inference = next_inference;
                    if output[0].0.size() > 0 {
                        break output[0].0.clone().to_vec();
                    }
                }
            };

            // 仅在[0..4096)范围内采样（Global相对域），不涉及EOS与阶段标签
            let vocab_global = if logits.len() < 4096 {
                logits.len()
            } else {
                4096
            };
            let next_id = self.sample_logits(&logits[..vocab_global], &args_global, None);

            // 追加到global输出（相对域 [0..4095]）
            global_tokens.push(next_id as i32);
            // 反馈到模型：+8196（GLOBAL_TOKEN_OFFSET）
            let feed_id = (next_id as i32 + GLOBAL_TOKEN_OFFSET) as u32;
            inference.batches[0].push(feed_id);
        }

        // === 切换到 Semantic 阶段 ===
        inference.batches[0].push(TTS_TAG_1 as u32);
        // 让标签生效，直到产生输出，并保留logits供首步使用
        let last_sem_logits: Vec<f32> = loop {
            let (next_inference, output) = self.runtime.infer(inference).await?;
            inference = next_inference;
            if output[0].0.size() > 0 {
                break output[0].0.clone().to_vec();
            }
        };

        // 语义阶段：限制最大生成步数为2048
        let semantic_limit: usize = usize::min(args.max_tokens, 2048);
        for i in 0..semantic_limit {
            // 取得当前语义阶段的logits：首步使用注入标签后的logits，其后每步从runtime获取
            let logits: Vec<f32> = if i == 0 {
                last_sem_logits.clone()
            } else {
                loop {
                    let (next_inference, output) = self.runtime.infer(inference).await?;
                    inference = next_inference;
                    if output[0].0.size() > 0 {
                        break output[0].0.clone().to_vec();
                    }
                }
            };

            // 语义阶段仅采样 [0..8192]（包含EOS），屏蔽TTS_TAG_*与其它域
            let mut logits_masked = logits.clone();
            for (i, v) in logits_masked.iter_mut().enumerate() {
                if i > TTS_EOS_TOKEN as usize {
                    *v = f32::NEG_INFINITY;
                }
            }
            for tag in [TTS_TAG_0, TTS_TAG_1, TTS_TAG_2] {
                let idx = tag as usize;
                if idx < logits_masked.len() {
                    logits_masked[idx] = f32::NEG_INFINITY;
                }
            }

            let next_id = self.sample_logits(&logits_masked, &args_sem, None);
            if next_id == TTS_EOS_TOKEN as usize {
                break;
            }

            // 追加到semantic输出（原始域 [0..8191]）
            semantic_tokens.push(next_id as i32);
            // 语义阶段反馈：直接反馈原始id（经验）
            inference.batches[0].push(next_id as u32);
        }

        Ok((global_tokens, semantic_tokens))
    }

    /// 重置运行时状态（当前Runtime按步推理，暂无显式重置需求，预留接口）
    pub fn reset(&mut self) {
        // 如果后续Runtime提供state重置API，可在此调用。
        // 目前每次推理都会重新构造输入批次，故此处为空实现。
    }

    /// 采样函数 - Nucleus(top-p) + top-k + temperature
    /// forbid_token: 可选禁止采样的token（如某些阶段的特殊符号）
    fn sample_logits(
        &mut self,
        logits: &[f32],
        args: &SamplerArgs,
        forbid_token: Option<usize>,
    ) -> usize {
        let vocab_size = logits.len();
        if vocab_size == 0 {
            return 0;
        }

        // 复制索引并可选过滤禁用token
        let mut indices: Vec<usize> = (0..vocab_size).collect();
        if let Some(ft) = forbid_token {
            indices.retain(|&i| i != ft);
        }
        if indices.is_empty() {
            return 0;
        }

        let temperature = args.temperature.max(0.1);
        let top_k = if args.top_k == 0 || args.top_k > indices.len() {
            indices.len()
        } else {
            args.top_k
        };
        let top_p = args.top_p.clamp(0.0, 1.0);

        // 快速路径：top_k==1或top_p极小，直接取最大logit
        if top_k == 1 || top_p < 1e-4 {
            let mut best = indices[0];
            let mut best_val = f32::NEG_INFINITY;
            for &i in &indices {
                let v = logits[i];
                if v > best_val {
                    best_val = v;
                    best = i;
                }
            }
            return best;
        }

        // 按logits降序排序（与softmax排序一致）
        indices.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if top_k < indices.len() {
            indices.truncate(top_k);
        }

        // 数值稳定的 softmax：减去最大值并clamp指数区间
        let inv_t = 1.0 / temperature;
        let mut scaled: Vec<f32> = indices.iter().map(|&i| logits[i] * inv_t).collect();
        let mut max_scaled = f32::NEG_INFINITY;
        for &v in &scaled {
            if v > max_scaled {
                max_scaled = v;
            }
        }
        let mut probs: Vec<f32> = scaled
            .into_iter()
            .map(|v| ((v - max_scaled).clamp(-80.0, 80.0)).exp())
            .collect();
        let mut sum: f32 = probs.iter().sum();
        if sum > 0.0 && sum.is_finite() {
            for p in &mut probs {
                *p /= sum;
            }
        } else {
            // 退化为均匀分布（极端数值情况下）
            let uniform = 1.0 / (probs.len() as f32).max(1.0);
            for p in &mut probs {
                *p = uniform;
            }
        }

        // top-p截断（在排序后概率空间中）
        if top_p < 1.0 {
            let mut cumsum = 0.0;
            let mut cutoff = probs.len();
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if cumsum >= top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            if cutoff < probs.len() {
                probs.truncate(cutoff);
                indices.truncate(cutoff);
            }
            // 再归一化
            sum = probs.iter().sum();
            if sum > 0.0 && sum.is_finite() {
                for p in &mut probs {
                    *p /= sum;
                }
            }
        }

        // 按概率采样（支持确定性RNG）
        let r: f32 = if let Some(rng) = &mut self.rng {
            rng.gen()
        } else {
            let mut rng = rand::thread_rng();
            rng.gen()
        };
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r <= cumsum {
                return indices[i];
            }
        }
        *indices.last().unwrap_or(&0)
    }
}
