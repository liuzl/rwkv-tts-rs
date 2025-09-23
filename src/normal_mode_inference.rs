use anyhow::Result;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tracing::warn;
use web_rwkv::runtime::infer::{RnnInput, RnnInputBatch, RnnOption};

use crate::shared_runtime::TtsInferContext;

/// 执行普通模式推理
pub async fn execute_normal_inference(
    infer_context: TtsInferContext,
    text_tokens: Vec<i32>,
    property_tokens: Vec<i32>,
    _rng: rand::rngs::StdRng,
    request: &crate::rwkv_sampler::TtsBatchRequest,
) -> Result<(Vec<i32>, Vec<i32>)> {
    let request_id = &infer_context.request_id;
    // 开始普通模式推理

    // 获取采样参数
    let _sampler_args = &request.args;

    // Acquire runtime semaphore for the entire inference to ensure isolation
    let _runtime_permit = infer_context
        .runtime_semaphore
        .acquire()
        .await
        .map_err(|e| anyhow::anyhow!("无法获取运行时信号量: {}", e))?;

    // 已获取信号量许可，开始推理

    // 获取runtime
    let runtime = &infer_context.runtime;
    let state = &infer_context.state;

    // 构建输入序列：属性tokens + TTS_TAG_2 + 文本tokens + TTS_TAG_0
    let mut input_tokens: Vec<i32> = Vec::new();
    input_tokens.extend_from_slice(&property_tokens);
    input_tokens.push(crate::rwkv_sampler::TTS_TAG_2);
    input_tokens.extend_from_slice(&text_tokens);
    input_tokens.push(crate::rwkv_sampler::TTS_TAG_0);

    // 调试：打印输入序列构建信息
    log::info!("🔍 [{}] 输入序列构建详情:", request_id);
    log::info!("   📝 属性tokens: {:?}", property_tokens);
    log::info!("   📝 文本tokens长度: {}", text_tokens.len());
    log::info!("   📝 完整输入序列长度: {}", input_tokens.len());
    log::info!(
        "   📝 输入序列前10个token: {:?}",
        &input_tokens[..std::cmp::min(10, input_tokens.len())]
    );

    // 构建完整输入序列

    // === Prefill 阶段 ===
    let input_tokens_u32: Vec<u32> = input_tokens.iter().map(|&t| t as u32).collect();
    let token_chunk_size = infer_context.options.token_chunk_size;

    // Prefill阶段 - 初始化独立状态

    // 创建独立的推理上下文
    let batch = RnnInputBatch::new(input_tokens_u32.clone(), RnnOption::Last);
    let mut inference = RnnInput::new(vec![batch], token_chunk_size);

    // 为批处理槽位0加载初始状态，确保状态隔离
    {
        let state_guard = state.lock().await;
        let initial_state = state_guard.init();
        state_guard.load(initial_state, 0)?;
        // 已为批处理槽位0加载初始状态
    }

    // 消化输入直到产生输出
    let last_logits: Vec<f32> = loop {
        let (remaining_input, output) = runtime.infer(inference.clone()).await?;
        inference = remaining_input;
        if !output.is_empty() && output[0].0.size() > 0 {
            break output[0].0.clone().to_vec();
        }
    };

    // 新增：根据logits长度推断词表大小，并校验属性token是否越界
    let vocab_size = last_logits.len();
    if !property_tokens.is_empty() {
        let mut out_of_range = vec![];
        for &t in &property_tokens {
            if (t as usize) >= vocab_size {
                out_of_range.push(t);
            }
        }
        if !out_of_range.is_empty() {
            log::warn!(
                "🚨 [{}] 检测到属性tokens超出词表范围，可能被模型忽略：越界token={:?}，词表大小={}。请核对TTS_SPECIAL_TOKEN_OFFSET是否与模型/词表匹配。",
                request_id,
                out_of_range,
                vocab_size
            );
        } else {
            log::info!(
                "✅ [{}] 属性tokens在词表范围内（vocab_size={}），将参与Prefill阶段。",
                request_id,
                vocab_size
            );
        }
    }

    // === Global 阶段 ===
    let mut global_tokens: Vec<i32> = Vec::new();
    let mut semantic_tokens: Vec<i32> = Vec::new();

    // 普通模式进行正常的生成流程（不使用预提取特征）
    // Global阶段使用固定参数（与Python版本一致）
    let args_global = crate::rwkv_sampler::SamplerArgs {
        temperature: 1.0, // Global阶段使用固定参数
        top_k: 20,
        top_p: 0.95,
        seed: infer_context.options.seed,
        max_tokens: 32, // Global阶段固定32个tokens
        voice_fidelity: infer_context.options.voice_fidelity,
        layered_randomness: infer_context.options.layered_randomness.clone(),
        token_chunk_size: infer_context.options.token_chunk_size,
    };

    let args_semantic = crate::rwkv_sampler::SamplerArgs {
        temperature: 1.0, // Semantic阶段使用固定参数
        top_p: 0.95,
        top_k: 80,
        seed: infer_context.options.seed,
        max_tokens: 2048,
        voice_fidelity: infer_context.options.voice_fidelity,
        layered_randomness: infer_context.options.layered_randomness.clone(),
        token_chunk_size: infer_context.options.token_chunk_size,
    };

    // 简化采样，移除优化组件

    // 创建独立的RNG用于不同阶段
    let mut global_rng = if args_global.layered_randomness.use_independent_seeds {
        if let Some(seed) = args_global.seed {
            // 用户提供了seed，使用确定性采样
            Some(StdRng::seed_from_u64(seed.wrapping_add(
                args_global.layered_randomness.global_seed_offset,
            )))
        } else {
            // 没有seed，创建随机RNG
            Some(StdRng::from_entropy())
        }
    } else {
        // 创建新的RNG实例，避免共享状态导致的不一致
        Some(if let Some(seed) = args_global.seed {
            StdRng::seed_from_u64(seed.wrapping_add(100))
        } else {
            StdRng::from_entropy()
        })
    };

    let mut semantic_rng = if args_semantic.layered_randomness.use_independent_seeds {
        if let Some(seed) = args_semantic.seed {
            // 用户提供了seed，使用确定性采样
            Some(StdRng::seed_from_u64(seed.wrapping_add(
                args_semantic.layered_randomness.semantic_seed_offset,
            )))
        } else {
            // 没有seed，创建随机RNG
            Some(StdRng::from_entropy())
        }
    } else {
        // 创建新的RNG实例，避免共享状态导致的不一致
        Some(if let Some(seed) = args_semantic.seed {
            StdRng::seed_from_u64(seed.wrapping_add(200))
        } else {
            StdRng::from_entropy()
        })
    };

    // RNG状态初始化

    // Global和Semantic阶段都使用固定参数（与Python版本一致）
    // 移除参数调整逻辑，直接使用固定值

    // 参数对比打印：Python vs Rust
    log::info!("🔍 [{}] 采样参数对比 (Python vs Rust):", request_id);
    log::info!("   📊 Global阶段:");
    log::info!("      Python: temperature=1.0, top_p=0.95, top_k=20");
    log::info!(
        "      Rust:   temperature={:.1}, top_p={:.2}, top_k={}",
        args_global.temperature,
        args_global.top_p,
        args_global.top_k
    );
    log::info!("   📊 Semantic阶段:");
    log::info!("      Python: temperature=1.0, top_p=0.95, top_k=80");
    log::info!(
        "      Rust:   temperature={:.1}, top_p={:.2}, top_k={}",
        args_semantic.temperature,
        args_semantic.top_p,
        args_semantic.top_k
    );

    // 验证参数一致性
    let global_match = (args_global.temperature - 1.0).abs() < 0.001
        && (args_global.top_p - 0.95).abs() < 0.001
        && args_global.top_k == 20;
    let semantic_match = (args_semantic.temperature - 1.0).abs() < 0.001
        && (args_semantic.top_p - 0.95).abs() < 0.001
        && args_semantic.top_k == 80;

    if global_match && semantic_match {
        log::info!("✅ [{}] 参数完全匹配Python版本！", request_id);
    } else {
        log::warn!(
            "⚠️ [{}] 参数与Python版本不匹配！Global匹配: {}, Semantic匹配: {}",
            request_id,
            global_match,
            semantic_match
        );
    }

    // 生成32个global tokens
    let global_tokens_size: usize = 32;

    for i in 0..global_tokens_size {
        let logits: Vec<f32> = if i == 0 {
            last_logits.clone()
        } else {
            // 继续推理获取logits - 使用现有inference上下文
            loop {
                let (next_inference, output) = runtime.infer(inference.clone()).await?;
                inference = next_inference;
                if output[0].0.size() > 0 {
                    break output[0].0.clone().to_vec();
                }
            }
        };

        // 仅在[0..4096)范围内采样
        let vocab_global = if logits.len() < 4096 {
            logits.len()
        } else {
            4096
        };

        // 直接使用原始logits，不进行增强处理
        let sampling_logits = logits[..vocab_global].to_vec();

        // 使用top-p/top-k采样器采样
        let next_id = crate::rwkv_sampler::sample_logits_with_top_p_k(
            &sampling_logits,
            args_global.temperature,
            args_global.top_p,
            args_global.top_k,
            None, // forbid_token
            &mut global_rng,
        );

        // 安全转换：确保token在有效范围内
        if next_id > i32::MAX as usize {
            warn!(
                "🚨 [{}] Global token {} 超出i32范围，跳过此token",
                request_id, next_id
            );
            continue;
        }

        // 额外检查：确保token在global范围内 [0..4096)
        if next_id >= 4096 {
            warn!(
                "🚨 [{}] Global token {} 超出范围[0..4096)，跳过此token",
                request_id, next_id
            );
            continue;
        }

        global_tokens.push(next_id as i32);

        // 回灌到模型：加上GLOBAL_TOKEN_OFFSET以进入Global域（与Python/zero-shot一致）
        let with_offset = (next_id as i32 + crate::rwkv_sampler::GLOBAL_TOKEN_OFFSET) as u32;
        inference.batches[0].push(with_offset);
        log::debug!(
            "🔧 [{}] 回灌Global token: raw={}, with_offset={}",
            request_id,
            next_id,
            with_offset
        );

        // Global token生成
    }

    // 记录Global阶段前若干个token，便于诊断开头漏字问题
    if !global_tokens.is_empty() {
        let head = std::cmp::min(8, global_tokens.len());
        log::info!(
            "🎯 [{}] Global阶段生成前{}个token: {:?}",
            request_id,
            head,
            &global_tokens[..head]
        );
    }

    // Global tokens生成完成

    // === 切换到 Semantic 阶段 ===
    inference.batches[0].push(crate::rwkv_sampler::TTS_TAG_1 as u32);
    // 切换到Semantic阶段

    // 让标签生效，直到产生输出，并保留logits供首步使用
    let last_sem_logits: Vec<f32> = loop {
        let (next_inference, output) = runtime.infer(inference).await?;
        inference = next_inference;
        if output[0].0.size() > 0 {
            break output[0].0.clone().to_vec();
        }
    };

    // 语义阶段：限制最大生成步数为2048
    let semantic_limit: usize = usize::min(request.args.max_tokens, 2048);
    // 开始生成semantic tokens

    for i in 0..semantic_limit {
        let logits: Vec<f32> = if i == 0 {
            last_sem_logits.clone()
        } else {
            loop {
                let (next_inference, output) = runtime.infer(inference.clone()).await?;
                inference = next_inference;
                if output[0].0.size() > 0 {
                    break output[0].0.clone().to_vec();
                }
            }
        };

        // 语义阶段仅采样 [0..8192]（包含EOS），屏蔽TTS_TAG_*与其它域
        let mut logits_masked = logits.clone();
        // 修复：不屏蔽EOS token，只屏蔽大于EOS token的部分
        for (j, v) in logits_masked.iter_mut().enumerate() {
            if j > crate::rwkv_sampler::TTS_EOS_TOKEN as usize {
                *v = f32::NEG_INFINITY;
            }
        }
        // 屏蔽TTS_TAG tokens，但保留EOS token
        for tag in [
            crate::rwkv_sampler::TTS_TAG_0,
            crate::rwkv_sampler::TTS_TAG_1,
            crate::rwkv_sampler::TTS_TAG_2,
        ] {
            let idx = tag as usize;
            if idx < logits_masked.len() {
                logits_masked[idx] = f32::NEG_INFINITY;
            }
        }

        // 注意：不屏蔽EOS token，让它能够被正常采样以终止生成

        // EOS token logits检查
        let _eos_logit = if (crate::rwkv_sampler::TTS_EOS_TOKEN as usize) < logits_masked.len() {
            logits_masked[crate::rwkv_sampler::TTS_EOS_TOKEN as usize]
        } else {
            f32::NEG_INFINITY
        };

        // 使用top-p/top-k采样器采样
        let next_id = crate::rwkv_sampler::sample_logits_with_top_p_k(
            &logits_masked,
            args_semantic.temperature,
            args_semantic.top_p,
            args_semantic.top_k,
            None, // forbid_token
            &mut semantic_rng,
        );

        // 检查是否遇到EOS token（必须在范围检查之前）
        if next_id == crate::rwkv_sampler::TTS_EOS_TOKEN as usize {
            // 遇到EOS token，停止生成
            break;
        }

        // 额外检查：确保token在语义范围内 [0..=8192]
        if next_id > crate::rwkv_sampler::TTS_EOS_TOKEN as usize {
            warn!(
                "🚨 [{}] Semantic token {} 超出范围[0..=8192]，跳过此token",
                request_id, next_id
            );
            continue;
        }

        let next_id_i32 = next_id as i32;
        semantic_tokens.push(next_id_i32);

        // 反馈到模型：直接使用原始ID（与C++代码一致）
        inference.batches[0].push(next_id as u32);
    }

    // 记录Semantic阶段前若干个token，辅助诊断“开头漏字”
    if !semantic_tokens.is_empty() {
        let head = std::cmp::min(12, semantic_tokens.len());
        log::info!(
            "🗣️ [{}] Semantic阶段生成前{}个token: {:?}",
            request_id,
            head,
            &semantic_tokens[..head]
        );
    } else {
        log::warn!(
            "⚠️ [{}] Semantic阶段未生成任何token（可能过早采样到EOS或输入序列构建异常）",
            request_id
        );
    }

    // 返回生成结果
    Ok((global_tokens, semantic_tokens))
}
