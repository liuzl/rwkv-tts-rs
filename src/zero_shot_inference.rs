use anyhow::Result;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tracing::warn;
use web_rwkv::runtime::infer::{RnnInput, RnnInputBatch, RnnOption};

use crate::shared_runtime::TtsInferContext;

/// 执行Zero-shot推理
pub async fn execute_zero_shot_inference(
    infer_context: TtsInferContext,
    text_tokens: Vec<i32>,
    property_tokens: Vec<i32>,
    rng: rand::rngs::StdRng,
    request: &crate::rwkv_sampler::TtsBatchRequest,
) -> Result<(Vec<i32>, Vec<i32>)> {
    let request_id = &infer_context.request_id;
    // 开始Zero-shot推理

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
    let token_chunk_size = infer_context.options.token_chunk_size;

    // === 验证和读取预提取的音色特征 ===
    let ref_global = request
        .ref_global_tokens
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Zero-shot模式需要预提取的global tokens"))?;
    let ref_semantic = request
        .ref_semantic_tokens
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Zero-shot模式需要预提取的semantic tokens"))?;

    // 文本tokens信息

    // 修正tokens范围，确保在有效范围内
    let corrected_global: Vec<i32> = ref_global.iter().map(|&t| t.clamp(0, 4095)).collect();
    let mut _corrected_semantic: Vec<i32> =
        ref_semantic.iter().map(|&t| t.clamp(0, 8192)).collect();

    // 修复：移除参考语义tokens末尾的EOS，避免模型在生成阶段立即结束
    let original_sem_len = _corrected_semantic.len();
    while let Some(&last) = _corrected_semantic.last() {
        if last == crate::rwkv_sampler::TTS_EOS_TOKEN {
            _corrected_semantic.pop();
        } else {
            break;
        }
    }
    let trimmed_count = original_sem_len.saturating_sub(_corrected_semantic.len());
    if trimmed_count > 0 {
        warn!(
            "🔧 [{}] 参考semantic末尾EOS已移除 {} 个，防止早停",
            request_id, trimmed_count
        );
    }

    if corrected_global != *ref_global {
        warn!("🔧 [{}] 已修正global tokens范围到[0..4096)", request_id);
    }
    if _corrected_semantic != *ref_semantic {
        warn!("🔧 [{}] 已修正semantic tokens范围到[0..8192]", request_id);
    }

    // 构建输入序列：属性tokens + TTS_TAG_2 + 文本tokens + TTS_TAG_0
    let mut input_tokens: Vec<i32> = Vec::new();
    input_tokens.extend_from_slice(&property_tokens);
    input_tokens.push(crate::rwkv_sampler::TTS_TAG_2);
    input_tokens.extend_from_slice(&text_tokens);
    input_tokens.push(crate::rwkv_sampler::TTS_TAG_0);
    // 加入预读取的global tokens（添加偏移）
    for &token in &corrected_global {
        input_tokens.push(token + crate::rwkv_sampler::GLOBAL_TOKEN_OFFSET);
    }
    input_tokens.push(crate::rwkv_sampler::TTS_TAG_1);
    // 跨语言克隆：跳过参考语义预填，仅保留global tokens影响音色
    // 说明：参考语义通常强绑定原语言内容，预填会导致继续原语种，不利于跨语言
    log::info!(
        "🌐 [{}] 跨语言克隆模式：禁用参考semantic预填，仅使用global tokens维持音色",
        request_id
    );

    // === Prefill 阶段（复制普通模式）===
    let input_tokens_u32: Vec<u32> = input_tokens.iter().map(|&t| t as u32).collect();

    // Prefill阶段 - 初始化独立状态

    // 创建独立的推理上下文
    let batch = RnnInputBatch::new(input_tokens_u32.clone(), RnnOption::Last);
    let mut inference = RnnInput::new(vec![batch], token_chunk_size);

    // 为批处理槽位0加载初始状态，确保状态隔离（优化：合并二次锁操作）
    {
        let state_guard = state.lock().await;
        let initial_state = state_guard.init();
        state_guard.load(initial_state, 0)?;
        drop(state_guard); // 显式释放锁
                           // 已为批处理槽位0加载初始状态
    }

    // 消化输入直到产生输出
    let last_sem_logits_prefill: Vec<f32> = loop {
        let (remaining_input, output) = runtime.infer(inference.clone()).await?;
        inference = remaining_input;
        if !output.is_empty() && output[0].0.size() > 0 {
            break output[0].0.clone().to_vec();
        }
    };

    // 预填完成后，直接在语义域继续采样（不再重复注入global或标签）
    let global_tokens: Vec<i32> = corrected_global.clone();
    let mut semantic_tokens: Vec<i32> = Vec::new();
    let last_sem_logits: Vec<f32> = last_sem_logits_prefill;

    // === Semantic tokens 生成阶段（复制普通模式参数和逻辑）===
    let semantic_limit: usize = usize::min(2048, 2048);
    // 加入最小生成长度约束（动态）：按文本长度自适应，避免首步采到EOS导致语义序列为空
    let min_semantic_len: usize = {
        let tlen = text_tokens.len();
        // 经验：语义token约为文本token的1/4～1/2，这里取保守的1/4
        // 下限8，上限64，兼顾短文本与长文本
        tlen.saturating_div(4).clamp(8, 64)
    };
    // 基于文本长度的“硬下限”，在达到该长度前一律禁止EOS
    // 经验初值：语义至少为文本token的1.8倍，避免还没读完就早停
    let hard_min_semantic_len: usize = {
        let tlen = text_tokens.len();
        let est = ((tlen as f32) * 1.8).ceil() as usize;
        // 不超过语义上限的90%，避免过长
        let upper = (semantic_limit as f32 * 0.9).floor() as usize;
        std::cmp::min(upper, std::cmp::max(min_semantic_len, est))
    };
    log::info!(
        "🛡️ [{}] Zero-shot最小语义长度: 动态={}，硬下限={} (text_tokens={})",
        request_id,
        min_semantic_len,
        hard_min_semantic_len,
        text_tokens.len()
    );

    // Zero-shot模式：跳过Global阶段，直接使用预提取的global_tokens
    // 设置Semantic阶段采样参数
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

    // 参数对比打印：Python vs Rust (Zero-shot模式)
    log::info!(
        "🔍 [{}] Zero-shot模式采样参数对比 (Python vs Rust):",
        request_id
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
    let semantic_match = (args_semantic.temperature - 1.0).abs() < 0.001
        && (args_semantic.top_p - 0.95).abs() < 0.001
        && args_semantic.top_k == 80;

    if semantic_match {
        log::info!(
            "✅ [{}] Zero-shot Semantic参数完全匹配Python版本！",
            request_id
        );
    } else {
        log::warn!(
            "⚠️ [{}] Zero-shot Semantic参数与Python版本不匹配！",
            request_id
        );
    }

    // 开始生成semantic tokens
    println!(
        "🎯 [{}] Zero-shot模式开始生成Semantic tokens，最大数量: {}",
        request_id, semantic_limit
    );

    // 简化采样，移除优化组件

    // 创建独立的RNG用于semantic阶段
    let semantic_rng = if args_semantic.layered_randomness.use_independent_seeds {
        if let Some(seed) = args_semantic.seed {
            // 用户提供了seed，使用确定性采样
            StdRng::seed_from_u64(
                seed.wrapping_add(args_semantic.layered_randomness.semantic_seed_offset),
            )
        } else {
            // 用户没有提供seed，使用随机采样
            StdRng::from_rng(rand::thread_rng()).expect("failed to seed StdRng")
        }
    } else {
        rng
    };

    let mut semantic_rng_opt = Some(semantic_rng);
    // EOS允许阈值判定：最近N步非EOS比例达到阈值才允许EOS
    let eos_window: usize = 12;
    let eos_ratio_threshold: f32 = 0.7;
    let mut recent_non_eos: Vec<bool> = Vec::with_capacity(eos_window);
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

        // 在最小长度内禁止EOS，确保至少生成若干语义tokens
        let eos_idx = crate::rwkv_sampler::TTS_EOS_TOKEN as usize;
        // 达到“硬下限”之前一律禁止EOS
        if i < hard_min_semantic_len && eos_idx < logits_masked.len() {
            logits_masked[eos_idx] = f32::NEG_INFINITY;
        }

        // 使用简单采样器采样
        let mut next_id = crate::rwkv_sampler::sample_logits(
            &logits_masked,
            &args_semantic,
            None, // forbid_token
            &mut semantic_rng_opt,
        );

        // 检查是否遇到EOS token（必须在范围检查之前）
        if next_id == crate::rwkv_sampler::TTS_EOS_TOKEN as usize {
            // 阈值判定：最近N步非EOS比例达到阈值才允许EOS
            let window_len = recent_non_eos.len();
            let non_eos_count = recent_non_eos.iter().filter(|&&b| b).count();
            let ratio = if window_len > 0 {
                non_eos_count as f32 / window_len as f32
            } else {
                0.0
            };
            let allow_eos = window_len >= eos_window && ratio >= eos_ratio_threshold;
            if allow_eos {
                log::info!(
                    "🛑 [{}] 允许EOS结束：recent_non_eos_ratio={:.2}, window={}",
                    request_id,
                    ratio,
                    window_len
                );
                break;
            } else {
                log::info!(
                    "⏭️ [{}] 阻止EOS：recent_non_eos_ratio={:.2}, window={}；继续采样",
                    request_id,
                    ratio,
                    window_len
                );
                // 屏蔽EOS后重新采样
                let eos_idx2 = crate::rwkv_sampler::TTS_EOS_TOKEN as usize;
                if eos_idx2 < logits_masked.len() {
                    logits_masked[eos_idx2] = f32::NEG_INFINITY;
                }
                next_id = crate::rwkv_sampler::sample_logits(
                    &logits_masked,
                    &args_semantic,
                    None,
                    &mut semantic_rng_opt,
                );
            }
        }

        // 额外检查：确保token在semantic范围内 [0..8192)（修复：应该是>8192而不是>=8192）
        if next_id > crate::rwkv_sampler::TTS_EOS_TOKEN as usize {
            warn!(
                "🚨 [{}] Token {} 超出semantic范围[0..8192]，停止生成以确保稳定性",
                request_id, next_id
            );
            break;
        }

        // 维护最近N步非EOS比例窗口
        let is_non_eos = next_id != crate::rwkv_sampler::TTS_EOS_TOKEN as usize;
        recent_non_eos.push(is_non_eos);
        if recent_non_eos.len() > eos_window {
            // 移除最早的一项（窗口固定长度）
            recent_non_eos.remove(0);
        }

        semantic_tokens.push(next_id as i32);

        // 反馈到模型：语义阶段直接使用原始token（不加偏移）
        inference.batches[0].push(next_id as u32);

        // 打印当前生成进度
        if (i + 1) % 16 == 0 || i == semantic_limit - 1 {
            println!(
                "📊 [{}] Zero-shot Semantic阶段: 已生成 {}/{} tokens",
                request_id,
                i + 1,
                semantic_limit
            );
        }
    }
    // 回退逻辑：如果语义为空（可能首步采到了EOS），强制从prefill的logits采样至少1个token
    if semantic_tokens.is_empty() {
        warn!(
            "⚠️ [{}] Zero-shot语义序列为空，应用回退采样确保至少1个token",
            request_id
        );
        let mut logits_masked = last_sem_logits.clone();
        let eos_idx = crate::rwkv_sampler::TTS_EOS_TOKEN as usize;
        if eos_idx < logits_masked.len() {
            logits_masked[eos_idx] = f32::NEG_INFINITY;
        }
        let next_id = crate::rwkv_sampler::sample_logits(
            &logits_masked,
            &args_semantic,
            None,
            &mut semantic_rng_opt,
        );
        if next_id <= crate::rwkv_sampler::TTS_EOS_TOKEN as usize {
            semantic_tokens.push(next_id as i32);
            inference.batches[0].push(next_id as u32);
        }
    }
    // TTS tokens生成完成
    println!(
        "✅ [{}] Zero-shot TTS生成完成 - Global tokens: {}, Semantic tokens: {}",
        request_id,
        global_tokens.len(),
        semantic_tokens.len()
    );
    Ok((global_tokens, semantic_tokens))
}
