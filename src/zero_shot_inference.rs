use anyhow::Result;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tracing::{debug, info, warn};
use web_rwkv::runtime::infer::{RnnInput, RnnInputBatch, RnnOption};

use crate::shared_runtime::TtsInferContext;

/// 执行Zero-shot推理（复制普通模式结构但跳过Global tokens生成）
pub async fn execute_zero_shot_inference(
    infer_context: &TtsInferContext,
    request: crate::rwkv_sampler::TtsBatchRequest,
    text_tokens: Vec<i32>,
    rng: Option<rand::rngs::StdRng>,
) -> Result<(Vec<i32>, Vec<i32>)> {
    let request_id = &infer_context.request_id;
    info!("🎯 [{}] 开始Zero-shot推理（复制普通模式结构）", request_id);

    // 获取runtime
    let runtime = &infer_context.runtime;
    let state = &infer_context.state;
    let token_chunk_size = request.args.token_chunk_size;

    // === 验证和读取预提取的音色特征 ===
    let ref_global = request
        .ref_global_tokens
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Zero-shot模式需要预提取的global tokens"))?;
    let ref_semantic = request
        .ref_semantic_tokens
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Zero-shot模式需要预提取的semantic tokens"))?;

    info!(
        "🎯 [{}] 预提取音色特征: global tokens {} 个, semantic tokens {} 个",
        request_id,
        ref_global.len(),
        ref_semantic.len()
    );

    // 修正tokens范围，确保在有效范围内
    let corrected_global: Vec<i32> = ref_global.iter().map(|&t| t.clamp(0, 4095)).collect();
    let _corrected_semantic: Vec<i32> = ref_semantic.iter().map(|&t| t.clamp(0, 8192)).collect();

    if corrected_global != *ref_global {
        warn!("🔧 [{}] 已修正global tokens范围到[0..4096)", request_id);
    }
    if _corrected_semantic != *ref_semantic {
        warn!("🔧 [{}] 已修正semantic tokens范围到[0..8192]", request_id);
    }

    // === 构建输入序列（包含预读取的semantic_tokens）===
    // 构建输入序列：属性tokens + TTS_TAG_2 + 文本tokens + TTS_TAG_0 + global_tokens + TTS_TAG_1 + semantic_tokens
    let mut input_tokens: Vec<i32> = Vec::new();
    input_tokens.push(crate::rwkv_sampler::TTS_TAG_2);
    input_tokens.extend_from_slice(&text_tokens);
    input_tokens.push(crate::rwkv_sampler::TTS_TAG_0);
    // 加入预读取的global tokens（添加偏移）
    for &token in &corrected_global {
        input_tokens.push(token + crate::rwkv_sampler::GLOBAL_TOKEN_OFFSET);
    }
    input_tokens.push(crate::rwkv_sampler::TTS_TAG_1);
    // 加入预读取的semantic tokens
    input_tokens.extend_from_slice(&_corrected_semantic);

    // === Prefill 阶段（复制普通模式）===
    let input_tokens_u32: Vec<u32> = input_tokens.iter().map(|&t| t as u32).collect();

    info!("🔧 [{}] Prefill阶段 - 初始化独立状态", request_id);

    // 创建独立的推理上下文
    let batch = RnnInputBatch::new(input_tokens_u32.clone(), RnnOption::Last);
    let mut inference = RnnInput::new(vec![batch], token_chunk_size);

    // 为批处理槽位0加载初始状态，确保状态隔离
    {
        let initial_state = state.lock().await.init();
        state.lock().await.load(initial_state, 0)?;
        info!("🔧 [{}] 已为批处理槽位0加载初始状态", request_id);
    }

    // 消化输入直到产生输出
    let _last_logits: Vec<f32> = loop {
        let (remaining_input, output) = runtime.infer(inference.clone()).await?;
        inference = remaining_input;
        if !output.is_empty() && output[0].0.size() > 0 {
            break output[0].0.clone().to_vec();
        }
    };

    // === Global 阶段：跳过生成，直接使用预提取的tokens ===
    let global_tokens: Vec<i32> = corrected_global.clone();
    let mut semantic_tokens: Vec<i32> = Vec::new();

    info!(
        "✅ [{}] 跳过Global tokens生成，直接使用预提取的tokens: {:?} (共{}个)",
        request_id,
        global_tokens,
        global_tokens.len()
    );

    // 将预提取的global tokens反馈到模型（不加偏移量，与普通模式一致）
    for &token in &global_tokens {
        inference.batches[0].push(token as u32);
    }

    info!("🔧 [{}] 已将预提取的global tokens反馈到模型", request_id);

    // === 切换到 Semantic 阶段（复制普通模式结构）===
    inference.batches[0].push(crate::rwkv_sampler::TTS_TAG_1 as u32);
    info!(
        "🔍 [{}] 切换到Semantic阶段，推入TTS_TAG_1={}",
        request_id,
        crate::rwkv_sampler::TTS_TAG_1
    );

    // 让标签生效，直到产生输出，并保留logits供首步使用
    let last_sem_logits: Vec<f32> = loop {
        let (next_inference, output) = runtime.infer(inference).await?;
        inference = next_inference;
        if output[0].0.size() > 0 {
            break output[0].0.clone().to_vec();
        }
    };

    // === Semantic tokens 生成阶段（复制普通模式参数和逻辑）===
    let semantic_limit: usize = usize::min(request.args.max_tokens, 2048);
    let mut args_semantic = request.args.clone();

    // Semantic阶段使用固定参数，与Python代码保持严格一致
    args_semantic.temperature = 1.0;
    args_semantic.top_p = 0.95;
    args_semantic.top_k = 80;

    info!(
        "🔍 [{}] 开始生成semantic tokens，最大限制: {}",
        request_id, semantic_limit
    );
    info!(
        "🔍 [{}] Semantic阶段采样参数: temperature={:.2}, top_p={:.2}, top_k={} (固定参数，与Python一致)",
        request_id, args_semantic.temperature, args_semantic.top_p, args_semantic.top_k
    );

    // 创建独立的RNG用于semantic阶段
    // 声音克隆场景也支持随机采样，根据seed参数决定采样方式
    let mut semantic_rng = if let Some(rng_instance) = rng {
        if request.args.layered_randomness.use_independent_seeds {
            if let Some(seed) = request.args.seed {
                // 用户提供了seed，使用确定性采样
                Some(StdRng::seed_from_u64(seed.wrapping_add(
                    request.args.layered_randomness.semantic_seed_offset,
                )))
            } else {
                // 没有seed，使用传入的RNG
                Some(rng_instance)
            }
        } else {
            Some(rng_instance)
        }
    } else {
        // 即使rng为None，也创建新的RNG用于随机采样（除非用户明确指定seed）
        if let Some(seed) = request.args.seed {

            Some(StdRng::seed_from_u64(seed.wrapping_add(
                request.args.layered_randomness.semantic_seed_offset,
            )))
        } else {

            Some(StdRng::from_entropy())
        }
    };

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


        // 直接使用屏蔽后的logits进行采样
        let next_id = crate::rwkv_sampler::sample_logits(
            &logits_masked,
            &args_semantic,
            None,
            &mut semantic_rng,
        );


        // 检查是否遇到EOS token（必须在范围检查之前）
        if next_id == crate::rwkv_sampler::TTS_EOS_TOKEN as usize {
            break;
        }

        // 额外检查：确保token在semantic范围内 [0..8192)（修复：应该是>8192而不是>=8192）
        if next_id > crate::rwkv_sampler::TTS_EOS_TOKEN as usize {
            warn!(
                "🚨 [{}] Token {} 超出semantic范围[0..8192]，跳过此token",
                request_id, next_id
            );
            continue;
        }

        semantic_tokens.push(next_id as i32);

        // 反馈到模型：语义阶段直接使用原始token（不加偏移）
        inference.batches[0].push(next_id as u32);
    }
    // 返回预提取的global tokens和新生成的semantic tokens（已排除预读取的semantic tokens）
    Ok((global_tokens, semantic_tokens))
}
