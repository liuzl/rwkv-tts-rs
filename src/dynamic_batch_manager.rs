use anyhow::Result;
use flume::{Receiver, Sender};

use rand::SeedableRng;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tracing::{error, warn};

// 导入拆分的模块
use crate::batch_types::*;

use crate::shared_runtime::*;

// 引入新的推理模块
use crate::normal_mode_inference::execute_normal_inference;
use crate::zero_shot_inference::execute_zero_shot_inference;

/// 动态批处理管理器
/// 负责收集请求、组织批次、协调推理工作线程
pub struct DynamicBatchManager {
    /// 配置
    config: DynamicBatchConfig,
    /// 请求发送通道
    request_tx: Sender<DynamicTtsRequest>,
    /// 共享运行时
    _shared_runtime: Arc<SharedRwkvRuntime>,
}

impl DynamicBatchManager {
    /// 创建新的动态批处理管理器
    pub async fn new(
        model_path: &str,
        vocab_path: &str,
        config: DynamicBatchConfig,
        quant_config: Option<std::collections::HashMap<usize, web_rwkv::runtime::model::Quant>>,
    ) -> Result<Self> {
        // 初始化动态批处理管理器
        // 配置信息

        // 创建共享运行时
        let shared_runtime = Arc::new(
            SharedRwkvRuntime::new(
                model_path.to_string(),
                vocab_path.to_string(),
                quant_config,
                config.clone(),
            )
            .await?,
        );

        // 创建请求通道
        let (request_tx, request_rx) = flume::unbounded();
        let (infer_tx, infer_rx) = flume::unbounded();

        // 启动核心运行时
        let shared_runtime_clone = shared_runtime.clone();
        let config_clone = config.clone();
        tokio::spawn(async move {
            Self::run_core_runtime(shared_runtime_clone, request_rx, infer_tx, config_clone).await;
        });

        // 启动推理工作线程
        for worker_id in 0..config.max_concurrent_batches {
            let infer_rx_clone = infer_rx.clone();
            let shared_runtime_clone = shared_runtime.clone();
            let config_clone = config.clone();
            tokio::spawn(async move {
                Self::infer_worker(
                    worker_id,
                    infer_rx_clone,
                    shared_runtime_clone,
                    config_clone,
                )
                .await;
            });
        }

        // 动态批处理管理器初始化完成

        Ok(Self {
            config,
            request_tx,
            _shared_runtime: shared_runtime,
        })
    }

    /// 生成TTS
    pub async fn generate_tts(
        &self,
        text: String,
        property_tokens: Vec<i32>,
        ref_global_tokens: Option<Vec<i32>>,
        ref_semantic_tokens: Option<Vec<i32>>,
        args: crate::rwkv_sampler::SamplerArgs,
    ) -> Result<(Vec<i32>, Vec<i32>)> {
        let (response_tx, response_rx) = oneshot::channel();

        let request = DynamicTtsRequest {
            text,
            property_tokens,
            ref_global_tokens,
            ref_semantic_tokens,
            args,
            response_tx,
            submitted_at: Instant::now(),
            batch_id: 0, // 将在收集阶段设置
        };

        self.request_tx
            .send_async(request)
            .await
            .map_err(|e| anyhow::anyhow!("发送请求失败: {}", e))?;

        response_rx
            .await
            .map_err(|e| anyhow::anyhow!("接收响应失败: {}", e))?
    }

    /// 核心运行时 - 负责收集请求并分发到推理工作线程
    async fn run_core_runtime(
        _shared_runtime: Arc<SharedRwkvRuntime>,
        request_rx: Receiver<DynamicTtsRequest>,
        infer_tx: Sender<InferBatch>,
        config: DynamicBatchConfig,
    ) {
        // 核心运行时启动

        // 启动请求收集工作线程
        tokio::spawn(Self::enqueue_worker(request_rx, infer_tx, config));

        // 保持运行时活跃
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }

    /// 请求收集工作线程
    async fn enqueue_worker(
        request_rx: Receiver<DynamicTtsRequest>,
        infer_tx: Sender<InferBatch>,
        config: DynamicBatchConfig,
    ) {
        // 请求收集工作线程启动
        let mut pending_requests = VecDeque::new();
        let mut batch_counter = 1usize;

        loop {
            let collect_start = Instant::now();

            // 收集请求的逻辑
            loop {
                let timeout = Duration::from_millis(config.collect_timeout_ms);
                match tokio::time::timeout(timeout, request_rx.recv_async()).await {
                    Ok(Ok(mut request)) => {
                        request.batch_id = batch_counter;
                        pending_requests.push_back(request);

                        // 激进地收集所有立即可用的请求（非阻塞）
                        let mut quick_collect_count = 0;
                        let _quick_collect_start = Instant::now();
                        while pending_requests.len() < config.max_batch_size
                            && quick_collect_count < 50
                        {
                            match request_rx.try_recv() {
                                Ok(mut req) => {
                                    req.batch_id = batch_counter;
                                    pending_requests.push_back(req);
                                    quick_collect_count += 1;
                                }
                                Err(_) => break,
                            }
                        }

                        if quick_collect_count > 0 {
                            // 快速收集到额外请求
                        }

                        // 如果收集到多个请求，立即处理
                        if pending_requests.len() > 1 {
                            break;
                        }

                        // 如果只有一个请求，只等待很短时间（10ms）收集更多请求
                        if pending_requests.len() == 1 && collect_start.elapsed().as_millis() >= 10
                        {
                            break;
                        }
                    }
                    Ok(Err(_)) => {
                        // 请求通道关闭，工作线程退出
                        return;
                    }
                    Err(_) => {
                        // 超时，如果有请求则立即处理
                        if !pending_requests.is_empty() {
                            break;
                        }
                    }
                }
            }

            // 处理收集到的请求
            if !pending_requests.is_empty() {
                let batch_id = batch_counter;
                batch_counter += 1;

                // 收集到批次请求

                Self::process_collected_batch(
                    pending_requests.drain(..).collect(),
                    &infer_tx,
                    batch_id,
                )
                .await;
            }
        }
    }

    /// 处理收集到的批次
    async fn process_collected_batch(
        requests: Vec<DynamicTtsRequest>,
        infer_tx: &Sender<InferBatch>,
        batch_id: usize,
    ) {
        let batch_size = requests.len();
        let (result_tx, result_rx) = flume::unbounded();

        // 开始处理批次

        // 转换为批处理请求
        let batch_requests: Vec<crate::rwkv_sampler::TtsBatchRequest> = requests
            .iter()
            .map(|req| crate::rwkv_sampler::TtsBatchRequest {
                text: req.text.clone(),
                property_tokens: req.property_tokens.clone(),
                ref_global_tokens: req.ref_global_tokens.clone(),
                ref_semantic_tokens: req.ref_semantic_tokens.clone(),
                args: req.args.clone(),
            })
            .collect();

        // 发送到推理队列
        let infer_batch = InferBatch::Run {
            batch_id,
            requests: batch_requests,
            sender: result_tx,
        };

        // 发送批次到推理队列
        if let Err(e) = infer_tx.send_async(infer_batch).await {
            error!("❌ 发送推理批次 {} 失败: {}", batch_id, e);
            // 发送错误给所有请求
            for request in requests {
                let _ = request
                    .response_tx
                    .send(Err(anyhow::anyhow!("推理队列发送失败")));
            }
            return;
        }

        // 等待推理结果
        // 等待批次推理结果
        match result_rx.recv_async().await {
            Ok(results) => {
                // 检查结果数量是否匹配
                if results.len() == batch_size {
                    // 分发结果
                    for (request, result) in requests.into_iter().zip(results.into_iter()) {
                        let _ = request.response_tx.send(Ok(result));
                    }
                    // 批次处理完成
                } else {
                    // 结果数量不匹配，可能是推理失败
                    error!(
                        "❌ 批次 {} 结果数量不匹配: 期望 {}, 实际 {}",
                        batch_id,
                        batch_size,
                        results.len()
                    );
                    // 发送错误给所有请求
                    for request in requests {
                        let _ = request
                            .response_tx
                            .send(Err(anyhow::anyhow!("推理失败，结果数量不匹配")));
                    }
                }
            }
            Err(e) => {
                error!("❌ 接收批次 {} 推理结果失败: {}", batch_id, e);
                // 发送错误给所有请求
                for request in requests {
                    let _ = request
                        .response_tx
                        .send(Err(anyhow::anyhow!("推理结果接收失败")));
                }
            }
        }
    }

    /// 推理工作线程 - 重构版：使用独立状态管理确保状态隔离
    /// 关键改进：每个请求创建独立的推理上下文，避免状态污染
    async fn infer_worker(
        _worker_id: usize,
        infer_rx: Receiver<InferBatch>,
        shared_runtime: Arc<SharedRwkvRuntime>,
        _config: DynamicBatchConfig,
    ) {
        // 推理工作线程启动，使用独立状态管理架构
        // 状态隔离：工作线程将为每个请求创建独立推理上下文

        while let Ok(batch) = infer_rx.recv_async().await {
            match batch {
                InferBatch::Run {
                    batch_id,
                    requests,
                    sender,
                } => {
                    let batch_size = requests.len();

                    // 工作线程开始推理批次

                    // 🔧 关键改进：为每个请求创建独立的推理上下文
                    // 确保完全的状态隔离，避免并发请求间的状态污染
                    let result = Self::process_batch_with_independent_contexts(
                        shared_runtime.clone(),
                        requests,
                        batch_id as u64,
                    )
                    .await;

                    match result {
                        Ok(results) => {
                            // 工作线程批次推理完成

                            if let Err(e) = sender.send_async(results).await {
                                error!("发送推理结果失败: {}", e);
                            }
                        }
                        Err(e) => {
                            error!("批次 {} 推理失败: {}", batch_id, e);
                            // 发送与请求数量匹配的错误结果
                            let error_results: Vec<(Vec<i32>, Vec<i32>)> =
                                (0..batch_size).map(|_| (vec![], vec![])).collect();
                            let _ = sender.send_async(error_results).await;
                        }
                    }
                }
                InferBatch::Result { batch_id, sender } => {
                    // 处理结果请求（如果需要）
                    warn!("收到结果请求 {}，当前实现不支持", batch_id);
                    let _ = sender.send(vec![]);
                }
            }
        }

        // 推理工作线程退出
    }

    /// 使用独立上下文处理批次
    /// 为每个请求创建独立的推理上下文，确保状态完全隔离
    async fn process_batch_with_independent_contexts(
        shared_runtime: Arc<SharedRwkvRuntime>,
        requests: Vec<crate::rwkv_sampler::TtsBatchRequest>,
        _batch_id: u64,
    ) -> Result<Vec<(Vec<i32>, Vec<i32>)>> {
        let batch_size = requests.len();
        let mut results = Vec::with_capacity(batch_size);

        // 为批次创建独立推理上下文

        // 为每个请求创建独立的推理上下文并顺序处理（避免GPU资源争用）
        // 注意：这里改为顺序处理而不是并行处理，因为GPU资源是有限的
        for request in requests.into_iter() {
            let shared_runtime_clone = shared_runtime.clone();
            // 统一使用全局请求ID命名：req_<number>
            let request_id = shared_runtime_clone.generate_request_id();

            // 检测是否为声音克隆场景
            let is_voice_cloning =
                request.ref_global_tokens.is_some() || request.ref_semantic_tokens.is_some();

            // 创建独立的推理上下文
            let options = TtsInferOptions {
                temperature: request.args.temperature,
                top_k: request.args.top_k,
                top_p: request.args.top_p,
                seed: if is_voice_cloning {
                    // 声音克隆时忽略用户提供的seed参数，确保确定性
                    // 声音克隆场景：忽略用户seed参数，使用确定性采样
                    None
                } else {
                    request.args.seed
                },
                voice_fidelity: request.args.voice_fidelity,
                layered_randomness: request.args.layered_randomness.clone(),
                sampling: None,
                token_chunk_size: request.args.token_chunk_size,
            };

            let infer_context = shared_runtime_clone
                .create_infer_context(request_id.clone(), request.text.clone(), options)
                .await?;

            // 保存状态ID用于清理
            let state_id = infer_context.state_id;

            // 执行独立推理
            let result = Self::execute_independent_inference(infer_context, request).await;

            // 清理状态
            shared_runtime_clone.cleanup_state(state_id).await;

            match result {
                Ok(res) => {
                    results.push(res);
                    // 请求处理完成
                }
                Err(e) => {
                    error!("❌ 请求 {} 处理失败: {}", request_id, e);
                    results.push((vec![], vec![]));
                }
            }
        }

        // 批次独立推理完成

        Ok(results)
    }

    /// 执行独立推理
    async fn execute_independent_inference(
        infer_context: TtsInferContext,
        request: crate::rwkv_sampler::TtsBatchRequest,
    ) -> Result<(Vec<i32>, Vec<i32>)> {
        let _request_id = &infer_context.request_id;
        // 开始独立推理

        // 检测是否为声音克隆场景
        let is_voice_cloning =
            request.ref_global_tokens.is_some() || request.ref_semantic_tokens.is_some();

        // 为本次请求创建独立RNG（可复现且互不干扰）
        let rng: rand::rngs::StdRng = if is_voice_cloning {
            // 声音克隆时不使用随机数，使用固定种子确保确定性
            // 声音克隆模式：使用固定种子确保结果一致性
            rand::rngs::StdRng::seed_from_u64(0) // 使用固定种子
        } else if let Some(seed) = request.args.seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_rng(rand::thread_rng()).expect("failed to seed StdRng")
        };

        // Acquire runtime semaphore for the entire inference to ensure isolation
        let _runtime_permit = infer_context
            .runtime_semaphore
            .acquire()
            .await
            .map_err(|e| anyhow::anyhow!("无法获取运行时信号量: {}", e))?;

        // 已获取信号量许可，开始推理

        // 获取tokenizer
        let tokenizer = &infer_context.tokenizer;

        // 编码文本
        let text_tokens_u32: Vec<u32> = tokenizer
            .encode(request.text.as_bytes())
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let text_tokens_raw: Vec<i32> = text_tokens_u32.into_iter().map(|t| t as i32).collect();

        // 根据C++代码逻辑，文本tokens直接使用原始ID，不需要任何偏移
        let text_tokens: Vec<i32> = text_tokens_raw.clone();

        // 文本编码结果

        // 释放原始tokens变量
        drop(text_tokens_raw);

        // 检测是否为Zero-shot模式（有预提取的音色特征）
        let is_zero_shot =
            request.ref_global_tokens.is_some() && request.ref_semantic_tokens.is_some();

        if is_zero_shot {
            // 检测到Zero-shot模式，调用专用推理函数
            return execute_zero_shot_inference(&infer_context, request, text_tokens, Some(rng))
                .await;
        }

        // 普通模式推理
        // 普通模式推理，调用专用推理函数
        return execute_normal_inference(&infer_context, request, text_tokens).await;
    }

    /// 获取配置
    pub fn config(&self) -> &DynamicBatchConfig {
        &self.config
    }
}

/// 全局动态批处理管理器单例
static GLOBAL_DYNAMIC_BATCH_MANAGER: std::sync::OnceLock<Arc<DynamicBatchManager>> =
    std::sync::OnceLock::new();

/// 初始化全局动态批处理管理器（支持量化配置）
pub async fn init_global_dynamic_batch_manager(
    model_path: &str,
    vocab_path: &str,
    config: DynamicBatchConfig,
    quant_config: Option<std::collections::HashMap<usize, web_rwkv::runtime::model::Quant>>,
) -> Result<()> {
    let manager = DynamicBatchManager::new(model_path, vocab_path, config, quant_config).await?;

    GLOBAL_DYNAMIC_BATCH_MANAGER
        .set(Arc::new(manager))
        .map_err(|_| anyhow::anyhow!("全局动态批处理管理器已经初始化"))?;

    Ok(())
}

/// 获取全局动态批处理管理器实例
pub fn get_global_dynamic_batch_manager() -> Result<Arc<DynamicBatchManager>> {
    GLOBAL_DYNAMIC_BATCH_MANAGER
        .get()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("全局动态批处理管理器未初始化"))
}
