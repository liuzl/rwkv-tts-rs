//! 批处理TTS管理器
//! 实现基于web-rwkv的全局Runtime管理和批处理请求队列

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::time::{timeout, Duration};
use tracing::{error, info, warn};

use crate::rwkv_sampler::{RwkvSampler, SamplerArgs, TtsBatchRequest};
use crate::ref_audio_utilities::RefAudioUtilities;

/// TTS请求结构
#[derive(Debug, Clone)]
pub struct TtsRequest {
    pub text: String,
    pub property_tokens: Vec<i32>,
    pub ref_global_tokens: Option<Vec<i32>>,
    pub ref_semantic_tokens: Option<Vec<i32>>,
    pub args: SamplerArgs,
    pub response_tx: oneshot::Sender<Result<(Vec<i32>, Vec<i32>)>>,
}

/// 批处理TTS管理器
pub struct BatchTtsManager {
    request_tx: mpsc::UnboundedSender<TtsRequest>,
    ref_audio_utilities: Arc<Mutex<Option<RefAudioUtilities>>>,
    inference_timeout_ms: u64,
}

impl BatchTtsManager {
    /// 创建新的批处理TTS管理器
    pub async fn new(
        model_path: &str,
        vocab_path: &str,
        batch_size: usize,
        batch_timeout_ms: u64,
        inference_timeout_ms: u64,
    ) -> Result<Self> {
        let (request_tx, request_rx) = mpsc::unbounded_channel();
        
        // 创建全局RWKV采样器，不使用量化配置
        let quant_config = None;
        let sampler = RwkvSampler::new(model_path, vocab_path, quant_config).await?;
        info!("全局RWKV采样器创建成功，batch_size: {}", batch_size);
        
        // 初始化参考音频工具
        let ref_audio_utilities = Arc::new(Mutex::new(
            RefAudioUtilities::new().ok()
        ));
        
        // 启动批处理工作线程
        let ref_audio_utils_clone = ref_audio_utilities.clone();
        tokio::spawn(async move {
            Self::batch_worker(
                sampler,
                request_rx,
                batch_size,
                batch_timeout_ms,
                inference_timeout_ms,
            ).await;
        });
        
        Ok(Self {
            request_tx,
            ref_audio_utilities: ref_audio_utils_clone,
            inference_timeout_ms,
        })
    }
    
    /// 提交TTS请求
    pub async fn generate_tts(
        &self,
        text: String,
        property_tokens: Vec<i32>,
        ref_global_tokens: Option<Vec<i32>>,
        ref_semantic_tokens: Option<Vec<i32>>,
        args: SamplerArgs,
    ) -> Result<(Vec<i32>, Vec<i32>)> {
        let (response_tx, response_rx) = oneshot::channel();
        
        let request = TtsRequest {
            text,
            property_tokens,
            ref_global_tokens,
            ref_semantic_tokens,
            args,
            response_tx,
        };
        
        // 发送请求到批处理队列
        self.request_tx.send(request)
            .map_err(|_| anyhow::anyhow!("批处理管理器已关闭"))?;
        
        // 等待响应
        let result = timeout(Duration::from_millis(self.inference_timeout_ms), response_rx).await
            .map_err(|_| anyhow::anyhow!("TTS请求超时，等待时间: {}ms", self.inference_timeout_ms))?
            .map_err(|_| anyhow::anyhow!("TTS请求被取消"))??;
        
        Ok(result)
    }
    
    /// 获取参考音频工具
    pub fn ref_audio_utilities(&self) -> Arc<Mutex<Option<RefAudioUtilities>>> {
        self.ref_audio_utilities.clone()
    }
    
    /// 批处理工作线程
    async fn batch_worker(
        mut sampler: RwkvSampler,
        mut request_rx: mpsc::UnboundedReceiver<TtsRequest>,
        batch_size: usize,
        batch_timeout_ms: u64,
        _inference_timeout_ms: u64,
    ) {
        let mut pending_requests = Vec::new();
        let batch_timeout = Duration::from_millis(batch_timeout_ms);
        
        info!("批处理工作线程启动，batch_size: {}, timeout: {}ms", batch_size, batch_timeout_ms);
        
        loop {
            // 收集请求直到达到批次大小或超时
            let should_process = if pending_requests.is_empty() {
                // 等待第一个请求
                match request_rx.recv().await {
                    Some(request) => {
                        pending_requests.push(request);
                        false // 继续收集更多请求
                    }
                    None => {
                        info!("请求通道关闭，批处理工作线程退出");
                        break;
                    }
                }
            } else {
                // 尝试收集更多请求直到批次满或超时
                match timeout(batch_timeout, request_rx.recv()).await {
                    Ok(Some(request)) => {
                        pending_requests.push(request);
                        pending_requests.len() >= batch_size
                    }
                    Ok(None) => {
                        info!("请求通道关闭，处理剩余请求后退出");
                        true // 处理剩余请求
                    }
                    Err(_) => {
                        // 超时，处理当前批次
                        true
                    }
                }
            };
            
            if should_process && !pending_requests.is_empty() {
                let batch_requests = std::mem::take(&mut pending_requests);
                let batch_count = batch_requests.len();
                
                info!("开始处理批次，请求数量: {}", batch_count);
                let start_time = std::time::Instant::now();
                
                // 处理批次
                Self::process_batch(&mut sampler, batch_requests).await;
                
                let elapsed = start_time.elapsed();
                info!("批次处理完成，耗时: {:.2}ms，平均每请求: {:.2}ms", 
                     elapsed.as_millis(), elapsed.as_millis() as f64 / batch_count as f64);
            }
        }
        
        // 处理剩余请求
        if !pending_requests.is_empty() {
            info!("处理剩余 {} 个请求", pending_requests.len());
            Self::process_batch(&mut sampler, pending_requests).await;
        }
    }
    
    /// 处理单个批次
    async fn process_batch(
        sampler: &mut RwkvSampler,
        requests: Vec<TtsRequest>,
    ) {
        let batch_size = requests.len();
        info!("🔄 处理批次，大小: {} (状态隔离模式)", batch_size);
        
        if requests.len() == 1 {
            // 单个请求，使用单独处理 - 确保状态隔离
            let request = requests.into_iter().next().unwrap();
            
            // 关键修复：单个请求处理前也进行状态重置
            sampler.reset();
            info!("🔄 单个请求处理前已重置状态");
            
            let result = sampler.generate_tts_tokens(
                &request.text,
                &request.property_tokens,
                request.ref_global_tokens.as_deref(),
                request.ref_semantic_tokens.as_deref(),
                &request.args,
            ).await;
            
            // 单个请求处理后也进行状态重置
            sampler.reset();
            info!("🔄 单个请求处理后已重置状态");
            
            if let Err(_) = request.response_tx.send(result) {
                warn!("无法发送单个请求响应，接收方已关闭");
            }
        } else {
            // 多个请求，使用批处理 - 批处理函数内部已有状态管理
            let batch_requests: Vec<TtsBatchRequest> = requests.iter().map(|req| {
                TtsBatchRequest {
                    text: req.text.clone(),
                    property_tokens: req.property_tokens.clone(),
                    ref_global_tokens: req.ref_global_tokens.clone(),
                    ref_semantic_tokens: req.ref_semantic_tokens.clone(),
                    args: req.args.clone(),
                }
            }).collect();
            
            match sampler.generate_tts_tokens_batch(batch_requests).await {
                Ok(results) => {
                    // 发送结果给各个请求
                    for (request, result) in requests.into_iter().zip(results.into_iter()) {
                        if let Err(_) = request.response_tx.send(Ok(result)) {
                            warn!("无法发送批处理请求响应，接收方已关闭");
                        }
                    }
                }
                Err(e) => {
                    error!("批处理失败: {}", e);
                    // 发送错误给所有请求
                    for request in requests {
                        if let Err(_) = request.response_tx.send(Err(anyhow::anyhow!("批处理失败: {}", e))) {
                            warn!("无法发送错误响应，接收方已关闭");
                        }
                    }
                }
            }
        }
    }
}