//! 批处理相关的共用类型定义和配置
//!
//! 本模块包含动态批处理系统中使用的核心类型定义，包括：
//! - 请求和批次结构体
//! - 配置参数
//! - 状态标识符
//! - 推理选项

use anyhow::Result;
use flume::Sender;
use std::time::Instant;
use tokio::sync::oneshot;

use crate::rwkv_sampler::{SamplerArgs, TtsBatchRequest};

/// TTS请求项，包含完整的请求信息和响应通道
#[derive(Debug)]
pub struct DynamicTtsRequest {
    pub text: String,
    pub property_tokens: Vec<i32>,
    pub ref_global_tokens: Option<Vec<i32>>,
    pub ref_semantic_tokens: Option<Vec<i32>>,
    pub args: SamplerArgs,
    pub response_tx: oneshot::Sender<Result<(Vec<i32>, Vec<i32>)>>,
    pub submitted_at: Instant,
    pub batch_id: usize,
}

/// 推理批次
#[derive(Debug)]
pub enum InferBatch {
    /// 执行推理
    Run {
        batch_id: usize,
        requests: Vec<TtsBatchRequest>,
        sender: Sender<Vec<(Vec<i32>, Vec<i32>)>>,
    },
    /// 获取结果
    Result {
        batch_id: usize,
        sender: oneshot::Sender<Vec<(Vec<i32>, Vec<i32>)>>,
    },
}

/// 动态批处理配置
#[derive(Debug, Clone)]
pub struct DynamicBatchConfig {
    /// 最小批处理大小
    pub min_batch_size: usize,
    /// 最大批处理大小
    pub max_batch_size: usize,
    /// 批处理收集超时时间（毫秒）
    pub collect_timeout_ms: u64,
    /// 推理超时时间（毫秒）
    pub inference_timeout_ms: u64,
    /// 最大并发批次数
    pub max_concurrent_batches: usize,
    /// 信号量许可数量（基于硬件和负载调整）
    pub semaphore_permits: usize,
}

impl Default for DynamicBatchConfig {
    fn default() -> Self {
        Self {
            min_batch_size: 1,
            max_batch_size: 10,
            collect_timeout_ms: 50,
            inference_timeout_ms: 60000,
            max_concurrent_batches: 4, // 合理的默认并发数
            semaphore_permits: 3,      // 信号量许可数量略小于并发数
        }
    }
}

/// TTS状态ID，用于标识不同的状态实例
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TtsStateId(pub u64);

/// TTS推理选项
#[derive(Debug, Clone)]
pub struct TtsInferOptions {
    /// 温度参数
    pub temperature: f32,
    /// top_k参数
    pub top_k: usize,
    /// top_p参数
    pub top_p: f32,
    /// 随机种子
    pub seed: Option<u64>,
    /// 音色保真度 (0.0-1.0)
    pub voice_fidelity: f32,
    /// 分层随机性配置
    pub layered_randomness: crate::rwkv_sampler::LayeredRandomnessConfig,
    /// 采样配置（可选）
    pub sampling: Option<std::collections::HashMap<String, serde_json::Value>>,
}

impl Default for TtsInferOptions {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            seed: None,
            voice_fidelity: 0.8,
            layered_randomness: crate::rwkv_sampler::LayeredRandomnessConfig::default(),
            sampling: None,
        }
    }
}
