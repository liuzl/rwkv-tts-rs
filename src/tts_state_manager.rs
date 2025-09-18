//! TTS状态管理器
//! 参考ai00-core实现独立状态管理，每个请求获得独立状态副本

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::rwkv_sampler::RwkvSampler;

use std::sync::atomic::{AtomicU64, Ordering};

/// 统一使用batch_types中的类型，去重
pub use crate::batch_types::{TtsInferOptions, TtsStateId};

/// TTS推理上下文，类似ai00-core的GenerateContext
pub struct TtsInferContext {
    /// 请求ID
    pub request_id: String,
    /// 状态ID
    pub state_id: TtsStateId,
    /// 输入文本
    pub text: String,
    /// 采样器实例（每个上下文独立）
    pub sampler: RwkvSampler,
    /// 推理选项
    pub options: TtsInferOptions,
}

/// TTS状态管理器，负责管理独立的状态实例
pub struct TtsStateManager {
    /// 模型路径
    model_path: String,
    /// 词汇表路径
    vocab_path: String,
    /// 量化配置
    quant_config: Option<HashMap<usize, web_rwkv::runtime::model::Quant>>,
    /// 状态ID生成器
    state_id_generator: AtomicU64,
    /// 活跃状态统计
    active_states: Arc<RwLock<HashMap<TtsStateId, String>>>,
}

impl TtsStateManager {
    /// 创建新的状态管理器
    pub async fn new(
        model_path: String,
        vocab_path: String,
        quant_config: Option<HashMap<usize, web_rwkv::runtime::model::Quant>>,
    ) -> Result<Self> {
        Ok(Self {
            model_path,
            vocab_path,
            quant_config,
            state_id_generator: AtomicU64::new(1),
            active_states: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// 创建新的推理上下文，每个请求获得独立的状态副本
    pub async fn create_infer_context(
        &self,
        request_id: String,
        text: String,
        options: TtsInferOptions,
    ) -> Result<TtsInferContext> {
        // 生成唯一的状态ID
        let state_id = TtsStateId(self.state_id_generator.fetch_add(1, Ordering::SeqCst));

        // 创建独立的采样器实例
        let sampler = RwkvSampler::new(
            &self.model_path,
            &self.vocab_path,
            self.quant_config.clone(),
            options.token_chunk_size,
        )
        .await?;

        // 采样器参数已在构造时设置

        // 记录活跃状态
        {
            let mut active = self.active_states.write().await;
            active.insert(state_id, request_id.clone());
        }

        Ok(TtsInferContext {
            request_id,
            state_id,
            text,
            sampler,
            options,
        })
    }

    /// 清理状态
    pub async fn cleanup_state(&self, state_id: TtsStateId) {
        let mut active = self.active_states.write().await;
        active.remove(&state_id);
    }

    /// 获取状态管理器统计信息
    pub async fn stats(&self) -> TtsStateStats {
        let active = self.active_states.read().await;
        TtsStateStats {
            active_states: active.len(),
        }
    }
}

/// 状态管理器统计信息
#[derive(Debug, Clone)]
pub struct TtsStateStats {
    /// 活跃状态数量
    pub active_states: usize,
}

impl std::fmt::Display for TtsStateStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TTS State Manager: {} active states", self.active_states)
    }
}
