use crate::ref_audio_utilities::RefAudioUtilities;
use crate::rwkv_sampler::{RwkvSampler, SamplerArgs, TtsBatchRequest};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use web_rwkv::runtime::model::Quant;

/// 全局RwkvSampler管理器，支持批处理并发
pub struct GlobalSamplerManager {
    sampler: Arc<Mutex<RwkvSampler>>,
    ref_audio_utils: Arc<Mutex<RefAudioUtilities>>,
    max_batch_size: usize,
}

impl GlobalSamplerManager {
    /// 创建新的全局采样器管理器
    pub async fn new(
        model_path: &str,
        vocab_path: &str,
        ref_audio_utils: RefAudioUtilities,
        max_batch_size: Option<usize>,
    ) -> Result<Self> {
        // 创建默认采样参数
        let _sampler_args = SamplerArgs::default();

        // 不使用量化配置创建采样器
        let quant_config = None;
        let sampler = RwkvSampler::new(model_path, vocab_path, quant_config, 256).await?;

        Ok(Self {
            sampler: Arc::new(Mutex::new(sampler)),
            ref_audio_utils: Arc::new(Mutex::new(ref_audio_utils)),
            max_batch_size: max_batch_size.unwrap_or(8), // 默认batch size为8
        })
    }

    /// 创建新的全局采样器管理器（支持自定义量化配置）
    pub async fn new_with_quant(
        model_path: &str,
        vocab_path: &str,
        ref_audio_utils: RefAudioUtilities,
        max_batch_size: Option<usize>,
        quant_config: Option<HashMap<usize, Quant>>,
    ) -> Result<Self> {
        // 创建默认采样参数
        let _sampler_args = SamplerArgs::default();

        // 使用指定的量化配置创建采样器
        let sampler = RwkvSampler::new(model_path, vocab_path, quant_config, 256).await?;

        Ok(Self {
            sampler: Arc::new(Mutex::new(sampler)),
            ref_audio_utils: Arc::new(Mutex::new(ref_audio_utils)),
            max_batch_size: max_batch_size.unwrap_or(8), // 默认batch size为8
        })
    }

    /// 获取采样器的引用
    pub fn get_sampler(&self) -> Arc<Mutex<RwkvSampler>> {
        self.sampler.clone()
    }

    /// 获取参考音频工具的引用
    pub fn get_ref_audio_utils(&self) -> Arc<Mutex<RefAudioUtilities>> {
        self.ref_audio_utils.clone()
    }

    /// 获取最大批处理大小
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// 批处理生成TTS tokens
    pub async fn generate_tts_tokens_batch(
        &self,
        requests: Vec<TtsBatchRequest>,
    ) -> Result<Vec<(Vec<i32>, Vec<i32>)>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }

        if requests.len() > self.max_batch_size {
            return Err(anyhow::anyhow!(
                "Batch size {} exceeds maximum {}",
                requests.len(),
                self.max_batch_size
            ));
        }

        let mut sampler = self.sampler.lock().await;
        sampler.generate_tts_tokens_batch(requests).await
    }

    /// 重置采样器状态
    pub async fn reset(&self) -> Result<()> {
        let mut sampler = self.sampler.lock().await;
        sampler.reset();
        Ok(())
    }
}

/// 全局采样器管理器的单例实例
static GLOBAL_SAMPLER_MANAGER: std::sync::OnceLock<Arc<GlobalSamplerManager>> =
    std::sync::OnceLock::new();

/// 初始化全局采样器管理器
pub async fn init_global_sampler_manager(
    model_path: &str,
    vocab_path: &str,
    ref_audio_utils: RefAudioUtilities,
    max_batch_size: Option<usize>,
) -> Result<()> {
    let manager =
        GlobalSamplerManager::new(model_path, vocab_path, ref_audio_utils, max_batch_size).await?;

    GLOBAL_SAMPLER_MANAGER
        .set(Arc::new(manager))
        .map_err(|_| anyhow::anyhow!("Global sampler manager already initialized"))?;

    Ok(())
}

/// 初始化全局采样器管理器（支持自定义量化配置）
pub async fn init_global_sampler_manager_with_quant(
    model_path: &str,
    vocab_path: &str,
    ref_audio_utils: RefAudioUtilities,
    max_batch_size: Option<usize>,
    quant_config: Option<HashMap<usize, Quant>>,
) -> Result<()> {
    let manager = GlobalSamplerManager::new_with_quant(
        model_path,
        vocab_path,
        ref_audio_utils,
        max_batch_size,
        quant_config,
    )
    .await?;

    GLOBAL_SAMPLER_MANAGER
        .set(Arc::new(manager))
        .map_err(|_| anyhow::anyhow!("Global sampler manager already initialized"))?;

    Ok(())
}

/// 获取全局采样器管理器实例
pub fn get_global_sampler_manager() -> Result<Arc<GlobalSamplerManager>> {
    GLOBAL_SAMPLER_MANAGER
        .get()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Global sampler manager not initialized"))
}
