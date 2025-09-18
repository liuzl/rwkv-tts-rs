//! 共享的RWKV运行时模块
//!
//! 本模块包含共享的RWKV Runtime实例和推理上下文管理，
//! 参考ai00-core的设计，使用共享Runtime和独立状态来优化内存使用。

use anyhow::Result;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock, Semaphore};
// 删除未使用的导入
use web_rwkv::runtime::loader::Loader;
use web_rwkv::runtime::model::{Bundle, State};
use web_rwkv::{runtime::v7, tokenizer::Tokenizer};

use crate::batch_types::{DynamicBatchConfig, TtsInferOptions, TtsStateId};

/// TTS推理上下文，类似ai00-core的GenerateContext
#[derive(Clone)]
pub struct TtsInferContext {
    /// 请求ID
    pub request_id: String,
    /// 状态ID
    pub state_id: TtsStateId,
    /// 输入文本
    pub text: String,
    /// 推理选项
    pub options: TtsInferOptions,
    /// 分词器引用
    pub tokenizer: Arc<Tokenizer>,
    /// Runtime引用
    pub runtime: Arc<web_rwkv::runtime::TokioRuntime<web_rwkv::runtime::infer::Rnn>>,
    /// 模型状态（独立副本）- 重新添加以确保状态隔离
    pub state: Arc<Mutex<Box<dyn State + Send + Sync>>>,
    /// Serialize runtime.infer calls for correctness under concurrency
    pub runtime_semaphore: Arc<Semaphore>,
}

/// 共享的RWKV Runtime实例
/// 参考ai00-core的设计，使用共享Runtime和独立状态
pub struct SharedRwkvRuntime {
    /// 共享的Runtime实例
    runtime: Arc<web_rwkv::runtime::TokioRuntime<web_rwkv::runtime::infer::Rnn>>,
    /// 共享的模型Bundle（用于创建状态）
    model_bundle: Arc<v7::Bundle<f32>>,
    /// 共享的分词器
    tokenizer: Arc<Tokenizer>,
    /// 全局请求ID生成器（用于统一日志命名：req_<number>）
    request_id_generator: AtomicU64,
    /// 状态ID生成器
    state_id_generator: AtomicU64,
    /// 活跃状态统计
    active_states: Arc<RwLock<HashMap<TtsStateId, String>>>,
    /// 模型路径
    #[allow(dead_code)]
    model_path: String,
    /// 词汇表路径
    #[allow(dead_code)]
    vocab_path: String,
    /// A semaphore to control concurrent inference calls
    /// The number of permits should be configured based on GPU capabilities
    runtime_semaphore: Arc<Semaphore>,
}

impl SharedRwkvRuntime {
    /// 创建新的共享Runtime（支持量化配置）
    pub async fn new(
        model_path: String,
        vocab_path: String,
        quant_config: Option<HashMap<usize, web_rwkv::runtime::model::Quant>>,
        config: DynamicBatchConfig, // 添加配置参数
    ) -> Result<Self> {
        // 初始化共享RWKV Runtime
        // 配置信息

        // 创建WebRWKV上下文和模型
        use web_rwkv::context::{ContextBuilder, InstanceExt};
        use web_rwkv::runtime::model::{ContextAutoLimits, ModelBuilder};
        use web_rwkv::wgpu::{Instance, PowerPreference};

        // 检测模型格式并加载
        let model_file_path = if Path::new(&model_path).is_dir() {
            // 如果是目录，优先尝试SafeTensors格式
            let safetensors_path = Path::new(&model_path).join("rwkvtts-Int8_22.safetensors");
            let prefab_path = Path::new(&model_path).join("rwkvtts-Int8_22.prefab");
            if safetensors_path.exists() {
                safetensors_path
            } else if prefab_path.exists() {
                prefab_path
            } else {
                return Err(anyhow::anyhow!(
                    "No supported model file found in directory: {}",
                    model_path
                ));
            }
        } else {
            PathBuf::from(&model_path)
        };

        let file = std::fs::File::open(&model_file_path)
            .map_err(|e| anyhow::anyhow!("Failed to open model file: {}", e))?;
        let data = unsafe { Mmap::map(&file) }
            .map_err(|e| anyhow::anyhow!("Failed to map model file: {}", e))?;

        // 尝试检测格式并获取模型信息
        let (load_type, info) = if let Ok(safetensors) = SafeTensors::deserialize(&data) {
            // SafeTensors格式
            let actual_info = Loader::info(&safetensors)
                .map_err(|e| anyhow::anyhow!("Failed to get SafeTensors model info: {}", e))?;

            // 检查版本
            if actual_info.version != web_rwkv::runtime::model::ModelVersion::V7 {
                return Err(anyhow::anyhow!(
                    "Only V7 models are supported, got: {:?}",
                    actual_info.version
                ));
            }

            // SafeTensors模型信息记录

            ("safetensors", actual_info)
        } else {
            // 假设为prefab格式，为V7模型创建默认info（实际加载时会验证）
            // 检测到prefab格式，使用V7模型默认配置
            let default_info = web_rwkv::runtime::model::ModelInfo {
                version: web_rwkv::runtime::model::ModelVersion::V7,
                num_vocab: 65536,
                num_layer: 32,
                num_emb: 2048,
                num_head: 32,
                num_hidden: 2048,
                custom: web_rwkv::runtime::model::ModelCustomInfo::None,
            };
            ("prefab", default_info)
        };

        // 模型格式确定

        // 创建GPU实例和适配器
        let instance = Instance::default();
        let adapter = instance
            .adapter(PowerPreference::HighPerformance)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get adapter: {}", e))?;

        // 创建上下文
        let context = ContextBuilder::new(adapter)
            .auto_limits(&info)
            .build()
            .await?;

        // 根据格式构建模型Bundle
        let model = if load_type == "safetensors" {
            // SafeTensors格式
            let safetensors = SafeTensors::deserialize(&data)
                .map_err(|e| anyhow::anyhow!("Failed to deserialize SafeTensors: {}", e))?;
            let mut builder = ModelBuilder::new(&context, safetensors);
            if let Some(ref quant) = quant_config {
                builder = builder.quant(quant.clone());
            }
            builder.build_v7().await?
        } else {
            // prefab格式 - 使用cbor4ii和Seed直接反序列化
            use cbor4ii::{core::utils::SliceReader, serde::Deserializer};
            use serde::de::DeserializeSeed;
            use web_rwkv::tensor::serialization::Seed;

            let reader = SliceReader::new(&data);
            let mut deserializer = Deserializer::new(reader);
            let seed = Seed::<web_rwkv::context::Context, v7::Model>::new(&context);
            seed.deserialize(&mut deserializer)
                .map_err(|e| anyhow::anyhow!("Failed to deserialize prefab model: {}", e))?
        };
        let model_bundle = Arc::new(v7::Bundle::new(model, config.max_concurrent_batches));

        // 使用配置中的信号量许可数量
        let semaphore_permits = config.semaphore_permits;
        // 设置信号量许可数量

        // 创建TokioRuntime实例
        let runtime = Arc::new(web_rwkv::runtime::TokioRuntime::new((*model_bundle).clone()).await);

        // 创建分词器 - 读取词汇表文件内容
        let vocab_content = std::fs::read_to_string(&vocab_path)
            .map_err(|e| anyhow::anyhow!("Failed to read vocab file {}: {}", vocab_path, e))?;
        let tokenizer = Arc::new(
            Tokenizer::new(&vocab_content)
                .map_err(|e| anyhow::anyhow!("Failed to parse vocabulary: {}", e))?,
        );

        // 共享RWKV Runtime初始化完成

        Ok(Self {
            runtime,
            model_bundle,
            tokenizer,
            request_id_generator: AtomicU64::new(1),
            state_id_generator: AtomicU64::new(1),
            active_states: Arc::new(RwLock::new(HashMap::new())),
            model_path,
            vocab_path,
            // 使用配置中的信号量许可数量
            runtime_semaphore: Arc::new(Semaphore::new(semaphore_permits)),
        })
    }

    /// 生成全局唯一的请求ID（用于统一日志命名）
    pub fn generate_request_id(&self) -> String {
        let id = self.request_id_generator.fetch_add(1, Ordering::SeqCst);
        format!("req_{}", id)
    }

    /// 创建新的推理上下文，每个请求获得独立的副本
    pub async fn create_infer_context(
        &self,
        request_id: String,
        text: String,
        options: TtsInferOptions,
    ) -> Result<TtsInferContext> {
        // 生成唯一的状态ID
        let state_id = TtsStateId(self.state_id_generator.fetch_add(1, Ordering::SeqCst));

        // 创建独立的的状态副本
        let state = Arc::new(Mutex::new(
            Box::new(self.model_bundle.state()) as Box<dyn State + Send + Sync>
        ));

        // 记录活跃状态（优化：减少锁持有时间）
        {
            let mut active = self.active_states.write().await;
            active.insert(state_id, request_id.clone());
            drop(active); // 显式释放锁
        }

        // 创建推理上下文

        Ok(TtsInferContext {
            request_id,
            state_id,
            text,
            options,
            tokenizer: self.tokenizer.clone(),
            runtime: self.runtime.clone(),
            state, // 添加独立状态
            runtime_semaphore: self.runtime_semaphore.clone(),
        })
    }

    /// 清理状态（优化：减少锁持有时间）
    pub async fn cleanup_state(&self, state_id: TtsStateId) {
        {
            let mut active = self.active_states.write().await;
            active.remove(&state_id);
            drop(active); // 显式释放锁
        }
        // 清理状态
    }

    /// 获取分词器
    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }

    /// 获取模型Bundle
    pub fn model_bundle(&self) -> &Arc<v7::Bundle<f32>> {
        &self.model_bundle
    }

    /// 获取Runtime实例
    pub fn runtime(&self) -> &Arc<web_rwkv::runtime::TokioRuntime<web_rwkv::runtime::infer::Rnn>> {
        &self.runtime
    }

    /// 获取状态统计信息
    pub async fn stats(&self) -> crate::tts_state_manager::TtsStateStats {
        let active = self.active_states.read().await;
        crate::tts_state_manager::TtsStateStats {
            active_states: active.len(),
        }
    }
}
