//! 推理状态管理器 - 优化推理调用频率
//!
//! 该模块实现了智能的推理状态管理，通过以下策略减少runtime.infer()调用频率：
//! 1. 批量推理：一次推理生成多个token的logits
//! 2. 状态缓存：缓存推理状态，避免重复计算
//! 3. 预测性推理：基于历史模式预测下一步需要的logits
//! 4. 异步推理：在后台预先计算可能需要的logits

// 移除性能监控组件
use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use web_rwkv::runtime::infer::RnnInput;
use web_rwkv::runtime::{infer::Rnn, TokioRuntime};

// Define types for compatibility
#[allow(dead_code)]
type Runtime = TokioRuntime<Rnn>;

/// 推理状态缓存条目
#[derive(Debug, Clone)]
struct InferenceStateEntry {
    #[allow(dead_code)]
    inference: RnnInput,
    logits_sequence: Vec<Vec<f32>>,
    #[allow(dead_code)]
    token_sequence: Vec<u32>,
    created_at: Instant,
    last_accessed: Instant,
    #[allow(dead_code)]
    access_count: u32,
}

/// 推理状态管理器配置
#[derive(Clone, Debug)]
pub struct InferenceStateConfig {
    /// 最大缓存条目数
    pub max_cache_entries: usize,
    /// 缓存条目最大存活时间
    pub max_entry_age: Duration,
    /// 批量推理大小
    pub batch_inference_size: usize,
    /// 预测性推理窗口大小
    pub prediction_window: usize,
    /// 启用异步预推理
    pub enable_async_pre_inference: bool,
    /// 状态相似度阈值
    pub state_similarity_threshold: f32,
}

impl Default for InferenceStateConfig {
    fn default() -> Self {
        Self {
            max_cache_entries: 100,
            max_entry_age: Duration::from_secs(300), // 5分钟
            batch_inference_size: 8,                 // 一次推理生成8个token的logits
            prediction_window: 16,                   // 预测未来16个token
            enable_async_pre_inference: true,
            state_similarity_threshold: 0.95,
        }
    }
}

/// 推理状态管理器
pub struct InferenceStateManager {
    /// 配置参数
    config: InferenceStateConfig,
    /// 状态缓存
    state_cache: Arc<RwLock<HashMap<String, InferenceStateEntry>>>,
    /// LRU访问队列
    access_queue: Arc<Mutex<VecDeque<String>>>,
    /// 统计信息
    stats: Arc<Mutex<InferenceStateStats>>,
    /// 异步推理任务句柄
    #[allow(dead_code)]
    async_tasks: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>>,
    // 移除性能监控器
}

/// 推理状态统计信息
#[derive(Default, Debug, Clone)]
pub struct InferenceStateStats {
    /// 总推理请求数
    pub total_requests: usize,
    /// 缓存命中数
    pub cache_hits: usize,
    /// 批量推理节省的调用数
    pub batch_inference_savings: usize,
    /// 预测性推理命中数
    pub prediction_hits: usize,
    /// 平均推理延迟
    pub average_inference_latency: Duration,
    /// 缓存清理次数
    pub cache_evictions: usize,
}

impl InferenceStateManager {
    /// 创建新的推理状态管理器
    pub fn new(config: InferenceStateConfig) -> Self {
        Self {
            config,
            state_cache: Arc::new(RwLock::new(HashMap::new())),
            access_queue: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(Mutex::new(InferenceStateStats::default())),
            async_tasks: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// 生成状态键
    #[allow(dead_code)]
    fn generate_state_key(&self, tokens: &[u32], context_id: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        tokens.hash(&mut hasher);
        context_id.hash(&mut hasher);
        format!("state_{}_{}", context_id, hasher.finish())
    }

    /// 智能推理：根据缓存状态决定是否需要实际推理
    pub async fn smart_inference(
        &self,
        runtime: &mut Box<dyn web_rwkv::runtime::Runtime<Rnn> + Send + Sync>,
        inference: RnnInput,
        context_id: &str,
        batch_size: usize,
    ) -> Result<(RnnInput, Vec<Vec<f32>>), anyhow::Error> {
        let start_time = Instant::now();

        let mut cached_logits = Vec::new();
        let mut current_inference = inference;

        // 尝试从缓存获取
        if let Some(entry) = self.state_cache.read().await.get(context_id) {
            if entry.created_at.elapsed() < self.config.max_entry_age {
                cached_logits.extend_from_slice(&entry.logits_sequence);
                if cached_logits.len() >= batch_size {
                    // 缓存足够，直接返回
                    cached_logits.truncate(batch_size);
                    return Ok((current_inference, cached_logits));
                }
            }
        }

        // 缓存不足，执行批量推理
        let remaining_tokens = batch_size - cached_logits.len();
        let mut new_logits = Vec::new();

        for _ in 0..remaining_tokens {
            let (next_inference, output) = runtime.infer(current_inference).await?;
            current_inference = next_inference;

            if !output.is_empty() && output[0].0.size() > 0 {
                let logits = output[0].0.clone().to_vec();
                new_logits.push(logits.clone());
                cached_logits.push(logits);
            }
        }

        // 缓存推理结果
        self.update_cache(context_id, &new_logits).await;

        // 移除性能统计

        Ok((current_inference, cached_logits))
    }

    /// 更新缓存
    async fn update_cache(&self, context_id: &str, logits: &[Vec<f32>]) {
        let entry = InferenceStateEntry {
            inference: RnnInput::new(vec![], 1024),
            logits_sequence: logits.to_vec(),
            token_sequence: Vec::new(),
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 1,
        };

        self.state_cache
            .write()
            .await
            .insert(context_id.to_string(), entry);

        // 清理过期缓存
        self.cleanup_expired_cache().await;
    }

    /// 清理过期缓存
    async fn cleanup_expired_cache(&self) {
        let mut cache = self.state_cache.write().await;
        let now = Instant::now();

        // 清理过期条目
        cache.retain(|_, entry| now.duration_since(entry.created_at) < self.config.max_entry_age);

        // 如果缓存仍然过大，驱逐旧条目
        if cache.len() > self.config.max_cache_entries {
            self.evict_old_entries(&mut cache).await;
        }
    }

    /// 驱逐旧条目
    async fn evict_old_entries(&self, cache: &mut HashMap<String, InferenceStateEntry>) {
        let now = Instant::now();
        let max_age = self.config.max_entry_age;

        // 收集需要删除的键
        let keys_to_remove: Vec<String> = cache
            .iter()
            .filter(|(_, entry)| now.duration_since(entry.last_accessed) > max_age)
            .map(|(key, _)| key.clone())
            .collect();

        // 删除过期条目
        for key in &keys_to_remove {
            cache.remove(key);
        }

        // 如果还是太多，使用LRU策略删除
        if cache.len() >= self.config.max_cache_entries {
            let mut queue = self.access_queue.lock().unwrap();
            while cache.len() >= self.config.max_cache_entries && !queue.is_empty() {
                if let Some(key) = queue.pop_front() {
                    cache.remove(&key);
                }
            }
        }

        // 更新统计信息
        {
            let mut stats = self.stats.lock().unwrap();
            stats.cache_evictions += keys_to_remove.len();
        }
    }

    /// 启动异步预推理
    #[allow(dead_code)]
    async fn start_async_pre_inference(
        &self,
        _runtime: &mut Box<dyn web_rwkv::runtime::Runtime<Rnn> + Send + Sync>,
        _inference: RnnInput,
        _context_id: &str,
    ) {
        // TODO: 实现异步预推理逻辑
        // 这里可以基于历史模式预测可能需要的下一个状态
        // 并在后台预先计算logits
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> InferenceStateStats {
        self.stats.lock().unwrap().clone()
    }

    /// 重置统计信息
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = InferenceStateStats::default();
    }

    /// 清理所有缓存
    pub async fn clear_cache(&self) {
        let mut cache = self.state_cache.write().await;
        cache.clear();

        let mut queue = self.access_queue.lock().unwrap();
        queue.clear();
    }

    /// 获取缓存命中率
    pub fn get_cache_hit_rate(&self) -> f64 {
        let stats = self.stats.lock().unwrap();
        if stats.total_requests == 0 {
            0.0
        } else {
            stats.cache_hits as f64 / stats.total_requests as f64
        }
    }

    /// 获取推理调用节省率
    pub fn get_inference_savings_rate(&self) -> f64 {
        let stats = self.stats.lock().unwrap();
        if stats.total_requests == 0 {
            0.0
        } else {
            (stats.cache_hits + stats.batch_inference_savings) as f64
                / (stats.total_requests + stats.batch_inference_savings) as f64
        }
    }
}

/// 全局推理状态管理器实例
static GLOBAL_INFERENCE_STATE_MANAGER: std::sync::OnceLock<InferenceStateManager> =
    std::sync::OnceLock::new();

/// 获取全局推理状态管理器
pub fn global_inference_state_manager() -> &'static InferenceStateManager {
    GLOBAL_INFERENCE_STATE_MANAGER
        .get_or_init(|| InferenceStateManager::new(InferenceStateConfig::default()))
}

/// 初始化全局推理状态管理器
pub fn init_global_inference_state_manager(config: InferenceStateConfig) {
    GLOBAL_INFERENCE_STATE_MANAGER
        .set(InferenceStateManager::new(config))
        .map_err(|_| "Global inference state manager already initialized")
        .ok();
}
