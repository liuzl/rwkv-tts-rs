//! StreamingInference - 流式推理优化组件
//!
//! 实现批量预取、流水线处理和智能调度，优化推理数据流

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;

/// 推理请求
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// 请求ID
    pub id: String,
    /// 输入token序列
    pub tokens: Vec<u32>,
    /// 上下文长度
    pub context_length: usize,
    /// 优先级（数值越大优先级越高）
    pub priority: u8,
    /// 创建时间
    pub created_at: Instant,
    /// 是否需要缓存结果
    pub cache_result: bool,
}

impl InferenceRequest {
    /// 创建新的推理请求
    pub fn new(id: String, tokens: Vec<u32>, context_length: usize) -> Self {
        Self {
            id,
            tokens,
            context_length,
            priority: 5, // 默认中等优先级
            created_at: Instant::now(),
            cache_result: true,
        }
    }

    /// 设置优先级
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// 设置是否缓存结果
    pub fn with_cache(mut self, cache_result: bool) -> Self {
        self.cache_result = cache_result;
        self
    }
}

/// 推理结果
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// 请求ID
    pub request_id: String,
    /// 推理结果logits
    pub logits: Vec<f32>,
    /// 推理耗时
    pub inference_time: Duration,
    /// 是否来自缓存
    pub from_cache: bool,
    /// 完成时间
    pub completed_at: Instant,
}

/// 批处理配置
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// 最大批大小
    pub max_batch_size: usize,
    /// 批处理超时时间
    pub batch_timeout: Duration,
    /// 启用动态批大小调整
    pub dynamic_batching: bool,
    /// 最小批大小（动态调整时）
    pub min_batch_size: usize,
    /// 预取窗口大小
    pub prefetch_window: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            batch_timeout: Duration::from_millis(50),
            dynamic_batching: true,
            min_batch_size: 2,
            prefetch_window: 4,
        }
    }
}

/// 流式推理统计信息
#[derive(Debug, Default, Clone)]
pub struct StreamingStats {
    /// 总请求数
    pub total_requests: u64,
    /// 批处理次数
    pub total_batches: u64,
    /// 缓存命中次数
    pub cache_hits: u64,
    /// 平均批大小
    pub avg_batch_size: f32,
    /// 平均推理时间（毫秒）
    pub avg_inference_time_ms: f32,
    /// 平均等待时间（毫秒）
    pub avg_wait_time_ms: f32,
    /// 预取命中次数
    pub prefetch_hits: u64,
    /// 流水线效率（0-1）
    pub pipeline_efficiency: f32,
}

impl StreamingStats {
    /// 计算缓存命中率
    pub fn cache_hit_rate(&self) -> f32 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.cache_hits as f32 / self.total_requests as f32
        }
    }

    /// 重置统计信息
    pub fn reset(&mut self) {
        *self = Default::default();
    }
}

/// 推理任务
struct InferenceTask {
    request: InferenceRequest,
    response_tx: oneshot::Sender<InferenceResult>,
}

/// StreamingInference - 流式推理管理器
pub struct StreamingInference {
    /// 配置
    config: BatchConfig,

    /// 请求队列
    request_queue: Arc<Mutex<VecDeque<InferenceTask>>>,
    /// 优先级队列
    priority_queues: Arc<Mutex<HashMap<u8, VecDeque<InferenceTask>>>>,
    /// 统计信息
    stats: Arc<RwLock<StreamingStats>>,
    /// 运行状态
    is_running: Arc<AtomicBool>,
    /// 活跃批处理数
    active_batches: Arc<AtomicUsize>,
    /// 预取队列
    prefetch_queue: Arc<Mutex<VecDeque<String>>>,
    /// 任务发送器
    task_tx: Option<mpsc::UnboundedSender<InferenceTask>>,
}

impl StreamingInference {
    /// 创建新的StreamingInference
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            request_queue: Arc::new(Mutex::new(VecDeque::new())),
            priority_queues: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(RwLock::new(StreamingStats::default())),
            is_running: Arc::new(AtomicBool::new(false)),
            active_batches: Arc::new(AtomicUsize::new(0)),
            prefetch_queue: Arc::new(Mutex::new(VecDeque::new())),
            task_tx: None,
        }
    }

    /// 启动流式推理服务
    pub async fn start(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.is_running.load(Ordering::Relaxed) {
            return Ok(());
        }

        self.is_running.store(true, Ordering::Relaxed);

        let (task_tx, mut task_rx) = mpsc::unbounded_channel::<InferenceTask>();
        self.task_tx = Some(task_tx);

        // 启动批处理调度器
        let _scheduler_handle = self.start_batch_scheduler().await;

        // 启动任务处理器
        let _processor_handle = {
            let _request_queue = Arc::clone(&self.request_queue);
            let priority_queues = Arc::clone(&self.priority_queues);
            let is_running = Arc::clone(&self.is_running);

            tokio::spawn(async move {
                while let Some(task) = task_rx.recv().await {
                    if !is_running.load(Ordering::Relaxed) {
                        break;
                    }

                    // 根据优先级分发任务
                    let priority = task.request.priority;
                    if let Ok(mut queues) = priority_queues.lock() {
                        queues
                            .entry(priority)
                            .or_insert_with(VecDeque::new)
                            .push_back(task);
                    }
                }
            })
        };

        Ok(())
    }

    /// 停止流式推理服务
    pub async fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);

        // 等待所有活跃批处理完成
        while self.active_batches.load(Ordering::Relaxed) > 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// 提交推理请求
    pub async fn submit_request(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>> {
        if !self.is_running.load(Ordering::Relaxed) {
            return Err("StreamingInference is not running".into());
        }

        // 创建响应通道
        let (response_tx, response_rx) = oneshot::channel();

        // 创建任务
        let task = InferenceTask {
            request,
            response_tx,
        };

        // 发送任务
        if let Some(ref task_tx) = self.task_tx {
            task_tx.send(task).map_err(|_| "Failed to send task")?;
        } else {
            return Err("Task sender not initialized".into());
        }

        // 等待结果
        let result = timeout(Duration::from_secs(30), response_rx)
            .await
            .map_err(|_| "Request timeout")?
            .map_err(|_| "Failed to receive response")?;

        // 更新统计信息
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_requests += 1;
        }

        Ok(result)
    }

    /// 启动批处理调度器
    async fn start_batch_scheduler(&self) -> tokio::task::JoinHandle<()> {
        let priority_queues = Arc::clone(&self.priority_queues);
        let stats = Arc::clone(&self.stats);
        let is_running = Arc::clone(&self.is_running);
        let active_batches = Arc::clone(&self.active_batches);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut batch_timer = tokio::time::interval(config.batch_timeout);

            while is_running.load(Ordering::Relaxed) {
                batch_timer.tick().await;

                // 收集批处理任务
                let batch = Self::collect_batch(&priority_queues, &config).await;

                if !batch.is_empty() {
                    active_batches.fetch_add(1, Ordering::Relaxed);

                    // 处理批处理
                    let stats_clone = Arc::clone(&stats);
                    let active_batches_clone = Arc::clone(&active_batches);

                    tokio::spawn(async move {
                        Self::process_batch(batch, stats_clone).await;
                        active_batches_clone.fetch_sub(1, Ordering::Relaxed);
                    });
                }
            }
        })
    }

    /// 收集批处理任务
    async fn collect_batch(
        priority_queues: &Arc<Mutex<HashMap<u8, VecDeque<InferenceTask>>>>,
        config: &BatchConfig,
    ) -> Vec<InferenceTask> {
        let mut batch = Vec::new();

        if let Ok(mut queues) = priority_queues.lock() {
            // 按优先级从高到低处理
            let mut priorities: Vec<u8> = queues.keys().cloned().collect();
            priorities.sort_by(|a, b| b.cmp(a)); // 降序排列

            for priority in priorities {
                if batch.len() >= config.max_batch_size {
                    break;
                }

                if let Some(queue) = queues.get_mut(&priority) {
                    while batch.len() < config.max_batch_size && !queue.is_empty() {
                        if let Some(task) = queue.pop_front() {
                            batch.push(task);
                        }
                    }
                }
            }
        }

        batch
    }

    /// 处理批处理
    async fn process_batch(batch: Vec<InferenceTask>, stats: Arc<RwLock<StreamingStats>>) {
        let batch_start = Instant::now();
        let batch_size = batch.len();

        // 模拟批处理推理（实际实现中需要调用真实的推理引擎）
        for task in batch {
            let inference_start = Instant::now();

            // 模拟推理计算
            tokio::time::sleep(Duration::from_millis(10)).await;

            // 生成模拟logits
            let vocab_size = 50257; // 示例词汇表大小
            let mut logits = Vec::with_capacity(vocab_size);
            for _i in 0..vocab_size {
                logits.push(rand::random::<f32>());
            }

            let inference_time = inference_start.elapsed();

            // 发送结果
            let result = InferenceResult {
                request_id: task.request.id,
                logits,
                inference_time,
                from_cache: false,
                completed_at: Instant::now(),
            };

            let _ = task.response_tx.send(result);
        }

        // 更新统计信息
        {
            let mut stats = stats.write().unwrap();
            stats.total_batches += 1;
            stats.avg_batch_size = (stats.avg_batch_size * (stats.total_batches - 1) as f32
                + batch_size as f32)
                / stats.total_batches as f32;

            let batch_time_ms = batch_start.elapsed().as_millis() as f32;
            stats.avg_inference_time_ms =
                (stats.avg_inference_time_ms * (stats.total_batches - 1) as f32 + batch_time_ms)
                    / stats.total_batches as f32;
        }
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> StreamingStats {
        self.stats.read().unwrap().clone()
    }

    /// 重置统计信息
    pub fn reset_stats(&self) {
        self.stats.write().unwrap().reset();
    }

    /// 动态调整批大小
    pub fn adjust_batch_size(&mut self, target_latency_ms: f32) {
        if !self.config.dynamic_batching {
            return;
        }

        let stats = self.get_stats();

        if stats.avg_inference_time_ms > target_latency_ms * 1.2 {
            // 延迟过高，减少批大小
            self.config.max_batch_size =
                (self.config.max_batch_size - 1).max(self.config.min_batch_size);
        } else if stats.avg_inference_time_ms < target_latency_ms * 0.8 {
            // 延迟较低，可以增加批大小
            self.config.max_batch_size = (self.config.max_batch_size + 1).min(16);
        }
    }

    /// 获取当前配置
    pub fn get_config(&self) -> &BatchConfig {
        &self.config
    }

    /// 更新配置
    pub fn update_config(&mut self, config: BatchConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_streaming_inference_basic() {
        let config = BatchConfig::default();
        let mut streaming = StreamingInference::new(config);

        streaming.start().await.unwrap();

        let request = InferenceRequest::new("test_1".to_string(), vec![1, 2, 3, 4], 10);

        let result = streaming.submit_request(request).await;
        assert!(result.is_ok());

        streaming.stop().await;
    }

    #[test]
    fn test_priority_handling() {
        let request_high =
            InferenceRequest::new("high".to_string(), vec![1, 2, 3], 5).with_priority(9);

        let request_low =
            InferenceRequest::new("low".to_string(), vec![4, 5, 6], 5).with_priority(1);

        assert!(request_high.priority > request_low.priority);
    }
}
