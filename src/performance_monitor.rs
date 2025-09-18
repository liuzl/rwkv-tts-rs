//! PerformanceMonitor - 性能监控和指标收集组件
//!
//! 提供全面的性能监控、指标收集和分析功能

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// 性能指标类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    /// 推理延迟
    InferenceLatency,
    /// 采样延迟
    SamplingLatency,
    /// 采样时间
    SamplingTime,
    /// 内存使用量
    MemoryUsage,
    /// 缓存命中率
    CacheHitRate,
    /// 缓存命中次数
    CacheHits,
    /// 快速路径命中次数
    FastPathHits,
    /// 完整路径回退次数
    FullPathFallbacks,
    /// 吞吐量（tokens/秒）
    Throughput,
    /// GPU利用率
    GpuUtilization,
    /// 批处理大小
    BatchSize,
    /// 队列长度
    QueueLength,
    /// 错误率
    ErrorRate,
    /// 预取命中率
    PrefetchHitRate,
}

impl MetricType {
    /// 获取指标单位
    pub fn unit(&self) -> &'static str {
        match self {
            MetricType::InferenceLatency
            | MetricType::SamplingLatency
            | MetricType::SamplingTime => "ms",
            MetricType::MemoryUsage => "MB",
            MetricType::CacheHitRate
            | MetricType::GpuUtilization
            | MetricType::ErrorRate
            | MetricType::PrefetchHitRate => "%",
            MetricType::Throughput => "tokens/s",
            MetricType::BatchSize
            | MetricType::QueueLength
            | MetricType::CacheHits
            | MetricType::FastPathHits
            | MetricType::FullPathFallbacks => "count",
        }
    }

    /// 获取指标描述
    pub fn description(&self) -> &'static str {
        match self {
            MetricType::InferenceLatency => "推理延迟时间",
            MetricType::SamplingLatency => "采样延迟时间",
            MetricType::SamplingTime => "采样执行时间",
            MetricType::MemoryUsage => "内存使用量",
            MetricType::CacheHitRate => "缓存命中率",
            MetricType::CacheHits => "缓存命中次数",
            MetricType::FastPathHits => "快速路径命中次数",
            MetricType::FullPathFallbacks => "完整路径回退次数",
            MetricType::Throughput => "处理吞吐量",
            MetricType::GpuUtilization => "GPU利用率",
            MetricType::BatchSize => "批处理大小",
            MetricType::QueueLength => "队列长度",
            MetricType::ErrorRate => "错误率",
            MetricType::PrefetchHitRate => "预取命中率",
        }
    }
}

/// 性能指标数据点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    /// 时间戳
    pub timestamp: u64,
    /// 指标值
    pub value: f64,
    /// 标签（可选）
    pub labels: HashMap<String, String>,
}

impl MetricPoint {
    /// 创建新的指标数据点
    pub fn new(value: f64) -> Self {
        Self {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            value,
            labels: HashMap::new(),
        }
    }

    /// 添加标签
    pub fn with_label(mut self, key: String, value: String) -> Self {
        self.labels.insert(key, value);
        self
    }

    /// 添加多个标签
    pub fn with_labels(mut self, labels: HashMap<String, String>) -> Self {
        self.labels.extend(labels);
        self
    }
}

/// 统计摘要
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsSummary {
    /// 样本数量
    pub count: usize,
    /// 平均值
    pub mean: f64,
    /// 最小值
    pub min: f64,
    /// 最大值
    pub max: f64,
    /// 标准差
    pub std_dev: f64,
    /// 百分位数
    pub percentiles: HashMap<String, f64>, // p50, p90, p95, p99
}

impl Default for StatsSummary {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            std_dev: 0.0,
            percentiles: HashMap::new(),
        }
    }
}

impl StatsSummary {
    /// 从数据点计算统计摘要
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        let count = values.len();
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;

        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // 计算标准差
        let variance: f64 = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        // 计算百分位数
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut percentiles = HashMap::new();
        percentiles.insert("p50".to_string(), Self::percentile(&sorted_values, 0.5));
        percentiles.insert("p90".to_string(), Self::percentile(&sorted_values, 0.9));
        percentiles.insert("p95".to_string(), Self::percentile(&sorted_values, 0.95));
        percentiles.insert("p99".to_string(), Self::percentile(&sorted_values, 0.99));

        Self {
            count,
            mean,
            min,
            max,
            std_dev,
            percentiles,
        }
    }

    /// 计算百分位数
    fn percentile(sorted_values: &[f64], p: f64) -> f64 {
        if sorted_values.is_empty() {
            return 0.0;
        }

        let index = (p * (sorted_values.len() - 1) as f64).round() as usize;
        sorted_values[index.min(sorted_values.len() - 1)]
    }
}

/// 性能监控配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// 是否启用监控
    pub enabled: bool,
    /// 指标保留时间（秒）
    pub retention_seconds: u64,
    /// 最大数据点数量
    pub max_data_points: usize,
    /// 采样间隔（毫秒）
    pub sampling_interval_ms: u64,
    /// 启用的指标类型
    pub enabled_metrics: Vec<MetricType>,
    /// 是否启用详细日志
    pub verbose_logging: bool,
    /// 性能报告间隔（秒）
    pub report_interval_seconds: u64,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            retention_seconds: 3600, // 1小时
            max_data_points: 10000,
            sampling_interval_ms: 100,
            enabled_metrics: vec![
                MetricType::InferenceLatency,
                MetricType::SamplingLatency,
                MetricType::MemoryUsage,
                MetricType::CacheHitRate,
                MetricType::Throughput,
            ],
            verbose_logging: false,
            report_interval_seconds: 300, // 5分钟
        }
    }
}

/// 性能计数器
#[derive(Debug, Default)]
struct PerformanceCounters {
    /// 总推理次数
    total_inferences: AtomicU64,
    /// 总采样次数
    total_samples: AtomicU64,
    /// 缓存命中次数
    cache_hits: AtomicU64,
    /// 缓存未命中次数
    cache_misses: AtomicU64,
    /// 错误次数
    error_count: AtomicU64,
    /// 预取命中次数
    prefetch_hits: AtomicU64,
    /// 预取未命中次数
    prefetch_misses: AtomicU64,
    /// 当前队列长度
    current_queue_length: AtomicUsize,
}

impl PerformanceCounters {
    /// 重置所有计数器
    fn reset(&self) {
        self.total_inferences.store(0, Ordering::Relaxed);
        self.total_samples.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.error_count.store(0, Ordering::Relaxed);
        self.prefetch_hits.store(0, Ordering::Relaxed);
        self.prefetch_misses.store(0, Ordering::Relaxed);
        self.current_queue_length.store(0, Ordering::Relaxed);
    }
}

/// PerformanceMonitor - 性能监控器
pub struct PerformanceMonitor {
    /// 配置
    config: MonitorConfig,
    /// 指标数据存储
    metrics: Arc<RwLock<HashMap<MetricType, VecDeque<MetricPoint>>>>,
    /// 性能计数器
    counters: PerformanceCounters,
    /// 开始时间
    start_time: Instant,
    /// 最后报告时间
    last_report_time: Arc<Mutex<Instant>>,
    /// 是否正在运行
    is_running: Arc<std::sync::atomic::AtomicBool>,
}

impl PerformanceMonitor {
    /// 创建新的性能监控器
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(HashMap::new())),
            counters: PerformanceCounters::default(),
            start_time: Instant::now(),
            last_report_time: Arc::new(Mutex::new(Instant::now())),
            is_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// 启动监控
    pub fn start(&self) {
        if !self.config.enabled {
            return;
        }

        self.is_running.store(true, Ordering::Relaxed);

        // 启动清理任务
        self.start_cleanup_task();

        // 启动报告任务
        if self.config.report_interval_seconds > 0 {
            self.start_report_task();
        }
    }

    /// 停止监控
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
    }

    /// 记录指标
    pub fn record_metric(&self, metric_type: MetricType, value: f64) {
        self.record_metric_with_labels(metric_type, value, HashMap::new());
    }

    /// 记录带标签的指标
    pub fn record_metric_with_labels(
        &self,
        metric_type: MetricType,
        value: f64,
        labels: HashMap<String, String>,
    ) {
        if !self.config.enabled || !self.config.enabled_metrics.contains(&metric_type) {
            return;
        }

        let point = MetricPoint::new(value).with_labels(labels);

        if let Ok(mut metrics) = self.metrics.write() {
            let data_points = metrics.entry(metric_type).or_insert_with(VecDeque::new);
            data_points.push_back(point);

            // 限制数据点数量
            if data_points.len() > self.config.max_data_points {
                data_points.pop_front();
            }
        }

        if self.config.verbose_logging {
            println!(
                "Recorded metric {:?}: {} {}",
                metric_type,
                value,
                metric_type.unit()
            );
        }
    }

    /// 记录推理延迟
    pub fn record_inference_latency(&self, duration: Duration) {
        self.counters
            .total_inferences
            .fetch_add(1, Ordering::Relaxed);
        self.record_metric(MetricType::InferenceLatency, duration.as_millis() as f64);
    }

    /// 记录采样延迟
    pub fn record_sampling_latency(&self, duration: Duration) {
        self.counters.total_samples.fetch_add(1, Ordering::Relaxed);
        self.record_metric(MetricType::SamplingLatency, duration.as_millis() as f64);
    }

    /// 记录缓存命中
    pub fn record_cache_hit(&self) {
        self.counters.cache_hits.fetch_add(1, Ordering::Relaxed);
        self.update_cache_hit_rate();
    }

    /// 记录缓存未命中
    pub fn record_cache_miss(&self) {
        self.counters.cache_misses.fetch_add(1, Ordering::Relaxed);
        self.update_cache_hit_rate();
    }

    /// 记录预取命中
    pub fn record_prefetch_hit(&self) {
        self.counters.prefetch_hits.fetch_add(1, Ordering::Relaxed);
        self.update_prefetch_hit_rate();
    }

    /// 记录预取未命中
    pub fn record_prefetch_miss(&self) {
        self.counters
            .prefetch_misses
            .fetch_add(1, Ordering::Relaxed);
        self.update_prefetch_hit_rate();
    }

    /// 记录错误
    pub fn record_error(&self) {
        self.counters.error_count.fetch_add(1, Ordering::Relaxed);
        self.update_error_rate();
    }

    /// 更新队列长度
    pub fn update_queue_length(&self, length: usize) {
        self.counters
            .current_queue_length
            .store(length, Ordering::Relaxed);
        self.record_metric(MetricType::QueueLength, length as f64);
    }

    /// 记录内存使用量（MB）
    pub fn record_memory_usage(&self, memory_mb: f64) {
        self.record_metric(MetricType::MemoryUsage, memory_mb);
    }

    /// 记录吞吐量
    pub fn record_throughput(&self, tokens_per_second: f64) {
        self.record_metric(MetricType::Throughput, tokens_per_second);
    }

    /// 记录批处理大小
    pub fn record_batch_size(&self, size: usize) {
        self.record_metric(MetricType::BatchSize, size as f64);
    }

    /// 更新缓存命中率
    fn update_cache_hit_rate(&self) {
        let hits = self.counters.cache_hits.load(Ordering::Relaxed);
        let misses = self.counters.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;

        if total > 0 {
            let hit_rate = (hits as f64 / total as f64) * 100.0;
            self.record_metric(MetricType::CacheHitRate, hit_rate);
        }
    }

    /// 更新预取命中率
    fn update_prefetch_hit_rate(&self) {
        let hits = self.counters.prefetch_hits.load(Ordering::Relaxed);
        let misses = self.counters.prefetch_misses.load(Ordering::Relaxed);
        let total = hits + misses;

        if total > 0 {
            let hit_rate = (hits as f64 / total as f64) * 100.0;
            self.record_metric(MetricType::PrefetchHitRate, hit_rate);
        }
    }

    /// 更新错误率
    fn update_error_rate(&self) {
        let errors = self.counters.error_count.load(Ordering::Relaxed);
        let total_ops = self.counters.total_inferences.load(Ordering::Relaxed)
            + self.counters.total_samples.load(Ordering::Relaxed);

        if total_ops > 0 {
            let error_rate = (errors as f64 / total_ops as f64) * 100.0;
            self.record_metric(MetricType::ErrorRate, error_rate);
        }
    }

    /// 获取指标统计摘要
    pub fn get_metric_summary(&self, metric_type: MetricType) -> Option<StatsSummary> {
        if let Ok(metrics) = self.metrics.read() {
            if let Some(data_points) = metrics.get(&metric_type) {
                let values: Vec<f64> = data_points.iter().map(|p| p.value).collect();
                return Some(StatsSummary::from_values(&values));
            }
        }
        None
    }

    /// 获取所有指标的统计摘要
    pub fn get_all_summaries(&self) -> HashMap<MetricType, StatsSummary> {
        let mut summaries = HashMap::new();

        if let Ok(metrics) = self.metrics.read() {
            for (&metric_type, data_points) in metrics.iter() {
                let values: Vec<f64> = data_points.iter().map(|p| p.value).collect();
                summaries.insert(metric_type, StatsSummary::from_values(&values));
            }
        }

        summaries
    }

    /// 获取实时性能统计
    pub fn get_realtime_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        stats.insert(
            "total_inferences".to_string(),
            self.counters.total_inferences.load(Ordering::Relaxed) as f64,
        );
        stats.insert(
            "total_samples".to_string(),
            self.counters.total_samples.load(Ordering::Relaxed) as f64,
        );
        stats.insert(
            "cache_hits".to_string(),
            self.counters.cache_hits.load(Ordering::Relaxed) as f64,
        );
        stats.insert(
            "cache_misses".to_string(),
            self.counters.cache_misses.load(Ordering::Relaxed) as f64,
        );
        stats.insert(
            "error_count".to_string(),
            self.counters.error_count.load(Ordering::Relaxed) as f64,
        );
        stats.insert(
            "prefetch_hits".to_string(),
            self.counters.prefetch_hits.load(Ordering::Relaxed) as f64,
        );
        stats.insert(
            "prefetch_misses".to_string(),
            self.counters.prefetch_misses.load(Ordering::Relaxed) as f64,
        );
        stats.insert(
            "current_queue_length".to_string(),
            self.counters.current_queue_length.load(Ordering::Relaxed) as f64,
        );
        stats.insert(
            "uptime_seconds".to_string(),
            self.start_time.elapsed().as_secs() as f64,
        );

        // 计算派生指标
        let hits = self.counters.cache_hits.load(Ordering::Relaxed);
        let misses = self.counters.cache_misses.load(Ordering::Relaxed);
        let total_cache_ops = hits + misses;
        if total_cache_ops > 0 {
            stats.insert(
                "cache_hit_rate".to_string(),
                (hits as f64 / total_cache_ops as f64) * 100.0,
            );
        }

        let prefetch_hits = self.counters.prefetch_hits.load(Ordering::Relaxed);
        let prefetch_misses = self.counters.prefetch_misses.load(Ordering::Relaxed);
        let total_prefetch_ops = prefetch_hits + prefetch_misses;
        if total_prefetch_ops > 0 {
            stats.insert(
                "prefetch_hit_rate".to_string(),
                (prefetch_hits as f64 / total_prefetch_ops as f64) * 100.0,
            );
        }

        stats
    }

    /// 重置所有统计信息
    pub fn reset(&self) {
        self.counters.reset();

        if let Ok(mut metrics) = self.metrics.write() {
            metrics.clear();
        }
    }

    /// 生成性能报告
    pub fn generate_report(&self) -> String {
        let summaries = self.get_all_summaries();
        let realtime_stats = self.get_realtime_stats();

        let mut report = String::new();
        report.push_str("=== 性能监控报告 ===\n");
        report.push_str(&format!(
            "运行时间: {:.2} 秒\n",
            self.start_time.elapsed().as_secs_f64()
        ));
        report.push_str("\n--- 实时统计 ---\n");

        for (key, value) in &realtime_stats {
            report.push_str(&format!("{}: {:.2}\n", key, value));
        }

        report.push_str("\n--- 指标摘要 ---\n");
        for (metric_type, summary) in &summaries {
            report.push_str(&format!("{}:\n", metric_type.description()));
            report.push_str(&format!("  样本数: {}\n", summary.count));
            report.push_str(&format!(
                "  平均值: {:.2} {}\n",
                summary.mean,
                metric_type.unit()
            ));
            report.push_str(&format!(
                "  最小值: {:.2} {}\n",
                summary.min,
                metric_type.unit()
            ));
            report.push_str(&format!(
                "  最大值: {:.2} {}\n",
                summary.max,
                metric_type.unit()
            ));
            report.push_str(&format!("  标准差: {:.2}\n", summary.std_dev));

            if let Some(p50) = summary.percentiles.get("p50") {
                report.push_str(&format!("  P50: {:.2} {}\n", p50, metric_type.unit()));
            }
            if let Some(p95) = summary.percentiles.get("p95") {
                report.push_str(&format!("  P95: {:.2} {}\n", p95, metric_type.unit()));
            }
            if let Some(p99) = summary.percentiles.get("p99") {
                report.push_str(&format!("  P99: {:.2} {}\n", p99, metric_type.unit()));
            }
            report.push('\n');
        }

        report
    }

    /// 启动清理任务
    fn start_cleanup_task(&self) {
        let metrics = Arc::clone(&self.metrics);
        let retention_seconds = self.config.retention_seconds;
        let is_running = Arc::clone(&self.is_running);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60)); // 每分钟清理一次

            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;

                let cutoff_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64
                    - (retention_seconds * 1000);

                if let Ok(mut metrics) = metrics.write() {
                    for data_points in metrics.values_mut() {
                        while let Some(point) = data_points.front() {
                            if point.timestamp < cutoff_time {
                                data_points.pop_front();
                            } else {
                                break;
                            }
                        }
                    }
                }
            }
        });
    }

    /// 启动报告任务
    fn start_report_task(&self) {
        let last_report_time = Arc::clone(&self.last_report_time);
        let report_interval = Duration::from_secs(self.config.report_interval_seconds);
        let is_running = Arc::clone(&self.is_running);
        let _metrics = Arc::clone(&self.metrics);
        let _counters_ptr = &self.counters as *const PerformanceCounters;
        let start_time = self.start_time;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(report_interval);

            while is_running.load(Ordering::Relaxed) {
                interval.tick().await;

                // 生成简化的报告
                let uptime = start_time.elapsed().as_secs_f64();
                let report = format!("Performance Report:\nUptime: {:.2} seconds\n", uptime);
                println!("{}", report);

                if let Ok(mut last_time) = last_report_time.lock() {
                    *last_time = Instant::now();
                }
            }
        });
    }

    /// 获取配置
    pub fn get_config(&self) -> &MonitorConfig {
        &self.config
    }

    /// 更新配置
    pub fn update_config(&mut self, config: MonitorConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_point_creation() {
        let point =
            MetricPoint::new(42.0).with_label("component".to_string(), "sampler".to_string());

        assert_eq!(point.value, 42.0);
        assert_eq!(point.labels.get("component"), Some(&"sampler".to_string()));
    }

    #[test]
    fn test_stats_summary() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let summary = StatsSummary::from_values(&values);

        assert_eq!(summary.count, 5);
        assert_eq!(summary.mean, 3.0);
        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.max, 5.0);
    }

    #[test]
    fn test_performance_monitor() {
        let config = MonitorConfig::default();
        let monitor = PerformanceMonitor::new(config);

        monitor.record_metric(MetricType::InferenceLatency, 100.0);
        monitor.record_cache_hit();
        monitor.record_cache_miss();

        let stats = monitor.get_realtime_stats();
        assert_eq!(stats.get("cache_hits"), Some(&1.0));
        assert_eq!(stats.get("cache_misses"), Some(&1.0));
        assert_eq!(stats.get("cache_hit_rate"), Some(&50.0));
    }

    #[test]
    fn test_metric_types() {
        assert_eq!(MetricType::InferenceLatency.unit(), "ms");
        assert_eq!(MetricType::MemoryUsage.unit(), "MB");
        assert_eq!(MetricType::CacheHitRate.unit(), "%");
        assert_eq!(MetricType::Throughput.unit(), "tokens/s");
    }
}
