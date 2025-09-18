//! LogitsCache - 智能缓存机制减少runtime.infer()调用频率
//!
//! 通过缓存logits结果和智能预测，显著减少昂贵的推理调用

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// 缓存键，基于上下文状态和输入token序列
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// 输入token序列的哈希值
    pub token_sequence_hash: u64,
    /// 上下文长度
    pub context_length: usize,
    /// 模型状态的简化哈希（如果可获取）
    pub state_hash: Option<u64>,
}

impl CacheKey {
    /// 从token序列创建缓存键
    pub fn from_tokens(tokens: &[u32], context_length: usize) -> Self {
        let mut hasher = DefaultHasher::new();
        tokens.hash(&mut hasher);
        let token_sequence_hash = hasher.finish();

        Self {
            token_sequence_hash,
            context_length,
            state_hash: None,
        }
    }

    /// 设置状态哈希
    pub fn with_state_hash(mut self, state_hash: u64) -> Self {
        self.state_hash = Some(state_hash);
        self
    }
}

/// 缓存条目
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// 缓存的logits
    pub logits: Vec<f32>,
    /// 创建时间
    pub created_at: Instant,
    /// 访问次数
    pub access_count: u32,
    /// 最后访问时间
    pub last_accessed: Instant,
    /// 缓存权重（基于访问频率和时间）
    pub weight: f32,
}

impl CacheEntry {
    /// 创建新的缓存条目
    pub fn new(logits: Vec<f32>) -> Self {
        let now = Instant::now();
        Self {
            logits,
            created_at: now,
            access_count: 1,
            last_accessed: now,
            weight: 1.0,
        }
    }

    /// 更新访问信息
    pub fn update_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = Instant::now();

        // 更新权重：结合访问频率和时间衰减
        let time_factor = 1.0
            / (1.0
                + self
                    .last_accessed
                    .duration_since(self.created_at)
                    .as_secs_f32()
                    / 3600.0);
        self.weight = (self.access_count as f32) * time_factor;
    }

    /// 检查是否过期
    pub fn is_expired(&self, max_age: Duration) -> bool {
        self.last_accessed.elapsed() > max_age
    }
}

/// LogitsCache配置
#[derive(Debug, Clone)]
pub struct LogitsCacheConfig {
    /// 最大缓存条目数
    pub max_entries: usize,
    /// 缓存条目最大存活时间
    pub max_age: Duration,
    /// 启用智能预取
    pub enable_prefetch: bool,
    /// 预取窗口大小
    pub prefetch_window: usize,
    /// 缓存命中率阈值（低于此值时调整策略）
    pub hit_rate_threshold: f32,
}

impl Default for LogitsCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            max_age: Duration::from_secs(300), // 5分钟
            enable_prefetch: true,
            prefetch_window: 3,
            hit_rate_threshold: 0.6,
        }
    }
}

/// LogitsCache统计信息
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    /// 总查询次数
    pub total_queries: u64,
    /// 缓存命中次数
    pub cache_hits: u64,
    /// 缓存未命中次数
    pub cache_misses: u64,
    /// 缓存驱逐次数
    pub evictions: u64,
    /// 预取命中次数
    pub prefetch_hits: u64,
    /// 平均查询时间（微秒）
    pub avg_query_time_us: f64,
}

impl CacheStats {
    /// 计算缓存命中率
    pub fn hit_rate(&self) -> f32 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.cache_hits as f32 / self.total_queries as f32
        }
    }

    /// 重置统计信息
    pub fn reset(&mut self) {
        *self = Default::default();
    }
}

/// LogitsCache - 智能缓存机制
pub struct LogitsCache {
    /// 缓存存储
    cache: Arc<RwLock<HashMap<CacheKey, CacheEntry>>>,
    /// 配置
    config: LogitsCacheConfig,
    /// 统计信息
    stats: Arc<RwLock<CacheStats>>,
    /// 预取队列
    prefetch_queue: Arc<RwLock<Vec<CacheKey>>>,
}

impl Default for LogitsCache {
    fn default() -> Self {
        Self::new(LogitsCacheConfig::default())
    }
}

impl LogitsCache {
    /// 创建新的缓存实例
    pub fn new(config: LogitsCacheConfig) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(RwLock::new(CacheStats::default())),
            prefetch_queue: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// 查询缓存
    pub fn get(&self, key: &CacheKey) -> Option<Vec<f32>> {
        let start_time = Instant::now();

        // 更新统计信息
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_queries += 1;
        }

        let result = {
            let mut cache = self.cache.write().unwrap();
            if let Some(entry) = cache.get_mut(key) {
                entry.update_access();
                Some(entry.logits.clone())
            } else {
                None
            }
        };

        // 更新统计信息
        {
            let mut stats = self.stats.write().unwrap();
            let query_time = start_time.elapsed().as_micros() as f64;
            stats.avg_query_time_us = (stats.avg_query_time_us * (stats.total_queries - 1) as f64
                + query_time)
                / stats.total_queries as f64;

            if result.is_some() {
                stats.cache_hits += 1;
            } else {
                stats.cache_misses += 1;
            }
        }

        result
    }

    /// 插入缓存
    pub fn insert(&self, key: CacheKey, logits: Vec<f32>) {
        let mut cache = self.cache.write().unwrap();

        // 检查是否需要清理过期条目
        if cache.len() >= self.config.max_entries {
            self.evict_entries(&mut cache);
        }

        // 插入新条目
        cache.insert(key.clone(), CacheEntry::new(logits));

        // 智能预取
        if self.config.enable_prefetch {
            self.schedule_prefetch(key);
        }
    }

    /// 清理过期和低权重条目
    fn evict_entries(&self, cache: &mut HashMap<CacheKey, CacheEntry>) {
        #[allow(unused_variables)]
        let now = Instant::now();
        let mut to_remove = Vec::new();

        // 收集过期条目
        for (key, entry) in cache.iter() {
            if entry.is_expired(self.config.max_age) {
                to_remove.push(key.clone());
            }
        }

        // 如果过期条目不够，按权重排序移除低权重条目
        let target_remove_count = if cache.len() >= self.config.max_entries {
            // 当缓存满时，至少移除一半条目为新条目腾出空间
            (cache.len() / 2).max(1)
        } else {
            (cache.len() / 4).max(1)
        };

        if to_remove.len() < target_remove_count {
            let mut entries: Vec<_> = cache.iter().collect();
            entries.sort_by(|a, b| {
                a.1.weight
                    .partial_cmp(&b.1.weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let additional_remove_count = target_remove_count - to_remove.len();
            for (key, _) in entries.iter().take(additional_remove_count) {
                if !to_remove.contains(key) {
                    to_remove.push((*key).clone());
                }
            }
        }

        // 执行移除
        let mut eviction_count = 0;
        for key in to_remove {
            cache.remove(&key);
            eviction_count += 1;
        }

        // 更新统计信息
        {
            let mut stats = self.stats.write().unwrap();
            stats.evictions += eviction_count;
        }
    }

    /// 调度预取
    fn schedule_prefetch(&self, key: CacheKey) {
        if let Ok(mut prefetch_queue) = self.prefetch_queue.write() {
            // 简单的预取策略：基于token序列预测下一个可能的键
            for i in 1..=self.config.prefetch_window {
                let mut prefetch_key = key.clone();
                prefetch_key.context_length += i;

                if !prefetch_queue.contains(&prefetch_key) {
                    prefetch_queue.push(prefetch_key);
                }
            }

            // 限制预取队列大小
            if prefetch_queue.len() > self.config.max_entries / 10 {
                prefetch_queue.truncate(self.config.max_entries / 10);
            }
        }
    }

    /// 获取预取队列中的下一个键
    pub fn get_next_prefetch_key(&self) -> Option<CacheKey> {
        self.prefetch_queue.write().unwrap().pop()
    }

    /// 清空缓存
    pub fn clear(&self) {
        self.cache.write().unwrap().clear();
        self.prefetch_queue.write().unwrap().clear();
        self.stats.write().unwrap().reset();
    }

    /// 获取缓存统计信息
    pub fn get_stats(&self) -> CacheStats {
        self.stats.read().unwrap().clone()
    }

    /// 获取缓存大小
    pub fn size(&self) -> usize {
        self.cache.read().unwrap().len()
    }

    /// 检查缓存健康状态
    pub fn health_check(&self) -> bool {
        let stats = self.get_stats();
        stats.hit_rate() >= self.config.hit_rate_threshold
    }

    /// 优化缓存配置（基于运行时统计）
    pub fn optimize_config(&mut self) {
        let stats = self.get_stats();

        // 如果命中率低，增加缓存大小
        if stats.hit_rate() < self.config.hit_rate_threshold {
            self.config.max_entries = (self.config.max_entries as f64 * 1.2) as usize;
            self.config.max_age =
                Duration::from_secs((self.config.max_age.as_secs() as f64 * 1.1) as u64);
        }

        // 如果命中率很高，可以适当减少缓存大小以节省内存
        if stats.hit_rate() > 0.9 {
            self.config.max_entries = (self.config.max_entries as f64 * 0.9) as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic_operations() {
        let cache = LogitsCache::default();
        let key = CacheKey::from_tokens(&[1, 2, 3], 10);
        let logits = vec![0.1, 0.2, 0.3, 0.4];

        // 测试插入和查询
        assert!(cache.get(&key).is_none());
        cache.insert(key.clone(), logits.clone());
        assert_eq!(cache.get(&key), Some(logits));

        // 测试统计信息
        let stats = cache.get_stats();
        assert_eq!(stats.total_queries, 2);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
    }

    #[test]
    fn test_cache_eviction() {
        let config = LogitsCacheConfig {
            max_entries: 2,
            max_age: Duration::from_millis(1),
            ..Default::default()
        };
        let cache = LogitsCache::new(config);

        // 插入超过最大条目数的数据
        for i in 0..5 {
            let key = CacheKey::from_tokens(&[i], 10);
            cache.insert(key, vec![i as f32]);
        }

        // 验证缓存大小被限制
        assert!(cache.size() <= 2);
    }

    #[test]
    fn test_cache_key_generation() {
        let key1 = CacheKey::from_tokens(&[1, 2, 3], 10);
        let key2 = CacheKey::from_tokens(&[1, 2, 3], 10);
        let key3 = CacheKey::from_tokens(&[1, 2, 4], 10);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }
}
