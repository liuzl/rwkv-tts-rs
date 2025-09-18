//! VecPool - 高性能对象池优化内存分配
//!
//! 通过复用Vec对象减少内存分配和释放开销，显著提升性能

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};

/// 池监控报告
#[derive(Debug, Clone)]
pub struct PoolMonitoringReport {
    /// 总内存使用量（字节）
    pub total_memory_usage: usize,
    /// 整体命中率
    pub overall_hit_rate: f32,
    /// 平均使用率
    pub average_utilization: f32,
    /// 各池效率评分
    pub efficiency_scores: Vec<(String, f32)>,
    /// 详细统计信息
    pub pool_stats: std::collections::HashMap<String, PoolStats>,
}

/// Vec对象池统计信息
#[derive(Debug, Default, Clone)]
pub struct PoolStats {
    /// 总分配次数
    pub total_allocations: usize,
    /// 池命中次数（复用）
    pub pool_hits: usize,
    /// 池未命中次数（新分配）
    pub pool_misses: usize,
    /// 当前池大小
    pub current_pool_size: usize,
    /// 峰值池大小
    pub peak_pool_size: usize,
    /// 池使用率（当前大小/最大大小）
    pub utilization_rate: f32,
    /// 平均Vec容量
    pub avg_vec_capacity: f32,
    /// 内存使用量估算（字节）
    pub estimated_memory_usage: usize,
}

impl PoolStats {
    /// 计算池命中率
    pub fn hit_rate(&self) -> f32 {
        if self.total_allocations == 0 {
            0.0
        } else {
            self.pool_hits as f32 / self.total_allocations as f32
        }
    }

    /// 计算池效率（综合命中率和使用率）
    pub fn efficiency_score(&self) -> f32 {
        let hit_rate = self.hit_rate();
        let utilization = self.utilization_rate;
        // 平衡命中率和使用率，避免过度缓存
        (hit_rate * 0.7 + utilization * 0.3).min(1.0)
    }

    /// 检查是否需要调整池大小
    pub fn needs_resize(&self) -> Option<bool> {
        if self.total_allocations < 100 {
            return None; // 样本不足
        }

        // 使用率过低且命中率不高，建议缩小
        if self.utilization_rate < 0.3 && self.hit_rate() < 0.5 {
            Some(false) // 建议缩小
        }
        // 使用率很高且命中率高，建议扩大
        else if self.utilization_rate > 0.8 && self.hit_rate() > 0.7 {
            Some(true) // 建议扩大
        } else {
            None // 保持当前大小
        }
    }

    /// 重置统计信息
    pub fn reset(&mut self) {
        *self = Default::default();
    }
}

/// 泛型Vec对象池
pub struct VecPool<T> {
    /// 存储可复用的Vec
    pool: Arc<Mutex<VecDeque<Vec<T>>>>,
    /// 最大池大小
    max_pool_size: usize,
    /// 统计信息
    stats: Arc<Mutex<PoolStats>>,
    /// 当前池大小计数器
    current_size: Arc<AtomicUsize>,
}

impl<T> VecPool<T> {
    /// 创建新的VecPool
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pool: Arc::new(Mutex::new(VecDeque::new())),
            max_pool_size,
            stats: Arc::new(Mutex::new(PoolStats::default())),
            current_size: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// 从池中获取Vec，如果池为空则创建新的
    pub fn get(&self) -> Vec<T> {
        let mut stats = self.stats.lock().unwrap();
        stats.total_allocations += 1;

        if let Ok(mut pool) = self.pool.lock() {
            if let Some(mut vec) = pool.pop_front() {
                vec.clear(); // 清空内容但保留容量
                stats.pool_hits += 1;
                self.current_size.fetch_sub(1, Ordering::Relaxed);
                return vec;
            }
        }

        // 池为空，创建新Vec
        stats.pool_misses += 1;
        Vec::new()
    }

    /// 获取指定容量的Vec
    pub fn get_with_capacity(&self, capacity: usize) -> Vec<T> {
        let mut vec = self.get();
        if vec.capacity() < capacity {
            vec.reserve(capacity - vec.capacity());
        }
        vec
    }

    /// 将Vec归还到池中
    pub fn return_vec(&self, vec: Vec<T>) {
        if vec.capacity() == 0 {
            return; // 不存储零容量的Vec
        }

        let current_size = self.current_size.load(Ordering::Relaxed);
        if current_size < self.max_pool_size {
            if let Ok(mut pool) = self.pool.lock() {
                pool.push_back(vec);
                let new_size = self.current_size.fetch_add(1, Ordering::Relaxed) + 1;

                // 更新峰值大小
                if let Ok(mut stats) = self.stats.lock() {
                    stats.current_pool_size = new_size;
                    if new_size > stats.peak_pool_size {
                        stats.peak_pool_size = new_size;
                    }
                }
            }
        }
        // 如果池已满，Vec会被自动释放
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> PoolStats {
        let stats = self.stats.lock().unwrap();
        let pool = self.pool.lock().unwrap();

        let current_size = pool.len();
        let utilization_rate = if self.max_pool_size > 0 {
            current_size as f32 / self.max_pool_size as f32
        } else {
            0.0
        };

        // 计算平均容量和内存使用量
        let (avg_capacity, memory_usage) = if !pool.is_empty() {
            let total_capacity: usize = pool.iter().map(|v| v.capacity()).sum();
            let avg_cap = total_capacity as f32 / pool.len() as f32;
            let memory = total_capacity * std::mem::size_of::<T>();
            (avg_cap, memory)
        } else {
            (0.0, 0)
        };

        PoolStats {
            total_allocations: stats.total_allocations,
            pool_hits: stats.pool_hits,
            pool_misses: stats.pool_misses,
            current_pool_size: current_size,
            peak_pool_size: stats.peak_pool_size,
            utilization_rate,
            avg_vec_capacity: avg_capacity,
            estimated_memory_usage: memory_usage,
        }
    }

    /// 清空池
    pub fn clear(&self) {
        if let Ok(mut pool) = self.pool.lock() {
            pool.clear();
            self.current_size.store(0, Ordering::Relaxed);
        }
    }

    /// 预热池（预分配指定数量的Vec）
    pub fn warm_up(&self, count: usize, capacity: usize) {
        for _ in 0..count.min(self.max_pool_size) {
            let vec = Vec::with_capacity(capacity);
            self.return_vec(vec);
        }
    }
}

impl<T> Clone for VecPool<T> {
    fn clone(&self) -> Self {
        Self {
            pool: Arc::clone(&self.pool),
            max_pool_size: self.max_pool_size,
            stats: Arc::clone(&self.stats),
            current_size: Arc::clone(&self.current_size),
        }
    }
}

/// RAII包装器，自动归还Vec到池
pub struct PooledVec<T> {
    vec: Option<Vec<T>>,
    pool: VecPool<T>,
}

impl<T> PooledVec<T> {
    /// 创建新的PooledVec
    pub fn new(pool: VecPool<T>) -> Self {
        let vec = pool.get();
        Self {
            vec: Some(vec),
            pool,
        }
    }

    /// 创建指定容量的PooledVec
    pub fn with_capacity(pool: VecPool<T>, capacity: usize) -> Self {
        let vec = pool.get_with_capacity(capacity);
        Self {
            vec: Some(vec),
            pool,
        }
    }

    /// 获取内部Vec的可变引用
    pub fn get_mut(&mut self) -> &mut Vec<T> {
        self.vec.as_mut().unwrap()
    }

    /// 获取内部Vec的不可变引用
    pub fn get_ref(&self) -> &Vec<T> {
        self.vec.as_ref().unwrap()
    }

    /// 提取内部Vec（不会自动归还到池）
    pub fn into_inner(mut self) -> Vec<T> {
        self.vec.take().unwrap()
    }
}

impl<T> Drop for PooledVec<T> {
    fn drop(&mut self) {
        if let Some(vec) = self.vec.take() {
            self.pool.return_vec(vec);
        }
    }
}

impl<T> std::ops::Deref for PooledVec<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        self.vec.as_ref().unwrap()
    }
}

impl<T> std::ops::DerefMut for PooledVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.vec.as_mut().unwrap()
    }
}

/// 全局Vec池集合
pub struct GlobalVecPools {
    /// f32类型的Vec池
    f32_pool: VecPool<f32>,
    /// usize类型的Vec池
    usize_pool: VecPool<usize>,
    /// u32类型的Vec池
    u32_pool: VecPool<u32>,
    /// i32类型的Vec池
    i32_pool: VecPool<i32>,
}

impl Default for GlobalVecPools {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalVecPools {
    /// 创建新的全局Vec池集合（针对TTS场景优化）
    pub fn new() -> Self {
        Self {
            // TTS场景中f32向量使用最频繁（logits、概率等），增大池容量
            f32_pool: VecPool::new(500),
            // 索引向量也较常用，适中容量
            usize_pool: VecPool::new(200),
            // token序列使用频繁，增大容量
            u32_pool: VecPool::new(300),
            // 其他整数向量使用较少
            i32_pool: VecPool::new(150),
        }
    }

    /// 获取f32 Vec
    pub fn get_f32_vec(&self, capacity: usize) -> Vec<f32> {
        self.f32_pool.get_with_capacity(capacity)
    }

    /// 归还f32 Vec
    pub fn return_f32_vec(&self, vec: Vec<f32>) {
        self.f32_pool.return_vec(vec);
    }

    /// 获取usize Vec
    pub fn get_usize_vec(&self, capacity: usize) -> Vec<usize> {
        self.usize_pool.get_with_capacity(capacity)
    }

    /// 归还usize Vec
    pub fn return_usize_vec(&self, vec: Vec<usize>) {
        self.usize_pool.return_vec(vec);
    }

    /// 获取u32 Vec
    pub fn get_u32_vec(&self, capacity: usize) -> Vec<u32> {
        self.u32_pool.get_with_capacity(capacity)
    }

    /// 归还u32 Vec
    pub fn return_u32_vec(&self, vec: Vec<u32>) {
        self.u32_pool.return_vec(vec);
    }

    /// 获取i32 Vec
    pub fn get_i32_vec(&self, capacity: usize) -> Vec<i32> {
        self.i32_pool.get_with_capacity(capacity)
    }

    /// 归还i32 Vec
    pub fn return_i32_vec(&self, vec: Vec<i32>) {
        self.i32_pool.return_vec(vec);
    }

    /// 预热所有池（针对TTS场景优化）
    pub fn warm_up(&self) {
        // TTS场景中logits向量通常较大（词汇表大小），预热更大容量
        self.f32_pool.warm_up(50, 65536); // 64K词汇表
                                          // 索引向量通常较小
        self.usize_pool.warm_up(30, 1024);
        // token序列长度适中
        self.u32_pool.warm_up(40, 2048);
        // 其他向量较小
        self.i32_pool.warm_up(25, 512);
    }

    /// 获取所有池的统计信息
    pub fn get_all_stats(&self) -> std::collections::HashMap<String, PoolStats> {
        let mut stats = std::collections::HashMap::new();
        stats.insert("f32".to_string(), self.f32_pool.get_stats());
        stats.insert("usize".to_string(), self.usize_pool.get_stats());
        stats.insert("u32".to_string(), self.u32_pool.get_stats());
        stats.insert("i32".to_string(), self.i32_pool.get_stats());
        stats
    }

    /// 获取综合监控报告
    pub fn get_monitoring_report(&self) -> PoolMonitoringReport {
        let all_stats = self.get_all_stats();

        let mut total_memory = 0;
        let mut total_allocations = 0;
        let mut total_hits = 0;
        let mut avg_utilization = 0.0;
        let mut efficiency_scores = Vec::new();

        for (pool_type, stats) in &all_stats {
            total_memory += stats.estimated_memory_usage;
            total_allocations += stats.total_allocations;
            total_hits += stats.pool_hits;
            avg_utilization += stats.utilization_rate;
            efficiency_scores.push((pool_type.clone(), stats.efficiency_score()));
        }

        avg_utilization /= all_stats.len() as f32;
        let overall_hit_rate = if total_allocations > 0 {
            total_hits as f32 / total_allocations as f32
        } else {
            0.0
        };

        PoolMonitoringReport {
            total_memory_usage: total_memory,
            overall_hit_rate,
            average_utilization: avg_utilization,
            efficiency_scores,
            pool_stats: all_stats,
        }
    }

    /// 清空所有池
    pub fn clear_all(&self) {
        self.f32_pool.clear();
        self.usize_pool.clear();
        self.u32_pool.clear();
        self.i32_pool.clear();
    }
}

/// 全局Vec池实例
static GLOBAL_VEC_POOLS: OnceLock<GlobalVecPools> = OnceLock::new();

/// 获取全局Vec池实例
pub fn global_vec_pools() -> &'static GlobalVecPools {
    GLOBAL_VEC_POOLS.get_or_init(|| {
        let pools = GlobalVecPools::new();
        pools.warm_up(); // 预热池
        pools
    })
}

/// 便利宏：自动管理Vec的生命周期
#[macro_export]
macro_rules! with_pooled_vec {
    ($pool:expr, $capacity:expr, $vec_name:ident, $body:block) => {{
        let mut $vec_name = $pool.get_with_capacity($capacity);
        let result = $body;
        $pool.return_vec($vec_name);
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_pool_basic_operations() {
        let pool = VecPool::<i32>::new(10);

        // 测试获取和归还
        let vec1 = pool.get();
        assert_eq!(vec1.len(), 0);

        let mut vec2 = vec1;
        vec2.push(42);
        pool.return_vec(vec2);

        // 再次获取应该复用之前的Vec（但内容已清空）
        let vec3 = pool.get();
        assert_eq!(vec3.len(), 0);
        assert!(vec3.capacity() > 0); // 容量应该保留

        let stats = pool.get_stats();
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.pool_hits, 1);
        assert_eq!(stats.pool_misses, 1);
    }

    #[test]
    fn test_pooled_vec_raii() {
        let pool = VecPool::<i32>::new(10);

        {
            let mut pooled_vec = PooledVec::new(pool.clone());
            pooled_vec.push(1);
            pooled_vec.push(2);
            assert_eq!(pooled_vec.len(), 2);
        } // PooledVec在这里自动归还到池

        // 验证Vec已归还
        let stats = pool.get_stats();
        assert_eq!(stats.current_pool_size, 1);
    }

    #[test]
    fn test_global_vec_pools() {
        let pools = global_vec_pools();

        let mut f32_vec = pools.get_f32_vec(100);
        f32_vec.push(std::f32::consts::PI);
        assert_eq!(f32_vec[0], std::f32::consts::PI);

        pools.return_f32_vec(f32_vec);

        let stats = pools.get_all_stats();
        assert!(stats.contains_key("f32"));
    }

    #[test]
    fn test_pool_capacity_management() {
        let pool = VecPool::<i32>::new(2);

        // 填满池
        let vec1 = pool.get();
        let vec2 = pool.get();
        pool.return_vec(vec1);
        pool.return_vec(vec2);

        // 尝试归还第三个Vec（应该被丢弃）
        let vec3 = pool.get();
        pool.return_vec(vec3);

        let stats = pool.get_stats();
        assert!(stats.current_pool_size <= 2);
    }

    #[test]
    fn test_with_pooled_vec_macro() {
        let pool = VecPool::<i32>::new(10);

        let result = with_pooled_vec!(pool, 100, vec, {
            vec.push(1);
            vec.push(2);
            vec.len()
        });

        assert_eq!(result, 2);

        // 验证Vec已自动归还
        let stats = pool.get_stats();
        assert_eq!(stats.current_pool_size, 1);
    }
}
