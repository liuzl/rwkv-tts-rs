use anyhow::Result;
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;

/// ONNX会话池内部结构，负责会话与空闲索引管理
struct OnnxSessionPoolInner {
    sessions: Vec<Arc<Session>>,
    free_indices: Mutex<Vec<usize>>, // LIFO 栈，存放空闲的会话索引
    semaphore: Arc<Semaphore>,       // 控制总并发数，不超过会话数量
}

/// ONNX会话池，支持并发访问（保证每个会话在任一时刻仅被一个请求持有）
pub struct OnnxSessionPool {
    inner: Arc<OnnxSessionPoolInner>,
    pool_size: usize,
}

impl OnnxSessionPool {
    /// 创建新的ONNX会话池
    pub fn new(model_path: &str, pool_size: usize) -> Result<Self> {
        if !Path::new(model_path).exists() {
            return Err(anyhow::anyhow!("Model file not found: {}", model_path));
        }

        let mut sessions = Vec::with_capacity(pool_size);

        // 根据CPU核心数动态设置线程数，提升性能
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4); // 默认4核

        // 优化线程配置：根据CPU核心数动态调整，最大化CPU利用率
        let inter_threads = std::cmp::min(cpu_cores, 8); // 最多8个inter线程
        let intra_threads = std::cmp::max(cpu_cores / 2, 4); // 至少4个intra线程，最多使用一半核心

        for _ in 0..pool_size {
            let mut builder =
                Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;

            // 在非macOS系统上设置线程数以提升性能
            #[cfg(not(target_os = "macos"))]
            {
                builder = builder
                    .with_inter_threads(inter_threads)?
                    .with_intra_threads(intra_threads)?;
            }

            // macOS系统保持默认设置以避免mutex问题
            #[cfg(target_os = "macos")]
            {
                builder = builder.with_inter_threads(1)?;
            }

            let session = builder.commit_from_file(model_path)?;
            sessions.push(Arc::new(session));
        }

        let free_indices: Vec<usize> = (0..pool_size).rev().collect();

        Ok(Self {
            inner: Arc::new(OnnxSessionPoolInner {
                sessions,
                free_indices: Mutex::new(free_indices),
                semaphore: Arc::new(Semaphore::new(pool_size)),
            }),
            pool_size,
        })
    }

    /// 获取一个可用的会话（独占）
    pub async fn acquire_session(&self) -> Result<SessionGuard> {
        // 先获取一个并发许可，确保总持有数量不超过会话数量
        let permit = self
            .inner
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to acquire session: {}", e))?;

        // 优化：减少锁持有时间，快速获取索引后立即释放锁
        let index = {
            let mut stack = self
                .inner
                .free_indices
                .lock()
                .map_err(|_| anyhow::anyhow!("Poisoned mutex in OnnxSessionPool"))?;
            let idx = stack
                .pop()
                .ok_or_else(|| anyhow::anyhow!("No free session index available"))?;
            drop(stack); // 显式释放锁
            idx
        };

        Ok(SessionGuard {
            inner: self.inner.clone(),
            index,
            _permit: permit,
        })
    }

    /// 批量获取会话（减少锁竞争，提高并发性能）
    pub async fn acquire_sessions_batch(&self, count: usize) -> Result<Vec<SessionGuard>> {
        if count > self.pool_size {
            return Err(anyhow::anyhow!(
                "请求的会话数量 {} 超过池大小 {}",
                count,
                self.pool_size
            ));
        }

        let mut sessions = Vec::with_capacity(count);
        let mut permits = Vec::with_capacity(count);

        // 批量获取许可
        for _ in 0..count {
            let permit = self
                .inner
                .semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|e| anyhow::anyhow!("获取会话许可失败: {}", e))?;
            permits.push(permit);
        }

        // 批量获取会话索引（单次锁操作）
        let mut indices = Vec::with_capacity(count);
        {
            let mut stack = self
                .inner
                .free_indices
                .lock()
                .map_err(|_| anyhow::anyhow!("Poisoned mutex in OnnxSessionPool"))?;

            if stack.len() < count {
                return Err(anyhow::anyhow!(
                    "可用会话不足: 需要 {} 个，但只有 {} 个可用",
                    count,
                    stack.len()
                ));
            }

            for _ in 0..count {
                indices.push(stack.pop().unwrap());
            }
        }

        // 创建会话守卫
        for index in indices {
            sessions.push(SessionGuard {
                inner: self.inner.clone(),
                index,
                _permit: permits.pop().unwrap(),
            });
        }

        Ok(sessions)
    }

    /// 获取池大小
    pub fn pool_size(&self) -> usize {
        self.pool_size
    }
}

/// 会话守卫，自动归还会话索引与并发许可
pub struct SessionGuard {
    inner: Arc<OnnxSessionPoolInner>,
    index: usize,
    _permit: tokio::sync::OwnedSemaphorePermit,
}

impl SessionGuard {
    /// 获取会话引用
    pub fn session(&self) -> &Session {
        &self.inner.sessions[self.index]
    }

    /// 获取会话的可变引用
    pub fn session_mut(&mut self) -> &mut Session {
        // 通过会话索引保证独占访问，转为可变引用是安全的
        unsafe {
            let ptr = Arc::as_ptr(&self.inner.sessions[self.index]) as *mut Session;
            &mut *ptr
        }
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        // 优化：快速归还索引并立即释放锁
        if let Ok(mut stack) = self.inner.free_indices.lock() {
            stack.push(self.index);
            drop(stack); // 显式释放锁
        }
        // _permit 在此处自动 Drop，释放并发许可
    }
}

/// 并发ONNX模型管理器
pub struct ConcurrentOnnxManager {
    bicodec_tokenize_pool: OnnxSessionPool,
    wav2vec2_pool: OnnxSessionPool,
    bicodec_detokenize_pool: OnnxSessionPool,
}

impl ConcurrentOnnxManager {
    /// 创建新的并发ONNX管理器
    pub fn new(
        bicodec_tokenize_path: &str,
        wav2vec2_path: &str,
        bicodec_detokenize_path: &str,
        pool_size: Option<usize>,
    ) -> Result<Self> {
        let pool_size = pool_size.unwrap_or(4); // 默认每个模型4个会话

        let bicodec_tokenize_pool = OnnxSessionPool::new(bicodec_tokenize_path, pool_size)?;
        let wav2vec2_pool = OnnxSessionPool::new(wav2vec2_path, pool_size)?;
        let bicodec_detokenize_pool = OnnxSessionPool::new(bicodec_detokenize_path, pool_size)?;

        Ok(Self {
            bicodec_tokenize_pool,
            wav2vec2_pool,
            bicodec_detokenize_pool,
        })
    }

    /// 获取BiCodec Tokenize会话
    pub async fn acquire_bicodec_tokenize_session(&self) -> Result<SessionGuard> {
        self.bicodec_tokenize_pool.acquire_session().await
    }

    /// 获取Wav2Vec2会话
    pub async fn acquire_wav2vec2_session(&self) -> Result<SessionGuard> {
        self.wav2vec2_pool.acquire_session().await
    }

    /// 获取BiCodec Detokenize会话
    pub async fn acquire_bicodec_detokenize_session(&self) -> Result<SessionGuard> {
        self.bicodec_detokenize_pool.acquire_session().await
    }

    /// 批量获取BiCodec Tokenize会话
    pub async fn acquire_bicodec_tokenize_sessions_batch(
        &self,
        count: usize,
    ) -> Result<Vec<SessionGuard>> {
        self.bicodec_tokenize_pool
            .acquire_sessions_batch(count)
            .await
    }

    /// 批量获取Wav2Vec2会话
    pub async fn acquire_wav2vec2_sessions_batch(&self, count: usize) -> Result<Vec<SessionGuard>> {
        self.wav2vec2_pool.acquire_sessions_batch(count).await
    }

    /// 批量获取BiCodec Detokenize会话
    pub async fn acquire_bicodec_detokenize_sessions_batch(
        &self,
        count: usize,
    ) -> Result<Vec<SessionGuard>> {
        self.bicodec_detokenize_pool
            .acquire_sessions_batch(count)
            .await
    }

    /// 获取池统计信息
    pub fn get_pool_stats(&self) -> (usize, usize, usize) {
        (
            self.bicodec_tokenize_pool.pool_size(),
            self.wav2vec2_pool.pool_size(),
            self.bicodec_detokenize_pool.pool_size(),
        )
    }
}

/// 全局ONNX管理器单例
static GLOBAL_ONNX_MANAGER: std::sync::OnceLock<Arc<ConcurrentOnnxManager>> =
    std::sync::OnceLock::new();

/// 初始化全局ONNX管理器
pub fn init_global_onnx_manager(
    bicodec_tokenize_path: &str,
    wav2vec2_path: &str,
    bicodec_detokenize_path: &str,
    pool_size: Option<usize>,
) -> Result<()> {
    let manager = ConcurrentOnnxManager::new(
        bicodec_tokenize_path,
        wav2vec2_path,
        bicodec_detokenize_path,
        pool_size,
    )?;

    GLOBAL_ONNX_MANAGER
        .set(Arc::new(manager))
        .map_err(|_| anyhow::anyhow!("Global ONNX manager already initialized"))?;

    Ok(())
}

/// 获取全局ONNX管理器实例
pub fn get_global_onnx_manager() -> Result<Arc<ConcurrentOnnxManager>> {
    GLOBAL_ONNX_MANAGER
        .get()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Global ONNX manager not initialized"))
}
