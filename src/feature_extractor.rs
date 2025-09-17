//! 特征提取模块
//!
//! 本模块负责处理TTS推理中的特征预提取功能，
//! 包括文本预处理、token化和特征缓存等。

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};
// use web_rwkv::runtime::model::State; // 暂时注释掉未使用的导入
use web_rwkv::tokenizer::Tokenizer;

use crate::batch_types::TtsInferOptions;
use crate::shared_runtime::TtsInferContext;

/// 特征提取器，负责预处理和特征缓存
pub struct FeatureExtractor {
    /// 特征缓存
    feature_cache: Arc<Mutex<HashMap<String, Vec<u16>>>>,
    /// 最大缓存大小
    max_cache_size: usize,
}

impl FeatureExtractor {
    /// 创建新的特征提取器
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            feature_cache: Arc::new(Mutex::new(HashMap::new())),
            max_cache_size,
        }
    }

    /// 预处理文本并提取特征
    pub async fn extract_features(&self, context: &TtsInferContext) -> Result<Vec<u16>> {
        let text = &context.text;
        let tokenizer = &context.tokenizer;

        // 检查缓存
        if let Some(cached_tokens) = self.get_cached_features(text).await {
            debug!("🎯 使用缓存特征: {} tokens", cached_tokens.len());
            return Ok(cached_tokens);
        }

        // 预处理文本
        let processed_text = self.preprocess_text(text, &context.options)?;

        // Token化
        let tokens = self.tokenize_text(&processed_text, tokenizer)?;

        // 缓存结果
        self.cache_features(text.clone(), tokens.clone()).await;

        info!("✅ 提取特征完成: {} -> {} tokens", text.len(), tokens.len());
        Ok(tokens)
    }

    /// 预处理文本
    fn preprocess_text(&self, text: &str, _options: &TtsInferOptions) -> Result<String> {
        let mut processed = text.to_string();

        // 基本清理
        processed = processed.trim().to_string();

        // 预处理文本（简化版本）
        processed = processed.trim().to_string();
        processed = processed.replace("\n", " ").replace("\t", " ");
        // 合并多个空格为单个空格
        while processed.contains("  ") {
            processed = processed.replace("  ", " ");
        }

        debug!("📝 文本预处理: '{}' -> '{}'", text, processed);
        Ok(processed)
    }

    /// Token化文本
    fn tokenize_text(&self, text: &str, tokenizer: &Tokenizer) -> Result<Vec<u16>> {
        let tokens = tokenizer
            .encode(text.as_bytes())
            .map_err(|e| anyhow::anyhow!("Token化失败: {}", e))?
            .iter()
            .map(|&id| id as u16)
            .collect::<Vec<u16>>();

        debug!(
            "🔤 Token化完成: {} chars -> {} tokens",
            text.len(),
            tokens.len()
        );
        Ok(tokens)
    }

    /// 从缓存获取特征
    async fn get_cached_features(&self, text: &str) -> Option<Vec<u16>> {
        let cache = self.feature_cache.lock().await;
        cache.get(text).cloned()
    }

    /// 缓存特征
    async fn cache_features(&self, text: String, tokens: Vec<u16>) {
        let mut cache = self.feature_cache.lock().await;

        // 检查缓存大小限制
        if cache.len() >= self.max_cache_size {
            // 简单的LRU策略：清理一半缓存
            let keys_to_remove: Vec<_> = cache.keys().take(cache.len() / 2).cloned().collect();
            for key in keys_to_remove {
                cache.remove(&key);
            }
            warn!("🧹 缓存已满，清理了 {} 个条目", cache.len() / 2);
        }

        cache.insert(text, tokens);
        debug!("💾 缓存特征，当前缓存大小: {}", cache.len());
    }

    /// 清理缓存
    pub async fn clear_cache(&self) {
        let mut cache = self.feature_cache.lock().await;
        cache.clear();
        info!("🧹 清理特征缓存");
    }

    /// 获取缓存统计
    pub async fn cache_stats(&self) -> FeatureCacheStats {
        let cache = self.feature_cache.lock().await;
        FeatureCacheStats {
            size: cache.len(),
            max_size: self.max_cache_size,
        }
    }
}

/// 特征缓存统计信息
#[derive(Debug, Clone)]
pub struct FeatureCacheStats {
    pub size: usize,
    pub max_size: usize,
}

/// 预提取特征处理器
/// 负责批量预处理和特征提取
pub struct PreExtractProcessor {
    feature_extractor: FeatureExtractor,
}

impl PreExtractProcessor {
    /// 创建新的预提取处理器
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            feature_extractor: FeatureExtractor::new(max_cache_size),
        }
    }

    /// 批量预提取特征
    pub async fn batch_pre_extract(&self, contexts: &[TtsInferContext]) -> Result<Vec<Vec<u16>>> {
        let mut results = Vec::with_capacity(contexts.len());

        for context in contexts {
            let tokens = self.feature_extractor.extract_features(context).await?;
            results.push(tokens);
        }

        info!("✅ 批量预提取完成: {} 个请求", contexts.len());
        Ok(results)
    }

    /// 获取特征提取器引用
    pub fn feature_extractor(&self) -> &FeatureExtractor {
        &self.feature_extractor
    }
}

/// 特征提取相关的辅助函数
pub mod utils {
    use super::*;

    /// 验证token序列的有效性
    pub fn validate_tokens(tokens: &[u16], vocab_size: usize) -> Result<()> {
        for &token in tokens {
            if token as usize >= vocab_size {
                return Err(anyhow::anyhow!(
                    "无效token: {} >= vocab_size({})",
                    token,
                    vocab_size
                ));
            }
        }
        Ok(())
    }

    /// 计算token序列的统计信息
    pub fn token_stats(tokens: &[u16]) -> TokenStats {
        if tokens.is_empty() {
            return TokenStats {
                count: 0,
                unique_count: 0,
                min_token: 0,
                max_token: 0,
            };
        }

        let mut unique_tokens = std::collections::HashSet::new();
        let mut min_token = tokens[0];
        let mut max_token = tokens[0];

        for &token in tokens {
            unique_tokens.insert(token);
            min_token = min_token.min(token);
            max_token = max_token.max(token);
        }

        TokenStats {
            count: tokens.len(),
            unique_count: unique_tokens.len(),
            min_token,
            max_token,
        }
    }

    /// Token统计信息
    #[derive(Debug, Clone)]
    pub struct TokenStats {
        pub count: usize,
        pub unique_count: usize,
        pub min_token: u16,
        pub max_token: u16,
    }
}
