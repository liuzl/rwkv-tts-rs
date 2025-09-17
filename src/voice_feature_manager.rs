//! 音色特征管理模块
//! 实现音色特征的提取、保存、加载和管理功能

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::fs as async_fs;
use uuid::Uuid;

/// 音色特征数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceFeature {
    /// 音色唯一标识符
    pub id: String,
    /// 音色名称
    pub name: String,
    /// 原始提示词
    pub prompt_text: String,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 全局令牌
    pub global_tokens: Vec<i32>,
    /// 语义令牌
    pub semantic_tokens: Vec<i32>,
    /// 音频时长（秒）
    pub audio_duration: f32,
    /// 采样率
    pub sample_rate: u32,
    /// 文件校验和
    pub checksum: String,
}

/// 音色元数据（用于列表显示）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceMetadata {
    pub id: String,
    pub name: String,
    pub prompt_text: String,
    pub created_at: DateTime<Utc>,
    pub file_path: String,
    pub file_size: u64,
    pub checksum: String,
}

/// 音色列表容器
#[derive(Debug, Serialize, Deserialize)]
pub struct VoicesMetadata {
    pub voices: Vec<VoiceMetadata>,
}

/// 音色特征管理器
#[derive(Debug)]
pub struct VoiceFeatureManager {
    /// RAF目录路径
    raf_dir: PathBuf,
    /// 元数据文件路径
    metadata_file: PathBuf,
    /// 内存中的音色缓存
    voice_cache: Arc<Mutex<HashMap<String, VoiceFeature>>>,
}

impl VoiceFeatureManager {
    /// 创建新的音色特征管理器
    pub fn new<P: AsRef<Path>>(raf_dir: P) -> Result<Self> {
        let raf_dir = raf_dir.as_ref().to_path_buf();
        let metadata_file = raf_dir.join("voices_metadata.json");

        // 确保RAF目录存在
        if !raf_dir.exists() {
            fs::create_dir_all(&raf_dir)?;
        }

        // 确保temp目录存在
        let temp_dir = raf_dir.join("temp");
        if !temp_dir.exists() {
            fs::create_dir_all(&temp_dir)?;
        }

        let upload_temp_dir = temp_dir.join("upload_temp_files");
        if !upload_temp_dir.exists() {
            fs::create_dir_all(&upload_temp_dir)?;
        }

        Ok(Self {
            raf_dir,
            metadata_file,
            voice_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// 生成新的音色ID
    fn generate_voice_id() -> String {
        let now = Utc::now();
        let timestamp = now.format("%Y%m%d_%H%M%S");
        let uuid_short = Uuid::new_v4().to_string()[..8].to_string();
        format!("voice_{}_{}", timestamp, uuid_short)
    }

    /// 计算数据的SHA256校验和
    fn calculate_checksum(data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }

    /// 保存音色特征到文件（只支持JSON文本格式）
    pub async fn save_voice_feature(
        &self,
        name: String,
        prompt_text: String,
        global_tokens: Vec<i32>,
        semantic_tokens: Vec<i32>,
        audio_duration: f32,
        sample_rate: u32,
    ) -> Result<String> {
        let voice_id = Self::generate_voice_id();
        let created_at = Utc::now();

        let voice_feature = VoiceFeature {
            id: voice_id.clone(),
            name: name.clone(),
            prompt_text: prompt_text.clone(),
            created_at,
            global_tokens,
            semantic_tokens,
            audio_duration,
            sample_rate,
            checksum: String::new(), // 将在序列化后计算
        };

        // 使用JSON文本格式序列化
        let serialized_data = serde_json::to_vec_pretty(&voice_feature)?;
        let checksum = Self::calculate_checksum(&serialized_data);

        // 更新校验和
        let mut voice_feature = voice_feature;
        voice_feature.checksum = checksum.clone();

        // 重新序列化包含校验和的数据
        let final_data = serde_json::to_vec_pretty(&voice_feature)?;

        // 保存到RAF文件（只使用JSON格式）
        let raf_file_path = self.raf_dir.join(format!("{}.raf.json", voice_id));

        // 确保父目录存在
        if let Some(parent_dir) = raf_file_path.parent() {
            async_fs::create_dir_all(parent_dir).await?;
        }

        async_fs::write(&raf_file_path, &final_data).await?;

        // 更新元数据
        let metadata = VoiceMetadata {
            id: voice_id.clone(),
            name,
            prompt_text,
            created_at,
            file_path: raf_file_path.to_string_lossy().to_string(),
            file_size: final_data.len() as u64,
            checksum,
        };

        self.add_voice_metadata(metadata).await?;

        // 添加到缓存
        {
            let mut cache = self.voice_cache.lock().unwrap();
            cache.insert(voice_id.clone(), voice_feature);
        }

        Ok(voice_id)
    }

    /// 加载音色特征从文件（只支持JSON文本格式）
    pub async fn load_voice_feature(&self, voice_id: &str) -> Result<VoiceFeature> {
        // 先检查缓存
        {
            let cache = self.voice_cache.lock().unwrap();
            if let Some(voice_feature) = cache.get(voice_id) {
                return Ok(voice_feature.clone());
            }
        }

        // 从文件加载（只使用JSON格式）
        let raf_file_path = self.raf_dir.join(format!("{}.raf.json", voice_id));
        if !raf_file_path.exists() {
            return Err(anyhow!("音色特征文件不存在: {}", voice_id));
        }

        let data = async_fs::read(&raf_file_path).await?;
        let voice_feature: VoiceFeature = serde_json::from_slice(&data)?;

        // 验证校验和
        let mut temp_feature = voice_feature.clone();
        temp_feature.checksum = String::new();
        let temp_data = serde_json::to_vec_pretty(&temp_feature)?;
        let calculated_checksum = Self::calculate_checksum(&temp_data);

        if calculated_checksum != voice_feature.checksum {
            return Err(anyhow!("音色特征文件校验和不匹配: {}", voice_id));
        }

        // 添加到缓存
        {
            let mut cache = self.voice_cache.lock().unwrap();
            cache.insert(voice_id.to_string(), voice_feature.clone());
        }

        Ok(voice_feature)
    }

    /// 获取所有音色列表
    pub async fn list_voices(&self) -> Result<Vec<VoiceMetadata>> {
        if !self.metadata_file.exists() {
            return Ok(Vec::new());
        }

        let content = async_fs::read_to_string(&self.metadata_file).await?;
        let metadata: VoicesMetadata = serde_json::from_str(&content)?;
        Ok(metadata.voices)
    }

    /// 删除音色特征
    pub async fn delete_voice(&self, voice_id: &str) -> Result<()> {
        // 删除RAF文件
        let raf_file_path = self.raf_dir.join(format!("{}.raf.json", voice_id));
        if raf_file_path.exists() {
            async_fs::remove_file(&raf_file_path).await?;
        }

        // 从元数据中移除
        self.remove_voice_metadata(voice_id).await?;

        // 从缓存中移除
        {
            let mut cache = self.voice_cache.lock().unwrap();
            cache.remove(voice_id);
        }

        Ok(())
    }

    /// 重命名音色
    pub async fn rename_voice(&self, voice_id: &str, new_name: String) -> Result<()> {
        // 更新缓存中的名称
        {
            let mut cache = self.voice_cache.lock().unwrap();
            if let Some(voice_feature) = cache.get_mut(voice_id) {
                voice_feature.name = new_name.clone();
            }
        }

        // 重新加载并保存特征文件
        let mut voice_feature = self.load_voice_feature(voice_id).await?;
        voice_feature.name = new_name.clone();

        // 重新序列化并保存
        let mut temp_feature = voice_feature.clone();
        temp_feature.checksum = String::new();
        let serialized_data = serde_json::to_vec_pretty(&temp_feature)?;
        let checksum = Self::calculate_checksum(&serialized_data);
        voice_feature.checksum = checksum;

        let final_data = serde_json::to_vec_pretty(&voice_feature)?;
        let raf_file_path = self.raf_dir.join(format!("{}.raf.json", voice_id));

        // 确保父目录存在
        if let Some(parent_dir) = raf_file_path.parent() {
            async_fs::create_dir_all(parent_dir).await?;
        }

        async_fs::write(&raf_file_path, &final_data).await?;

        // 更新元数据
        self.update_voice_metadata_name(voice_id, new_name).await?;

        Ok(())
    }

    /// 添加音色元数据
    async fn add_voice_metadata(&self, metadata: VoiceMetadata) -> Result<()> {
        let mut voices_metadata = if self.metadata_file.exists() {
            let content = async_fs::read_to_string(&self.metadata_file).await?;
            serde_json::from_str::<VoicesMetadata>(&content)?
        } else {
            VoicesMetadata { voices: Vec::new() }
        };

        voices_metadata.voices.push(metadata);

        let content = serde_json::to_string_pretty(&voices_metadata)?;

        // 确保元数据文件的父目录存在
        if let Some(parent_dir) = self.metadata_file.parent() {
            async_fs::create_dir_all(parent_dir).await?;
        }

        async_fs::write(&self.metadata_file, content).await?;

        Ok(())
    }

    /// 移除音色元数据
    async fn remove_voice_metadata(&self, voice_id: &str) -> Result<()> {
        if !self.metadata_file.exists() {
            return Ok(());
        }

        let content = async_fs::read_to_string(&self.metadata_file).await?;
        let mut voices_metadata: VoicesMetadata = serde_json::from_str(&content)?;

        voices_metadata.voices.retain(|v| v.id != voice_id);

        let content = serde_json::to_string_pretty(&voices_metadata)?;

        // 确保元数据文件的父目录存在
        if let Some(parent_dir) = self.metadata_file.parent() {
            async_fs::create_dir_all(parent_dir).await?;
        }

        async_fs::write(&self.metadata_file, content).await?;

        Ok(())
    }

    /// 更新音色元数据名称
    async fn update_voice_metadata_name(&self, voice_id: &str, new_name: String) -> Result<()> {
        if !self.metadata_file.exists() {
            return Err(anyhow!("元数据文件不存在"));
        }

        let content = async_fs::read_to_string(&self.metadata_file).await?;
        let mut voices_metadata: VoicesMetadata = serde_json::from_str(&content)?;

        if let Some(voice_meta) = voices_metadata.voices.iter_mut().find(|v| v.id == voice_id) {
            voice_meta.name = new_name;
        }

        let content = serde_json::to_string_pretty(&voices_metadata)?;

        // 确保元数据文件的父目录存在
        if let Some(parent_dir) = self.metadata_file.parent() {
            async_fs::create_dir_all(parent_dir).await?;
        }

        async_fs::write(&self.metadata_file, content).await?;

        Ok(())
    }

    /// 清理缓存
    pub fn clear_cache(&self) {
        let mut cache = self.voice_cache.lock().unwrap();
        cache.clear();
    }

    /// 获取RAF目录路径
    pub fn get_raf_dir(&self) -> &Path {
        &self.raf_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_voice_feature_manager() {
        let temp_dir = TempDir::new().unwrap();
        let manager = VoiceFeatureManager::new(temp_dir.path()).unwrap();

        // 测试保存音色特征
        let voice_id = manager
            .save_voice_feature(
                "测试音色".to_string(),
                "这是一个测试音色".to_string(),
                vec![1, 2, 3, 4, 5],
                vec![6, 7, 8, 9, 10],
                5.0,
                16000,
            )
            .await
            .unwrap();

        // 测试加载音色特征
        let loaded_feature = manager.load_voice_feature(&voice_id).await.unwrap();
        assert_eq!(loaded_feature.name, "测试音色");
        assert_eq!(loaded_feature.global_tokens, vec![1, 2, 3, 4, 5]);

        // 测试列表音色
        let voices = manager.list_voices().await.unwrap();
        assert_eq!(voices.len(), 1);
        assert_eq!(voices[0].name, "测试音色");

        // 测试重命名
        manager
            .rename_voice(&voice_id, "新名称".to_string())
            .await
            .unwrap();
        let renamed_feature = manager.load_voice_feature(&voice_id).await.unwrap();
        assert_eq!(renamed_feature.name, "新名称");

        // 测试删除
        manager.delete_voice(&voice_id).await.unwrap();
        let voices_after_delete = manager.list_voices().await.unwrap();
        assert_eq!(voices_after_delete.len(), 0);
    }
}
