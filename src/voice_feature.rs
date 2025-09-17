//! 音色特征管理模块
//!
//! 提供音色特征的提取、保存、加载和管理功能

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tokio::fs as async_fs;
use tracing::{debug, info, warn};

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
    /// Global tokens（已处理，无需偏移）
    pub global_tokens: Vec<i32>,
    /// Semantic tokens
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
    /// 音色唯一标识符
    pub id: String,
    /// 音色名称
    pub name: String,
    /// 原始提示词
    pub prompt_text: String,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 特征文件路径
    pub file_path: String,
    /// 文件大小（字节）
    pub file_size: u64,
    /// 文件校验和
    pub checksum: String,
}

/// 音色元数据集合
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VoiceMetadataCollection {
    /// 音色列表
    pub voices: Vec<VoiceMetadata>,
}

/// 音色特征管理器
pub struct VoiceFeatureManager {
    /// RAF目录路径
    raf_dir: PathBuf,
    /// 元数据文件路径
    metadata_file: PathBuf,
}

impl VoiceFeatureManager {
    /// 创建新的音色特征管理器
    pub fn new<P: AsRef<Path>>(raf_dir: P) -> Result<Self> {
        let raf_dir = raf_dir.as_ref().to_path_buf();
        let metadata_file = raf_dir.join("voices_metadata.json");

        // 确保RAF目录存在
        if !raf_dir.exists() {
            fs::create_dir_all(&raf_dir).map_err(|e| anyhow!("创建RAF目录失败: {}", e))?;
            info!("创建RAF目录: {:?}", raf_dir);
        }

        // 确保临时目录存在
        let temp_dir = raf_dir.join("temp").join("upload_temp_files");
        if !temp_dir.exists() {
            fs::create_dir_all(&temp_dir).map_err(|e| anyhow!("创建临时目录失败: {}", e))?;
            info!("创建临时目录: {:?}", temp_dir);
        }

        Ok(Self {
            raf_dir,
            metadata_file,
        })
    }

    /// 生成新的音色ID
    pub fn generate_voice_id() -> String {
        let now = Utc::now();
        let timestamp = now.format("%Y%m%d_%H%M%S");
        let random_suffix = rand::random::<u16>();
        format!("voice_{}_{:04x}", timestamp, random_suffix)
    }

    /// 保存音色特征到RAF文件
    pub async fn save_voice_feature(&self, voice_feature: &VoiceFeature) -> Result<String> {
        let file_name = format!("{}.raf", voice_feature.id);
        let file_path = self.raf_dir.join(&file_name);

        // 序列化音色特征数据
        let data = bincode::serde::encode_to_vec(voice_feature, bincode::config::standard())
            .map_err(|e| anyhow!("序列化音色特征失败: {}", e))?;

        // 写入文件
        async_fs::write(&file_path, &data)
            .await
            .map_err(|e| anyhow!("写入音色特征文件失败: {}", e))?;

        // 计算文件大小
        let file_size = data.len() as u64;

        // 更新元数据
        let metadata = VoiceMetadata {
            id: voice_feature.id.clone(),
            name: voice_feature.name.clone(),
            prompt_text: voice_feature.prompt_text.clone(),
            created_at: voice_feature.created_at,
            file_path: file_name.clone(),
            file_size,
            checksum: voice_feature.checksum.clone(),
        };

        self.add_voice_metadata(metadata).await?;

        info!("保存音色特征文件: {:?}", file_path);
        Ok(file_name)
    }

    /// 从RAF文件加载音色特征
    pub async fn load_voice_feature(&self, voice_id: &str) -> Result<VoiceFeature> {
        let file_name = format!("{}.raf", voice_id);
        let file_path = self.raf_dir.join(&file_name);

        if !file_path.exists() {
            return Err(anyhow!("音色特征文件不存在: {}", voice_id));
        }

        // 读取文件数据
        let data = async_fs::read(&file_path)
            .await
            .map_err(|e| anyhow!("读取音色特征文件失败: {}", e))?;

        // 反序列化音色特征数据
        let (voice_feature, _): (VoiceFeature, usize) =
            bincode::serde::decode_from_slice(&data, bincode::config::standard())
                .map_err(|e| anyhow!("反序列化音色特征失败: {}", e))?;

        debug!(
            "加载音色特征: {} (global: {}, semantic: {})",
            voice_id,
            voice_feature.global_tokens.len(),
            voice_feature.semantic_tokens.len()
        );

        Ok(voice_feature)
    }

    /// 获取音色列表
    pub async fn list_voices(&self) -> Result<Vec<VoiceMetadata>> {
        let metadata_collection = self.load_metadata_collection().await?;
        Ok(metadata_collection.voices)
    }

    /// 删除音色特征
    pub async fn delete_voice(&self, voice_id: &str) -> Result<()> {
        let file_name = format!("{}.raf", voice_id);
        let file_path = self.raf_dir.join(&file_name);

        // 删除文件
        if file_path.exists() {
            async_fs::remove_file(&file_path)
                .await
                .map_err(|e| anyhow!("删除音色特征文件失败: {}", e))?;
            info!("删除音色特征文件: {:?}", file_path);
        }

        // 从元数据中移除
        self.remove_voice_metadata(voice_id).await?;

        Ok(())
    }

    /// 重命名音色
    pub async fn rename_voice(&self, voice_id: &str, new_name: &str) -> Result<()> {
        let mut metadata_collection = self.load_metadata_collection().await?;

        // 查找并更新音色名称
        let voice_metadata = metadata_collection
            .voices
            .iter_mut()
            .find(|v| v.id == voice_id)
            .ok_or_else(|| anyhow!("音色不存在: {}", voice_id))?;

        voice_metadata.name = new_name.to_string();

        // 保存更新后的元数据
        self.save_metadata_collection(&metadata_collection).await?;

        info!("重命名音色: {} -> {}", voice_id, new_name);
        Ok(())
    }

    /// 加载元数据集合
    async fn load_metadata_collection(&self) -> Result<VoiceMetadataCollection> {
        if !self.metadata_file.exists() {
            return Ok(VoiceMetadataCollection::default());
        }

        let content = async_fs::read_to_string(&self.metadata_file)
            .await
            .map_err(|e| anyhow!("读取元数据文件失败: {}", e))?;

        let collection: VoiceMetadataCollection =
            serde_json::from_str(&content).map_err(|e| anyhow!("解析元数据文件失败: {}", e))?;

        Ok(collection)
    }

    /// 保存元数据集合
    async fn save_metadata_collection(&self, collection: &VoiceMetadataCollection) -> Result<()> {
        let content = serde_json::to_string_pretty(collection)
            .map_err(|e| anyhow!("序列化元数据失败: {}", e))?;

        async_fs::write(&self.metadata_file, content)
            .await
            .map_err(|e| anyhow!("写入元数据文件失败: {}", e))?;

        Ok(())
    }

    /// 添加音色元数据
    async fn add_voice_metadata(&self, metadata: VoiceMetadata) -> Result<()> {
        let mut collection = self.load_metadata_collection().await?;

        // 检查是否已存在相同ID的音色
        if collection.voices.iter().any(|v| v.id == metadata.id) {
            return Err(anyhow!("音色ID已存在: {}", metadata.id));
        }

        collection.voices.push(metadata);
        self.save_metadata_collection(&collection).await?;

        Ok(())
    }

    /// 移除音色元数据
    async fn remove_voice_metadata(&self, voice_id: &str) -> Result<()> {
        let mut collection = self.load_metadata_collection().await?;

        let original_len = collection.voices.len();
        collection.voices.retain(|v| v.id != voice_id);

        if collection.voices.len() == original_len {
            warn!("尝试删除不存在的音色元数据: {}", voice_id);
        }

        self.save_metadata_collection(&collection).await?;

        Ok(())
    }

    /// 获取RAF目录路径
    pub fn raf_dir(&self) -> &Path {
        &self.raf_dir
    }

    /// 获取临时目录路径
    pub fn temp_dir(&self) -> PathBuf {
        self.raf_dir.join("temp").join("upload_temp_files")
    }
}

/// 计算数据的SHA256校验和
pub fn calculate_checksum(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

/// 从音频数据提取音色特征（占位符实现）
/// 实际实现需要调用音频处理和模型推理
pub async fn extract_voice_feature_from_audio(
    audio_data: &[u8],
    prompt_text: &str,
    voice_name: &str,
) -> Result<VoiceFeature> {
    // TODO: 实现实际的音色特征提取逻辑
    // 这里是占位符实现，需要集成实际的音频处理和模型推理

    let voice_id = VoiceFeatureManager::generate_voice_id();
    let checksum = calculate_checksum(audio_data);

    // 占位符：生成模拟的tokens
    let global_tokens: Vec<i32> = (0..32).map(|i| (i * 100) % 4096).collect();
    let semantic_tokens: Vec<i32> = (0..64).map(|i| (i * 50) % 8192).collect();

    let voice_feature = VoiceFeature {
        id: voice_id,
        name: voice_name.to_string(),
        prompt_text: prompt_text.to_string(),
        created_at: Utc::now(),
        global_tokens,
        semantic_tokens,
        audio_duration: 3.0, // 占位符
        sample_rate: 16000,  // 占位符
        checksum,
    };

    info!("提取音色特征完成: {} ({})", voice_name, voice_feature.id);
    Ok(voice_feature)
}
