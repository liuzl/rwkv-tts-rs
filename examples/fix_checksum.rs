use serde_json;
use sha2::{Digest, Sha256};
use std::fs;

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct VoiceFeature {
    id: String,
    name: String,
    prompt_text: String,
    created_at: String,
    global_tokens: Vec<i32>,
    semantic_tokens: Vec<i32>,
    audio_duration: f32,
    sample_rate: u32,
    checksum: String,
}

fn calculate_checksum(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let files = [
        "./raf/voice_20241218_123456_test001.raf.json",
        "./raf/voice_20241218_123456_test002.raf.json",
    ];

    for file_path in &files {
        println!("处理文件: {}", file_path);

        // 读取文件
        let content = fs::read_to_string(file_path)?;
        let mut voice_feature: VoiceFeature = serde_json::from_str(&content)?;

        // 清空校验和
        voice_feature.checksum = String::new();

        // 计算校验和
        let serialized_data = serde_json::to_vec_pretty(&voice_feature)?;
        let checksum = calculate_checksum(&serialized_data);

        // 更新校验和
        voice_feature.checksum = checksum;

        // 重新保存文件
        let final_data = serde_json::to_string_pretty(&voice_feature)?;
        fs::write(file_path, final_data)?;

        println!("已更新校验和: {}", voice_feature.checksum);
    }

    println!("所有文件校验和已更新完成！");
    Ok(())
}