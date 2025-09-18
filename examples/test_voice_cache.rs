//! 测试音色缓存功能
//! 验证VoiceFeatureManager的内存缓存机制

use anyhow::Result;
use rwkv_tts_rs::voice_feature_manager::VoiceFeatureManager;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== 音色缓存功能测试 ===");

    // 1. 创建VoiceFeatureManager实例
    let raf_dir = "./raf";
    println!("创建VoiceFeatureManager实例，RAF目录: {}", raf_dir);

    let manager = VoiceFeatureManager::new(raf_dir)?;

    // 2. 测试预加载功能
    println!("\n--- 测试预加载功能 ---");
    let preload_start = Instant::now();
    manager.preload_all_voices().await?;
    let preload_time = preload_start.elapsed();
    println!("预加载完成，耗时: {:?}", preload_time);

    // 3. 获取缓存统计信息
    let stats = manager.get_cache_stats();
    println!("\n--- 缓存统计信息 ---");
    println!("缓存中的音色数量: {}", stats.total_voices);
    println!("缓存命中次数: {}", stats.cache_hits);
    println!("缓存未命中次数: {}", stats.cache_misses);
    println!("最后刷新时间: {}", stats.last_refresh);
    println!("缓存命中率: {:.2}%", manager.get_cache_hit_rate() * 100.0);

    // 4. 列出所有可用音色
    println!("\n--- 可用音色列表 ---");
    let voices = manager.list_voices().await?;
    if voices.is_empty() {
        println!("未找到任何音色文件");
        println!("提示: 请确保在 {} 目录下有 .raf.json 文件", raf_dir);
    } else {
        for (i, voice) in voices.iter().enumerate() {
            println!("{}. ID: {}, 名称: {}", i + 1, voice.id, voice.name);
            println!("   创建时间: {}", voice.created_at);
            println!("   文件大小: {} bytes", voice.file_size);
        }

        // 5. 测试从缓存获取tokens
        if let Some(first_voice) = voices.first() {
            println!("\n--- 测试缓存获取tokens ---");
            let voice_id = &first_voice.id;

            // 第一次获取（可能从文件加载）
            let start1 = Instant::now();
            match manager.get_voice_tokens(voice_id).await {
                Ok((global_tokens, semantic_tokens)) => {
                    let time1 = start1.elapsed();
                    println!("第一次获取 voice_id: {}", voice_id);
                    println!("  global_tokens 长度: {}", global_tokens.len());
                    println!("  semantic_tokens 长度: {}", semantic_tokens.len());
                    println!("  耗时: {:?}", time1);

                    // 第二次获取（应该从缓存获取）
                    let start2 = Instant::now();
                    match manager.get_voice_tokens(voice_id).await {
                        Ok((global_tokens2, semantic_tokens2)) => {
                            let time2 = start2.elapsed();
                            println!("第二次获取 voice_id: {}", voice_id);
                            println!("  global_tokens 长度: {}", global_tokens2.len());
                            println!("  semantic_tokens 长度: {}", semantic_tokens2.len());
                            println!("  耗时: {:?}", time2);

                            // 验证数据一致性
                            if global_tokens == global_tokens2
                                && semantic_tokens == semantic_tokens2
                            {
                                println!("  ✓ 数据一致性验证通过");
                            } else {
                                println!("  ✗ 数据一致性验证失败");
                            }

                            // 性能对比
                            if time2 < time1 {
                                let speedup = time1.as_nanos() as f64 / time2.as_nanos() as f64;
                                println!("  ✓ 缓存加速: {:.2}x", speedup);
                            }
                        }
                        Err(e) => println!("第二次获取失败: {}", e),
                    }
                }
                Err(e) => println!("第一次获取失败: {}", e),
            }
        }
    }

    // 6. 最终缓存统计
    let final_stats = manager.get_cache_stats();
    println!("\n--- 最终缓存统计 ---");
    println!("缓存命中次数: {}", final_stats.cache_hits);
    println!("缓存未命中次数: {}", final_stats.cache_misses);
    println!("缓存命中率: {:.2}%", manager.get_cache_hit_rate() * 100.0);
    println!("缓存中的音色数量: {}", manager.get_cached_voice_count());

    println!("\n=== 测试完成 ===");
    Ok(())
}
