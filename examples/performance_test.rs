use rwkv_tts_rs::voice_feature_manager::VoiceFeatureManager;
use std::time::Instant;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 音色缓存性能测试 ===");

    // 创建VoiceFeatureManager实例
    let manager = VoiceFeatureManager::new("./raf")?;
    println!("创建VoiceFeatureManager实例完成");

    // 测试预加载性能
    println!("\n--- 预加载性能测试 ---");
    let start = Instant::now();
    manager.preload_all_voices().await?;
    let preload_time = start.elapsed();
    println!("预加载耗时: {:?}", preload_time);

    let voices = manager.list_voices().await?;
    if voices.is_empty() {
        println!("警告: 没有找到音色文件，无法进行性能测试");
        return Ok(());
    }

    let test_voice_id = &voices[0].id;
    println!("使用测试音色ID: {}", test_voice_id);

    // 测试缓存命中性能
    println!("\n--- 缓存命中性能测试 ---");
    let iterations = 1000;
    let start = Instant::now();

    for i in 0..iterations {
        let _tokens = manager.get_voice_tokens(test_voice_id).await?;
        if i % 100 == 0 {
            print!(".");
        }
    }
    println!();

    let cache_hit_time = start.elapsed();
    let avg_time_per_call = cache_hit_time / iterations;

    println!("缓存命中测试完成:");
    println!("  总耗时: {:?}", cache_hit_time);
    println!("  平均每次调用: {:?}", avg_time_per_call);
    println!(
        "  每秒可处理: {:.0} 次调用",
        1_000_000_000.0 / avg_time_per_call.as_nanos() as f64
    );

    // 测试缓存统计
    println!("\n--- 缓存统计信息 ---");
    let stats = manager.get_cache_stats();
    println!("缓存中的音色数量: {}", stats.total_voices);
    println!("缓存命中次数: {}", stats.cache_hits);
    println!("缓存未命中次数: {}", stats.cache_misses);
    println!("缓存命中率: {:.2}%", manager.get_cache_hit_rate() * 100.0);

    // 测试内存使用情况
    println!("\n--- 内存使用情况 ---");
    println!("缓存中的音色数量: {}", manager.get_cached_voice_count());

    // 测试刷新缓存性能
    println!("\n--- 缓存刷新性能测试 ---");
    let start = Instant::now();
    manager.refresh_cache().await?;
    let refresh_time = start.elapsed();
    println!("缓存刷新耗时: {:?}", refresh_time);

    println!("\n=== 性能测试完成 ===");
    Ok(())
}
