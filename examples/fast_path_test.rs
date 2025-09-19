//! 测试FastSampler快速路径是否被正确触发

use rand::rngs::StdRng;
use rand::SeedableRng;
use rwkv_tts_rs::fast_sampler::{FastSampler, SamplingConfig};

fn main() {
    let sampler = FastSampler::new();

    // 测试1：确定性采样（极低温度）
    println!("=== 测试1：确定性采样（温度=0.01） ===");
    let config_deterministic = SamplingConfig {
        temperature: 0.01,
        top_p: 0.9,
        top_k: 50,
        use_fast_path: true,
        fast_path_threshold: 0.7,
        use_simd: false,
    };

    let logits = vec![1.0, 5.0, 2.0, 0.5, 1.5]; // 明显的最大值在索引1
    let mut rng = Some(StdRng::seed_from_u64(42));

    sampler.reset_stats();

    for i in 0..10 {
        let result = sampler.optimized_sample(&logits, &config_deterministic, None, &mut rng);
        println!("采样 {}: 结果={}", i + 1, result);
    }

    let stats = sampler.get_stats();
    println!("统计信息:");
    println!("  总采样次数: {}", stats.total_samples);
    println!("  快速路径命中: {}", stats.fast_path_hits);
    println!("  确定性采样: {}", stats.deterministic_samples);
    println!(
        "  快速路径命中率: {:.2}%",
        stats.fast_path_hit_rate() * 100.0
    );

    // 测试2：单峰分布
    println!("\n=== 测试2：单峰分布 ===");
    let config_single_peak = SamplingConfig {
        temperature: 1.0,
        top_p: 0.9,
        top_k: 50,
        use_fast_path: true,
        fast_path_threshold: 0.7,
        use_simd: false,
    };

    let logits_peak = vec![0.0, 10.0, 1.0, 0.5, 0.8]; // 非常明显的峰值

    sampler.reset_stats();

    for i in 0..10 {
        let result = sampler.optimized_sample(&logits_peak, &config_single_peak, None, &mut rng);
        println!("采样 {}: 结果={}", i + 1, result);
    }

    let stats = sampler.get_stats();
    println!("统计信息:");
    println!("  总采样次数: {}", stats.total_samples);
    println!("  快速路径命中: {}", stats.fast_path_hits);
    println!("  确定性采样: {}", stats.deterministic_samples);
    println!(
        "  快速路径命中率: {:.2}%",
        stats.fast_path_hit_rate() * 100.0
    );

    // 测试3：普通分布（应该不触发快速路径）
    println!("\n=== 测试3：普通分布（不应触发快速路径） ===");
    let config_normal = SamplingConfig {
        temperature: 1.0,
        top_p: 0.9,
        top_k: 50,
        use_fast_path: true,
        fast_path_threshold: 0.7,
        use_simd: false,
    };

    let logits_normal = vec![2.0, 2.1, 2.2, 2.0, 1.9]; // 相对平均的分布

    sampler.reset_stats();

    for i in 0..10 {
        let result = sampler.optimized_sample(&logits_normal, &config_normal, None, &mut rng);
        println!("采样 {}: 结果={}", i + 1, result);
    }

    let stats = sampler.get_stats();
    println!("统计信息:");
    println!("  总采样次数: {}", stats.total_samples);
    println!("  快速路径命中: {}", stats.fast_path_hits);
    println!("  确定性采样: {}", stats.deterministic_samples);
    println!(
        "  快速路径命中率: {:.2}%",
        stats.fast_path_hit_rate() * 100.0
    );

    // 测试4：直接测试try_fast_path方法
    println!("\n=== 测试4：直接测试try_fast_path方法 ===");
    let result = sampler.try_fast_path(&logits, &config_deterministic, None);
    println!("确定性采样快速路径结果: {:?}", result);

    let result = sampler.try_fast_path(&logits_peak, &config_single_peak, None);
    println!("单峰分布快速路径结果: {:?}", result);

    let result = sampler.try_fast_path(&logits_normal, &config_normal, None);
    println!("普通分布快速路径结果: {:?}", result);
}
