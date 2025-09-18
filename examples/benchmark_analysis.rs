//! 分析基准测试中快速路径的命中率

use rwkv_tts_rs::fast_sampler::{FastSampler, SamplingConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// 生成测试用的logits数据（与基准测试相同）
fn generate_test_logits(vocab_size: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut logits: Vec<f32> = (0..vocab_size)
        .map(|_| rng.gen_range(-5.0..5.0))
        .collect();
    
    // 为了更好地测试快速路径，在一些位置创建明显的峰值
    if vocab_size > 100 {
        // 创建一些高概率的token
        for i in 0..10 {
            let idx = (i * vocab_size / 10) % vocab_size;
            logits[idx] = rng.gen_range(8.0..12.0); // 高logit值
        }
    }
    
    logits
}

fn main() {
    let vocab_size = 50257; // GPT-2词汇表大小
    let logits = generate_test_logits(vocab_size, 42);
    
    let configs = vec![
        (
            "low_temp",
            SamplingConfig {
                temperature: 0.1,
                top_p: 0.9,
                top_k: 50,
                use_fast_path: true,
                fast_path_threshold: 0.7,
                use_simd: false,
            },
        ),
        (
            "high_temp",
            SamplingConfig {
                temperature: 2.0,
                top_p: 0.9,
                top_k: 50,
                use_fast_path: true,
                fast_path_threshold: 0.7,
                use_simd: false,
            },
        ),
        (
            "strict_top_k",
            SamplingConfig {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 10,
                use_fast_path: true,
                fast_path_threshold: 0.7,
                use_simd: false,
            },
        ),
        (
            "strict_top_p",
            SamplingConfig {
                temperature: 1.0,
                top_p: 0.5,
                top_k: 0,
                use_fast_path: true,
                fast_path_threshold: 0.7,
                use_simd: false,
            },
        ),
    ];
    
    for (name, config) in configs {
        println!("=== 分析 {} 场景 ===", name);
        
        let fast_sampler = FastSampler::new();
        let mut rng = Some(StdRng::seed_from_u64(12345));
        
        // 运行1000次采样
        for _ in 0..1000 {
            let logits_copy = logits.clone();
            fast_sampler.optimized_sample(&logits_copy, &config, None, &mut rng);
        }
        
        let stats = fast_sampler.get_stats();
        println!("  总采样次数: {}", stats.total_samples);
        println!("  快速路径命中: {}", stats.fast_path_hits);
        println!("  确定性采样: {}", stats.deterministic_samples);
        println!("  快速路径命中率: {:.2}%", stats.fast_path_hit_rate() * 100.0);
        println!();
    }
    
    // 分析logits分布
    println!("=== Logits分布分析 ===");
    let mut sorted_logits = logits.clone();
    sorted_logits.sort_by(|a, b| b.partial_cmp(a).unwrap());
    
    println!("前10个最大logits: {:?}", &sorted_logits[0..10]);
    println!("最大logit: {:.2}", sorted_logits[0]);
    println!("第二大logit: {:.2}", sorted_logits[1]);
    println!("差值: {:.2}", sorted_logits[0] - sorted_logits[1]);
    
    // 计算softmax概率
    let max_logit = sorted_logits[0];
    let temp = 0.1; // 低温度
    let exp_max = ((max_logit - max_logit) / temp).exp(); // = 1.0
    let exp_second = ((sorted_logits[1] - max_logit) / temp).exp();
    let prob_max = exp_max / (exp_max + exp_second);
    
    println!("在温度0.1下，最大token的概率: {:.4}", prob_max);
    println!("是否超过0.7阈值: {}", prob_max > 0.7);
}