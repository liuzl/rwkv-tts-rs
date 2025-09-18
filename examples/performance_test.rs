//! 性能测试示例
//!
//! 这个示例演示了LLM推理优化的效果，包括：
//! 1. VecPool对象池的内存分配优化
//! 2. FastSampler的采样优化
//! 3. 性能监控和统计

use rand::{rngs::StdRng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rwkv_tts_rs::fast_sampler::{FastSampler, SamplingConfig};
use rwkv_tts_rs::performance_monitor::{MetricType, MonitorConfig, PerformanceMonitor};
use rwkv_tts_rs::vec_pool::global_vec_pools;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    println!("🚀 RWKV-TTS LLM推理性能优化测试");
    println!("{}", "=".repeat(50));

    // 测试VecPool性能
    test_vec_pool_performance();

    // 测试FastSampler性能
    test_fast_sampler_performance();

    // 测试性能监控
    test_performance_monitoring();

    println!("\n✅ 所有性能测试完成！");
}

/// 测试VecPool对象池的性能
fn test_vec_pool_performance() {
    println!("\n📊 测试VecPool对象池性能...");

    let iterations = 10000;
    let vec_size = 50257; // GPT-2词汇表大小

    // 测试标准Vec分配
    let start = Instant::now();
    for _ in 0..iterations {
        let _vec: Vec<f32> = vec![0.0; vec_size];
    }
    let standard_duration = start.elapsed();

    // 测试VecPool分配
    let start = Instant::now();
    for _ in 0..iterations {
        let _vec = global_vec_pools().get_f32_vec(vec_size);
    }
    let pool_duration = start.elapsed();

    let speedup = standard_duration.as_nanos() as f64 / pool_duration.as_nanos() as f64;

    println!("  标准Vec分配: {:?}", standard_duration);
    println!("  VecPool分配: {:?}", pool_duration);
    println!("  性能提升: {:.2}x", speedup);

    if speedup > 1.0 {
        println!("  ✅ VecPool优化生效！");
    } else {
        println!("  ⚠️  VecPool优化效果不明显（可能需要更多迭代）");
    }
}

/// 测试FastSampler的性能
fn test_fast_sampler_performance() {
    println!("\n🎯 测试FastSampler采样性能...");

    let vocab_size = 50257;
    let iterations = 1000;

    // 生成测试logits
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    use rand::Rng;
    let base_logits: Vec<f32> = (0..vocab_size)
        .map(|_| rng.gen_range(-10.0..10.0))
        .collect();

    // 配置FastSampler
    let config = SamplingConfig {
        temperature: 1.0,
        top_p: 0.9,
        top_k: 50,
        use_fast_path: true,
        fast_path_threshold: 0.1,
        use_simd: false, // 在测试中禁用SIMD以确保一致性
    };

    let config_monitor = MonitorConfig::default();
    let _monitor = Arc::new(PerformanceMonitor::new(config_monitor));
    let fast_sampler = FastSampler::new();

    // 测试FastSampler性能
    let start = Instant::now();
    for _ in 0..iterations {
        let logits = base_logits.clone();
        let sample_rng = StdRng::seed_from_u64(12345);
        let _token = fast_sampler.optimized_sample(&logits, &config, None, &mut Some(sample_rng));
    }
    let fast_duration = start.elapsed();

    // 测试朴素采样性能
    let start = Instant::now();
    for _ in 0..iterations {
        let mut logits = base_logits.clone();
        let mut sample_rng = StdRng::seed_from_u64(12345);
        let _token = naive_sample(&mut logits, &config, &mut sample_rng);
    }
    let naive_duration = start.elapsed();

    let speedup = naive_duration.as_nanos() as f64 / fast_duration.as_nanos() as f64;

    println!("  FastSampler: {:?}", fast_duration);
    println!("  朴素采样: {:?}", naive_duration);
    println!("  性能提升: {:.2}x", speedup);

    // 显示采样统计
    let stats = fast_sampler.get_stats();
    let fast_path_ratio = if stats.total_samples > 0 {
        stats.fast_path_hits as f64 / stats.total_samples as f64
    } else {
        0.0
    };
    println!("  快速路径使用率: {:.1}%", fast_path_ratio * 100.0);
    println!("  总采样次数: {}", stats.total_samples);
    println!("  快速路径命中: {}", stats.fast_path_hits);

    if speedup > 1.0 {
        println!("  ✅ FastSampler优化生效！");
    } else {
        println!("  ⚠️  FastSampler优化效果不明显");
    }
}

/// 测试性能监控功能
fn test_performance_monitoring() {
    println!("\n📈 测试性能监控功能...");

    let config = MonitorConfig::default();
    let monitor = Arc::new(PerformanceMonitor::new(config));

    // 模拟一些性能指标
    monitor.record_metric(MetricType::SamplingLatency, 10.0);
    monitor.record_metric(MetricType::InferenceLatency, 5.0);
    monitor.record_cache_hit();
    monitor.record_cache_miss();

    // 获取性能报告
    let report = monitor.generate_report();
    println!("  性能报告:\n{}", report);

    // 获取实时统计
    let stats = monitor.get_realtime_stats();
    if let Some(cache_hits) = stats.get("cache_hits") {
        println!("  缓存命中次数: {}", cache_hits);
    }
    if let Some(cache_hit_rate) = stats.get("cache_hit_rate") {
        println!("  缓存命中率: {:.1}%", cache_hit_rate);
    }

    println!("  ✅ 性能监控正常工作！");
}

/// 朴素的采样实现（用于性能对比）
fn naive_sample(logits: &mut [f32], config: &SamplingConfig, rng: &mut StdRng) -> usize {
    use rand::Rng;
    use std::cmp::Ordering;

    // 应用温度
    if config.temperature != 1.0 {
        for logit in logits.iter_mut() {
            *logit /= config.temperature;
        }
    }

    // 转换为概率
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut probs: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for prob in probs.iter_mut() {
        *prob /= sum;
    }

    // Top-k过滤
    if config.top_k > 0 && config.top_k < probs.len() {
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(Ordering::Equal));

        for &idx in indices.iter().skip(config.top_k) {
            probs[idx] = 0.0;
        }
    }

    // Top-p过滤
    if config.top_p < 1.0 {
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(Ordering::Equal));

        let mut cumulative = 0.0;
        let mut cutoff = probs.len();

        for (i, &idx) in indices.iter().enumerate() {
            cumulative += probs[idx];
            if cumulative >= config.top_p {
                cutoff = i + 1;
                break;
            }
        }

        for &idx in indices.iter().skip(cutoff) {
            probs[idx] = 0.0;
        }
    }

    // 重新归一化
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for prob in probs.iter_mut() {
            *prob /= sum;
        }
    }

    // 采样
    let random_value = rng.gen::<f32>();
    let mut cumulative = 0.0;

    for (i, &prob) in probs.iter().enumerate() {
        cumulative += prob;
        if random_value <= cumulative {
            return i;
        }
    }

    // 确定性采样：返回概率最高的token
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
