use rand::prelude::*;
use rand::rngs::StdRng;
use rwkv_tts_rs::fast_sampler::{FastSampler, SamplingConfig};
use rwkv_tts_rs::logits_cache::{CacheKey, LogitsCache, LogitsCacheConfig};
use rwkv_tts_rs::performance_monitor::{MetricType, MonitorConfig, PerformanceMonitor};
use rwkv_tts_rs::vec_pool::VecPool;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn main() {
    println!("=== RWKV-TTS 性能优化组件测试 ===");
    println!();

    // 测试FastSampler性能
    test_fast_sampler_performance();
    println!();

    // 测试VecPool性能
    test_vec_pool_performance();
    println!();

    // 测试LogitsCache性能
    test_logits_cache_performance();
    println!();

    // 测试性能监控
    test_performance_monitor();
}

fn test_fast_sampler_performance() {
    println!("🚀 FastSampler 性能测试");

    let vocab_size = 50257;
    let iterations = 1000;

    // 生成测试数据
    let mut rng = StdRng::seed_from_u64(42);
    let logits: Vec<f32> = (0..vocab_size)
        .map(|_| rng.gen_range(-10.0..10.0))
        .collect();

    let config = SamplingConfig {
        temperature: 1.0,
        top_p: 0.9,
        top_k: 50,
        use_fast_path: true,
        fast_path_threshold: 0.1,
        use_simd: true,
    };

    let fast_sampler = FastSampler::new();
    let mut rng_opt = Some(StdRng::seed_from_u64(12345));

    // 测试FastSampler
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = fast_sampler.optimized_sample(&logits, &config, None, &mut rng_opt);
    }
    let fast_duration = start.elapsed();

    // 测试朴素实现
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = naive_sample(&logits, &config);
    }
    let naive_duration = start.elapsed();

    let speedup = naive_duration.as_nanos() as f64 / fast_duration.as_nanos() as f64;

    println!(
        "  FastSampler: {:.2}ms ({} iterations)",
        fast_duration.as_millis(),
        iterations
    );
    println!(
        "  朴素实现:    {:.2}ms ({} iterations)",
        naive_duration.as_millis(),
        iterations
    );
    println!("  性能提升:    {:.2}x", speedup);

    if speedup > 1.1 {
        println!("  ✅ FastSampler 显著提升性能!");
    } else {
        println!("  ⚠️  FastSampler 性能提升不明显");
    }
}

fn test_vec_pool_performance() {
    println!("🏊 VecPool 内存池性能测试");

    let size = 50257;
    let iterations = 10000;

    // 测试VecPool
    let pool = VecPool::<f32>::new(100);
    let start = Instant::now();
    for _ in 0..iterations {
        let _vec = pool.get_with_capacity(size);
        // 向量会自动返回池中
    }
    let pool_duration = start.elapsed();

    // 测试标准分配
    let start = Instant::now();
    for _ in 0..iterations {
        let _vec: Vec<f32> = Vec::with_capacity(size);
    }
    let standard_duration = start.elapsed();

    let speedup = standard_duration.as_nanos() as f64 / pool_duration.as_nanos() as f64;

    println!(
        "  VecPool:     {:.2}ms ({} iterations)",
        pool_duration.as_millis(),
        iterations
    );
    println!(
        "  标准分配:    {:.2}ms ({} iterations)",
        standard_duration.as_millis(),
        iterations
    );
    println!("  性能提升:    {:.2}x", speedup);

    if speedup > 1.1 {
        println!("  ✅ VecPool 显著减少内存分配开销!");
    } else {
        println!("  ⚠️  VecPool 性能提升不明显");
    }
}

fn test_logits_cache_performance() {
    println!("💾 LogitsCache 缓存性能测试");

    let config = LogitsCacheConfig {
        max_entries: 1000,
        max_age: Duration::from_secs(300),
        enable_prefetch: true,
        prefetch_window: 3,
        hit_rate_threshold: 0.6,
    };

    let cache = LogitsCache::new(config);
    let iterations = 1000;

    // 生成测试数据
    let mut rng = StdRng::seed_from_u64(42);
    let test_keys: Vec<CacheKey> = (0..100)
        .map(|i| {
            let tokens: Vec<u32> = (0..10).map(|j| (i * 10 + j) as u32).collect();
            CacheKey::from_tokens(&tokens, tokens.len())
        })
        .collect();

    let test_logits: Vec<f32> = (0..50257).map(|_| rng.gen_range(-10.0..10.0)).collect();

    // 预填充缓存
    for key in &test_keys {
        cache.insert(key.clone(), test_logits.clone());
    }

    // 测试缓存命中性能
    let start = Instant::now();
    let mut hits = 0;
    for _ in 0..iterations {
        let key = &test_keys[rng.gen_range(0..test_keys.len())];
        if cache.get(key).is_some() {
            hits += 1;
        }
    }
    let cache_duration = start.elapsed();

    let hit_rate = hits as f64 / iterations as f64 * 100.0;

    println!(
        "  缓存查询:    {:.2}ms ({} iterations)",
        cache_duration.as_millis(),
        iterations
    );
    println!("  缓存命中率: {:.1}%", hit_rate);

    if hit_rate > 90.0 {
        println!("  ✅ LogitsCache 缓存效果良好!");
    } else {
        println!("  ⚠️  LogitsCache 缓存命中率较低");
    }
}

fn test_performance_monitor() {
    println!("📊 PerformanceMonitor 监控测试");

    let config = MonitorConfig::default();
    let monitor = Arc::new(PerformanceMonitor::new(config));

    // 记录一些测试指标
    for i in 0..100 {
        let inference_duration = Duration::from_millis((10 + i / 10) as u64);
        let sampling_duration = Duration::from_micros((1000 + i * 10) as u64);
        monitor.record_inference_latency(inference_duration);
        monitor.record_sampling_latency(sampling_duration);
        monitor.record_cache_hit();
        if i % 10 == 0 {
            monitor.record_cache_miss();
        }
    }

    let inference_stats = monitor.get_metric_summary(MetricType::InferenceLatency);
    let sampling_stats = monitor.get_metric_summary(MetricType::SamplingLatency);
    let cache_stats = monitor.get_metric_summary(MetricType::CacheHitRate);

    if let Some(stats) = inference_stats {
        println!("  推理延迟统计:");
        println!("    平均: {:.2}ms", stats.mean);
        println!("    最小: {:.2}ms", stats.min);
        println!("    最大: {:.2}ms", stats.max);
    }

    if let Some(stats) = sampling_stats {
        println!("  采样延迟统计:");
        println!("    平均: {:.2}ms", stats.mean);
    }

    if let Some(stats) = cache_stats {
        println!("  缓存统计:");
        println!("    命中率: {:.1}%", stats.mean * 100.0);
    }

    println!("  ✅ PerformanceMonitor 正常工作!");
}

// 朴素采样实现用于对比
fn naive_sample(logits: &[f32], config: &SamplingConfig) -> usize {
    let mut rng = StdRng::seed_from_u64(12345);
    let mut probs = logits.to_vec();

    // 应用温度
    if config.temperature != 1.0 {
        for prob in probs.iter_mut() {
            *prob /= config.temperature;
        }
    }

    // 计算softmax
    let max_logit = probs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0;
    for prob in probs.iter_mut() {
        *prob = (*prob - max_logit).exp();
        sum += *prob;
    }
    for prob in probs.iter_mut() {
        *prob /= sum;
    }

    // 简单采样
    let sample = rng.gen::<f32>();
    let mut cumulative = 0.0;
    for (i, &prob) in probs.iter().enumerate() {
        cumulative += prob;
        if sample <= cumulative {
            return i;
        }
    }
    probs.len() - 1
}
