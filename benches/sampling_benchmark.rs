use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_chacha::ChaCha8Rng;
use rwkv_tts_rs::fast_sampler::{FastSampler, SamplingConfig};
use rwkv_tts_rs::performance_monitor::{MonitorConfig, PerformanceMonitor};
use rwkv_tts_rs::vec_pool::VecPool;
use std::sync::Arc;

// 朴素采样实现
fn naive_sample(logits: &mut [f32], config: &SamplingConfig, rng: &mut impl Rng) -> usize {
    // 应用温度
    if config.temperature != 1.0 {
        for logit in logits.iter_mut() {
            *logit /= config.temperature;
        }
    }

    // 计算softmax
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0;
    for logit in logits.iter_mut() {
        *logit = (*logit - max_logit).exp();
        sum += *logit;
    }
    for logit in logits.iter_mut() {
        *logit /= sum;
    }

    // 简单的多项式采样
    let sample = rng.gen::<f32>();
    let mut cumulative = 0.0;
    for (i, &prob) in logits.iter().enumerate() {
        cumulative += prob;
        if sample <= cumulative {
            return i;
        }
    }
    logits.len() - 1
}

/// 生成测试用的logits数据
fn generate_test_logits(vocab_size: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    use rand::Rng;

    let mut logits: Vec<f32> = (0..vocab_size).map(|_| rng.gen_range(-5.0..5.0)).collect();

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

/// 基准测试：FastSampler vs 原始采样
fn benchmark_fast_sampler(c: &mut Criterion) {
    let vocab_size = 50257; // GPT-2词汇表大小
    let logits = generate_test_logits(vocab_size, 42);

    let config = SamplingConfig {
        temperature: 1.0,
        top_p: 0.9,
        top_k: 50,
        use_fast_path: true,
        fast_path_threshold: 0.7, // 提高阈值以更容易触发快速路径
        use_simd: false,          // 在基准测试中禁用SIMD以确保一致性
    };

    let monitor_config = MonitorConfig::default();
    let _monitor = Arc::new(PerformanceMonitor::new(monitor_config));
    let fast_sampler = FastSampler::new();
    let mut rng = Some(StdRng::seed_from_u64(12345));

    c.bench_function("fast_sampler_optimized", |b| {
        b.iter(|| {
            let logits_copy = logits.clone();
            fast_sampler.optimized_sample(
                black_box(&logits_copy),
                &config,
                None,
                black_box(&mut rng),
            )
        })
    });

    c.bench_function("naive_sampling", |b| {
        b.iter(|| {
            let mut logits_copy = logits.clone();
            let mut rng_local = StdRng::seed_from_u64(12345);
            naive_sample(
                black_box(&mut logits_copy),
                &config,
                black_box(&mut rng_local),
            )
        })
    });
}

/// 基准测试：VecPool内存分配优化
fn benchmark_vec_pool(c: &mut Criterion) {
    let size = 50257;

    c.bench_function("vec_pool_allocation", |b| {
        b.iter(|| {
            let pool = VecPool::<f32>::new(100);
            let _vec = pool.get_with_capacity(black_box(size));
            // 向量会在作用域结束时自动返回池中
        })
    });

    c.bench_function("standard_allocation", |b| {
        b.iter(|| {
            let _vec: Vec<f32> = vec![0.0; black_box(size)];
            // 标准分配，每次都会重新分配内存
        })
    });
}

/// 基准测试：不同词汇表大小的性能
fn benchmark_vocab_sizes(c: &mut Criterion) {
    let vocab_sizes = [1000, 10000, 50257, 100000];

    let config = SamplingConfig {
        temperature: 1.0,
        top_p: 0.9,
        top_k: 50,
        use_fast_path: true,
        fast_path_threshold: 0.7, // 提高阈值以更容易触发快速路径
        use_simd: false,
    };

    let monitor_config = MonitorConfig::default();
    let _monitor = Arc::new(PerformanceMonitor::new(monitor_config));

    for &vocab_size in &vocab_sizes {
        let logits = generate_test_logits(vocab_size, 42);
        let fast_sampler = FastSampler::new();
        let mut rng = Some(StdRng::seed_from_u64(12345));

        c.bench_function(&format!("fast_sampler_vocab_{}", vocab_size), |b| {
            b.iter(|| {
                let logits_copy = logits.clone();
                fast_sampler.optimized_sample(
                    black_box(&logits_copy),
                    &config,
                    None,
                    black_box(&mut rng),
                )
            })
        });
    }
}

/// 基准测试：不同采样参数的性能
fn benchmark_sampling_params(c: &mut Criterion) {
    let vocab_size = 50257;
    let logits = generate_test_logits(vocab_size, 42);

    let configs = vec![
        (
            "low_temp",
            SamplingConfig {
                temperature: 0.1,
                top_p: 0.9,
                top_k: 50,
                use_fast_path: true,
                fast_path_threshold: 0.7, // 提高阈值
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
                fast_path_threshold: 0.7, // 提高阈值
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
                fast_path_threshold: 0.7, // 提高阈值
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
                fast_path_threshold: 0.7, // 提高阈值
                use_simd: false,
            },
        ),
    ];

    let monitor_config = MonitorConfig::default();
    let _monitor = Arc::new(PerformanceMonitor::new(monitor_config));

    for (name, config) in configs {
        let config_clone = config.clone();
        let fast_sampler = FastSampler::new();
        let mut rng = Some(StdRng::seed_from_u64(12345));

        c.bench_function(&format!("sampling_{}", name), |b| {
            b.iter(|| {
                let logits_copy = logits.clone();
                fast_sampler.optimized_sample(
                    black_box(&logits_copy),
                    &config_clone,
                    None,
                    black_box(&mut rng),
                )
            })
        });
    }
}

criterion_group!(
    benches,
    benchmark_fast_sampler,
    benchmark_vec_pool,
    benchmark_vocab_sizes,
    benchmark_sampling_params
);
criterion_main!(benches);
