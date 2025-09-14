//! RWKV TTS HTTP Server
//! 基于Salvo框架的高并发TTS服务器，提供REST API和Web UI界面

use anyhow::Result;
use base64::Engine;
use clap::{Arg, Command};
use rust_embed::RustEmbed;
use salvo::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;
// 添加模型下载相关导入
// use hf_hub::api::tokio::Api; // Now using ApiBuilder::from_env().build()
use tokio::fs;

/// 嵌入的静态资源
#[derive(RustEmbed)]
#[folder = "static/"]
struct Assets;

// 移除未使用的导入
// Logger功能暂时禁用

use rwkv_tts_rs::lightweight_tts_pipeline::{LightweightTtsPipeline, LightweightTtsPipelineArgs};
use web_rwkv::runtime::model::Quant;

/// TTS请求参数
#[derive(Debug, Deserialize)]
struct TtsRequest {
    text: String,
    #[allow(dead_code)]
    speaker: Option<String>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    #[allow(dead_code)]
    speed: Option<f32>,
    #[allow(dead_code)]
    zero_shot: Option<bool>,
    ref_audio_path: Option<String>,
    seed: Option<u64>,
    // 添加新的高级选项
    age: Option<String>,
    gender: Option<String>,
    emotion: Option<String>,
    pitch: Option<String>,
    // 添加提示词字段
    prompt_text: Option<String>,
}

/// TTS响应
#[derive(Debug, Serialize)]
struct TtsResponse {
    success: bool,
    message: String,
    audio_base64: Option<String>,
    duration_ms: Option<u64>,
    rtf: Option<f64>,
}

/// 错误响应
#[derive(Debug, Serialize)]
struct ErrorResponse {
    success: bool,
    error: String,
}

/// 服务器状态
#[derive(Debug, Serialize)]
struct ServerStatus {
    status: String,
    version: String,
    uptime_seconds: u64,
    total_requests: u64,
}

/// 将f32音频样本转换为WAV格式的字节数据
fn convert_samples_to_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let mut wav_data = Vec::new();

    // 分析音频数据范围以确定合适的缩放因子
    let max_abs = samples.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale_factor = if max_abs > 0.0 {
        // 如果最大值超过1.0，需要归一化；如果小于1.0，需要放大
        if max_abs > 1.0 {
            1.0 / max_abs
        } else {
            // 对于小幅度信号，适度放大但不超过安全范围
            (0.8 / max_abs).min(10.0)
        }
    } else {
        1.0
    };

    info!(
        "音频数据分析: max_abs={:.6}, scale_factor={:.6}",
        max_abs, scale_factor
    );

    // WAV文件头
    wav_data.extend_from_slice(b"RIFF");
    let file_size = 36 + samples.len() * 2; // 16位音频
    wav_data.extend_from_slice(&(file_size as u32).to_le_bytes());
    wav_data.extend_from_slice(b"WAVE");

    // fmt chunk
    wav_data.extend_from_slice(b"fmt ");
    wav_data.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    wav_data.extend_from_slice(&1u16.to_le_bytes()); // audio format (PCM)
    wav_data.extend_from_slice(&1u16.to_le_bytes()); // num channels (mono)
    wav_data.extend_from_slice(&sample_rate.to_le_bytes()); // sample rate
    wav_data.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    wav_data.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav_data.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data chunk
    wav_data.extend_from_slice(b"data");
    wav_data.extend_from_slice(&(samples.len() * 2).to_le_bytes() as &[u8]);

    // 音频数据 (转换f32到i16，应用动态缩放)
    for sample in samples {
        let scaled_sample = sample * scale_factor;
        let sample_i16 = (scaled_sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        wav_data.extend_from_slice(&sample_i16.to_le_bytes());
    }

    wav_data
}

/// 计算实时因子(RTF)
fn calculate_rtf(audio_data: &[f32], processing_time: std::time::Duration) -> f64 {
    let audio_duration = audio_data.len() as f64 / 16000.0; // 假设16kHz采样率
    let processing_seconds = processing_time.as_secs_f64();
    if audio_duration > 0.0 {
        processing_seconds / audio_duration
    } else {
        0.0
    }
}

/// 应用状态
#[derive(Debug, Clone)]
struct AppState {
    #[allow(dead_code)]
    start_time: std::time::Instant,
    #[allow(dead_code)]
    model_path: String,
    #[allow(dead_code)]
    vocab_path: String,
    tts_pipeline: Arc<LightweightTtsPipeline>,
}

/// 全局应用状态
static GLOBAL_APP_STATE: std::sync::OnceLock<AppState> = std::sync::OnceLock::new();

/// 初始化全局应用状态
fn init_global_app_state(app_state: AppState) {
    GLOBAL_APP_STATE.set(app_state).expect("应用状态已初始化");
}

/// 获取全局应用状态
fn get_global_app_state() -> AppState {
    GLOBAL_APP_STATE.get().expect("应用状态未初始化").clone()
}

/// 处理TTS请求（支持文件上传）
#[handler]
async fn handle_tts(req: &mut Request, res: &mut Response) -> Result<(), StatusError> {
    // 检查是否是multipart请求（文件上传）
    if req
        .content_type()
        .map(|ct| ct.type_() == "multipart")
        .unwrap_or(false)
    {
        // 处理multipart表单数据（包含文件上传）
        handle_tts_with_file_upload(req, res).await
    } else {
        // 处理普通的JSON请求
        handle_tts_json(req, res).await
    }
}

/// 处理带文件上传的TTS请求
async fn handle_tts_with_file_upload(
    req: &mut Request,
    res: &mut Response,
) -> Result<(), StatusError> {
    let total_start = std::time::Instant::now();

    // 解析multipart表单数据
    let parse_start = std::time::Instant::now();
    req.parse_form::<()>().await.map_err(|e| {
        error!("表单数据解析失败: {}", e);
        StatusError::bad_request()
    })?;
    let parse_time = parse_start.elapsed();

    // 提取文本和其他参数
    let text: String = req.form("text").await.unwrap_or_default();
    let _speaker: String = req.form("speaker").await.unwrap_or_default();
    let temperature: f32 = req
        .form("temperature")
        .await
        .unwrap_or("1.0".to_string())
        .parse()
        .unwrap_or(1.0);
    let top_p: f32 = req
        .form("top_p")
        .await
        .unwrap_or("0.3".to_string())
        .parse()
        .unwrap_or(0.3);
    let _speed: f32 = req
        .form("speed")
        .await
        .unwrap_or("1.0".to_string())
        .parse()
        .unwrap_or(1.0);
    let zero_shot: bool = req
        .form("zero_shot")
        .await
        .unwrap_or("false".to_string())
        .parse()
        .unwrap_or(false);
    let ref_audio_path: String = req.form("ref_audio_path").await.unwrap_or_default();
    let seed_str: String = req.form("seed").await.unwrap_or_default();
    let seed: Option<u64> = if seed_str.is_empty() {
        None
    } else {
        seed_str.parse().ok()
    };
    let age: String = req.form("age").await.unwrap_or("youth-adult".to_string());
    let gender: String = req.form("gender").await.unwrap_or("male".to_string());
    let emotion: String = req.form("emotion").await.unwrap_or("NEUTRAL".to_string());
    let pitch: String = req
        .form("pitch")
        .await
        .unwrap_or("medium_pitch".to_string());
    let prompt_text: String = req.form("prompt_text").await.unwrap_or_default();

    info!(
        "🎯 收到TTS请求(带文件上传): text='{}', ref_audio_path='{:?}'",
        text, ref_audio_path
    );
    info!(
        "  ⏱️  请求解析耗时: {:.2}ms",
        parse_time.as_secs_f64() * 1000.0
    );

    // 处理文件上传
    let uploaded_file_path = req
        .file("refAudioFile")
        .await
        .map(|_file| "uploaded_file_path".to_string());
    if uploaded_file_path.is_some() {
        info!("  📁 文件上传处理完成");
    }

    // 确定最终使用的参考音频路径
    let final_ref_audio_path = if let Some(ref uploaded_path) = uploaded_file_path {
        uploaded_path.clone()
    } else {
        ref_audio_path
    };

    // 获取应用状态和创建参数
    let setup_start = std::time::Instant::now();
    let app_state = get_global_app_state();

    let pipeline_args = LightweightTtsPipelineArgs {
        text: text.clone(),
        ref_audio_path: final_ref_audio_path.clone(),
        zero_shot: !final_ref_audio_path.is_empty() || zero_shot,
        temperature,
        top_p,
        top_k: 100,
        max_tokens: 8000,
        seed,
        // 添加新的高级选项并进行类型转换
        age,
        gender,
        emotion,
        // 音调和语速需要转换为数值
        pitch: match pitch.as_str() {
            "low_pitch" => 150.0,
            "medium_pitch" => 200.0,
            "high_pitch" => 250.0,
            "very_high_pitch" => 300.0,
            _ => 200.0, // 默认中音调
        },
        speed: 4.2, // 默认语速
        // 添加提示词
        prompt_text,
        ..Default::default()
    };
    let setup_time = setup_start.elapsed();
    info!(
        "  ⏱️  参数设置耗时: {:.2}ms",
        setup_time.as_secs_f64() * 1000.0
    );

    // TTS生成（主要处理时间）
    let tts_start = std::time::Instant::now();
    let audio_data = match app_state.tts_pipeline.generate_speech(&pipeline_args).await {
        Ok(data) => data,
        Err(e) => {
            error!("生成TTS音频失败: {}", e);
            res.status_code(StatusCode::INTERNAL_SERVER_ERROR);
            res.render(Json(ErrorResponse {
                success: false,
                error: format!("生成TTS音频失败: {}", e),
            }));
            return Ok(());
        }
    };
    let tts_time = tts_start.elapsed();
    info!(
        "  ⏱️  TTS生成耗时: {:.2}ms",
        tts_time.as_secs_f64() * 1000.0
    );

    // 音频格式转换
    let convert_start = std::time::Instant::now();
    let wav_data = convert_samples_to_wav(&audio_data, 16000);
    let convert_time = convert_start.elapsed();
    info!(
        "  ⏱️  WAV转换耗时: {:.2}ms",
        convert_time.as_secs_f64() * 1000.0
    );

    // Base64编码
    let encode_start = std::time::Instant::now();
    let base64_audio = base64::engine::general_purpose::STANDARD.encode(&wav_data);
    let encode_time = encode_start.elapsed();
    info!(
        "  ⏱️  Base64编码耗时: {:.2}ms",
        encode_time.as_secs_f64() * 1000.0
    );

    // 计算总体性能指标
    let total_time = total_start.elapsed();
    let rtf = calculate_rtf(&audio_data, total_time);
    let audio_duration = audio_data.len() as f64 / 16000.0;

    info!("📊 TTS请求完成统计:");
    info!("  ⏱️  总耗时: {:.2}ms", total_time.as_secs_f64() * 1000.0);
    info!("  🎵 音频时长: {:.2}s", audio_duration);
    info!("  📈 RTF: {:.3}", rtf);
    info!("  📦 音频样本数: {}", audio_data.len());
    info!("  💾 WAV文件大小: {} bytes", wav_data.len());
    info!("  📝 Base64大小: {} chars", base64_audio.len());

    // 性能分析和优化建议
    let tts_percentage = tts_time.as_secs_f64() / total_time.as_secs_f64() * 100.0;
    let convert_percentage = convert_time.as_secs_f64() / total_time.as_secs_f64() * 100.0;
    let encode_percentage = encode_time.as_secs_f64() / total_time.as_secs_f64() * 100.0;

    info!("🔍 性能分析:");
    info!("  - TTS生成: {:.1}%", tts_percentage);
    info!("  - WAV转换: {:.1}%", convert_percentage);
    info!("  - Base64编码: {:.1}%", encode_percentage);
    info!(
        "  - 其他开销: {:.1}%",
        100.0 - tts_percentage - convert_percentage - encode_percentage
    );

    if rtf > 0.3 {
        info!("⚠️  服务器性能提示: RTF > 0.3，建议优化:");
        if tts_percentage > 90.0 {
            info!(
                "   - TTS生成占用{:.1}%时间，主要瓶颈在模型推理",
                tts_percentage
            );
        }
        if convert_percentage > 5.0 {
            info!(
                "   - WAV转换占用{:.1}%时间，考虑优化音频处理",
                convert_percentage
            );
        }
        if encode_percentage > 5.0 {
            info!(
                "   - Base64编码占用{:.1}%时间，考虑流式传输",
                encode_percentage
            );
        }
    }

    // 构建响应
    let response_start = std::time::Instant::now();
    res.render(Json(TtsResponse {
        success: true,
        message: "TTS生成成功".to_string(),
        audio_base64: Some(base64_audio),
        duration_ms: Some(total_time.as_millis() as u64),
        rtf: Some(rtf),
    }));
    let response_time = response_start.elapsed();
    info!(
        "  ⏱️  响应构建耗时: {:.2}ms",
        response_time.as_secs_f64() * 1000.0
    );

    // 清理上传的临时文件
    if let Some(uploaded_path) = uploaded_file_path {
        tokio::spawn(async move {
            // 等待一段时间后删除临时文件
            tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
            if let Err(e) = tokio::fs::remove_file(&uploaded_path).await {
                warn!("删除临时文件失败: {}: {}", uploaded_path, e);
            } else {
                info!("临时文件已清理: {}", uploaded_path);
            }
        });
    }

    Ok(())
}

/// 处理JSON格式的TTS请求（原有逻辑）
async fn handle_tts_json(req: &mut Request, res: &mut Response) -> Result<(), StatusError> {
    let total_start = std::time::Instant::now();

    // 1. 解析JSON请求
    let parse_start = std::time::Instant::now();
    let tts_request: TtsRequest = match req.parse_json().await {
        Ok(request) => request,
        Err(e) => {
            error!("JSON解析失败: {}", e);
            res.status_code(StatusCode::BAD_REQUEST);
            res.render(Json(ErrorResponse {
                success: false,
                error: format!("JSON解析失败: {}", e),
            }));
            return Ok(());
        }
    };
    let parse_time = parse_start.elapsed();

    info!(
        "🎯 收到TTS请求: text='{}', ref_audio_path='{:?}'",
        tts_request.text, tts_request.ref_audio_path
    );
    info!(
        "  ⏱️  请求解析耗时: {:.2}ms",
        parse_time.as_secs_f64() * 1000.0
    );

    // 2. 获取应用状态和创建参数
    let setup_start = std::time::Instant::now();
    let app_state = get_global_app_state();

    let pipeline_args = LightweightTtsPipelineArgs {
        text: tts_request.text.clone(),
        ref_audio_path: tts_request.ref_audio_path.clone().unwrap_or_default(),
        zero_shot: tts_request.ref_audio_path.is_some(),
        temperature: tts_request.temperature.unwrap_or(1.0),
        top_p: tts_request.top_p.unwrap_or(0.8),
        top_k: 100,
        max_tokens: 8000,
        seed: tts_request.seed,
        // 添加新的高级选项并进行类型转换
        age: tts_request.age.unwrap_or("youth-adult".to_string()),
        gender: tts_request.gender.unwrap_or("male".to_string()),
        emotion: tts_request.emotion.unwrap_or("NEUTRAL".to_string()),
        // 音调和语速需要转换为数值
        pitch: match tts_request.pitch.as_deref() {
            Some("low_pitch") => 150.0,
            Some("medium_pitch") => 200.0,
            Some("high_pitch") => 250.0,
            Some("very_high_pitch") => 300.0,
            _ => 200.0, // 默认中音调
        },
        speed: 4.2, // 默认语速
        // 添加提示词
        prompt_text: tts_request.prompt_text.unwrap_or_default(),
        ..Default::default()
    };
    let setup_time = setup_start.elapsed();
    info!(
        "  ⏱️  参数设置耗时: {:.2}ms",
        setup_time.as_secs_f64() * 1000.0
    );

    // 3. TTS生成（主要处理时间）
    let tts_start = std::time::Instant::now();
    let audio_data = match app_state.tts_pipeline.generate_speech(&pipeline_args).await {
        Ok(data) => data,
        Err(e) => {
            error!("生成TTS音频失败: {}", e);
            res.status_code(StatusCode::INTERNAL_SERVER_ERROR);
            res.render(Json(ErrorResponse {
                success: false,
                error: format!("生成TTS音频失败: {}", e),
            }));
            return Ok(());
        }
    };
    let tts_time = tts_start.elapsed();
    info!(
        "  ⏱️  TTS生成耗时: {:.2}ms",
        tts_time.as_secs_f64() * 1000.0
    );

    // 4. 音频格式转换
    let convert_start = std::time::Instant::now();
    let wav_data = convert_samples_to_wav(&audio_data, 16000);
    let convert_time = convert_start.elapsed();
    info!(
        "  ⏱️  WAV转换耗时: {:.2}ms",
        convert_time.as_secs_f64() * 1000.0
    );

    // 5. Base64编码
    let encode_start = std::time::Instant::now();
    let base64_audio = base64::engine::general_purpose::STANDARD.encode(&wav_data);
    let encode_time = encode_start.elapsed();
    info!(
        "  ⏱️  Base64编码耗时: {:.2}ms",
        encode_time.as_secs_f64() * 1000.0
    );

    // 6. 计算总体性能指标
    let total_time = total_start.elapsed();
    let rtf = calculate_rtf(&audio_data, total_time);
    let audio_duration = audio_data.len() as f64 / 16000.0;

    info!("📊 TTS请求完成统计:");
    info!("  ⏱️  总耗时: {:.2}ms", total_time.as_secs_f64() * 1000.0);
    info!("  🎵 音频时长: {:.2}s", audio_duration);
    info!("  📈 RTF: {:.3}", rtf);
    info!("  📦 音频样本数: {}", audio_data.len());
    info!("  💾 WAV文件大小: {} bytes", wav_data.len());
    info!("  📝 Base64大小: {} chars", base64_audio.len());

    // 性能分析和优化建议
    let tts_percentage = tts_time.as_secs_f64() / total_time.as_secs_f64() * 100.0;
    let convert_percentage = convert_time.as_secs_f64() / total_time.as_secs_f64() * 100.0;
    let encode_percentage = encode_time.as_secs_f64() / total_time.as_secs_f64() * 100.0;

    info!("🔍 性能分析:");
    info!("  - TTS生成: {:.1}%", tts_percentage);
    info!("  - WAV转换: {:.1}%", convert_percentage);
    info!("  - Base64编码: {:.1}%", encode_percentage);
    info!(
        "  - 其他开销: {:.1}%",
        100.0 - tts_percentage - convert_percentage - encode_percentage
    );

    if rtf > 0.3 {
        info!("⚠️  服务器性能提示: RTF > 0.3，建议优化:");
        if tts_percentage > 90.0 {
            info!(
                "   - TTS生成占用{:.1}%时间，主要瓶颈在模型推理",
                tts_percentage
            );
        }
        if convert_percentage > 5.0 {
            info!(
                "   - WAV转换占用{:.1}%时间，考虑优化音频处理",
                convert_percentage
            );
        }
        if encode_percentage > 5.0 {
            info!(
                "   - Base64编码占用{:.1}%时间，考虑流式传输",
                encode_percentage
            );
        }
    }

    // 7. 构建响应
    let response_start = std::time::Instant::now();
    res.render(Json(TtsResponse {
        success: true,
        message: "TTS生成成功".to_string(),
        audio_base64: Some(base64_audio),
        duration_ms: Some(total_time.as_millis() as u64),
        rtf: Some(rtf),
    }));
    let response_time = response_start.elapsed();
    info!(
        "  ⏱️  响应构建耗时: {:.2}ms",
        response_time.as_secs_f64() * 1000.0
    );

    Ok(())
}

/// 获取服务器状态
#[handler]
async fn handle_status(res: &mut Response) {
    let status = ServerStatus {
        status: "running".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: 0, // 简化处理
        total_requests: 0, // 简化处理
    };

    res.render(Json(status));
}

/// 提供Web UI界面
#[handler]
async fn handle_web_ui(_req: &mut Request, res: &mut Response) {
    match Assets::get("index.html") {
        Some(content) => {
            let html = std::str::from_utf8(content.data.as_ref())
                .unwrap_or("<h1>Error reading embedded HTML</h1>");
            res.render(Text::Html(html.to_string()));
        }
        None => {
            res.render(Text::Html("<h1>Web UI not found</h1>".to_string()));
        }
    }
}

/// 处理嵌入的静态文件
#[handler]
async fn handle_static_files(req: &mut Request, res: &mut Response) {
    let path = req.param::<String>("**path").unwrap_or_default();

    match Assets::get(&path) {
        Some(content) => {
            // 根据文件扩展名设置Content-Type
            let content_type = match path.split('.').next_back() {
                Some("html") => "text/html",
                Some("css") => "text/css",
                Some("js") => "application/javascript",
                Some("png") => "image/png",
                Some("jpg") | Some("jpeg") => "image/jpeg",
                Some("gif") => "image/gif",
                Some("svg") => "image/svg+xml",
                Some("ico") => "image/x-icon",
                _ => "application/octet-stream",
            };

            res.headers_mut()
                .insert("content-type", content_type.parse().unwrap());

            // 复制数据以避免生命周期问题
            let data = content.data.to_vec();
            res.write_body(data).unwrap();
        }
        None => {
            res.status_code(StatusCode::NOT_FOUND);
            res.render(Text::Plain("File not found"));
        }
    }
}

/// 健康检查
#[handler]
async fn handle_health(_req: &mut Request, res: &mut Response) {
    res.render(Json(serde_json::json!({
        "status": "healthy",
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    })));
}

/// CORS中间件
#[handler]
async fn cors_handler(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
    ctrl: &mut FlowCtrl,
) {
    res.headers_mut()
        .insert("Access-Control-Allow-Origin", "*".parse().unwrap());
    res.headers_mut().insert(
        "Access-Control-Allow-Methods",
        "GET, POST, OPTIONS".parse().unwrap(),
    );
    res.headers_mut().insert(
        "Access-Control-Allow-Headers",
        "Content-Type, Authorization".parse().unwrap(),
    );
    ctrl.call_next(req, depot, res).await;
}

/// 中间件：请求日志
#[handler]
async fn request_logger(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
    ctrl: &mut FlowCtrl,
) {
    let start = std::time::Instant::now();
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    ctrl.call_next(req, depot, res).await;

    let duration = start.elapsed();
    let status = res.status_code.unwrap_or(StatusCode::OK);

    info!("{} {} {} - {:?}", method, path, status.as_u16(), duration);
}

/// 解析量化类型字符串
fn parse_quant_type(s: &str) -> Result<Quant> {
    let quant_type = match s.to_lowercase().as_str() {
        "none" => Quant::None,
        "int8" => Quant::Int8,
        "nf4" => {
            warn!("NF4量化可能在某些GPU上存在兼容性问题，建议使用Int8量化");
            Quant::NF4
        }
        "sf4" => {
            warn!("SF4量化可能在某些GPU上存在兼容性问题，建议使用Int8量化");
            Quant::SF4
        }
        _ => {
            return Err(anyhow::anyhow!(
                "不支持的量化类型: {}. 支持的类型: none, int8, nf4, sf4",
                s
            ))
        }
    };

    // 验证量化类型兼容性
    if matches!(quant_type, Quant::NF4 | Quant::SF4) {
        info!(
            "使用实验性量化类型: {:?}，如遇到问题请改用 int8",
            quant_type
        );
    }

    Ok(quant_type)
}

/// 创建量化配置
fn create_quant_config(quant_layers: usize, quant_type: Quant) -> Option<HashMap<usize, Quant>> {
    if quant_layers == 0 || matches!(quant_type, Quant::None) {
        return None;
    }

    let mut config = HashMap::new();
    for layer in 0..quant_layers {
        config.insert(layer, quant_type);
    }
    Some(config)
}

/// 从Hugging Face下载模型文件
async fn download_models_from_hf() -> Result<()> {
    info!("开始从Hugging Face下载模型文件...");

    // 创建模型目录
    let model_dir = PathBuf::from("assets/model");
    fs::create_dir_all(&model_dir).await?;

    // 定义多个镜像地址
    let mirrors = [
        "https://huggingface.co", // 官方地址，优先使用
        "https://hf-mirror.com",  // 中国镜像，备用
    ];

    // 需要下载的文件列表
    let files_to_download = vec![
        "rwkvtts-Int8_22.prefab",
        "tokenizer.json",
        "BiCodecTokenize_static_qdq.onnx",
        "wav2vec2-large-xlsr-53_static_qdq.onnx",
        "BiCodecDetokenize_static_qdq.onnx",
    ];

    for filename in files_to_download {
        let local_path = model_dir.join(filename);

        // 如果文件已存在，跳过下载
        if local_path.exists() {
            info!("文件已存在，跳过下载: {}", filename);
            continue;
        }

        info!("正在下载: {}", filename);
        let mut download_success = false;
        let mut last_error = None;

        // 尝试每个镜像
        for (index, mirror_url) in mirrors.iter().enumerate() {
            info!("尝试镜像 {}/{}: {}", index + 1, mirrors.len(), mirror_url);

            // 清除现有的HF_ENDPOINT环境变量
            std::env::remove_var("HF_ENDPOINT");

            // 设置环境变量HF_ENDPOINT来配置镜像
            std::env::set_var("HF_ENDPOINT", mirror_url);
            info!("设置环境变量 HF_ENDPOINT={}", mirror_url);

            // 添加小延迟确保环境变量生效
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

            // 验证环境变量是否正确设置
            if let Ok(current_endpoint) = std::env::var("HF_ENDPOINT") {
                info!("HF_ENDPOINT已设置为: {}", current_endpoint);
            } else {
                warn!("设置HF_ENDPOINT失败: {}", mirror_url);
                continue;
            }

            // 为每个镜像创建新的API客户端
            let api = match hf_hub::api::tokio::ApiBuilder::from_env().build() {
                Ok(api) => api,
                Err(e) => {
                    warn!("初始化API客户端失败 ({}): {}", mirror_url, e);
                    // 失败时清理环境变量
                    std::env::remove_var("HF_ENDPOINT");
                    last_error = Some(e.into());
                    continue;
                }
            };

            let repo = api.model("cgisky/rwkv-tts".to_string());

            // 设置超时时间
            let download_future = repo.get(filename);
            let timeout_duration = std::time::Duration::from_secs(300); // 5分钟超时

            match tokio::time::timeout(timeout_duration, download_future).await {
                Ok(Ok(file_path)) => match fs::copy(&file_path, &local_path).await {
                    Ok(_) => {
                        let file_size = fs::metadata(&local_path).await?.len();
                        info!(
                            "下载完成: {} ({} bytes) - 使用镜像: {}",
                            filename, file_size, mirror_url
                        );
                        download_success = true;
                        break;
                    }
                    Err(e) => {
                        warn!("文件复制失败 ({}): {}", mirror_url, e);
                        last_error = Some(e.into());
                    }
                },
                Ok(Err(e)) => {
                    warn!("下载失败 ({}): {}", mirror_url, e);
                    last_error = Some(e.into());
                }
                Err(_) => {
                    warn!(
                        "下载超时 ({}): 超过{}秒",
                        mirror_url,
                        timeout_duration.as_secs()
                    );
                    last_error = Some(anyhow::anyhow!("下载超时"));
                }
            }
        }

        if !download_success {
            let error_msg = match last_error {
                Some(e) => format!("所有镜像都失败了，最后一个错误: {}", e),
                None => "所有镜像都失败了，未知错误".to_string(),
            };
            error!("下载文件失败: {} - {}", filename, error_msg);
            return Err(anyhow::anyhow!(
                "下载文件失败: {} - {}",
                filename,
                error_msg
            ));
        }
    }

    // 清理环境变量
    std::env::remove_var("HF_ENDPOINT");
    info!("所有模型文件下载完成！");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // 解析命令行参数
    let matches = Command::new("RWKV TTS Server")
        .version(env!("CARGO_PKG_VERSION"))
        .about("基于RWKV的高性能TTS服务器")
        .arg(
            Arg::new("quant-layers")
                .long("quant-layers")
                .value_name("NUMBER")
                .help("指定量化层数")
                .default_value("24"),
        )
        .arg(
            Arg::new("quant-type")
                .long("quant-type")
                .value_name("TYPE")
                .help("指定量化类型 (none, int8, nf4, sf4)。推荐使用 int8 以获得最佳稳定性")
                .default_value("int8"),
        )
        .arg(
            Arg::new("model-path")
                .long("model-path")
                .value_name("PATH")
                .help("模型文件路径")
                .default_value("assets/model/rwkvtts-Int8_22.prefab"),
        )
        .arg(
            Arg::new("vocab-path")
                .long("vocab-path")
                .value_name("PATH")
                .help("词汇表文件路径")
                .default_value("assets/model/tokenizer.json"),
        )
        .arg(
            Arg::new("batch-size")
                .long("batch-size")
                .value_name("NUMBER")
                .help("批处理最大大小")
                .default_value("10"),
        )
        .arg(
            Arg::new("batch-timeout")
                .long("batch-timeout")
                .value_name("MS")
                .help("批处理超时时间（毫秒）")
                .default_value("20"),
        )
        .arg(
            Arg::new("inference-timeout")
                .long("inference-timeout")
                .value_name("MS")
                .help("推理超时时间（毫秒）")
                .default_value("120000"),
        )
        .arg(
            Arg::new("port")
                .long("port")
                .value_name("PORT")
                .help("服务器监听端口")
                .default_value("3000"),
        )
        .get_matches();

    // 初始化日志，过滤掉ort和web-rwkv的调试输出
    let filter = EnvFilter::new("info")
        .add_directive("ort=warn".parse().unwrap())
        .add_directive("web_rwkv=warn".parse().unwrap())
        .add_directive("naga=warn".parse().unwrap())
        .add_directive("wgpu=warn".parse().unwrap());

    tracing_subscriber::fmt().with_env_filter(filter).init();

    info!("启动RWKV TTS HTTP服务器...");

    // 获取命令行参数
    let model_path = matches.get_one::<String>("model-path").unwrap();
    let vocab_path = matches.get_one::<String>("vocab-path").unwrap();
    let quant_layers: usize = matches
        .get_one::<String>("quant-layers")
        .unwrap()
        .parse()
        .map_err(|e| anyhow::anyhow!("无效的量化层数: {}", e))?;
    let quant_type_str = matches.get_one::<String>("quant-type").unwrap();
    let quant_type = parse_quant_type(quant_type_str)?;

    // 创建量化配置
    let quant_config = create_quant_config(quant_layers, quant_type);

    // 打印量化配置信息
    match &quant_config {
        Some(config) => {
            info!("🔧 量化配置: {} 层使用 {:?} 量化", config.len(), quant_type);
        }
        None => {
            info!("🔧 未启用量化");
        }
    }

    // 验证模型文件路径，如果不存在则尝试下载
    let model_missing = !Path::new(model_path).exists();
    let vocab_missing = !Path::new(vocab_path).exists();
    let onnx_files = [
        "assets/model/BiCodecTokenize_static_qdq.onnx",
        "assets/model/wav2vec2-large-xlsr-53_static_qdq.onnx",
        "assets/model/BiCodecDetokenize_static_qdq.onnx",
    ];
    let onnx_missing = onnx_files.iter().any(|path| !Path::new(path).exists());

    if model_missing || vocab_missing || onnx_missing {
        warn!("检测到缺失的模型文件，尝试从Hugging Face下载...");
        if model_missing {
            warn!("模型文件不存在: {}", model_path);
        }
        if vocab_missing {
            warn!("词汇表文件不存在: {}", vocab_path);
        }
        if onnx_missing {
            warn!("ONNX模型文件缺失");
        }

        // 尝试下载模型
        match download_models_from_hf().await {
            Ok(()) => {
                info!("模型下载成功，继续启动服务器...");
            }
            Err(e) => {
                error!("模型下载失败: {}", e);
                return Err(anyhow::anyhow!("无法获取必要的模型文件: {}", e));
            }
        }

        // 再次验证文件是否存在
        if !Path::new(model_path).exists() {
            return Err(anyhow::anyhow!("下载后模型文件仍不存在: {}", model_path));
        }
        if !Path::new(vocab_path).exists() {
            return Err(anyhow::anyhow!("下载后词汇表文件仍不存在: {}", vocab_path));
        }
        for onnx_path in &onnx_files {
            if !Path::new(onnx_path).exists() {
                return Err(anyhow::anyhow!("下载后ONNX文件仍不存在: {}", onnx_path));
            }
        }
    }

    info!("模型路径验证成功: {}", model_path);
    info!("词汇表路径验证成功: {}", vocab_path);

    // 架构优化：移除全局RwkvSampler管理器，避免与动态批处理管理器的重复初始化
    // 动态批处理管理器已经内置了共享Runtime架构，无需额外的全局管理器

    // 从命令行参数获取批处理配置
    let batch_size: usize = matches
        .get_one::<String>("batch-size")
        .unwrap()
        .parse()
        .expect("无效的批处理大小");

    info!("初始化ONNX会话池（使用量化模型）...");
    rwkv_tts_rs::onnx_session_pool::init_global_onnx_manager(
        "assets/model/BiCodecTokenize_static_qdq.onnx",
        "assets/model/wav2vec2-large-xlsr-53_static_qdq.onnx",
        "assets/model/BiCodecDetokenize_static_qdq.onnx",
        Some(4),
    )
    .map_err(|e| anyhow::anyhow!("初始化ONNX管理器失败: {}", e))?;

    info!("初始化动态批处理管理器...");
    // 获取批处理超时配置
    let batch_timeout: u64 = matches
        .get_one::<String>("batch-timeout")
        .unwrap()
        .parse()
        .expect("无效的批处理超时时间");

    // 获取推理超时配置
    let inference_timeout: u64 = matches
        .get_one::<String>("inference-timeout")
        .unwrap()
        .parse()
        .expect("无效的推理超时时间");

    // 自动计算最大并发批次数
    let max_concurrent_batches: usize = if batch_size <= 10 {
        10
    } else {
        std::cmp::max(8, batch_size / 10)
    };

    // 创建动态批处理配置
    let dynamic_batch_config = rwkv_tts_rs::dynamic_batch_manager::DynamicBatchConfig {
        min_batch_size: 1,
        max_batch_size: batch_size,              // 可配置的批处理大小
        collect_timeout_ms: batch_timeout,       // 可配置的超时时间
        inference_timeout_ms: inference_timeout, // 可配置的推理超时时间
        max_concurrent_batches,                  // 可配置的并发批次数
        semaphore_permits: (max_concurrent_batches * 3 / 4).clamp(1, 8), // 信号量许可数量略小于并发数
    };
    info!(
        "动态批处理配置: 最大大小={}, 收集超时={}ms, 推理超时={}ms, 最大并发批次={}（自动计算）",
        batch_size, batch_timeout, inference_timeout, max_concurrent_batches
    );
    rwkv_tts_rs::dynamic_batch_manager::init_global_dynamic_batch_manager(
        model_path,
        vocab_path,
        dynamic_batch_config,
        quant_config,
    )
    .await
    .map_err(|e| anyhow::anyhow!("初始化动态批处理管理器失败: {}", e))?;

    // 创建轻量级TTS流水线
    let tts_pipeline = Arc::new(LightweightTtsPipeline::new());

    let app_state = AppState {
        start_time: std::time::Instant::now(),
        model_path: model_path.to_string(),
        vocab_path: vocab_path.to_string(),
        tts_pipeline,
    };

    // 初始化全局应用状态
    init_global_app_state(app_state);

    // 创建路由
    let router = Router::new()
        .hoop(cors_handler)
        .push(Router::with_path("/").get(handle_web_ui))
        .push(Router::with_path("/api/status").get(handle_status))
        .push(Router::with_path("/api/tts").post(handle_tts))
        .push(Router::with_path("/api/health").get(handle_health))
        .push(Router::with_path("/static/<**path>").get(handle_static_files));

    // 注意：现在静态文件已嵌入到二进制文件中，不再依赖外部static目录

    // 创建服务
    let service = Service::new(router).hoop(request_logger);

    let port: u16 = matches
        .get_one::<String>("port")
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(3000);

    let acceptor = TcpListener::new(format!("0.0.0.0:{port}")).bind().await;

    info!("服务器启动成功，监听端口: http://0.0.0.0:{}", port);
    info!("Web UI: http://localhost:{}", port);
    info!("API文档: http://localhost:{}/api/status", port);
    info!("TTS服务已就绪，使用预加载的全局模型实例，支持高并发访问");

    Server::new(acceptor).serve(service).await;

    Ok(())
}
