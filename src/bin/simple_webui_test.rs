//! 简化版WebUI RTF显示测试服务器
//! 用于测试WebUI中RTF显示功能，不依赖于复杂的TTS逻辑

use salvo::prelude::*;
use salvo::serve_static::StaticDir;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// TTS请求参数
#[derive(Debug, Deserialize)]
struct TtsRequest {
    text: String,
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

/// 生成模拟的WAV音频数据
fn generate_mock_wav() -> Vec<u8> {
    // 创建一个简单的WAV文件头和模拟音频数据
    let mut wav_data = Vec::new();

    // WAV文件头 (44字节)
    wav_data.extend_from_slice(b"RIFF");
    let file_size = 36 + 4000 * 2; // 16位音频，4000个样本
    wav_data.extend_from_slice(&(file_size as u32).to_le_bytes());
    wav_data.extend_from_slice(b"WAVE");

    // fmt chunk
    wav_data.extend_from_slice(b"fmt ");
    wav_data.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    wav_data.extend_from_slice(&1u16.to_le_bytes()); // audio format (PCM)
    wav_data.extend_from_slice(&1u16.to_le_bytes()); // num channels (mono)
    wav_data.extend_from_slice(&16000u32.to_le_bytes()); // sample rate
    wav_data.extend_from_slice(&(16000u32 * 2).to_le_bytes()); // byte rate
    wav_data.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav_data.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data chunk
    wav_data.extend_from_slice(b"data");
    wav_data.extend_from_slice(&((4000 * 2) as u32).to_le_bytes() as &[u8]);

    // 模拟音频数据 (4000个16位样本)
    for i in 0..4000 {
        let sample = ((i as f32 / 100.0).sin() * 10000.0) as i16;
        wav_data.extend_from_slice(&sample.to_le_bytes());
    }

    wav_data
}

/// 计算实时因子(RTF)
fn calculate_rtf(audio_duration: f64, processing_time: Duration) -> f64 {
    let processing_seconds = processing_time.as_secs_f64();
    if audio_duration > 0.0 {
        processing_seconds / audio_duration
    } else {
        0.0
    }
}

/// 处理TTS请求
#[handler]
async fn handle_tts(req: &mut Request, res: &mut Response) {
    // 解析请求
    let tts_request: TtsRequest = match req.parse_json().await {
        Ok(request) => request,
        Err(_) => {
            res.status_code(StatusCode::BAD_REQUEST);
            res.render(Json(TtsResponse {
                success: false,
                message: "JSON解析失败".to_string(),
                audio_base64: None,
                duration_ms: None,
                rtf: None,
            }));
            return;
        }
    };

    println!("收到TTS请求: text='{}'", tts_request.text);

    // 模拟处理时间 (100-300ms)
    let processing_time = Duration::from_millis(100 + (tts_request.text.len() as u64) % 200);
    tokio::time::sleep(processing_time).await;

    // 生成模拟音频数据
    let wav_data = generate_mock_wav();
    let base64_audio =
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &wav_data);

    // 计算RTF (假设音频时长为0.25秒)
    let audio_duration = 0.25; // 4000个样本 / 16000Hz = 0.25秒
    let rtf = calculate_rtf(audio_duration, processing_time);

    // 构建响应
    res.render(Json(TtsResponse {
        success: true,
        message: "TTS生成成功".to_string(),
        audio_base64: Some(base64_audio),
        duration_ms: Some(processing_time.as_millis() as u64),
        rtf: Some(rtf),
    }));

    println!(
        "TTS请求处理完成: 处理时间={}ms, RTF={:.3}",
        processing_time.as_millis(),
        rtf
    );
}

/// 提供Web UI界面
#[handler]
async fn handle_web_ui(_req: &mut Request, res: &mut Response) {
    let html = std::fs::read_to_string("static/index.html")
        .unwrap_or_else(|_| "<h1>Web UI not found</h1>".to_string());
    res.render(Text::Html(html));
}

#[tokio::main]
async fn main() {
    println!("启动WebUI RTF测试服务器...");

    // 创建路由
    let router = Router::new()
        .push(Router::with_path("/").get(handle_web_ui))
        .push(Router::with_path("/api/tts").post(handle_tts))
        .push(Router::with_path("/static/<**path>").get(StaticDir::new(["static"])));

    let service = Service::new(router);
    let acceptor = TcpListener::new("0.0.0.0:3000").bind().await;

    println!("服务器启动成功，监听端口: http://0.0.0.0:3000");
    println!("Web UI: http://localhost:3000");

    Server::new(acceptor).serve(service).await;
}
