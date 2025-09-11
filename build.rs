use std::env;
use std::path::PathBuf;

fn main() {
    // 首先尝试从环境变量获取 ONNX Runtime 库路径（CI 环境）
    if let Ok(ort_lib_location) = env::var("ORT_LIB_LOCATION") {
        let lib_dir = PathBuf::from(ort_lib_location);
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=onnxruntime");
        println!("cargo:rustc-env=ORT_LIB_LOCATION={}", lib_dir.display());
        println!("cargo:rerun-if-changed={}", lib_dir.display());
        return;
    }
    
    // 获取项目根目录
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    
    // 根据目标平台确定 ONNX Runtime 目录名
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| "unknown".to_string());
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| "unknown".to_string());
    
    let onnx_dir_name = match (target_os.as_str(), target_arch.as_str()) {
        ("windows", "x86_64") => "onnxruntime-win-x64-1.22.1",
        ("linux", "x86_64") => "onnxruntime-linux-x64-1.22.0",
        ("macos", "x86_64") => "onnxruntime-osx-x86_64-1.22.0",
        _ => {
            // 默认回退到 Windows 版本（本地开发环境）
            "onnxruntime-win-x64-1.22.1"
        }
    };
    
    // 尝试多个可能的路径
    let primary = PathBuf::from(&manifest_dir).join("第三方库源码").join(onnx_dir_name);
    let fallback = PathBuf::from(&manifest_dir).join(onnx_dir_name);
    let onnx_dir = if primary.exists() { primary } else { fallback };
    
    // 设置ONNX Runtime库路径
    let lib_dir = onnx_dir.join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    
    // 链接ONNX Runtime库
    println!("cargo:rustc-link-lib=onnxruntime");
    
    // 设置环境变量供运行时使用
    println!("cargo:rustc-env=ORT_LIB_LOCATION={}", lib_dir.display());
    
    // 如果库文件发生变化，重新构建
    println!("cargo:rerun-if-changed={}", lib_dir.display());
    
    // 设置DLL路径（Windows）
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-env=PATH={};{}", lib_dir.display(), env::var("PATH").unwrap_or_default());
    }
}