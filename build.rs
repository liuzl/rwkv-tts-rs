use std::env;
use std::path::PathBuf;

fn main() {
    // 获取项目根目录
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let onnx_dir = PathBuf::from(&manifest_dir).join("onnxruntime-win-x64-1.22.1");
    
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