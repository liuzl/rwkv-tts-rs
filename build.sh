
#!/bin/bash

# 安装huggingface_hub
pip install huggingface_hub

# 下载模型（后台运行）
python download_models.py &

# 编译项目
cargo build --release --bin rwkvtts_server

# 等待所有后台进程完成
wait