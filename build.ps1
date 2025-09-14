# 安装huggingface_hub
Start-Job -ScriptBlock { pip install huggingface_hub }

# 下载模型
Start-Job -ScriptBlock { python download_models.py }

# 编译项目
Start-Job -ScriptBlock { cargo build --release --bin rwkvtts_server }

# 等待所有作业完成
Get-Job | Wait-Job