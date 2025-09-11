pip install -U huggingface_hub

Start-Job { huggingface-cli download cgisky/rwkv-tts --repo-type model --local-dir assets/model/ --resume-download --local-dir-use-symlinks False }
Start-Job { cargo build --release }

Get-Job | Wait-Job | Receive-Job