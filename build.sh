
pip install -U huggingface_hub

huggingface-cli download cgisky/rwkv-tts --repo-type model --local-dir assets/model/ --resume-download --local-dir-use-symlinks False &

cargo build --release &

wait