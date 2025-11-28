pip install uv
uv venv --python 3.11
source .venv/bin/activate

uv pip install -r requirements.txt
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
uv pip install flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
# uv pip install flash-attn==2.8.3 --no-build-isolation
uv pip install jupyter
uv pip uninstall pynvml
uv pip install nvidia-ml-py


mkdir -p model_weights
if [ ! -d "model_weights/Qwen3-1.7B" ]; then
    huggingface-cli download --resume-download Qwen/Qwen3-1.7B --local-dir model_weights/Qwen3-1.7B
fi

if [ ! -d "model_weights/Qwen3-4B" ]; then
    huggingface-cli download --resume-download Qwen/Qwen3-4B --local-dir model_weights/Qwen3-4B
fi

if [ ! -d "model_weights/Qwen3-4B" ]; then
    huggingface-cli download --resume-download Qwen/Qwen3-4B-Base --local-dir model_weights/Qwen3-4B-Base
fi

if [ ! -d "model_weights/Qwen2.5-Math-7B" ]; then
    huggingface-cli download --resume-download Qwen/Qwen2.5-Math-7B --local-dir model_weights/Qwen2.5-Math-7B
fi

sudo apt update
sudo apt install ffmpeg -y
sudo apt install nodejs npm -y
sudo npm install -g n
curl -fsSL https://claude.ai/install.sh | bash