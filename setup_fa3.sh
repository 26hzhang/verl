#!/bin/bash
set -e

# clone and switch version
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.8.3

# init submodule
git submodule update --init csrc/cutlass

# install dependencies
uv pip install ninja packaging einops

# compile
cd hopper
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=64 python setup.py bdist_wheel

# install
uv pip install dist/flash_attn_3-*.whl

echo "Flash Attention 3 compile finished!"