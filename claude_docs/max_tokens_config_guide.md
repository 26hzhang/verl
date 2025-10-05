# VeRL Token Length Configuration Guide

本文档详细介绍VeRL框架中关于token长度配置的三个关键参数：`max_prompt_length`、`max_response_length` 和 `max_num_batched_tokens`。

## 参数概览

| 参数名 | 用途 | 默认值 | 配置文件位置 |
|--------|------|--------|-------------|
| `max_prompt_length` | 单个prompt的最大token长度 | 512 | `verl/trainer/config/data/legacy_data.yaml` |
| `max_response_length` | 单个response的最大token长度 | 512 | `verl/trainer/config/data/legacy_data.yaml` |
| `max_num_batched_tokens` | 批次中所有序列的token总数上限 | 8192 | `verl/trainer/config/rollout/rollout.yaml` |

## 1. max_prompt_length

### 定义与用途
控制训练和推理过程中单个输入prompt的最大token长度。

### 配置文件位置
```yaml
# verl/trainer/config/data/legacy_data.yaml
max_prompt_length: 512
```

### 对应数据类
```python
# verl/trainer/config/data_config.py
@dataclass
class DataConfig(BaseConfig):
    max_prompt_length: int = 512
```

### 使用示例
```bash
# 在训练脚本中设置
python3 -m verl.trainer.main_ppo \
    data.max_prompt_length=2048 \
    # 其他参数...
```

## 2. max_response_length

### 定义与用途
控制训练和推理过程中单个模型响应的最大token长度。

### 配置文件位置
```yaml
# verl/trainer/config/data/legacy_data.yaml
max_response_length: 512
```

### 对应数据类
```python
# verl/trainer/config/data_config.py
@dataclass
class DataConfig(BaseConfig):
    max_response_length: int = 512
```

### 使用示例
```bash
# 在训练脚本中设置
python3 -m verl.trainer.main_ppo \
    data.max_response_length=16384 \
    # 其他参数...
```

## 3. max_num_batched_tokens

### 定义与用途
控制推理引擎（vLLM/SGLang）在单个批次中处理的token总数上限，直接影响显存使用和推理吞吐量。

### 配置文件位置
```yaml
# verl/trainer/config/rollout/rollout.yaml:53
# max number of tokens in a batch
max_num_batched_tokens: 8192
```

### 对应数据类
```python
# verl/workers/config/rollout.py:97
@dataclass
class RolloutConfig(BaseConfig):
    max_num_batched_tokens: int = 8192
```

### 使用示例
```bash
# 在训练脚本中设置
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.rollout.max_num_batched_tokens=18432 \
    # 其他参数...
```

## 参数关系详解

### 1. 基本计算关系

```python
# 单个序列的最大长度
max_model_len = max_prompt_length + max_response_length

# 示例：
# max_prompt_length = 2048
# max_response_length = 16384  
# max_model_len = 2048 + 16384 = 18432
```

### 2. 关键约束条件

在启用`enable_chunked_prefill`时，存在强制约束：

```python
# verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:278-282
if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
    raise ValueError(
        "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, "
        "please increase max_num_batched_tokens or disable chunked prefill"
    )
```

**约束条件**：`max_num_batched_tokens >= max_model_len`（当启用chunked_prefill时）

### 3. 配置引用关系

rollout配置会自动引用data配置：

```yaml
# verl/trainer/config/rollout/rollout.yaml:21-25
# typically the same as data max prompt length
# same as data.max_prompt_length if it exists
prompt_length: ${oc.select:data.max_prompt_length,512}

# typically the same as data max response length
# same as data.max_response_length if it exists  
response_length: ${oc.select:data.max_response_length,512}
```

## 配置文件层次结构

```
主配置文件：
├── verl/trainer/config/ppo_trainer.yaml
│   ├── defaults:
│   │   ├── data@data: legacy_data          # → max_prompt_length, max_response_length
│   │   └── rollout@actor_rollout_ref.rollout: rollout  # → max_num_batched_tokens
│   
└── 生成的完整配置：
    └── verl/trainer/config/_generated_ppo_trainer.yaml
```

## 实际配置示例

### 示例1：7B模型配置
```bash
# examples/tuning/7b/qwen2-7b_grpo-lora_1_h100_fsdp_vllm.sh
python3 -m verl.trainer.main_ppo \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.rollout.max_model_len=1536 \        # 512+1024=1536
    actor_rollout_ref.rollout.max_num_batched_tokens=1536 \  # 等于max_model_len
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
```

### 示例2：DAPO多轮对话配置  
```bash
# examples/sglang_multiturn/run_qwen3_4b_dapo_multiturn.sh
python3 -m verl.trainer.main_ppo \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    # max_model_len = 2048 + 16384 = 18432
    # 默认max_num_batched_tokens=8192 < 18432，需要调整
```

**注意**：在示例2中，由于`max_num_batched_tokens`默认为8192，小于`max_model_len`(18432)，如果启用`enable_chunked_prefill`会报错。

## 配置建议

### 1. 显存充足时的推荐配置

```bash
python3 -m verl.trainer.main_ppo \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=20480 \  # 略大于max_model_len
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
```

### 2. 显存受限时的推荐配置

```bash
python3 -m verl.trainer.main_ppo \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \   # 较小值节省显存
    actor_rollout_ref.rollout.enable_chunked_prefill=False \  # 禁用chunked_prefill
```

### 3. 配置规则总结

1. **启用chunked_prefill时**：
   - `max_num_batched_tokens >= max_prompt_length + max_response_length`
   - 提供更高的推理吞吐量

2. **禁用chunked_prefill时**：  
   - `max_num_batched_tokens`可以设置为较小值
   - 节省显存，但可能降低吞吐量

3. **显存优化**：
   - 减小`max_num_batched_tokens`来降低显存占用
   - 减小`max_response_length`来减少单序列长度

## 性能影响分析

### max_num_batched_tokens 对性能的影响

| 数值大小 | 显存占用 | 推理吞吐量 | 适用场景 |
|----------|----------|------------|----------|
| 小（< 8192） | 低 | 较低 | 显存受限环境 |
| 中（8192-16384） | 中等 | 中等 | 平衡配置 |
| 大（> 16384） | 高 | 高 | 显存充足，追求高吞吐量 |

### 长度参数对训练的影响

- **max_prompt_length**：影响输入上下文长度，过短可能截断重要信息
- **max_response_length**：影响生成长度，需要根据任务特点设置
- 两者之和决定了单个样本的最大token消耗

## 故障排除

### 常见错误1：chunked_prefill约束违反
```
ValueError: Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len,
please increase max_num_batched_tokens or disable chunked prefill
```

**解决方案**：
```bash
# 方案1：增加max_num_batched_tokens
actor_rollout_ref.rollout.max_num_batched_tokens=20480 \

# 方案2：禁用chunked_prefill  
actor_rollout_ref.rollout.enable_chunked_prefill=False \
```

### 常见错误2：显存不足
```
OutOfMemoryError: CUDA out of memory
```

**解决方案**：
```bash
# 减小批次token数量
actor_rollout_ref.rollout.max_num_batched_tokens=4096 \

# 减小GPU内存利用率
actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \

# 减小序列长度
data.max_response_length=8192 \
```

## 监控和调试

### 查看当前配置
```bash
# 打印完整配置
python3 -m verl.trainer.main_ppo --cfg job

# 只打印相关配置段
python3 -m verl.trainer.main_ppo --cfg job | grep -A 10 -B 5 "max_.*_length\|max_num_batched_tokens"
```

### 显存使用监控
```bash
# 监控GPU显存使用
nvidia-smi -l 1

# 在训练脚本中添加显存监控
export CUDA_LAUNCH_BLOCKING=1
```

## 相关文件参考

- **配置文件**：
  - `verl/trainer/config/data/legacy_data.yaml`
  - `verl/trainer/config/rollout/rollout.yaml`
  - `verl/trainer/config/ppo_trainer.yaml`

- **实现文件**：
  - `verl/workers/config/rollout.py`
  - `verl/trainer/config/data_config.py`
  - `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

- **示例脚本**：
  - `examples/tuning/7b/qwen2-7b_grpo-lora_1_h100_fsdp_vllm.sh`
  - `examples/sglang_multiturn/run_qwen3_4b_dapo_multiturn.sh`

---

*最后更新：2025-09-16*