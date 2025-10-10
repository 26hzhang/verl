# Agent Loop 架构详解

## 概述

`agent_loop.py` 是 verl 框架中的一个实验性模块，用于实现 Agent 循环（Agent Loop）功能。该模块允许 Agent 与兼容 OpenAI 的 LLM 服务器进行交互，并与各种环境（如工具调用、多模态输入等）进行集成。

## 核心组件

### 1. AsyncLLMServerManager

**功能**: 管理多个兼容 OpenAI 的 LLM 服务器，提供负载均衡和会话粘性功能。

**主要特性**:
- **负载均衡**: 使用最少请求（least requests）算法进行负载均衡
- **会话粘性**: 将多轮对话发送到同一服务器，实现自动前缀缓存
- **LRU 缓存**: 使用 LRU 缓存维护 request_id 到服务器的映射

**核心方法**:
- `_choose_server()`: 根据 request_id 选择服务器，实现会话粘性
- `generate()`: 异步生成 token，支持多模态输入

### 2. AgentLoopBase (抽象基类)

**功能**: 定义 Agent 循环的基本接口，所有具体的 Agent 循环实现都需要继承此类。

**主要方法**:
- `init_class()`: 类级别的初始化，用于共享重型初始化工作
- `run()`: 抽象方法，需要子类实现具体的 Agent 循环逻辑

**设计模式**:
- 使用装饰器模式 `@register(agent_name)` 注册 Agent 循环实现
- 支持 Hydra 配置系统进行动态实例化

### 3. BatchExecutor

**功能**: 批处理执行器，用于收集请求并批量执行，提高处理效率。

**主要特性**:
- 异步任务提交
- 自动批处理收集
- 支持微批次大小配置
- 使用线程池进行后台处理

### 4. RewardManagerWorker

**功能**: 奖励管理器 Worker，用于异步计算奖励分数。

**主要特性**:
- 异步奖励计算，与 Agent 循环并行执行
- 支持批处理执行器优化
- 使用线程池避免阻塞

### 5. AgentLoopWorker

**功能**: Agent 循环 Worker，负责处理一批消息并在 Agent 循环中运行每条消息。

**主要职责**:
- 管理 LLM 服务器连接
- 处理多轮对话和工具调用
- 序列生成和后处理
- 奖励计算集成
- 多模态输入处理

**核心方法**:
- `generate_sequences()`: 从 Agent 循环生成序列
- `_run_agent_loop()`: 运行单个 Agent 循环实例
- `_postprocess()`: 后处理和批次组装

### 6. AgentLoopManager

**功能**: 管理一组 Agent 循环 Worker，提供高层次的接口。

**主要职责**:
- 初始化和管理 LLM 服务器
- 分发任务给多个 Worker
- 性能指标收集
- 服务器生命周期管理（wake_up/sleep）

## 数据流程

### 1. 输入处理
```
输入批次 (DataProto)
    ↓
分割成多个块
    ↓
分配给不同的 AgentLoopWorker
```

### 2. Agent 循环执行
```
每个样本:
    ↓
创建 Agent 循环实例
    ↓
与 LLM 服务器交互
    ↓
工具调用（如果需要）
    ↓
生成响应
```

### 3. 序列处理
```
原始输出 (AgentLoopOutput)
    ↓
填充处理 (padding)
    ↓
掩码计算 (masks)
    ↓
位置编码 (position_ids)
    ↓
内部格式 (_InternalAgentLoopOutput)
```

### 4. 批次组装
```
多个内部输出
    ↓
张量拼接
    ↓
奖励分数处理
    ↓
最终批次 (DataProto)
```

## 关键特性

### 1. 多轮对话支持

支持复杂的多轮对话，包括：
- 用户消息
- 助手响应
- 工具调用和响应

响应掩码示例：
```
responses:     |<- LLM生成 ->|<- 工具调用 ->|<- LLM生成 ->|<- 填充 ->|
response_mask: | 1, 1, ..., 1 | 0, 0, ..., 0 | 1, 1, ..., 1 | 0, 0, ..., 0|
```

### 2. 多模态支持

- 目前支持 Qwen2VL 的图像处理
- 自动处理多模态输入的位置编码
- 支持图像和视频输入

### 3. 异步处理

- 使用 asyncio 进行并发处理
- 支持异步奖励计算
- Ray 分布式计算集成

### 4. 性能优化

- 批处理执行器减少开销
- 负载均衡提高吞吐量
- 会话粘性优化缓存利用
- 服务器睡眠/唤醒机制节省资源

## 配置示例

```yaml
actor_rollout_ref:
  rollout:
    agent:
      num_workers: 4  # Agent循环Worker数量
      agent_loop_config_path: "path/to/config.yaml"  # Agent循环配置路径
      custom_async_server:  # 自定义服务器配置（可选）
        path: "module.path"
        name: "ServerClass"
    prompt_length: 512  # 提示词最大长度
    response_length: 512  # 响应最大长度
    temperature: 0.7  # 采样温度
    top_p: 0.9  # Top-p采样
    calculate_log_probs: true  # 是否计算对数概率
```

## 扩展指南

### 1. 实现自定义 Agent 循环

```python
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, register, AgentLoopOutput

@register("my_custom_agent")
class MyCustomAgent(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        # 实现你的Agent逻辑
        prompt_ids = kwargs.get("prompt_ids", [])

        # 与LLM服务器交互
        output = await self.server_manager.generate(
            request_id="unique_id",
            prompt_ids=prompt_ids,
            sampling_params=sampling_params
        )

        # 返回AgentLoopOutput
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids,
            response_mask=[1] * len(output.token_ids),
            num_turns=1,
            metrics=AgentLoopMetrics()
        )
```

### 2. 注册 Agent 配置

在配置文件中添加：
```yaml
- name: "my_custom_agent"
  _target_: "module.path.MyCustomAgent"
  custom_param: "value"
```

## 注意事项

1. **内存管理**: 注意批次大小和序列长度配置，避免 OOM
2. **异步处理**: 确保正确处理异步操作，避免死锁
3. **错误处理**: 实现适当的错误处理和重试机制
4. **性能监控**: 使用提供的性能指标进行监控和优化

## 未来改进方向

1. 支持更多多模态处理器
2. 实现服务器压力感知的负载均衡
3. 增强错误恢复机制
4. 扩展性能指标收集