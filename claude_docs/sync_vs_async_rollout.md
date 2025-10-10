# VeRL Rollout模式分析：同步 vs 异步

本文档详细分析了VeRL框架中两种rollout模式的实现原理和使用场景。

## 概述

在`verl/trainer/ppo/ray_trainer.py`中，VeRL支持两种rollout模式：
- **同步模式** (`async_rollout_mode = False`)：使用`actor_rollout_wg.generate_sequences()`
- **异步模式** (`async_rollout_mode = True`)：使用`async_rollout_manager.generate_sequences()`

## 1. 同步模式 (Synchronous Rollout)

### 调用路径
```python
# ray_trainer.py:1198
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
```

### 实现细节

同步模式直接使用`ActorRolloutWorkerGroup`进行推理生成：

1. **生成流程** (`sglang_rollout.py:788-957`)：
   ```python
   def generate_sequences(self, prompts: DataProto) -> DataProto:
       # 预处理输入
       idx_list = [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)]

       # 主进程执行推理
       if self._tp_rank == 0:
           loop = asyncio.get_event_loop()
           output = loop.run_until_complete(
               self._engine.async_generate(
                   input_ids=idx_list,
                   sampling_params=request_sampling_params,
                   return_logprob=True,
               )
           )

       # 同步所有进程
       dist.barrier()
       output = broadcast_pyobj(output, ...)

       # 后处理并返回结果
       return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
   ```

2. **特点**：
   - 使用`asyncio.run_until_complete`阻塞等待结果
   - 通过`dist.barrier()`确保所有进程同步
   - 批量处理所有样本
   - 简单直接，适合单轮生成

### 适用场景
- 标准的PPO训练流程
- 单轮对话生成
- 不需要外部工具调用
- 批次内样本处理时间相近

## 2. 异步模式 (Asynchronous Rollout)

### 调用路径
```python
# ray_trainer.py:1200
gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
```

### 实现架构

异步模式使用`AgentLoopManager`管理多个异步工作进程：

```
AgentLoopManager
├── AsyncLLMServerManager (负载均衡器)
│   └── 多个 AsyncEngine 实例
└── AgentLoopWorker[] (工作进程池)
    └── 每个处理一部分样本
```

### 核心组件

1. **AgentLoopManager** (`agent_loop.py:886-1110`)：
   ```python
   class AgentLoopManager:
       def __init__(self, config, worker_group, rm_wg=None):
           # 初始化LLM服务器
           self._initialize_llm_servers()
           # 初始化agent循环worker
           self._init_agent_loop_workers()

       def generate_sequences(self, prompts: DataProto) -> DataProto:
           # 分块处理
           chunks = prompts.chunk(len(self.agent_loop_workers))
           outputs = ray.get([
               worker.generate_sequences.remote(chunk)
               for worker, chunk in zip(self.agent_loop_workers, chunks)
           ])
           return DataProto.concat(outputs)
   ```

2. **AsyncLLMServerManager** (`agent_loop.py:48-127`)：
   - 最少请求负载均衡
   - 会话粘性（同一request_id路由到同一服务器）
   - LRU缓存管理request映射

3. **AgentLoopWorker** (`agent_loop.py:452-859`)：
   ```python
   async def generate_sequences(self, batch: DataProto) -> DataProto:
       # 为每个样本创建异步任务
       tasks = []
       for i in range(len(batch)):
           kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
           tasks.append(asyncio.create_task(
               self._run_agent_loop(sampling_params, trajectory_info[i], **kwargs)
           ))

       # 并发执行所有任务
       outputs = await asyncio.gather(*tasks)
       return self._postprocess(outputs)
   ```

### 高级功能

1. **多轮对话支持**：
   - 状态机管理：`PENDING → RUNNING → TOOL_CALLING → INTERACTING`
   - 最大轮数控制：`max_assistant_turns`, `max_user_turns`

2. **工具调用支持**：
   ```python
   if self._function_call_parser.has_tool_call(content):
       tool_calls = self._function_call_parser.parse_non_stream(content)
       tool_call_results = await asyncio.gather(*[
           self._tool_map[tool_call.function.name].execute(...)
           for tool_call in parsed_tool_calls
       ])
   ```

3. **动态交互**：
   - 支持与外部环境交互
   - 可配置的interaction模式

### 适用场景
- Agent任务（需要工具调用）
- 多轮对话系统
- 样本处理时间差异大的场景
- 需要与外部系统交互的任务

## 对比总结

| 特性 | 同步模式 | 异步模式 |
|------|---------|----------|
| **并发性** | 串行处理 | 并行处理 |
| **复杂度** | 简单 | 复杂 |
| **功能支持** | 基础生成 | 工具调用、多轮对话、环境交互 |
| **资源利用** | 单引擎实例 | 多服务器实例，负载均衡 |
| **延迟** | 受最慢样本影响 | 独立处理，整体更快 |
| **内存占用** | 较低 | 较高（多实例） |
| **适用场景** | 标准RL训练 | 复杂Agent任务 |

## 配置方式

在配置文件中通过`rollout.mode`参数选择：

```yaml
actor_rollout_ref:
  rollout:
    mode: "sync"  # 或 "async"
    # 异步模式特有配置
    agent:
      num_workers: 4  # AgentLoopWorker数量
      agent_loop_config_path: "path/to/config.yaml"
```

## 性能考虑

1. **同步模式**：
   - 优点：实现简单，资源占用少
   - 缺点：批次内最慢样本决定整体速度

2. **异步模式**：
   - 优点：更好的并行性，支持复杂交互
   - 缺点：资源占用高，实现复杂

## 选择建议

- 使用**同步模式**当：
  - 进行标准的PPO/RLHF训练
  - 样本是简单的单轮生成
  - 资源有限

- 使用**异步模式**当：
  - 需要工具调用功能
  - 实现多轮对话系统
  - 样本生成时间差异大
  - 需要与外部环境交互

## 相关代码位置

- 模式选择逻辑：`verl/trainer/ppo/ray_trainer.py:951-958`
- 同步实现：`verl/workers/rollout/sglang_rollout/sglang_rollout.py:788-957`
- 异步管理器：`verl/experimental/agent_loop/agent_loop.py:886-1110`
- 异步工作器：`verl/experimental/agent_loop/agent_loop.py:452-859`