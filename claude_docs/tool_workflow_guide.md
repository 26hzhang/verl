# VeRL Tool Workflow Guide: GSM8K Tool Example

本文档以 `Gsm8kTool` 为例，详细说明 VeRL 框架中工具的创建、执行和奖励计算的完整流程。

## 目录

1. [概述](#概述)
2. [工具初始化流程](#工具初始化流程)
3. [工具执行流程](#工具执行流程)
4. [奖励计算机制](#奖励计算机制)
5. [关键文件和修改点](#关键文件和修改点)
6. [自定义工具开发指南](#自定义工具开发指南)

## 概述

VeRL 中的工具系统用于在多轮对话中为模型提供外部功能调用能力。工具系统遵循 OpenAI 的函数调用规范，支持：
- 动态工具加载和初始化
- 异步工具执行
- 基于工具结果的奖励计算
- 多工具并行处理

## 工具初始化流程

### 1. 配置文件定义

工具配置存储在 YAML 文件中，例如 `examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml`:

```yaml
tools:
  - class_name: "verl.tools.gsm8k_tool.Gsm8kTool"
    config:
      type: native
    tool_schema:
      type: "function"
      function:
        name: "calc_gsm8k_reward"
        description: "A tool for calculating the reward of gsm8k"
        parameters:
          type: "object"
          properties:
            answer:
              type: "string"
              description: "The model's answer to the GSM8K math problem"
          required: ["answer"]
```

### 2. 初始化调用链

```
run_qwen2.5-3b_gsm8k_multiturn.sh
    ↓
verl.trainer.main_ppo
    ↓
SGLangRollout.__init__ (verl/workers/rollout/sglang_rollout/sglang_rollout.py:421-426)
    ↓
_initialize_tools (verl/workers/rollout/sglang_rollout/sglang_rollout.py:657-705)
    ↓
initialize_tools_from_config (verl/tools/utils/tool_registry.py:82-130)
    ↓
get_tool_class + 动态实例化
```

### 3. 关键函数解析

#### `initialize_tools_from_config` (verl/tools/utils/tool_registry.py:82-130)
- 加载 YAML 配置文件
- 遍历工具配置列表
- 动态导入工具类
- 创建工具实例

```python
def initialize_tools_from_config(tools_config_file):
    tools_config = OmegaConf.load(tools_config_file)
    tool_list = []

    for tool_config in tools_config.tools:
        cls_name = tool_config.class_name
        tool_type = ToolType(tool_config.config.type)
        tool_cls = get_tool_class(cls_name)

        if tool_type == ToolType.NATIVE:
            tool_schema = OpenAIFunctionToolSchema.model_validate(
                OmegaConf.to_container(tool_config.tool_schema)
            )
            tool = tool_cls(
                config=OmegaConf.to_container(tool_config.config),
                tool_schema=tool_schema,
            )
            tool_list.append(tool)
```

#### `_initialize_tools` (verl/workers/rollout/sglang_rollout/sglang_rollout.py:657-705)
- 构建工具映射表 (`tool_map`)
- 初始化工具调用解析器
- 返回工具相关的所有组件

```python
def _initialize_tools(self, config, processing_class):
    if config.multi_turn.tool_config_path is None:
        return [], {}, None, [], None

    tool_list = initialize_tools_from_config(config.multi_turn.tool_config_path)
    tool_schemas = [tool.get_openai_tool_schema().model_dump() for tool in tool_list]
    tool_map = {tool.name: tool for tool in tool_list}

    return tool_schemas, tool_map, tool_call_parser_type, sgl_tools, function_call_parser
```

## 工具执行流程

### 1. 请求生命周期

```
AsyncRolloutRequest 创建
    ↓
PENDING 状态 → _handle_pending_state
    ↓ (创建工具实例)
GENERATING 状态 → 模型生成响应
    ↓ (解析工具调用)
TOOL_CALLING 状态 → 执行工具
    ↓ (多轮交互)
FINISHED 状态 → 计算奖励并释放资源
```

### 2. 工具创建阶段

在 `_handle_pending_state` (verl/workers/rollout/sglang_rollout/sglang_rollout.py:1203-1214):

```python
async def _handle_pending_state(self, _req: AsyncRolloutRequest):
    if _req.tool_schemas is not None:
        tool_creation_coroutines = []
        for tool_schema in _req.tool_schemas:
            tool = self._tool_map[tool_schema.function.name]
            create_kwargs = _req.tools_kwargs[tool.name].get("create_kwargs", {})
            tool_creation_coroutines.append(
                tool.create(_req.request_id, **create_kwargs)
            )
        tool_creation_results = await asyncio.gather(*tool_creation_coroutines)
```

`Gsm8kTool.create` 方法会：
- 为每个请求创建独立的工具实例
- 存储 ground_truth 答案用于后续评分
- 初始化奖励为 0.0

### 3. 工具执行阶段

当模型生成工具调用时，在 `_handle_tool_calling_state` (verl/workers/rollout/sglang_rollout/sglang_rollout.py:1011-1021):

```python
if _req.messages[-1].tool_calls is not None:
    parsed_tool_calls = _req.messages[-1].tool_calls
    tool_call_results = await asyncio.gather(
        *[
            self._tool_map[tool_call.function.name].execute(
                _req.request_id,
                tool_call.function.arguments,
                **_req.tools_kwargs.get(tool_call.function.name, {}).get("execute_kwargs", {})
            )
            for tool_call in parsed_tool_calls
        ]
    )
```

`Gsm8kTool.execute` 方法会：
- 解析模型提供的答案
- 计算当前答案的奖励
- 返回工具响应和增量奖励

### 4. 工具响应处理

工具执行结果会被添加到对话历史中，供模型进行下一轮生成：

```python
tool_responses = [result[0] for result in tool_call_results]
tool_rewards = [result[1] for result in tool_call_results]
_req.add_tool_response_messages(self.processing_class, tool_responses)
```

## 奖励计算机制

### 1. 增量奖励机制

在 `Gsm8kTool.execute` 中实现了增量奖励：

```python
async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
    answer = parameters.get("answer", "")
    reward = await self.calc_reward(instance_id)

    # 如果新答案没有改进，给予惩罚
    tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05

    # 更新存储的最佳奖励
    self._instance_dict[instance_id]["reward"] = reward

    return ToolResponse(text=f"Current parsed {answer=} {reward=}"), tool_reward, {}
```

### 2. 最终奖励计算

在请求完成时 (verl/workers/rollout/sglang_rollout/sglang_rollout.py:1154-1165):

```python
async def calc_reward_and_release_fn(name, tool):
    reward = await tool.calc_reward(
        _req.request_id,
        **_req.tools_kwargs[name].get("calc_reward_kwargs", {})
    )
    await tool.release(
        _req.request_id,
        **_req.tools_kwargs[name].get("release_kwargs", {})
    )
    return name, reward

tool_reward_tasks = []
for name in _req.tools_kwargs.keys():
    tool = self._tool_map[name]
    tool_reward_tasks.append(calc_reward_and_release_fn(name, tool))

tool_reward_scores = await asyncio.gather(*tool_reward_tasks)
```

### 3. 奖励评分逻辑

`Gsm8kTool.calc_reward` 使用灵活的评分方法：

```python
async def calc_reward(self, instance_id: str, **kwargs) -> float:
    return gsm8k.compute_score(
        self._instance_dict[instance_id]["response"],
        self._instance_dict[instance_id]["ground_truth"],
        method="flexible",
        format_score=0.0,  # 格式错误得分
        score=1.0,         # 正确答案得分
    )
```

## 关键文件和修改点

### 核心文件列表

| 文件路径 | 功能描述 | 常见修改场景 |
|---------|---------|-------------|
| `verl/tools/base_tool.py` | 工具基类定义 | 添加新的工具接口方法 |
| `verl/tools/gsm8k_tool.py` | GSM8K 工具实现 | 修改评分逻辑或响应格式 |
| `verl/tools/utils/tool_registry.py` | 工具注册和加载 | 支持新的工具类型（如 MCP） |
| `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | Rollout 工作节点 | 修改工具执行流程或并发策略 |
| `verl/utils/reward_score/gsm8k.py` | GSM8K 评分逻辑 | 调整答案解析或评分规则 |
| `examples/sglang_multiturn/config/tool_config/*.yaml` | 工具配置文件 | 配置新工具或修改现有工具参数 |

### 修改工具行为的关键点

1. **修改工具初始化参数**
   - 编辑 YAML 配置文件中的 `config` 部分
   - 在工具类的 `__init__` 方法中添加参数处理

2. **调整奖励计算策略**
   - 修改 `execute` 方法中的增量奖励逻辑
   - 更新 `calc_reward` 方法的评分算法

3. **改变工具响应格式**
   - 修改 `ToolResponse` 的返回内容
   - 调整 `tool_schema` 中的参数定义

4. **优化并发执行**
   - 在 `SGLangRollout` 中调整 `asyncio.gather` 的使用
   - 实现工具级别的并发控制

## 自定义工具开发指南

### 1. 创建新工具类

```python
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

class MyCustomTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances = {}

    async def create(self, instance_id: str = None, **kwargs):
        # 初始化工具实例
        pass

    async def execute(self, instance_id: str, parameters: dict, **kwargs):
        # 执行工具逻辑
        pass

    async def calc_reward(self, instance_id: str, **kwargs):
        # 计算奖励
        pass

    async def release(self, instance_id: str, **kwargs):
        # 释放资源
        pass
```

### 2. 配置工具

创建 `config/tool_config/my_tool_config.yaml`:

```yaml
tools:
  - class_name: "path.to.MyCustomTool"
    config:
      type: native
      custom_param: value
    tool_schema:
      type: "function"
      function:
        name: "my_custom_function"
        description: "Description of the tool"
        parameters:
          type: "object"
          properties:
            param1:
              type: "string"
              description: "Parameter description"
          required: ["param1"]
```

### 3. 在训练脚本中使用

```bash
actor_rollout_ref.rollout.multi_turn.tool_config_path="path/to/my_tool_config.yaml"
```

## 调试和监控

### 日志位置
- 工具初始化日志：搜索 "Initialize tools from configuration"
- 工具执行日志：在各工具的 `execute` 方法中添加
- 奖励计算日志：搜索 "tool_reward_scores"

### 常见问题排查
1. 工具未被调用：检查模型是否正确生成了工具调用格式
2. 奖励计算异常：验证 ground_truth 是否正确传递
3. 并发问题：检查工具实例是否线程安全

## 总结

VeRL 的工具系统提供了灵活的扩展机制，通过继承 `BaseTool` 并实现核心方法，可以轻松集成各种外部功能。关键在于理解：
- 工具生命周期管理（创建、执行、释放）
- 异步执行模型
- 奖励信号的设计和传递
- 与 SGLang 推理引擎的集成

通过本指南，开发者可以快速理解工具系统的工作原理，并开发自定义工具来支持特定的强化学习任务。