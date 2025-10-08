# Tool System Complete Guide

本文档详细说明 VERL 中工具系统的完整构建流程，包括工具实现、配置文件、数据集构建以及它们之间的相互关系。

## 目录
- [概述](#概述)
- [工具生命周期](#工具生命周期)
- [组件详解](#组件详解)
  - [1. 工具实现 (Python)](#1-工具实现-python)
  - [2. 工具配置 (YAML)](#2-工具配置-yaml)
  - [3. 数据集构建](#3-数据集构建)
  - [4. 运行时执行流程](#4-运行时执行流程)
- [完整示例](#完整示例)
- [高级用例](#高级用例)
- [最佳实践](#最佳实践)

---

## 概述

VERL 工具系统遵循**生命周期管理模式**，每个工具实例在训练的一个轨迹（trajectory）中经历四个阶段：

```
CREATE → EXECUTE (多次) → CALC_REWARD → RELEASE
```

### 核心设计理念

1. **实例隔离**：每个训练样本创建独立的工具实例，避免状态污染
2. **参数分离**：
   - `create_kwargs`：数据驱动的静态参数（如 ground_truth）
   - `execute_kwargs`：执行时的额外配置参数
   - `parameters`：模型生成的动态参数
3. **异步执行**：所有工具方法都是异步的，支持高并发

---

## 工具生命周期

### 阶段详解

#### 1. CREATE - 工具实例创建
```python
async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]
```
- **时机**：rollout 开始前，`PENDING` 状态时调用
- **作用**：初始化工具实例状态，存储样本特定的上下文
- **参数来源**：`create_kwargs` (从数据集读取)
- **返回**：实例 ID 和可选的初始响应消息

#### 2. EXECUTE - 工具执行
```python
async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]
```
- **时机**：模型生成 tool call 后，`TOOL_CALLING` 状态时调用
- **作用**：执行工具逻辑，更新状态，返回结果
- **参数来源**：
  - `parameters`：模型生成的 tool call arguments
  - `kwargs`：`execute_kwargs` (从数据集读取)
- **返回**：工具响应、即时奖励、指标字典
- **可调用次数**：多次（支持多轮交互）

#### 3. CALC_REWARD - 计算最终奖励
```python
async def calc_reward(self, instance_id: str, **kwargs) -> float
```
- **时机**：rollout 结束后，计算轨迹总奖励时调用
- **作用**：基于整个轨迹的状态计算最终奖励
- **参数来源**：`calc_reward_kwargs` (从数据集读取)
- **返回**：最终奖励值

#### 4. RELEASE - 释放资源
```python
async def release(self, instance_id: str, **kwargs) -> None
```
- **时机**：奖励计算完成后
- **作用**：清理资源，删除实例状态
- **参数来源**：`release_kwargs` (从数据集读取)

---

## 组件详解

### 1. 工具实现 (Python)

#### 基础模板

```python
# verl/tools/your_tool.py
from typing import Any, Optional
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse


class YourTool(BaseTool):
    """Your tool description."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        # 存储所有实例的状态
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """定义工具的 OpenAI 函数调用格式"""
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,  # create_kwargs 参数
        **kwargs
    ) -> tuple[str, ToolResponse]:
        """创建工具实例，存储样本特定的上下文"""
        if instance_id is None:
            instance_id = str(uuid4())

        # 初始化实例状态
        self._instance_dict[instance_id] = {
            "ground_truth": ground_truth,
            "history": [],
            "reward": 0.0,
        }

        # 可选：返回初始消息给模型
        return instance_id, ToolResponse()

    @rollout_trace_op  # 记录执行轨迹
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],  # 模型生成的参数
        **kwargs  # execute_kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """执行工具逻辑"""
        # 从模型参数中提取输入
        user_input = parameters.get("input", "")

        # 执行工具逻辑
        result = self._do_something(instance_id, user_input)

        # 更新状态
        self._instance_dict[instance_id]["history"].append({
            "input": user_input,
            "result": result,
        })

        # 计算即时奖励（step reward）
        step_reward = self._calculate_step_reward(instance_id)

        # 返回响应消息、奖励、指标
        return (
            ToolResponse(text=f"Result: {result}"),
            step_reward,
            {"metric_name": 0.0}  # 可选的指标
        )

    async def calc_reward(
        self,
        instance_id: str,
        **kwargs  # calc_reward_kwargs
    ) -> float:
        """计算最终奖励"""
        state = self._instance_dict[instance_id]
        # 基于完整状态计算奖励
        final_reward = self._compute_final_score(state)
        return final_reward

    async def release(
        self,
        instance_id: str,
        **kwargs  # release_kwargs
    ) -> None:
        """清理资源"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

    # ===== 辅助方法 =====

    def _do_something(self, instance_id: str, user_input: str):
        """工具的核心逻辑"""
        pass

    def _calculate_step_reward(self, instance_id: str) -> float:
        """计算即时奖励"""
        return 0.0

    def _compute_final_score(self, state: dict) -> float:
        """计算最终奖励"""
        return 0.0
```

#### 关键点说明

1. **`_instance_dict`**：使用字典存储所有实例状态，键为 `instance_id`
2. **`@rollout_trace_op`**：装饰器用于记录执行轨迹，方便调试和分析
3. **参数命名**：
   - `create()` 的参数会从 `create_kwargs` 传入
   - `execute()` 的 `parameters` 来自模型生成
   - `execute()` 的 `**kwargs` 来自 `execute_kwargs`
4. **异步方法**：所有生命周期方法必须是 `async`

---

### 2. 工具配置 (YAML)

工具配置文件定义工具的元信息和 OpenAI 函数调用格式。

#### 配置文件结构

```yaml
# examples/sglang_multiturn/config/tool_config/your_tool_config.yaml

tools:
  - class_name: "verl.tools.your_tool.YourTool"  # 工具类的完整路径
    config:
      type: native  # 工具类型：native | mcp | sandbox_fusion
      # 其他工具特定配置
      timeout: 30
      max_retries: 3

    tool_schema:  # OpenAI Function Calling 格式
      type: "function"
      function:
        name: "your_tool_name"  # 必须与工具类中的 name 一致
        description: "Detailed description of what the tool does"
        parameters:
          type: "object"
          properties:
            input:  # 模型需要提供的参数
              type: "string"
              description: "Description of this parameter"
            option:
              type: "integer"
              description: "Optional parameter with default"
              default: 1
          required: ["input"]  # 必需参数列表
```

#### 多工具配置示例

```yaml
# 支持同时配置多个工具
tools:
  - class_name: "verl.tools.geo3k_tool.Geo3kTool"
    config:
      type: native
    tool_schema:
      type: "function"
      function:
        name: "calc_geo3k_reward"
        description: "Calculate reward for geometry problems"
        parameters:
          type: "object"
          properties:
            answer:
              type: "string"
              description: "The answer in \\boxed{} format"
          required: ["answer"]

  - class_name: "verl.tools.search_tool.SearchTool"
    config:
      type: native
      api_key: "${SEARCH_API_KEY}"  # 支持环境变量
    tool_schema:
      type: "function"
      function:
        name: "web_search"
        description: "Search the web for information"
        parameters:
          type: "object"
          properties:
            query:
              type: "string"
              description: "Search query"
          required: ["query"]
```

#### 配置字段说明

- **`class_name`**：工具类的完整 Python 路径
- **`config`**：工具初始化配置
  - `type`：工具类型（native/mcp/sandbox_fusion）
  - 其他字段由具体工具定义
- **`tool_schema`**：OpenAI Function Calling 格式
  - 定义模型如何调用工具
  - `parameters` 定义模型生成的参数格式

---

### 3. 数据集构建

数据集预处理脚本需要为每个样本添加 `tools_kwargs` 字段。

#### 完整数据集预处理示例

```python
# examples/data_preprocess/your_dataset_w_tool.py

import argparse
import os
import datasets

def make_map_fn(split):
    def process_fn(example, idx):
        # 1. 提取原始数据字段
        question = example.pop("question")
        answer = example.pop("answer")
        images = example.pop("images", [])

        # 2. 构建提示信息
        prompt = f"{question}\nPlease solve step by step."

        # 3. 构建数据项
        data = {
            "data_source": "your_dataset",
            "prompt": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use tools when necessary.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "images": images,
            "ability": "reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer,
                "question": question,

                # ===== 工具配置的关键字段 =====
                "need_tools_kwargs": True,  # 标记需要使用工具

                "tools_kwargs": {
                    # 工具名称必须与 tool_schema 中的 name 一致
                    "your_tool_name": {
                        # CREATE 阶段参数：样本特定的上下文
                        "create_kwargs": {
                            "ground_truth": answer,  # 正确答案
                            "metadata": {"difficulty": "hard"},
                        },

                        # EXECUTE 阶段参数：执行时的额外配置（可选）
                        "execute_kwargs": {
                            "timeout": 30,
                            "verbose": True,
                        },

                        # CALC_REWARD 阶段参数（可选）
                        "calc_reward_kwargs": {
                            "weight": 1.0,
                        },

                        # RELEASE 阶段参数（可选）
                        "release_kwargs": {},
                    },

                    # 多工具支持
                    "another_tool": {
                        "create_kwargs": {
                            "config_param": "value",
                        },
                    },
                },
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/your_dataset_w_tool")
    args = parser.parse_args()

    # 加载数据集
    dataset = datasets.load_dataset("your/dataset")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # 应用预处理
    train_dataset = train_dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        num_proc=8
    )
    test_dataset = test_dataset.map(
        function=make_map_fn("test"),
        with_indices=True,
        num_proc=8
    )

    # 保存为 parquet 格式
    train_dataset.to_parquet(os.path.join(args.local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_save_dir, "test.parquet"))
```

#### `tools_kwargs` 字段说明

```python
"tools_kwargs": {
    "<tool_name>": {  # 必须与 YAML 配置中的 function.name 一致
        "create_kwargs": {...},    # create() 方法的 **kwargs
        "execute_kwargs": {...},   # execute() 方法的 **kwargs
        "calc_reward_kwargs": {...},  # calc_reward() 方法的 **kwargs
        "release_kwargs": {...},   # release() 方法的 **kwargs
    }
}
```

#### 关键点

1. **`need_tools_kwargs: True`**：必须设置，标记该样本使用工具
2. **工具名称匹配**：`tools_kwargs` 的键必须与 YAML 中的 `function.name` 一致
3. **参数用途区分**：
   - `create_kwargs`：数据驱动的参数（如 ground_truth）
   - `execute_kwargs`：配置参数（如 timeout）
   - `calc_reward_kwargs` 和 `release_kwargs`：通常为空
4. **多工具支持**：可为多个工具配置不同的 kwargs

---

### 4. 运行时执行流程

#### 完整调用链路

```
数据加载 → 工具初始化 → Rollout 执行 → 奖励计算
```

#### 详细流程图

```
1. 训练启动
   ├─ 加载工具配置 YAML
   ├─ 实例化工具类 (YourTool.__init__)
   └─ 注册到 tool_map

2. 数据加载
   ├─ 读取 parquet 文件
   ├─ 提取 tools_kwargs 字段
   └─ 构建 AsyncRolloutRequest

3. Rollout 开始 (PENDING 状态)
   ├─ 调用 tool.create(instance_id, **create_kwargs)
   ├─ 存储 instance_id
   └─ 可选：将 ToolResponse 添加到对话历史

4. 模型生成 (RUNNING 状态)
   ├─ 模型生成 text 或 tool_calls
   └─ 如果生成 tool_calls → 进入 TOOL_CALLING 状态

5. 工具执行 (TOOL_CALLING 状态)
   ├─ 解析 tool_call.function.arguments (模型生成的 parameters)
   ├─ 调用 tool.execute(instance_id, parameters, **execute_kwargs)
   ├─ 获取 ToolResponse, step_reward, metrics
   ├─ 将 ToolResponse 添加到对话历史
   ├─ 累积 step_reward 到 tool_reward
   └─ 返回 RUNNING 状态（继续生成）

6. Rollout 结束
   ├─ 达到终止条件（max_turns / EOS / length limit）
   └─ 进入奖励计算阶段

7. 奖励计算
   ├─ 对每个工具调用 tool.calc_reward(instance_id, **calc_reward_kwargs)
   ├─ 获取 final_reward
   ├─ 汇总：total_reward = step_rewards + final_rewards
   └─ 调用 tool.release(instance_id, **release_kwargs)

8. 训练更新
   └─ 使用 total_reward 进行 PPO/GRPO 更新
```

#### 关键代码位置

**工具创建** (`verl/workers/rollout/sglang_rollout/sglang_rollout.py:1067-1077`)
```python
async def _handle_pending_state(self, _req: AsyncRolloutRequest):
    if _req.tool_schemas is not None:
        for tool_schema in _req.tool_schemas:
            tool = self._tool_map[tool_schema.function.name]
            create_kwargs = _req.tools_kwargs[tool.name].get("create_kwargs", {})
            instance_id, tool_response = await tool.create(_req.request_id, **create_kwargs)
```

**工具执行** (`verl/workers/rollout/sglang_rollout/sglang_rollout.py:856-860`)
```python
tool_results = await asyncio.gather(*[
    self._tool_map[tool_call.function.name].execute(
        _req.request_id,
        tool_call.function.arguments,  # 模型生成的参数
        **_req.tools_kwargs.get(tool_call.function.name, {}).get("execute_kwargs", {})
    )
    for tool_call in parsed_tool_calls
])
```

**奖励计算与释放** (`verl/workers/rollout/sglang_rollout/sglang_rollout.py:1016-1018`)
```python
async def calc_reward_and_release_fn(name: str, tool: BaseTool):
    reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {}))
    await tool.release(_req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {}))
    return name, reward
```

---

## 完整示例

### 示例：Geometry3K 工具

#### 1. 工具实现

```python
# verl/tools/geo3k_tool.py
from typing import Any, Optional
from uuid import uuid4

from verl.utils.reward_score import geo3k
from verl.utils.rollout_trace import rollout_trace_op
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse


class Geo3kTool(BaseTool):
    """A tool for calculating rewards for Geometry3K problems."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        # 存储正确答案，用于后续奖励计算
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }

        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        # 从模型参数中获取答案
        answer = parameters.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)

        # 保存模型的回答
        self._instance_dict[instance_id]["response"] = answer

        # 计算当前奖励
        reward = await self.calc_reward(instance_id)

        # 如果没有改进答案，施加惩罚
        tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05

        # 更新最佳奖励
        self._instance_dict[instance_id]["reward"] = reward

        return (
            ToolResponse(text=f"Current parsed {answer=} {reward=}"),
            tool_reward,
            {}
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # 使用规则方法计算奖励
        return geo3k.compute_score(
            self._instance_dict[instance_id]["response"],
            self._instance_dict[instance_id]["ground_truth"],
            use_boxed=False,
            format_score=0.0,
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        # 清理实例状态
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
```

#### 2. 工具配置

```yaml
# examples/sglang_multiturn/config/tool_config/geo3k_tool_config.yaml
tools:
  - class_name: "verl.tools.geo3k_tool.Geo3kTool"
    config:
      type: native
    tool_schema:
      type: "function"
      function:
        name: "calc_geo3k_reward"
        description: "A tool for calculating the reward of geo3k. (1.0 if parsed answer is correct, 0.0 if parsed answer is incorrect or not correctly parsed)"
        parameters:
          type: "object"
          properties:
            answer:
              type: "string"
              description: "The model's answer to the geo3k problem, must be a digits"
          required: ["answer"]
```

#### 3. 数据集预处理

```python
# examples/data_preprocess/geo3k_multiturn_w_tool.py

def make_map_fn(split):
    def process_fn(example, idx):
        problem = example.pop("problem")
        answer = example.pop("answer")
        images = example.pop("images")

        prompt = problem + " " + instruction_following

        data = {
            "data_source": "hiyouga/geometry3k",
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "You are a math expert. "
                        "You should use the `calc_geo3k_reward` tool "
                        "after solving the question."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "images": images,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer,
                "question": problem,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "calc_geo3k_reward": {
                        "create_kwargs": {"ground_truth": answer},
                        # execute_kwargs, calc_reward_kwargs, release_kwargs 都为空
                    },
                },
            },
        }
        return data

    return process_fn
```

#### 4. 训练配置

```yaml
# examples/sglang_multiturn/config/geo3k_multiturn_megatron_grpo.yaml

# 数据配置
data:
  train_files:
    - /path/to/geo3k_multiturn_w_tool/train.parquet
  val_files:
    - /path/to/geo3k_multiturn_w_tool/test.parquet

# 工具配置
actor_rollout_ref:
  rollout:
    tool_config: config/tool_config/geo3k_tool_config.yaml
    multi_turn:
      enable: true
      max_assistant_turns: 5

# 奖励配置
reward_model:
  enable: false  # 使用工具奖励而非奖励模型
```

#### 5. 运行训练

```bash
python -m verl.trainer.main_grpo \
  --config-name geo3k_multiturn_megatron_grpo \
  data.train_files='["/path/to/train.parquet"]' \
  data.val_files='["/path/to/test.parquet"]'
```

---

## 高级用例

### 多工具协同示例

#### Video-Holmes 数据集（视频问答）

**数据集预处理**（`video_holmes_multiturn_w_tool.py`）
```python
"tools_kwargs": {
    # 工具1：视频帧提取
    "execute_frame_extraction": {
        "create_kwargs": {
            "video_path": "/path/to/video.mp4",
            "num_frames": 4,
            "resolution": 224,
        },
    },

    # 工具2：目标检测
    "execute_object_detection": {
        "create_kwargs": {
            "resolution": 448,
        },
    },

    # 工具3：帧描述生成
    "execute_frame_caption": {
        "create_kwargs": {
            "model_name": "blip2",
        },
    },

    # 工具4：答案验证与奖励计算
    "calc_video_reward": {
        "create_kwargs": {
            "ground_truth": answer,
        },
    },
},
```

**工具调用流程**：
```
1. CREATE 所有工具实例
   ├─ execute_frame_extraction: 准备视频路径
   ├─ execute_object_detection: 准备检测器
   ├─ execute_frame_caption: 准备描述模型
   └─ calc_video_reward: 准备正确答案

2. 模型推理
   ├─ 调用 execute_frame_extraction → 返回帧图像
   ├─ 调用 execute_object_detection → 返回检测结果
   ├─ 调用 execute_frame_caption → 返回帧描述
   ├─ 模型综合信息生成答案
   └─ 调用 calc_video_reward → 验证答案并返回反馈

3. 奖励计算
   └─ calc_video_reward.calc_reward() → 最终奖励

4. 释放资源
   └─ 所有工具调用 release()
```

### 带外部 API 的工具

#### 搜索工具示例

```python
# verl/tools/search_tool.py
class SearchTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.api_key = config.get("api_key")
        self.search_engine = config.get("engine", "google")
        self._instance_dict = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs):
        if instance_id is None:
            instance_id = str(uuid4())

        self._instance_dict[instance_id] = {
            "search_history": [],
            "total_calls": 0,
        }

        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        timeout: int = 30,  # execute_kwargs
        max_results: int = 5,  # execute_kwargs
        **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        query = parameters.get("query", "")

        # 调用外部搜索 API
        results = await self._call_search_api(
            query,
            timeout=timeout,
            max_results=max_results
        )

        # 更新状态
        self._instance_dict[instance_id]["search_history"].append({
            "query": query,
            "results": results,
        })
        self._instance_dict[instance_id]["total_calls"] += 1

        # 计算即时奖励（可能是负的，作为成本）
        api_cost = -0.01  # 每次调用的成本

        # 格式化响应
        formatted_results = "\n".join([
            f"{i+1}. {r['title']}: {r['snippet']}"
            for i, r in enumerate(results)
        ])

        return (
            ToolResponse(text=f"Search results:\n{formatted_results}"),
            api_cost,
            {"num_results": len(results), "total_calls": self._instance_dict[instance_id]["total_calls"]}
        )

    async def _call_search_api(self, query: str, timeout: int, max_results: int):
        # 实际的 API 调用逻辑
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://api.{self.search_engine}.com/search",
                params={"q": query, "num": max_results},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=timeout
            ) as response:
                return await response.json()
```

**数据集配置**：
```python
"tools_kwargs": {
    "web_search": {
        "create_kwargs": {},  # 无需样本特定参数
        "execute_kwargs": {
            "timeout": 30,
            "max_results": 5,
        },
    },
},
```

**工具配置**：
```yaml
tools:
  - class_name: "verl.tools.search_tool.SearchTool"
    config:
      type: native
      api_key: "${SEARCH_API_KEY}"  # 从环境变量读取
      engine: "google"
    tool_schema:
      type: "function"
      function:
        name: "web_search"
        description: "Search the web for information"
        parameters:
          type: "object"
          properties:
            query:
              type: "string"
              description: "The search query"
          required: ["query"]
```

---

## 最佳实践

### 1. 参数设计原则

#### ✅ 正确的参数分配

```python
# CREATE_KWARGS - 样本特定的、静态的、数据驱动的
"create_kwargs": {
    "ground_truth": answer,           # ✅ 每个样本不同
    "problem_type": "geometry",       # ✅ 样本元数据
    "difficulty": "hard",             # ✅ 样本属性
}

# EXECUTE_KWARGS - 配置性的、全局的、可调的
"execute_kwargs": {
    "timeout": 30,                    # ✅ 执行超时配置
    "max_retries": 3,                 # ✅ 重试策略
    "verbose": True,                  # ✅ 调试选项
}

# PARAMETERS (模型生成) - 动态的、推理时决定的
{
    "answer": "42",                   # ✅ 模型生成的答案
    "confidence": 0.95,               # ✅ 模型的置信度
}
```

#### ❌ 错误的参数分配

```python
# 不要把模型生成的参数放在 create_kwargs 中
"create_kwargs": {
    "answer": "42",  # ❌ 这应该由模型生成
}

# 不要把样本特定的参数放在 execute_kwargs 中
"execute_kwargs": {
    "ground_truth": answer,  # ❌ 这应该在 create_kwargs
}
```

### 2. 工具状态管理

#### 使用实例字典

```python
class YourTool(BaseTool):
    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        # ✅ 使用字典管理多个实例
        self._instance_dict = {}

    async def create(self, instance_id, **kwargs):
        # ✅ 每个实例独立的状态
        self._instance_dict[instance_id] = {
            "data": [],
            "state": "initialized",
        }
```

#### 避免全局状态

```python
class BadTool(BaseTool):
    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        # ❌ 全局状态会被多个实例污染
        self.global_data = []
        self.global_state = "initialized"
```

### 3. 奖励设计模式

#### 即时奖励 vs 最终奖励

```python
@rollout_trace_op
async def execute(self, instance_id, parameters, **kwargs):
    # 即时奖励：步骤级的反馈
    step_reward = 0.0

    if self._is_action_valid(parameters):
        step_reward += 0.1  # 奖励有效操作
    else:
        step_reward -= 0.05  # 惩罚无效操作

    if self._is_action_redundant(instance_id, parameters):
        step_reward -= 0.02  # 惩罚重复操作

    return ToolResponse(...), step_reward, {}

async def calc_reward(self, instance_id, **kwargs):
    # 最终奖励：基于整个轨迹的评估
    state = self._instance_dict[instance_id]

    # 任务完成奖励
    if self._is_task_completed(state):
        final_reward = 1.0
    else:
        final_reward = 0.0

    # 效率奖励
    efficiency_bonus = 1.0 / (state["num_steps"] + 1)

    return final_reward + efficiency_bonus
```

#### 奖励组合策略

```python
# 总奖励 = Σ(即时奖励) + 最终奖励
total_reward = sum(step_rewards) + final_reward

# 示例：
# step_rewards = [0.1, -0.05, 0.1, 0.0]  # 每次 execute 返回
# final_reward = 1.0                      # calc_reward 返回
# total_reward = 0.15 + 1.0 = 1.15
```

### 4. 错误处理

```python
@rollout_trace_op
async def execute(self, instance_id, parameters, **kwargs):
    try:
        # 验证参数
        if not self._validate_parameters(parameters):
            return (
                ToolResponse(text="Invalid parameters"),
                -0.1,  # 惩罚
                {"error": "invalid_params"}
            )

        # 执行工具逻辑
        result = await self._do_work(instance_id, parameters)

        return (
            ToolResponse(text=f"Success: {result}"),
            0.1,
            {"status": "success"}
        )

    except TimeoutError:
        return (
            ToolResponse(text="Operation timed out"),
            -0.05,
            {"error": "timeout"}
        )

    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return (
            ToolResponse(text=f"Error: {str(e)}"),
            -0.1,
            {"error": "execution_failed"}
        )
```

### 5. 日志与调试

```python
import logging

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class YourTool(BaseTool):
    @rollout_trace_op  # 自动记录执行轨迹
    async def execute(self, instance_id, parameters, **kwargs):
        logger.info(f"Executing tool for instance {instance_id}")
        logger.debug(f"Parameters: {parameters}")

        result = await self._do_work(parameters)

        logger.info(f"Execution completed: {result}")
        return ToolResponse(text=result), 0.0, {}
```

### 6. 性能优化

#### 异步并发

```python
async def create(self, instance_id, **kwargs):
    # ✅ 并发初始化多个资源
    results = await asyncio.gather(
        self._load_model(),
        self._connect_to_api(),
        self._prepare_cache(),
    )

    self._instance_dict[instance_id] = {"resources": results}
    return instance_id, ToolResponse()
```

#### 资源复用

```python
class YourTool(BaseTool):
    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        # ✅ 共享的、重量级的资源
        self._shared_model = self._load_heavy_model()
        self._api_client = self._create_api_client()
        # ✅ 实例特定的、轻量级的状态
        self._instance_dict = {}
```

### 7. 测试工具

```python
# tests/experimental/agent_loop/test_your_tool.py
import pytest
from verl.tools.your_tool import YourTool
from verl.tools.schemas import OpenAIFunctionToolSchema


@pytest.mark.asyncio
async def test_tool_lifecycle():
    # 创建工具
    tool = YourTool(
        config={"type": "native"},
        tool_schema=OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {"name": "your_tool", "parameters": {...}},
        })
    )

    # 测试 CREATE
    instance_id, response = await tool.create(ground_truth="answer")
    assert instance_id is not None

    # 测试 EXECUTE
    tool_response, reward, metrics = await tool.execute(
        instance_id,
        {"input": "test"}
    )
    assert reward is not None

    # 测试 CALC_REWARD
    final_reward = await tool.calc_reward(instance_id)
    assert final_reward >= 0.0

    # 测试 RELEASE
    await tool.release(instance_id)
    assert instance_id not in tool._instance_dict
```

---

## 总结

### 组件关系图

```
┌─────────────────────────────────────────────────────────────┐
│                      Training Pipeline                       │
└─────────────────────────────────────────────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │   YAML   │   │  Python  │   │  Parquet │
        │  Config  │   │   Tool   │   │ Dataset  │
        └──────────┘   └──────────┘   └──────────┘
                │              │              │
                │   provides   │   contains   │
                │   schema     │   kwargs     │
                │              │              │
                └──────────────┼──────────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │  Rollout Worker    │
                    │  (sglang_rollout)  │
                    └────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
      ┌─────────┐        ┌─────────┐       ┌─────────┐
      │ CREATE  │───────▶│ EXECUTE │──────▶│ REWARD  │
      │ (once)  │        │ (multi) │       │ (once)  │
      └─────────┘        └─────────┘       └─────────┘
            │                  │                  │
            ▼                  ▼                  ▼
      create_kwargs      execute_kwargs   calc_reward_kwargs
      (from dataset)     (from dataset)   (from dataset)
                               │
                               ▼
                         ┌─────────┐
                         │ RELEASE │
                         │ (once)  │
                         └─────────┘
                               │
                               ▼
                      release_kwargs
                      (from dataset)
```

### 快速检查清单

开发新工具时，确保完成以下步骤：

- [ ] **Python 工具实现**
  - [ ] 继承 `BaseTool`
  - [ ] 实现 `create()`, `execute()`, `calc_reward()`, `release()`
  - [ ] 使用 `_instance_dict` 管理状态
  - [ ] 添加 `@rollout_trace_op` 装饰器到 `execute()`

- [ ] **YAML 配置文件**
  - [ ] 设置 `class_name` 为工具类完整路径
  - [ ] 配置 `tool_schema` (OpenAI 函数格式)
  - [ ] 确保 `function.name` 与工具类 `self.name` 一致

- [ ] **数据集预处理**
  - [ ] 设置 `need_tools_kwargs: True`
  - [ ] 添加 `tools_kwargs` 字典
  - [ ] 为每个工具配置 `create_kwargs`
  - [ ] 按需配置 `execute_kwargs`, `calc_reward_kwargs`, `release_kwargs`

- [ ] **训练配置**
  - [ ] 指定 `tool_config` 路径
  - [ ] 启用 `multi_turn.enable: true`
  - [ ] 设置 `max_assistant_turns`

- [ ] **测试**
  - [ ] 编写单元测试验证工具生命周期
  - [ ] 运行小规模训练验证集成

---

## 参考资料

- **工具基类**: `verl/tools/base_tool.py`
- **示例工具**:
  - `verl/tools/geo3k_tool.py` - 简单的奖励验证工具
  - `verl/tools/search_tool.py` - 外部 API 调用工具
  - `verl/tools/sandbox_fusion_tools.py` - 沙箱执行工具
- **Rollout 实现**: `verl/workers/rollout/sglang_rollout/sglang_rollout.py`
- **数据预处理示例**: `examples/data_preprocess/geo3k_multiturn_w_tool.py`
- **配置示例**: `examples/sglang_multiturn/config/tool_config/`

---

**文档版本**: 1.0
**最后更新**: 2025-10-07
