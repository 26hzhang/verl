# Tool Parser 注册和使用指南

本文档以 Geo3kTool 为例，详细说明 verl 中 tool parser 的注册机制。

## 目录

1. [概述](#概述)
2. [架构组件](#架构组件)
3. [注册流程](#注册流程)
4. [配置系统](#配置系统)
5. [运行时工作流程](#运行时工作流程)
6. [添加自定义 Tool Parser](#添加自定义-tool-parser)
7. [示例：Geo3kTool](#示例geo3ktool)

---

## 概述

verl 实现了 tool parser 的**注册表模式**，具有以下特性：
- 在导入时动态注册 tool parser
- 配置驱动的 parser 选择
- 可扩展支持不同的工具调用格式

系统关注点分离：
- **Tool Parsers（工具解析器）**：从模型输出中提取工具调用
- **Tools（工具）**：执行工具逻辑并计算奖励
- **Configuration（配置）**：基于 YAML 的声明式配置

---

## 架构组件

### 1. 核心类

#### `ToolParser` (抽象基类)

**位置：** `verl/experimental/agent_loop/tool_parser.py:42-72`

```python
class ToolParser(ABC):
    _registry: dict[str, type["ToolParser"]] = {}  # 类级别注册表

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    @abstractmethod
    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        """从模型响应的 token ID 中提取工具调用。"""
        raise NotImplementedError

    @classmethod
    def get_tool_parser(cls, name: str, tokenizer):
        """工厂方法：检索已注册的 parser。"""
        if name not in cls._registry:
            raise ValueError(f"Unknown tool parser: {name}")
        return cls._registry[name](tokenizer)

    @classmethod
    def register(cls, name: str):
        """装饰器：注册 parser 实现。"""
        def decorator(subclass: type[ToolParser]) -> type[ToolParser]:
            cls._registry[name] = subclass
            return subclass
        return decorator
```

**关键设计模式：**
- **抽象基类**：定义接口契约
- **注册表模式**：类级别的 `_registry` 字典存储实现
- **工厂方法**：`get_tool_parser()` 通过名称创建实例
- **装饰器注册**：`@ToolParser.register()` 自动注册子类

#### `FunctionCall` (数据模型)

**位置：** `verl/experimental/agent_loop/tool_parser.py:29-40`

```python
class FunctionCall(BaseModel):
    arguments: str  # JSON 格式的参数
    name: str       # 要调用的函数名称
```

#### `BaseTool` (抽象基类)

**位置：** `verl/tools/base_tool.py:23-92`

```python
class BaseTool(ABC):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        self.config = config
        self.tool_schema = tool_schema
        self.name = tool_schema.function.name

    @abstractmethod
    async def create(self, instance_id: str, **kwargs) -> tuple[str, ToolResponse]:
        """为轨迹初始化工具实例。"""
        pass

    @abstractmethod
    async def execute(self, instance_id: str, parameters: dict, **kwargs) -> tuple[ToolResponse, float, dict]:
        """执行工具并返回响应、奖励和元数据。"""
        pass

    @abstractmethod
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """计算工具使用的奖励。"""
        pass
```

---

## 注册流程

### 阶段 1：模块导入（自动注册）

当 Python 导入 `verl/experimental/agent_loop/tool_parser.py` 时：

```python
# 步骤 1：使用装饰器定义类
@ToolParser.register("hermes")
class HermesToolParser(ToolParser):
    ...

# 步骤 2：装饰器立即执行
# 等价于：
# HermesToolParser = ToolParser.register("hermes")(HermesToolParser)

# 步骤 3：更新注册表
# ToolParser._registry["hermes"] = HermesToolParser
```

**结果：** `_registry = {"hermes": HermesToolParser, ...}`

### 阶段 2：配置加载

**配置文件：** `verl/trainer/config/rollout/rollout.yaml:187`

```yaml
multi_turn:
  enable: False
  format: hermes  # 指定使用哪个 parser
  tool_config_path: null
  max_assistant_turns: null
  max_parallel_calls: 1
```

**Hydra 配置组合：**

```
verl/trainer/config/rollout/rollout.yaml (默认配置)
    ↓ 继承
verl/trainer/config/ppo_trainer.yaml
    ↓ 继承
examples/sglang_multiturn/config/geo3k_multiturn_grpo.yaml
    ↓ 命令行覆盖 (可选)
run_qwen2.5-3b_geo3k_multiturn.sh
```

### 阶段 3：运行时实例化

**位置：** `verl/experimental/agent_loop/tool_agent_loop.py:100`

```python
cls.tool_parser = ToolParser.get_tool_parser(
    config.actor_rollout_ref.rollout.multi_turn.format,  # "hermes"
    cls.tokenizer
)
```

**执行流程：**

```python
# 1. 调用工厂方法
ToolParser.get_tool_parser("hermes", tokenizer)

# 2. 在注册表中查找
cls._registry["hermes"]  # 返回 HermesToolParser 类

# 3. 实例化 parser
HermesToolParser(tokenizer)

# 4. 初始化 parser 状态
def __init__(self, tokenizer):
    super().__init__(tokenizer)
    self.tool_call_start_token = "<tool_call>"
    self.tool_call_end_token = "</tool_call>"
    self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
```

---

## 配置系统

### 工具配置文件

**文件：** `examples/sglang_multiturn/config/tool_config/geo3k_tool_config.yaml`

```yaml
tools:
  - class_name: "verl.tools.geo3k_tool.Geo3kTool"  # 完整的 Python 路径
    config:
      type: native  # 工具类型：native | mcp
    tool_schema:  # OpenAI 函数调用格式
      type: "function"
      function:
        name: "calc_geo3k_reward"
        description: "计算 geo3k 问题的奖励（正确为 1.0，错误为 0.0）"
        parameters:
          type: "object"
          properties:
            answer:
              type: "string"
              description: "模型的答案（必须是数字）"
          required: ["answer"]
```

### 工具初始化

**位置：** `verl/tools/utils/tool_registry.py:81-129`

```python
def initialize_tools_from_config(tools_config_file):
    tools_config = OmegaConf.load(tools_config_file)
    tool_list = []

    for tool_config in tools_config.tools:
        # 1. 动态类加载
        cls_name = tool_config.class_name  # "verl.tools.geo3k_tool.Geo3kTool"
        tool_cls = get_tool_class(cls_name)

        # 2. Schema 验证
        tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema)
        tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)

        # 3. 工具实例化
        tool = tool_cls(
            config=OmegaConf.to_container(tool_config.config),
            tool_schema=tool_schema
        )
        tool_list.append(tool)

    return tool_list
```

### 动态类加载

**位置：** `verl/tools/utils/tool_registry.py:67-78`

```python
def get_tool_class(cls_name):
    # 解析："verl.tools.geo3k_tool.Geo3kTool"
    module_name, class_name = cls_name.rsplit(".", 1)

    # 动态导入
    if module_name not in sys.modules:
        spec = importlib.util.find_spec(module_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = sys.modules[module_name]

    # 获取类对象
    tool_cls = getattr(module, class_name)
    return tool_cls
```

---

## 运行时工作流程

### 完整执行流程

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 配置加载                                                  │
├─────────────────────────────────────────────────────────────┤
│ • 加载 YAML: geo3k_tool_config.yaml                         │
│ • 解析 multi_turn.format: "hermes"                          │
│ • 解析 tool_config_path: "path/to/geo3k_tool_config.yaml"  │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 工具初始化                                                │
├─────────────────────────────────────────────────────────────┤
│ initialize_tools_from_config()                              │
│   ├─ 加载工具配置 YAML                                       │
│   ├─ get_tool_class("verl.tools.geo3k_tool.Geo3kTool")     │
│   ├─ 验证 OpenAIFunctionToolSchema                          │
│   └─ 实例化：Geo3kTool(config, tool_schema)                 │
│                                                             │
│ 结果：tools = {"calc_geo3k_reward": Geo3kTool()}            │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Parser 初始化                                             │
├─────────────────────────────────────────────────────────────┤
│ ToolParser.get_tool_parser("hermes", tokenizer)            │
│   ├─ 在 _registry 中查找：{"hermes": HermesToolParser}      │
│   └─ 实例化：HermesToolParser(tokenizer)                    │
│                                                             │
│ 结果：tool_parser = HermesToolParser(tokenizer)             │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 模型生成                                                  │
├─────────────────────────────────────────────────────────────┤
│ 模型输出：                                                   │
│ "The answer is <tool_call>                                  │
│  {\"name\": \"calc_geo3k_reward\", \"arguments\": {\"answer\": \"42\"}}│
│  </tool_call>"                                              │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. 工具调用提取                                              │
├─────────────────────────────────────────────────────────────┤
│ tool_parser.extract_tool_calls(response_ids)               │
│   ├─ 解码：tokenizer.decode(response_ids)                  │
│   ├─ 正则匹配：<tool_call>(.*?)</tool_call>                 │
│   ├─ 解析 JSON：json.loads(match)                          │
│   └─ 返回：(content, [FunctionCall(...)])                  │
│                                                             │
│ 结果：                                                       │
│   content = "The answer is"                                 │
│   function_calls = [                                        │
│     FunctionCall(                                           │
│       name="calc_geo3k_reward",                            │
│       arguments='{"answer": "42"}'                         │
│     )                                                       │
│   ]                                                         │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. 工具实例创建                                              │
├─────────────────────────────────────────────────────────────┤
│ tool = tools["calc_geo3k_reward"]                           │
│ instance_id, _ = await tool.create(                         │
│     instance_id="uuid-123",                                 │
│     ground_truth="42"                                       │
│ )                                                           │
│                                                             │
│ 内部状态：                                                   │
│   _instance_dict["uuid-123"] = {                            │
│     "response": "",                                         │
│     "ground_truth": "42",                                   │
│     "reward": 0.0                                           │
│   }                                                         │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. 工具执行                                                  │
├─────────────────────────────────────────────────────────────┤
│ response, tool_reward, _ = await tool.execute(              │
│     instance_id="uuid-123",                                 │
│     parameters={"answer": "42"}                             │
│ )                                                           │
│                                                             │
│ 执行步骤：                                                   │
│   1. 存储答案：_instance_dict["uuid-123"]["response"] = "42"│
│   2. 计算奖励：geo3k.compute_score("42", "42") → 1.0        │
│   3. 计算 tool_reward：                                     │
│        - 如果奖励提升：0.0                                   │
│        - 如果未提升：-0.05 (惩罚)                            │
│   4. 更新状态：_instance_dict["uuid-123"]["reward"] = 1.0   │
│                                                             │
│ 结果：                                                       │
│   response = ToolResponse(text="Current parsed answer='42' reward=1.0")│
│   tool_reward = 0.0                                         │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. 奖励计算                                                  │
├─────────────────────────────────────────────────────────────┤
│ 最终奖励组合：                                               │
│   • 任务奖励：1.0 (答案正确)                                 │
│   • 工具奖励：0.0 (无惩罚)                                   │
│   • KL 惩罚：单独计算                                        │
└─────────────────────────────────────────────────────────────┘
```

### 关键数据结构

```python
# 初始化后
tools = {
    "calc_geo3k_reward": Geo3kTool(
        config={"type": "native"},
        tool_schema=OpenAIFunctionToolSchema(...),
        _instance_dict={}
    )
}

tool_parser = HermesToolParser(
    tokenizer=tokenizer,
    tool_call_start_token="<tool_call>",
    tool_call_end_token="</tool_call>",
    tool_call_regex=re.compile(...)
)

# tool.create() 后
tool._instance_dict = {
    "uuid-123": {
        "response": "",
        "ground_truth": "42",
        "reward": 0.0
    }
}

# tool.execute() 后
tool._instance_dict = {
    "uuid-123": {
        "response": "42",
        "ground_truth": "42",
        "reward": 1.0
    }
}
```

---

## 添加自定义 Tool Parser

### 步骤 1：实现 Parser 类

创建 `verl/experimental/agent_loop/my_custom_parser.py`：

```python
import asyncio
import json
from verl.experimental.agent_loop.tool_parser import ToolParser, FunctionCall

@ToolParser.register("custom_format")
class CustomToolParser(ToolParser):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        # 初始化 parser 特定状态
        self.start_marker = "[TOOL]"
        self.end_marker = "[/TOOL]"

    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)

        # 自定义解析逻辑
        if self.start_marker not in text:
            return text, []

        # 提取标记之间的工具调用
        import re
        pattern = rf"{self.start_marker}(.*?){self.end_marker}"
        matches = re.findall(pattern, text, re.DOTALL)

        function_calls = []
        for match in matches:
            try:
                data = json.loads(match)
                function_calls.append(
                    FunctionCall(
                        name=data["function"],
                        arguments=json.dumps(data["args"])
                    )
                )
            except Exception as e:
                print(f"解析失败：{e}")

        # 从内容中移除工具调用标记
        content = re.sub(pattern, "", text, flags=re.DOTALL)
        return content, function_calls
```

### 步骤 2：导入 Parser 模块

确保在使用前导入你的 parser。添加到 `verl/experimental/agent_loop/__init__.py`：

```python
from .tool_parser import ToolParser, HermesToolParser
from .my_custom_parser import CustomToolParser  # 导入以触发注册
```

### 步骤 3：配置 Parser

更新配置文件或命令行：

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      format: custom_format  # 使用你注册的名称
```

或通过命令行：

```bash
python -m verl.trainer.main_ppo \
    actor_rollout_ref.rollout.multi_turn.format=custom_format \
    ...
```

### 步骤 4：更新 Chat Template

确保你的模型的 chat template 生成预期格式：

```python
custom_chat_template = """
{%- for message in messages %}
{%- if message.role == "assistant" %}
<|im_start|>assistant
{{ message.content }}
{%- for tool_call in message.tool_calls %}
[TOOL]{"function": "{{ tool_call.name }}", "args": {{ tool_call.arguments | tojson }}}[/TOOL]
{%- endfor %}
<|im_end|>
{%- endif %}
{%- endfor %}
"""
```

---

## 示例：Geo3kTool

### 完整实现

**文件：** `verl/tools/geo3k_tool.py:32-100`

```python
class Geo3kTool(BaseTool):
    """用于计算几何数学问题奖励的工具。"""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}  # 跟踪每个轨迹的状态

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs
    ) -> tuple[str, ToolResponse]:
        """为新轨迹初始化工具实例。"""
        if instance_id is None:
            instance_id = str(uuid4())

        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0
        }
        return instance_id, ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """执行工具并计算奖励。"""
        # 提取答案
        answer = parameters.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)

        # 存储响应
        self._instance_dict[instance_id]["response"] = answer

        # 计算奖励
        reward = await self.calc_reward(instance_id)

        # 对未改进的提交进行惩罚
        old_reward = self._instance_dict[instance_id]["reward"]
        tool_reward = 0.0 if reward > old_reward else -0.05

        # 更新存储的奖励
        self._instance_dict[instance_id]["reward"] = reward

        return (
            ToolResponse(text=f"Current parsed {answer=} {reward=}"),
            tool_reward,
            {}
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """通过与正确答案比较来计算奖励。"""
        return geo3k.compute_score(
            self._instance_dict[instance_id]["response"],
            self._instance_dict[instance_id]["ground_truth"],
            use_boxed=False,
            format_score=0.0
        )

    async def release(self, instance_id: str) -> None:
        """清理工具实例。"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
```

### 使用示例

**脚本：** `examples/sglang_multiturn/geo3k/run_qwen2.5-3b_geo3k_multiturn.sh`

```bash
#!/bin/bash
set -x

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='geo3k_multiturn_grpo' \
    data.train_files=$HOME/data/geo3k_multiturn_w_tool/train.parquet \
    data.val_files=$HOME/data/geo3k_multiturn_w_tool/test.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/geo3k_tool_config.yaml" \
    trainer.total_epochs=15
```

### 预期模型输出

```
<|im_start|>user
What is the length of the hypotenuse if the two legs are 3 and 4?<|im_end|>
<|im_start|>assistant
I need to calculate the hypotenuse using the Pythagorean theorem.
<tool_call>
{"name": "calc_geo3k_reward", "arguments": {"answer": "5"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
Current parsed answer='5' reward=1.0
</tool_response><|im_end|>
<|im_start|>assistant
The length of the hypotenuse is 5.<|im_end|>
```

---

## 配置参考

### 完整配置路径

对于 Geo3kTool，配置层次结构为：

```
1. 基础配置：verl/trainer/config/rollout/rollout.yaml
   └─ multi_turn.format: hermes (默认值)

2. 训练器配置：verl/trainer/config/ppo_trainer.yaml
   └─ 继承 rollout.yaml

3. 示例配置：examples/sglang_multiturn/config/geo3k_multiturn_grpo.yaml
   └─ 覆盖：multi_turn.enable: True
   └─ 覆盖：multi_turn.max_assistant_turns: 5

4. 运行脚本：examples/sglang_multiturn/geo3k/run_qwen2.5-3b_geo3k_multiturn.sh
   └─ 覆盖：multi_turn.tool_config_path: "path/to/geo3k_tool_config.yaml"
```

### 关键配置参数

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      enable: True                    # 启用多轮 agent loop
      format: hermes                  # Tool parser 格式
      tool_config_path: "path.yaml"   # 工具定义
      max_assistant_turns: 5          # 停止前的最大轮数
      max_parallel_calls: 1           # 每轮的工具数
      max_tool_response_length: 256   # 截断限制
      tool_response_truncate_side: middle  # left | middle | right
```

### 环境变量

```bash
# 控制日志详细程度
export VERL_LOGGING_LEVEL=DEBUG  # DEBUG | INFO | WARN | ERROR
```

---

## 故障排查

### 常见问题

1. **"Unknown tool parser: xyz"**
   - **原因：** Parser 未注册或未导入
   - **解决方法：** 确保在使用前导入 parser 模块

2. **Tool not found in tools dict**
   - **原因：** Schema 和函数调用之间的工具名称不匹配
   - **解决方法：** 验证 `tool_schema.function.name` 与模型输出匹配

3. **extract_tool_calls 中的 JSON 解析错误**
   - **原因：** 模型输出不符合预期格式
   - **解决方法：** 更新 chat template 或 parser 正则表达式

4. **Instance ID 错误**
   - **原因：** 在执行前未创建工具实例
   - **解决方法：** 确保在 `tool.execute()` 前调用 `tool.create()`

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查注册表
print(ToolParser._registry)  # {'hermes': <class 'HermesToolParser'>, ...}

# 检查工具 schema
print(tools["calc_geo3k_reward"].tool_schema.model_dump())

# 验证模型输出格式
print(tokenizer.decode(response_ids))
```

---

## 相关文件

- **Parser 基类：** `verl/experimental/agent_loop/tool_parser.py`
- **Tool 基类：** `verl/tools/base_tool.py`
- **Tool 注册表：** `verl/tools/utils/tool_registry.py`
- **Agent Loop：** `verl/experimental/agent_loop/tool_agent_loop.py`
- **Geo3k Tool：** `verl/tools/geo3k_tool.py`
- **配置：** `verl/trainer/config/rollout/rollout.yaml`
- **示例：** `examples/sglang_multiturn/config/geo3k_multiturn_grpo.yaml`

---

## 总结

verl 中的 tool parser 系统提供：

1. **声明式注册：** `@ToolParser.register()` 装饰器
2. **工厂模式：** `get_tool_parser()` 用于动态实例化
3. **配置驱动：** 基于 YAML 的 parser 和工具选择
4. **可扩展设计：** 易于添加自定义 parser 和工具
5. **类型安全：** 使用 Pydantic 模型进行 schema 验证

这种架构能够在多轮 RL 训练工作流程中灵活集成工具，同时保持清晰的关注点分离。
