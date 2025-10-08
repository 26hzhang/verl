# Tool Parser 注册流程详解（Step by Step）

本文档以 `geo3k_tool_config.yaml` 为例，详细分析从配置文件到工具注册的完整流程，包含每个步骤的代码片段、输入和输出。

---

## 前置条件

**配置文件：** `examples/sglang_multiturn/config/tool_config/geo3k_tool_config.yaml`

```yaml
tools:
  - class_name: "verl.tools.geo3k_tool.Geo3kTool"
    config:
      type: native
    tool_schema:
      type: "function"
      function:
        name: "calc_geo3k_reward"
        description: "A tool for calculating the reward of geo3k."
        parameters:
          type: "object"
          properties:
            answer:
              type: "string"
              description: "The model's answer to the geo3k problem"
          required: ["answer"]
```

---

## 完整流程图

```
启动训练
    ↓
ToolAgentLoop.initialize_class()
    ↓
initialize_tools_from_config()
    ├─ OmegaConf.load()
    ├─ get_tool_class()
    │   ├─ importlib 动态导入
    │   └─ getattr() 获取类
    ├─ OpenAIFunctionToolSchema.model_validate()
    └─ Geo3kTool(config, tool_schema)
    ↓
构建工具字典和 schemas
    ↓
ToolParser.get_tool_parser()
    ├─ 查找 _registry
    └─ 实例化 HermesToolParser
```

---

## Step 1: 加载配置文件

### 执行位置
`verl/experimental/agent_loop/tool_agent_loop.py:96-97`

### 代码片段
```python
tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
```

### 输入
```python
tool_config_path = "/path/to/examples/sglang_multiturn/config/tool_config/geo3k_tool_config.yaml"
```

### 输出
```python
# 触发 initialize_tools_from_config() 函数调用
```

---

## Step 2: 初始化工具配置

### 执行位置
`verl/tools/utils/tool_registry.py:81-82`

### 代码片段
```python
def initialize_tools_from_config(tools_config_file):
    tools_config = OmegaConf.load(tools_config_file)
    tool_list = []
```

### 输入
```python
tools_config_file = "/path/to/geo3k_tool_config.yaml"
```

### 输出（OmegaConf 对象）
```python
tools_config = DictConfig({
    'tools': [
        {
            'class_name': 'verl.tools.geo3k_tool.Geo3kTool',
            'config': {'type': 'native'},
            'tool_schema': {
                'type': 'function',
                'function': {
                    'name': 'calc_geo3k_reward',
                    'description': 'A tool for calculating the reward of geo3k.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'answer': {
                                'type': 'string',
                                'description': "The model's answer to the geo3k problem"
                            }
                        },
                        'required': ['answer']
                    }
                }
            }
        }
    ]
})
```

---

## Step 3: 遍历工具配置

### 执行位置
`verl/tools/utils/tool_registry.py:96-98`

### 代码片段
```python
for tool_config in tools_config.tools:
    cls_name = tool_config.class_name
    tool_type = ToolType(tool_config.config.type)
    tool_cls = get_tool_class(cls_name)
```

### 输入
```python
tool_config = DictConfig({
    'class_name': 'verl.tools.geo3k_tool.Geo3kTool',
    'config': {'type': 'native'},
    'tool_schema': {...}
})
```

### 中间变量
```python
cls_name = "verl.tools.geo3k_tool.Geo3kTool"
tool_type = ToolType.NATIVE  # enum 值
```

---

## Step 4: 动态加载工具类

### 执行位置
`verl/tools/utils/tool_registry.py:67-78`

### 代码片段
```python
def get_tool_class(cls_name):
    # 步骤 4.1: 分离模块名和类名
    module_name, class_name = cls_name.rsplit(".", 1)

    # 步骤 4.2: 检查模块是否已导入
    if module_name not in sys.modules:
        # 步骤 4.3: 查找模块规范
        spec = importlib.util.find_spec(module_name)
        # 步骤 4.4: 从规范创建模块对象
        module = importlib.util.module_from_spec(spec)
        # 步骤 4.5: 注册到 sys.modules
        sys.modules[module_name] = module
        # 步骤 4.6: 执行模块代码
        spec.loader.exec_module(module)
    else:
        module = sys.modules[module_name]

    # 步骤 4.7: 获取类对象
    tool_cls = getattr(module, class_name)
    return tool_cls
```

### 详细执行过程

#### 步骤 4.1: 分离模块名和类名
```python
# 输入
cls_name = "verl.tools.geo3k_tool.Geo3kTool"

# 执行
module_name, class_name = cls_name.rsplit(".", 1)

# 输出
module_name = "verl.tools.geo3k_tool"
class_name = "Geo3kTool"
```

#### 步骤 4.2-4.3: 查找模块
```python
# 输入
module_name = "verl.tools.geo3k_tool"

# 执行
spec = importlib.util.find_spec(module_name)

# 输出（ModuleSpec 对象）
spec = ModuleSpec(
    name='verl.tools.geo3k_tool',
    loader=<_frozen_importlib_external.SourceFileLoader>,
    origin='/path/to/verl/tools/geo3k_tool.py',
    ...
)
```

#### 步骤 4.4-4.6: 创建并执行模块
```python
# 输入
spec = ModuleSpec(...)

# 执行
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

# 这时会执行 geo3k_tool.py 文件中的所有代码
# 包括类定义、导入语句等
```

#### 步骤 4.7: 获取类对象
```python
# 输入
module = <module 'verl.tools.geo3k_tool'>
class_name = "Geo3kTool"

# 执行
tool_cls = getattr(module, class_name)

# 输出
tool_cls = <class 'verl.tools.geo3k_tool.Geo3kTool'>
```

### 最终输出
```python
# get_tool_class() 返回值
return Geo3kTool  # 类对象（不是实例）
```

---

## Step 5: 验证 Tool Schema

### 执行位置
`verl/tools/utils/tool_registry.py:105-106`

### 代码片段
```python
tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
```

### 详细执行过程

#### 步骤 5.1: 转换为 Python 字典
```python
# 输入（OmegaConf DictConfig）
tool_config.tool_schema = DictConfig({
    'type': 'function',
    'function': {
        'name': 'calc_geo3k_reward',
        'description': '...',
        'parameters': {...}
    }
})

# 执行
tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)

# 输出（原生 Python dict）
tool_schema_dict = {
    'type': 'function',
    'function': {
        'name': 'calc_geo3k_reward',
        'description': 'A tool for calculating the reward of geo3k.',
        'parameters': {
            'type': 'object',
            'properties': {
                'answer': {
                    'type': 'string',
                    'description': "The model's answer to the geo3k problem"
                }
            },
            'required': ['answer']
        }
    }
}
```

#### 步骤 5.2: Pydantic 验证和构造
```python
# 输入
tool_schema_dict = {'type': 'function', 'function': {...}}

# 执行
tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)

# 输出（Pydantic 模型实例）
tool_schema = OpenAIFunctionToolSchema(
    type='function',
    function=FunctionDefinition(
        name='calc_geo3k_reward',
        description='A tool for calculating the reward of geo3k.',
        parameters={
            'type': 'object',
            'properties': {
                'answer': {
                    'type': 'string',
                    'description': "The model's answer to the geo3k problem"
                }
            },
            'required': ['answer']
        }
    )
)
```

---

## Step 6: 实例化工具

### 执行位置
`verl/tools/utils/tool_registry.py:107-110`

### 代码片段
```python
tool = tool_cls(
    config=OmegaConf.to_container(tool_config.config, resolve=True),
    tool_schema=tool_schema,
)
tool_list.append(tool)
```

### 详细执行过程

#### 步骤 6.1: 准备参数
```python
# 转换 config
config_dict = OmegaConf.to_container(tool_config.config, resolve=True)
# 输出
config_dict = {'type': 'native'}

# tool_schema 已在 Step 5 创建
tool_schema = OpenAIFunctionToolSchema(...)
```

#### 步骤 6.2: 调用构造函数
```python
# 执行
tool = Geo3kTool(
    config={'type': 'native'},
    tool_schema=OpenAIFunctionToolSchema(...)
)
```

#### 步骤 6.3: Geo3kTool.__init__() 内部执行

**位置：** `verl/tools/geo3k_tool.py:41-62`

```python
def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
    # 步骤 6.3.1: 调用父类构造函数
    super().__init__(config, tool_schema)
    # BaseTool.__init__() 执行：
    #   self.config = {'type': 'native'}
    #   self.tool_schema = OpenAIFunctionToolSchema(...)
    #   self.name = 'calc_geo3k_reward'  # 从 tool_schema.function.name 提取

    # 步骤 6.3.2: 初始化实例字典
    self._instance_dict = {}
```

### 输出（Geo3kTool 实例）
```python
tool = Geo3kTool(
    config={'type': 'native'},
    tool_schema=OpenAIFunctionToolSchema(
        type='function',
        function=FunctionDefinition(
            name='calc_geo3k_reward',
            description='...',
            parameters={...}
        )
    ),
    name='calc_geo3k_reward',
    _instance_dict={}
)
```

#### 步骤 6.4: 添加到工具列表
```python
# 执行
tool_list.append(tool)

# 输出
tool_list = [
    Geo3kTool(name='calc_geo3k_reward', ...)
]
```

---

## Step 7: 返回工具列表

### 执行位置
`verl/tools/utils/tool_registry.py:129`

### 代码片段
```python
return tool_list
```

### 输出
```python
tool_list = [
    Geo3kTool(
        config={'type': 'native'},
        tool_schema=OpenAIFunctionToolSchema(...),
        name='calc_geo3k_reward',
        _instance_dict={}
    )
]
```

---

## Step 8: 构建工具字典

### 执行位置
`verl/experimental/agent_loop/tool_agent_loop.py:98`

### 代码片段
```python
cls.tools = {tool.name: tool for tool in tool_list}
```

### 执行过程
```python
# 输入
tool_list = [Geo3kTool(name='calc_geo3k_reward', ...)]

# 执行字典推导
cls.tools = {tool.name: tool for tool in tool_list}

# 等价于
cls.tools = {}
for tool in tool_list:
    cls.tools[tool.name] = tool

# 输出
cls.tools = {
    'calc_geo3k_reward': Geo3kTool(...)
}
```

---

## Step 9: 构建工具 Schema 列表

### 执行位置
`verl/experimental/agent_loop/tool_agent_loop.py:99`

### 代码片段
```python
cls.tool_schemas = [
    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
    for tool in tool_list
]
```

### 执行过程
```python
# 输入
tool_list = [Geo3kTool(name='calc_geo3k_reward', tool_schema=...)]

# 对每个工具执行 model_dump()
tool_schema_dict = tool.tool_schema.model_dump(
    exclude_unset=True,  # 排除未设置的字段
    exclude_none=True    # 排除值为 None 的字段
)

# 输出（Pydantic 转为字典）
tool_schema_dict = {
    'type': 'function',
    'function': {
        'name': 'calc_geo3k_reward',
        'description': 'A tool for calculating the reward of geo3k.',
        'parameters': {
            'type': 'object',
            'properties': {
                'answer': {
                    'type': 'string',
                    'description': "The model's answer to the geo3k problem"
                }
            },
            'required': ['answer']
        }
    }
}

# 最终输出
cls.tool_schemas = [
    {
        'type': 'function',
        'function': {
            'name': 'calc_geo3k_reward',
            'description': '...',
            'parameters': {...}
        }
    }
]
```

---

## Step 10: 初始化 Tool Parser

### 执行位置
`verl/experimental/agent_loop/tool_agent_loop.py:100`

### 代码片段
```python
cls.tool_parser = ToolParser.get_tool_parser(
    config.actor_rollout_ref.rollout.multi_turn.format,
    cls.tokenizer
)
```

### 详细执行过程

#### 步骤 10.1: 准备参数
```python
# 从配置获取 format
name = config.actor_rollout_ref.rollout.multi_turn.format  # "hermes"
tokenizer = cls.tokenizer  # 已初始化的 tokenizer 对象
```

#### 步骤 10.2: 调用工厂方法

**位置：** `verl/experimental/agent_loop/tool_parser.py:60-64`

```python
@classmethod
def get_tool_parser(cls, name: str, tokenizer):
    # 步骤 10.2.1: 查找注册表
    if name not in cls._registry:
        raise ValueError(f"Unknown tool parser: {name}")

    # 步骤 10.2.2: 获取 parser 类
    parser_cls = cls._registry[name]

    # 步骤 10.2.3: 实例化 parser
    return parser_cls(tokenizer)
```

#### 步骤 10.2.1: 查找注册表
```python
# 输入
name = "hermes"
cls._registry = {
    "hermes": HermesToolParser,
    # 其他已注册的 parser...
}

# 执行检查
if "hermes" not in cls._registry:  # False，已注册
    raise ValueError(...)

# 继续执行
```

#### 步骤 10.2.2: 获取 parser 类
```python
# 执行
parser_cls = cls._registry["hermes"]

# 输出
parser_cls = <class 'HermesToolParser'>  # 类对象
```

#### 步骤 10.2.3: 实例化 parser
```python
# 执行
return HermesToolParser(tokenizer)
```

#### 步骤 10.3: HermesToolParser.__init__() 内部执行

**位置：** `verl/experimental/agent_loop/tool_parser.py:79-84`

```python
def __init__(self, tokenizer) -> None:
    # 步骤 10.3.1: 调用父类构造函数
    super().__init__(tokenizer)
    # ToolParser.__init__() 执行：
    #   self.tokenizer = tokenizer

    # 步骤 10.3.2: 初始化 parser 特定属性
    self.tool_call_start_token = "<tool_call>"
    self.tool_call_end_token = "</tool_call>"
    self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
```

### 输出（HermesToolParser 实例）
```python
cls.tool_parser = HermesToolParser(
    tokenizer=<PreTrainedTokenizerFast>,
    tool_call_start_token="<tool_call>",
    tool_call_end_token="</tool_call>",
    tool_call_regex=re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
)
```

---

## Step 11: 完成初始化

### 执行位置
`verl/experimental/agent_loop/tool_agent_loop.py:101`

### 代码片段
```python
print(f"Initialized tools: {cls.tools}")
```

### 输出（控制台打印）
```
Initialized tools: {'calc_geo3k_reward': <verl.tools.geo3k_tool.Geo3kTool object at 0x...>}
```

---

## 最终状态总结

完成所有步骤后，`ToolAgentLoop` 类的状态：

```python
class ToolAgentLoop:
    # 工具字典：名称 -> 工具实例
    tools = {
        'calc_geo3k_reward': Geo3kTool(
            config={'type': 'native'},
            tool_schema=OpenAIFunctionToolSchema(...),
            name='calc_geo3k_reward',
            _instance_dict={}
        )
    }

    # 工具 Schema 列表（用于传递给模型）
    tool_schemas = [
        {
            'type': 'function',
            'function': {
                'name': 'calc_geo3k_reward',
                'description': 'A tool for calculating the reward of geo3k.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'answer': {
                            'type': 'string',
                            'description': "The model's answer"
                        }
                    },
                    'required': ['answer']
                }
            }
        }
    ]

    # Tool Parser 实例
    tool_parser = HermesToolParser(
        tokenizer=<PreTrainedTokenizerFast>,
        tool_call_start_token="<tool_call>",
        tool_call_end_token="</tool_call>",
        tool_call_regex=re.compile(...)
    )
```

---

## 运行时使用示例

### 场景：模型生成响应并调用工具

#### 输入（模型输出）
```
"The answer is 42. <tool_call>{\"name\": \"calc_geo3k_reward\", \"arguments\": {\"answer\": \"42\"}}</tool_call>"
```

#### Step 1: 解析工具调用
```python
# 调用
response_ids = tokenizer.encode(model_output)
content, function_calls = await tool_parser.extract_tool_calls(response_ids)

# tool_parser.extract_tool_calls() 内部执行：
# 1. 解码 token IDs
text = tokenizer.decode(response_ids)
# text = "The answer is 42. <tool_call>{...}</tool_call>"

# 2. 正则匹配
matches = tool_call_regex.findall(text)
# matches = ['{\"name\": \"calc_geo3k_reward\", \"arguments\": {\"answer\": \"42\"}}']

# 3. 解析 JSON
for match in matches:
    function_call = json.loads(match)
    # function_call = {"name": "calc_geo3k_reward", "arguments": {"answer": "42"}}

    function_calls.append(FunctionCall(
        name="calc_geo3k_reward",
        arguments='{"answer": "42"}'
    ))

# 4. 移除工具调用标记
content = tool_call_regex.sub("", text)
# content = "The answer is 42. "

# 输出
content = "The answer is 42. "
function_calls = [
    FunctionCall(name="calc_geo3k_reward", arguments='{"answer": "42"}')
]
```

#### Step 2: 获取工具实例
```python
# 根据名称查找工具
function_call = function_calls[0]
tool_name = function_call.name  # "calc_geo3k_reward"
tool = cls.tools[tool_name]

# 输出
tool = Geo3kTool(...)
```

#### Step 3: 创建工具实例
```python
# 调用
instance_id, _ = await tool.create(
    instance_id="uuid-abc-123",
    ground_truth="42"
)

# tool.create() 内部执行：
self._instance_dict["uuid-abc-123"] = {
    "response": "",
    "ground_truth": "42",
    "reward": 0.0
}

# 输出
instance_id = "uuid-abc-123"
```

#### Step 4: 执行工具
```python
# 解析参数
parameters = json.loads(function_call.arguments)
# parameters = {"answer": "42"}

# 调用
response, tool_reward, metadata = await tool.execute(
    instance_id="uuid-abc-123",
    parameters={"answer": "42"}
)

# tool.execute() 内部执行：
# 1. 提取并存储答案
answer = parameters["answer"]  # "42"
self._instance_dict["uuid-abc-123"]["response"] = "42"

# 2. 计算奖励
reward = await self.calc_reward("uuid-abc-123")
# geo3k.compute_score("42", "42") → 1.0

# 3. 计算工具奖励
old_reward = self._instance_dict["uuid-abc-123"]["reward"]  # 0.0
tool_reward = 0.0 if reward > old_reward else -0.05  # 0.0 (提升了)

# 4. 更新状态
self._instance_dict["uuid-abc-123"]["reward"] = 1.0

# 5. 返回
return (
    ToolResponse(text="Current parsed answer='42' reward=1.0"),
    0.0,
    {}
)

# 输出
response = ToolResponse(text="Current parsed answer='42' reward=1.0")
tool_reward = 0.0
metadata = {}
```

---

## 关键数据结构变化时间线

```
时间点 0: 启动
├─ tools = {}
├─ tool_schemas = []
└─ tool_parser = None

时间点 1: 加载配置后
├─ tool_config_path = "/path/to/geo3k_tool_config.yaml"
└─ 触发 initialize_tools_from_config()

时间点 2: 工具类加载后
├─ tool_cls = <class Geo3kTool>
└─ tool_schema = OpenAIFunctionToolSchema(...)

时间点 3: 工具实例化后
├─ tool_list = [Geo3kTool(...)]
└─ tool._instance_dict = {}

时间点 4: 构建字典后
├─ tools = {"calc_geo3k_reward": Geo3kTool(...)}
└─ tool_schemas = [{...}]

时间点 5: Parser 初始化后
├─ tool_parser = HermesToolParser(...)
└─ 系统就绪

时间点 6: 运行时 - 创建实例
└─ tool._instance_dict["uuid-123"] = {"response": "", "ground_truth": "42", "reward": 0.0}

时间点 7: 运行时 - 执行工具
└─ tool._instance_dict["uuid-123"] = {"response": "42", "ground_truth": "42", "reward": 1.0}
```

---

## 总结

这个 step-by-step 流程展示了从 YAML 配置文件到可用工具的完整转换过程：

1. **配置解析**：OmegaConf 加载 YAML → DictConfig 对象
2. **动态导入**：importlib 根据字符串路径导入类
3. **Schema 验证**：Pydantic 验证和构造 OpenAI 格式的工具描述
4. **实例化**：创建具体的工具实例（如 Geo3kTool）
5. **注册管理**：构建字典和列表便于查找和使用
6. **Parser 初始化**：基于注册表创建对应的解析器
7. **运行时使用**：解析模型输出、调用工具、计算奖励

每个步骤都有明确的输入、处理逻辑和输出，形成了一个完整的工具注册和使用链条。
