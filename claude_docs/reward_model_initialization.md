# VeRL Reward Model 初始化机制分析

本文档详细分析了VeRL框架中reward model的初始化机制，包括rule-based和model-based两种方式。

## 概述

VeRL支持两种类型的奖励计算：
1. **Rule-based Reward**：基于规则的奖励，通过预定义的计算函数评估模型输出
2. **Model-based Reward**：基于模型的奖励，使用训练好的神经网络模型评分

## 1. Rule-based Reward 初始化流程

### 1.1 调用链路

```
ray_trainer.py
├── 初始化时提供 reward_fn
├── compute_reward() / compute_reward_async()
└── load_reward_manager()
    ├── get_custom_reward_fn() → 自定义奖励函数
    └── default_compute_score → 默认计算函数
```

### 1.2 核心代码分析

#### 奖励管理器加载 (`verl/trainer/ppo/reward.py`)

```python
def load_reward_manager(
    config: DictConfig,
    tokenizer: Any,
    num_examine: int,
    **reward_kwargs: Any
) -> AbstractRewardManager:
    # 1. 获取奖励管理器类型（默认为naive）
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)

    # 2. 尝试加载自定义奖励函数
    compute_score = get_custom_reward_fn(config)

    # 3. 如果没有自定义函数，使用默认计算函数
    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        if sandbox_config:
            # 使用沙箱环境执行代码
            final_compute_score = partial(
                default_compute_score,
                sandbox_fusion_url=sandbox_url,
                ...
            )
        else:
            # 使用内置奖励函数
            final_compute_score = default_compute_score

    # 4. 实例化奖励管理器
    return reward_manager_cls(
        tokenizer=tokenizer,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs
    )
```

#### 默认计算函数 (`verl/utils/reward_score/__init__.py`)

```python
def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    ...
):
    # 根据数据源选择对应的评分函数
    if data_source == "openai/gsm8k":
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", ...]:
        from . import math_reward
        res = math_reward.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo":
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)
    # ... 更多数据源
    else:
        raise NotImplementedError(f"Reward function not implemented for {data_source}")

    return float(res)
```

### 1.3 自定义奖励函数

通过配置文件指定自定义奖励函数：

```yaml
custom_reward_function:
  path: "/path/to/my_reward.py"
  name: "compute_custom_reward"
  reward_kwargs:
    threshold: 0.9
    penalty: -1.0
```

自定义函数示例：
```python
def compute_custom_reward(data_source, solution_str, ground_truth, threshold=0.8):
    # 自定义评分逻辑
    score = evaluate_solution(solution_str, ground_truth)
    return score if score > threshold else 0.0
```

### 1.4 奖励管理器实现

`NaiveRewardManager` 是最基础的实现：

```python
@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    def __call__(self, data: DataProto, return_dict: bool = False):
        # 如果已有奖励分数，直接返回
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        # 否则计算奖励
        for i in range(len(data)):
            # 解码生成的文本
            response_str = self.tokenizer.decode(valid_response_ids)

            # 调用计算函数
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            # 在最后一个token位置设置奖励
            reward_tensor[i, valid_response_length - 1] = score
```

## 2. Model-based Reward 初始化流程

### 2.1 配置启用

```yaml
reward_model:
  enable: true  # 启用模型奖励
  strategy: "fsdp"  # 或 "fsdp2", "megatron"
  model:
    path: "/path/to/reward_model"
    fsdp_config:
      param_offload: true
```

### 2.2 初始化流程

在 `ray_trainer.py` 中：

```python
# 判断是否需要使用奖励模型
self.use_rm = need_reward_model(self.role_worker_mapping)

# 创建奖励模型worker
if self.use_rm:
    resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
    rm_cls = RayClassWithInitArgs(
        self.role_worker_mapping[Role.RewardModel],
        config=self.config.reward_model
    )
    self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

# 初始化模型
if self.use_rm:
    self.rm_wg = all_wg["rm"]
    self.rm_wg.init_model()
```

### 2.3 FSDP实现 (`verl/workers/fsdp_workers.py`)

```python
class FSDPRewardModel(FSDPWorkerBase):
    def _build_model(self, config):
        # 1. 下载模型到本地
        local_path = copy_to_local(config.model.path)

        # 2. 加载模型配置
        model_config = AutoConfig.from_pretrained(local_path)
        model_config.num_labels = 1  # 奖励模型输出单个分数

        # 3. 初始化模型
        reward_module = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=local_path,
            config=model_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # 4. 应用FSDP包装
        reward_module = FSDP(
            reward_module,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=sharding_strategy,
            cpu_offload=CPUOffload(offload_params=True),
            device_mesh=self.device_mesh,
        )

        return reward_module

    def compute_rm_score(self, data: DataProto):
        # 前向计算
        with torch.no_grad():
            output = self.reward_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            rm_score = output.logits.squeeze(-1)

        # 提取最后一个有效token的分数
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)
        rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]

        # 扩展为token级别奖励
        token_level_scores = self._expand_to_token_level(data, rm_score)

        return DataProto(batch={"rm_scores": token_level_scores})
```

## 3. 两种方式的使用场景

### 何时使用 Rule-based Reward

1. **验证阶段**：在`_validate()`中总是使用rule-based
   ```python
   if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
       return {}  # 跳过model-based验证
   ```

2. **训练阶段**：当没有配置reward model时
   ```python
   if self.use_rm and "rm_scores" not in batch.batch.keys():
       reward_tensor = self.rm_wg.compute_rm_score(batch)
   else:
       # 使用reward_fn（rule-based）
       reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
   ```

### 资源池配置

Model-based reward支持独立资源池：

```yaml
reward_model:
  enable: true
  enable_resource_pool: true  # 使用独立GPU资源
  n_gpus_per_node: 2
  nnodes: 1
```

## 4. 重要说明：数据中的style字段与self.use_rm的关系

### 4.1 self.use_rm的初始化机制

`self.use_rm` 的值完全由配置文件决定，与数据内容无关：

```python
# verl/trainer/ppo/utils.py
def need_reward_model(role_worker_mapping: dict[Role, WorkerType]) -> bool:
    """Given a role worker mapping, do we need reward model."""
    return Role.RewardModel in role_worker_mapping

# verl/trainer/main_ppo.py
# 只有在配置中启用奖励模型时才添加相应的工作节点
if config.reward_model.enable:
    self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
```

### 4.2 关键区别

1. **配置层面** (`config.reward_model.enable`)：
   - 决定是否初始化reward model worker
   - 决定 `self.use_rm` 的值
   - 在训练器初始化时确定，整个训练过程中不变

2. **数据层面** (`data["reward_model"]["style"]`)：
   - 标记每个样本应该使用哪种奖励计算方式
   - 可以是 "rule" 或 "model"
   - 在运行时使用，不影响初始化

### 4.3 实际影响

即使数据标记为 `{"style": "rule"}`：
- 如果 `config.reward_model.enable = True`，仍会初始化 reward model worker
- `self.use_rm = True`
- 但在验证时会跳过model-based样本的验证

```python
# 验证时的逻辑
if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
    return {}  # 跳过model-based验证，因为验证只支持rule-based
```

### 4.4 混合使用示例

配置文件：
```yaml
reward_model:
  enable: true  # 启用reward model，self.use_rm = True
```

数据文件可以包含混合样本：
```json
[
  {"prompt": "What is 2+2?", "reward_model": {"style": "rule", "ground_truth": "4"}},
  {"prompt": "Write a poem", "reward_model": {"style": "model"}}
]
```

在这种情况下：
- 两种样本都可以训练
- 只有第一个样本（rule-based）会参与验证
- reward model worker 始终被初始化（因为 enable=true）

## 5. 总结

| 特性 | Rule-based | Model-based |
|-----|------------|-------------|
| **初始化复杂度** | 简单 | 复杂（需要加载模型） |
| **计算速度** | 快 | 慢（需要神经网络前向） |
| **准确性** | 确定性（基于规则） | 学习性（基于训练） |
| **资源消耗** | CPU即可 | 需要GPU |
| **适用场景** | 数学题、编程题等有明确答案的任务 | 开放域对话、创意写作等主观任务 |
| **验证支持** | ✓ | ✗ |

### 最佳实践

1. **开发阶段**：先使用rule-based reward快速迭代
2. **优化阶段**：训练专门的reward model提升质量
3. **混合使用**：验证用rule-based，训练用model-based
4. **资源分配**：为reward model配置独立资源池避免竞争
5. **数据标记**：在数据中明确标记每个样本的reward计算方式（style字段）